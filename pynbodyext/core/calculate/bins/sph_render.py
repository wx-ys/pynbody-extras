"""SPH-smoothed views of a binned result: ``bins.sph_render["mass.sum"]``.

A normal query answers "what is in this cell" by counting the particles that fall
inside it.  The same query through :attr:`BinNDResult.sph_render` answers it by
*smearing* each particle with its SPH kernel instead, so a cell borrows from its
neighbours and the map is smooth where the strict binning was noisy::

    bins.sph_render["count"]  # kernel-integrated particle count
    bins.sph_render["mass.sum"]  # per-cell mass, smoothed
    bins.s.sph_render["vz.mean@mass"]  # stars only, mass-weighted mean vz

Semantics
---------
The query grammar does not change — ``"<field>.<transform>.<stat>@<weight>"``, as
parsed by :func:`~.statistics.parse_pipeline_key` — and neither does what the parts
mean.  Only cell membership changes: a particle at separation ``r`` from a cell
contributes with the weight

.. math::  w_{ij} = W(r_{ij}, h_i)\\, V_{\\rm cell}\\, [\\text{@weight}]

where ``W`` is pynbody's kernel (normalised so that ``∫ W dV = 1``), ``h_i`` is the
particle's smoothing length and ``V_cell`` the cell's volume (area, for a
two-dimensional projection).  The weights are dimensionless.  Two consequences of
that sum are worth knowing, because both differ from the strict query:

- **A render of a subregion is a genuine SPH estimate of that subregion**, so it
  includes the particles just outside the grid whose kernels reach in — the
  quantity in a cell is ``Σ`` over *neighbours*, not over the particles the grid
  happens to contain.  The totals agree with the particle number only when the
  grid contains every kernel.
- **The kernel is sampled at the cell centre, not integrated over the cell** (the
  same choice pynbody's own renderer makes), so the cells have to resolve it:
  with ``h`` of the order of a cell the totals come out right, and with a grid far
  coarser than the smoothing length they are inflated.  Raise ``smooth_floor`` or
  use a finer grid when that matters.

With those in mind:

- ``count`` is that sum — a *kernel-integrated* count, fractional by nature
  (``0.37`` particles may land in a cell), unlike ``bins["count"]``;
- ``sum`` is ``Σ_i w_ij f_i``, with the units of ``f`` (so
  ``sph_render["mass.sum"]`` is a mass per cell, like ``bins["mass.sum"]``);
- ``mean`` (and ``rms``, ``disp``) are the weighted versions, ``Σ w f / Σ w``,
  which also carries the units of ``f``; ``@mass`` weights them by mass exactly
  as it does for a strict query;
- ``transform``s act on the *particle* values before smoothing, so
  ``"vz.abs.mean@mass"`` is the mass-weighted mean of ``|vz|``.

Two engines share that query, because not every statistic is a sum, and only one of
them is on by default:

- **the kernel sums** — ``count``, ``sum`` and the mean-likes (``mean``, ``rms``,
  ``disp``) — accumulate over the whole kernel support, which is what pynbody's
  renderer does in C.  They are the default, and they are exact;
- **the quantiles** — ``median`` and the percentiles ``pXX`` — are weighted
  quantiles over **every particle the kernel reaches**, since a quantile is not
  linear in the weights and so cannot be a kernel sum.  They are built by
  *scattering*: each particle lays its kernel over the cells it touches — the same
  thing pynbody's renderer does internally, periodic images included — and the
  resulting ``(value, weight)`` pairs are reduced cell by cell with
  :func:`~.statistics.weighted_percentiles`, which is the strict query's own
  definition and takes a length per cell so nothing is padded.  A cell therefore
  sees exactly the neighbours its kernel sums see, and ``"vz.abs.p16@mass"`` is
  the mass-weighted 16th percentile of ``|vz|`` over all of them, while
  ``"vz.median"`` is the weighted median — which need not equal ``"vz.mean"``.
  (An earlier version looked at the *nearest* particles, capped at a fixed number;
  a real snapshot holds hundreds inside ``2h``, so that truncated the weight by up
  to 44% and moved medians by tens of km/s.  Scattering is why there is no such
  parameter now.)

  Scattering is not free — it must gather and sort every particle-cell pair, tens
  of times the cost of a kernel sum — so it is **opt-in**: ``SphRender(...,
  exact=True)``, or the :attr:`BinNDResult.sph_render_exact` shortcut.  With the
  default view a quantile query raises and says so, rather than being answered by
  something cheaper that means something else.  ``exact`` affects the quantiles
  only; the kernel sums are identical either way.

Requirements
------------
- **Two or three spatial axes.**  The bin axes must be ``x``, ``y`` and — for a
  three-dimensional result — ``z``, all distinct.  A radial or temperature axis
  has no SPH geometry to render onto.
- **Evenly spaced bins** (``mode="linear"``), because the renderer works on the
  cell grid: ``V_cell`` is a single number.  Anything else raises.
- **An SPH snapshot**: the particles need smoothing lengths, which pynbody
  derives from the particle distribution on demand as ``sim["smooth"]`` — so any
  snapshot with particles works, and setting ``smooth`` yourself overrides it.
  Zeros or non-finite values there are rejected rather than rendered.

``pynbody.sph`` is imported on first use, so importing the bins package stays
light.

Known upstream caveat
---------------------
pynbody's three-dimensional renderer sizes its z pixels with the *y* resolution
(``pixel_dz = (z2 - z1) / ny``), so a 3-D grid with ``nz != ny`` comes out
misplaced — and can lose contributions altogether.  That renderer is still the one
used here (its kernel, its wrapping and its conventions are what this view is meant
to reproduce), and a :class:`UserWarning` says so whenever ``nz != ny``; use equal
y and z bin counts until it is fixed upstream.
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .query import _wrap
from .statistics import (
    RMS,
    Dispersion,
    Mean,
    Median,
    Percentile,
    Sum,
    apply_pipeline,
    parse_pipeline_key,
    weighted_percentiles,
)

if TYPE_CHECKING:
    from pynbody.sph.kernels import KernelBase

    from .arrays import BinsArray
    from .result import BinNDResult

__all__ = ["BinSphRenderMixin", "SphRender"]

#: The spatial axes a render understands, in the order the renderers index them.
_SPATIAL = ("x", "y", "z")

#: Statistics built from one or two kernel sums (no neighbour lists needed).
_SUM_STATISTICS = (Sum, Mean, RMS, Dispersion)

#: Statistics that need the per-cell neighbour lists: a weighted quantile.
_QUANTILE_STATISTICS = (Median, Percentile)

#: Column of :attr:`BinResultModel.position` each spatial axis lives in.
_AXIS_COLUMN = {"x": 0, "y": 1, "z": 2}

#: Rows of cells the quantile engine scatters at a time.  Peak memory is set by the
#: particle-cell pairs inside one slab, so this trades passes over the particles
#: for memory.
_SLAB_ROWS = 32

#: Particle-cell pairs in one quantile before we warn about the cost.
_ENTRY_WARNING = 50_000_000


class SphRender:
    """Kernel-smoothed queries over a binned result.

    Reached as :attr:`BinNDResult.sph_render`; see the module docstring for what a
    query means here and which results support it.

    Parameters
    ----------
    result : BinNDResult
        The binned result whose particles are rendered.  Its ``model.sim`` is the
        particle set — a family sub-result (``bins.s``) renders only that family.
    kernel : str, pynbody.sph.kernels.KernelBase or None, optional
        SPH kernel, by pynbody name (``"CubicSpline"``, ``"WendlandC2"``) or as an
        instance; ``None`` takes pynbody's configured default.
    smooth_floor : float, default: 0.0
        Lower bound on the smoothing length, in the units of the axes.  Useful for
        a snapshot with a few very small ``h``; passed to the renderer unchanged.
    wrap : bool, default: True
        Whether to repeat particles across a periodic ``boxsize``, as pynbody's
        renderer does.  Turning it off restricts the render to the box.  The
        quantiles wrap the query into the box for the same reason (they search
        their own tree), so both engines agree across a boundary.
    exact : bool, default: False
        Whether the *quantiles* may use the exact scattering engine.  The kernel
        sums — ``count``, ``sum``, ``mean``, ``rms``, ``disp`` — never need it:
        pynbody's C renderer accumulates exactly those.  A quantile (``median``,
        ``pXX``) is not a kernel sum, so with ``exact=False`` it is refused rather
        than approximated; passing ``True`` builds each cell's complete set of
        ``(value, weight)`` pairs by scattering, which is exact and costs tens of
        times a kernel sum.

    Examples
    --------
    >>> bins.sph_render["mass.sum"]  # doctest: +SKIP
    >>> bins.s.sph_render["vz.mean@mass"]  # doctest: +SKIP
    """

    def __init__(
        self,
        result: BinNDResult,
        *,
        kernel: str | KernelBase | None = None,
        smooth_floor: float = 0.0,
        wrap: bool = True,
        exact: bool = False,
    ) -> None:
        self._result = result
        self._kernel_spec = kernel
        self._smooth_floor = float(smooth_floor)
        self._wrap = bool(wrap)
        self._exact = bool(exact)
        self._renders: dict[str, BinsArray] = {}
        self._plan: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"<SphRender {self._kernel_spec!r} of {self._result!r}>"

    # ------------------------------------------------------------------ query
    def __getitem__(self, key: str) -> BinsArray:
        """Render one query, e.g. ``"mass.sum"``, ``"count"``, ``"vz.mean@mass"``.

        Parameters
        ----------
        key : str
            A per-bin query in the usual grammar, or ``"count"`` for the
            kernel-integrated particle count.

        Returns
        -------
        BinsArray
            The smoothed values on the result's own bin grid, so ``.image``,
            ``.plot`` and the image layer work unchanged.
        """
        cached = self._renders.get(key)
        if cached is not None:
            return cached
        rendered = self._compute(key)
        self._renders[key] = rendered
        return rendered

    def _compute(self, key: str) -> BinsArray:
        sim = self._result.model.sim
        self._warn_about_z_resolution(self._layout())
        parsed = parse_pipeline_key(key)

        if parsed is None:
            if key != "count":
                raise KeyError(f"Unknown sph_render query {key!r}; use 'count' or a '<field>.<stat>' query.")
            statistic: Any = Sum("sum")
            units = None
            field = None
            values = np.ones(len(sim), dtype=float)
            weights = None
        else:
            field, transforms, statistic, weight = parsed
            if not isinstance(statistic, _SUM_STATISTICS + _QUANTILE_STATISTICS):
                raise NotImplementedError(
                    f"sph_render does not support the {statistic.key!r} statistic: it is neither a ratio "
                    "of kernel sums nor a weighted quantile. Use 'sum', 'mean', 'rms', 'disp', 'median' "
                    "or 'pXX'."
                )
            if isinstance(statistic, _QUANTILE_STATISTICS) and not self._exact:
                raise ValueError(
                    f"{statistic.key!r} is a weighted quantile, which no kernel sum can express, so it "
                    "needs the exact engine: construct the view with SphRender(..., exact=True), or call "
                    "bins.sph_render_exact[key].  The default reaches only the kernel sums (count, sum, "
                    "mean, rms, disp), which pynbody's renderer accumulates in C."
                )
            raw = sim[field]
            units = getattr(raw, "units", None)
            values = apply_pipeline(np.asarray(raw, dtype=float), transforms)
            weights = None if weight is None else np.asarray(sim[weight], dtype=float)

        out = np.asarray(self._statistic(statistic, values, weights), dtype=float)
        result = _wrap(self._result, out, name=key, field=field, mode=statistic.key)
        if units is not None:
            result.units = units
        return result

    def _statistic(self, statistic: Any, values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
        """One kernel sum for ``sum``/``count``, a ratio of two for the mean-likes."""
        if isinstance(statistic, _QUANTILE_STATISTICS):
            plan = self._layout()
            return self._to_axis_order(self._quantile(statistic, values, weights, plan), plan)
        weight = np.ones_like(values) if weights is None else weights
        if isinstance(statistic, Sum):
            return self._render(values * weight)
        total = self._render(weight)
        # Cells no kernel reaches come out NaN, as an empty bin does in a strict
        # query; dividing into them is expected, not a warning.
        with np.errstate(invalid="ignore", divide="ignore"):
            if isinstance(statistic, Mean):
                return self._render(values * weight) / total
            if isinstance(statistic, RMS):
                return np.sqrt(self._render(values * values * weight) / total)
            if isinstance(statistic, Dispersion):
                mean = self._render(values * weight) / total
                square = self._render(values * values * weight) / total
                return np.sqrt(np.clip(square - mean * mean, 0.0, None))
        raise TypeError(f"{statistic!r} has no kernel-weighted form.")

    def _quantile(
        self, statistic: Percentile | Median, values: np.ndarray, weights: np.ndarray | None, plan: dict[str, Any]
    ) -> np.ndarray:
        r"""A kernel-weighted quantile per cell, from every particle the kernel reaches.

        A quantile is not linear in the weights, so — unlike the kernel sums, which
        pynbody's renderer accumulates cell by cell on the fly — it needs each
        cell's *set* of ``(value, weight)`` pairs.  Those come from scattering: every
        particle lays its kernel over the cells it touches, exactly as the renderer
        does internally, its periodic images included, so a cell sees the same
        neighbours the kernel sums see and nothing is truncated.

        The pairs are reduced one *slab* of cells at a time, which bounds peak
        memory by the slab rather than the grid; within a slab they are sorted
        cell-major and handed to :func:`~.statistics.weighted_percentiles` with a
        length per cell, so no padding to the busiest cell is needed.  The
        definition is therefore the strict query's own, only smoothed:
        ``"vz.abs.p16@mass"`` is the mass-weighted 16th percentile of ``|vz|`` over
        everything the kernel reaches.
        """
        present = plan["present"]
        axes = plan["axes"]
        shape = tuple(axes[prop].nbins for prop in present)
        columns = [_AXIS_COLUMN[prop] for prop in present]
        centres = [_cell_centres(axes[prop]) for prop in present]
        origins = [float(axes[prop].edges[0]) for prop in present]
        widths = [plan["widths"][prop] for prop in present]
        strides = [int(np.prod(shape[index + 1 :], dtype=int)) for index in range(len(shape))]
        quantity = np.asarray(values, dtype=float)
        extra = np.ones_like(quantity) if weights is None else np.asarray(weights, dtype=float)

        particle, shift = self._images(plan)
        position = plan["position"][particle][:, columns] + shift
        smoothing = self._smoothing(plan["smooth"])[particle]
        support = 2.0 * smoothing
        value = quantity[particle]
        weight = extra[particle]
        kernel = self._kernel(projected=len(present) == 2)
        measure = plan["measure"]
        self._warn_if_expensive(position, support, widths, shape)

        out = np.empty(shape, dtype=float)
        row = strides[0]
        for low in range(0, shape[0], _SLAB_ROWS):
            high = min(low + _SLAB_ROWS, shape[0])
            cell, entry_value, entry_weight = _scatter(
                position,
                smoothing,
                support,
                value,
                weight,
                centres,
                origins,
                widths,
                strides,
                low,
                high,
                kernel,
                measure,
            )
            counts = np.bincount(cell, minlength=(high - low) * row)
            reduced = weighted_percentiles(entry_value, entry_weight, statistic.percentile, segments=counts)
            out[low:high] = np.asarray(reduced, dtype=float).reshape((high - low, *shape[1:]))
        return out

    def _images(self, plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Particle indices and per-axis shifts for every image that can reach the grid.

        A periodic box means a particle near one edge reaches cells at the other, so
        a handful of shifted copies are needed — the same repeats pynbody's renderer
        uses.  Only the particles an image brings within reach are kept, so the
        common case costs one pass and no copies.
        """
        present = plan["present"]
        columns = [_AXIS_COLUMN[prop] for prop in present]
        position = plan["position"][:, columns]
        support = 2.0 * self._smoothing(plan["smooth"])
        bounds = [(float(plan["axes"][prop].edges[0]), float(plan["axes"][prop].edges[-1])) for prop in present]
        parts: list[np.ndarray] = []
        shifts: list[np.ndarray] = []
        for shift in _offset_vectors(plan["boxsize"], [plan["axes"][prop] for prop in present]):
            reaches = np.ones(len(position), dtype=bool)
            for axis, (low, high) in enumerate(bounds):
                moved = position[:, axis] + shift[axis]
                reaches &= (moved + support > low) & (moved - support < high)
            if reaches.any():
                picked = np.flatnonzero(reaches)
                parts.append(picked)
                shifts.append(np.tile(shift, (len(picked), 1)))
        return np.concatenate(parts), np.concatenate(shifts)

    def _warn_if_expensive(
        self, position: np.ndarray, support: np.ndarray, widths: list[float], shape: tuple[int, ...]
    ) -> None:
        """Warn when a scatter would touch an unreasonable number of pairs."""
        reach = np.ones(len(position), dtype=float)
        for axis, width in enumerate(widths):
            reach *= np.clip(np.ceil(2.0 * support / width) + 1.0, 0.0, float(shape[axis]))
        entries = float(reach.sum())
        if entries > _ENTRY_WARNING:
            warnings.warn(
                f"this sph_render scatters about {entries / 1e6:.0f} million particle-cell pairs: the smoothing "
                "lengths are large next to the cells. Use smooth_floor, or a coarser grid, to bound it.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------ plan
    def _layout(self) -> dict[str, Any]:
        """Validate the result once, and cache what a render needs from it."""
        if self._plan is not None:
            return self._plan

        model = self._result.model
        axes = model.axes
        props = [str(axis.prop) for axis in axes]
        if len(axes) not in (2, 3):
            raise ValueError(
                f"sph_render needs 2 or 3 axes (a plane or a volume); this result has {len(axes)} ({props})."
            )
        if any(prop not in _SPATIAL for prop in props) or len(set(props)) != len(props):
            raise ValueError(
                f"sph_render renders onto the sky/volume, so its axes must be distinct spatial ones; "
                f"this result has {props} — rebuild the bins from x/y/z."
            )
        for axis in axes:
            if not _is_uniform(axis):
                label = axis.alias or axis.prop
                raise ValueError(
                    f"sph_render needs evenly spaced bins: axis {label!r} has bins of unequal width. Use mode='linear'."
                )

        sim = model.sim
        try:
            smooth = np.asarray(sim["smooth"], dtype=float)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "sph_render needs SPH smoothing lengths, and this particle set has no usable 'smooth' array: "
                "only some families carry one (gas does, the collisionless ones do not), so render a family "
                "or sub-result that does — bins.gas.sph_render[...] — or give a single-family binning."
            ) from exc
        if not np.all(np.isfinite(smooth)) or np.any(smooth <= 0.0):
            raise ValueError(
                "sph_render needs finite, positive smoothing lengths; this snapshot's 'smooth' array "
                "has zeros or non-finite values."
            )

        by_prop = {str(axis.prop): axis for axis in axes}
        widths = {prop: float(axis.edges[1] - axis.edges[0]) for prop, axis in by_prop.items()}
        present = [prop for prop in _SPATIAL if prop in by_prop]
        measure = float(np.prod([widths[prop] for prop in present]))

        self._plan = {
            "axes": by_prop,
            "present": present,
            "widths": widths,
            "measure": measure,
            "position": _position_array(sim, by_prop),
            "smooth": smooth,
            "boxsize": _boxsize(sim) if self._wrap else None,
        }
        return self._plan

    def _kernel(self, *, projected: bool) -> Any:
        from pynbody.sph import kernels

        kernel = kernels.create_kernel(self._kernel_spec)
        return kernel.projection() if projected else kernel

    def _smoothing(self, smooth: np.ndarray) -> np.ndarray:
        return np.maximum(smooth, self._smooth_floor)

    # ----------------------------------------------------------------- render
    def _render(self, quantity: np.ndarray) -> np.ndarray:
        """Kernel sum of *quantity* over the cells, in the result's axis order."""
        return self._to_axis_order(self._render_cells(quantity), self._layout())

    def _render_cells(self, quantity: np.ndarray) -> np.ndarray:
        """Kernel sum of *quantity*, in the order the cell grid is indexed here."""
        plan = self._layout()
        if len(plan["present"]) == 2:
            return self._render_projected(quantity, plan)
        return self._render_volume(quantity, plan)

    def _to_axis_order(self, rendered: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
        """Reorder an ``(x, y[, z])`` grid into the result's own axis order."""
        order = [plan["present"].index(str(axis.prop)) for axis in self._result.model.axes]
        return np.transpose(rendered, np.argsort(order))

    def _render_projected(self, quantity: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
        r"""Project along the missing axis with pynbody's C renderer.

        The column render uses pynbody's projected kernel (``∫W dz``), so the
        quantity pynbody accumulates is ``Σ_i q_i W_2D(r_ij, h_i)`` — the same
        convention pynbody uses for its own images, with ``mass``/``rho`` set to
        one so that nothing is weighted twice.  Multiplying by the cell area makes
        the weight dimensionless, and by the cell measure we keep the units of the
        strict query rather than pynbody's per-area ones.
        """
        from pynbody.sph import _render

        first, second = plan["present"]
        coord = {"x": 0, "y": 1, "z": 2}
        position = plan["position"]
        smooth = self._smoothing(plan["smooth"])
        first_axis, second_axis = plan["axes"][first], plan["axes"][second]
        ones = np.ones_like(quantity)

        image = _render.render_image(
            first_axis.nbins,
            second_axis.nbins,
            position[:, coord[first]],
            position[:, coord[second]],
            position[:, 2],
            smooth,
            float(first_axis.edges[0]),
            float(first_axis.edges[-1]),
            float(second_axis.edges[0]),
            float(second_axis.edges[-1]),
            0.0,
            0.0,
            quantity,
            ones,
            ones,
            0.0,
            np.inf,
            -np.inf,
            np.inf,
            self._smooth_floor,
            self._kernel(projected=True),
            self._wrap_offsets(first_axis, plan["boxsize"]),
            self._wrap_offsets(second_axis, plan["boxsize"]),
        )
        # render_image returns (second, first); the result indexes (first, second).
        return np.asarray(image).T * plan["measure"]

    def _render_volume(self, quantity: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
        r"""Render onto the 3-D cell grid with pynbody's own renderer.

        Same convention as the projected path — pynbody's kernel, ``mass``/``rho``
        set to one so nothing is weighted twice, multiplied by the cell volume to
        keep the units of the strict query.  pynbody's
        :func:`~pynbody.sph._render.to_3d_grid` sizes its z pixels with the *y*
        resolution (``pixel_dz = (z2 - z1) / ny``, `_render.pyx`), so anything but
        ``nz == ny`` is suspect; :meth:`_warn_about_z_resolution` says so rather
        than quietly returning a different map.
        """
        from pynbody.sph import _render

        axes = plan["axes"]
        position = plan["position"]
        smooth = self._smoothing(plan["smooth"])
        ones = np.ones_like(quantity)
        bounds = {prop: (float(axes[prop].edges[0]), float(axes[prop].edges[-1])) for prop in _SPATIAL}

        volume = _render.to_3d_grid(
            axes["x"].nbins,
            axes["y"].nbins,
            axes["z"].nbins,
            position[:, 0],
            position[:, 1],
            position[:, 2],
            smooth,
            bounds["x"][0],
            bounds["x"][1],
            bounds["y"][0],
            bounds["y"][1],
            bounds["z"][0],
            bounds["z"][1],
            quantity,
            ones,
            ones,
            0.0,
            np.inf,
            self._kernel(projected=False),
            self._wrap_offsets(axes["x"], plan["boxsize"]),
            self._wrap_offsets(axes["y"], plan["boxsize"]),
            self._wrap_offsets(axes["z"], plan["boxsize"]),
        )
        return np.asarray(volume) * plan["measure"]

    def _warn_about_z_resolution(self, plan: dict[str, Any]) -> None:
        """Flag a 3-D render that hits pynbody's z-pixel bug.

        ``to_3d_grid`` divides the z range by ``ny`` instead of ``nz``, so the z
        cells it uses are not the cells asked for unless the two happen to match:
        values are misplaced, and with small smoothing lengths they can vanish
        altogether.  Warning beats silently returning a different map; the fix is
        upstream, and until then equal y/z resolutions are the safe case.
        """
        if len(plan["present"]) < 3:
            return
        y_cells, z_cells = plan["axes"]["y"].nbins, plan["axes"]["z"].nbins
        if y_cells == z_cells:
            return
        warnings.warn(
            f"pynbody's 3-D renderer sizes its z pixels with the y resolution, so a grid with "
            f"nz={z_cells} != ny={y_cells} renders misplaced cells (and can lose contributions "
            "entirely). Use the same number of bins on y and z, or a 2-D projection, for now.",
            UserWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------ wrap
    def _wrap_offsets(self, axis: Any, boxsize: float | None) -> list[float]:
        """The offsets along one axis, as pynbody's renderer builds them."""
        if boxsize is None:
            return [0.0]
        span = float(axis.edges[-1] - axis.edges[0])
        repeats = int(round(span / (2.0 * boxsize))) + 1
        return list(np.linspace(-repeats * boxsize, repeats * boxsize, 2 * repeats + 1))


class BinSphRenderMixin:
    """Provides the :attr:`sph_render` view for :class:`~.result.BinNDResult`."""

    @property
    def sph_render(self) -> SphRender:
        """Kernel-smoothed queries over this result: ``bins.sph_render["mass.sum"]``.

        The same queries as ``bins[...]``, with particles smeared by their SPH
        kernel instead of counted in their cell: cells borrow from their
        neighbours, so a noisy map comes out smooth and a sparse one keeps its
        neighbours' signal.  ``bins.s.sph_render[...]`` does it for one family or
        sub-result.  The kernel sums (``count``, ``sum``, ``mean``, ``rms``,
        ``disp``) come from pynbody's renderer; the quantiles (``median``,
        ``pXX``) need :attr:`sph_render_exact`, since no kernel sum can express
        them.

        Returns
        -------
        SphRender
            A view; index it with a query such as ``"mass.sum"``, ``"count"`` or
            ``"vz.mean@mass"``.  See :class:`SphRender` for what each part means
            and which results can be rendered.

        Raises
        ------
        ValueError
            If the result is not two or three spatial axes with evenly spaced
            bins, or the snapshot has no smoothing lengths, or a quantile is asked
            for here rather than from :attr:`sph_render_exact`.  All raised on the
            first query, not on this property.

        Examples
        --------
        >>> bins.sph_render["count"]  # doctest: +SKIP
        >>> bins.s.sph_render["vz.mean"]  # doctest: +SKIP
        """
        render = self.__dict__.get("_sph_render")
        if render is None:
            render = SphRender(cast("BinNDResult", self))
            self._sph_render = render
        return render

    @property
    def sph_render_exact(self) -> SphRender:
        """As :attr:`sph_render`, but with the exact engine for quantiles enabled.

        ``bins.sph_render["mass.sum"]`` accumulates with pynbody's renderer, which
        is what a kernel sum is; ``bins.sph_render_exact["vz.median"]`` additionally
        allows the quantiles, which no kernel sum can express and which therefore
        have to scatter every particle over the cells its kernel reaches —
        ``SphRender(..., exact=True)`` under a shorter name.

        Returns
        -------
        SphRender
            The same view, with ``exact=True``.

        Examples
        --------
        >>> bins.sph_render_exact["vz.median"]  # doctest: +SKIP
        """
        render = self.__dict__.get("_sph_render_exact")
        if render is None:
            render = SphRender(cast("BinNDResult", self), exact=True)
            self._sph_render_exact = render
        return render


def _is_uniform(axis: Any) -> bool:
    edges = np.asarray(axis.edges, dtype=float)
    return bool(np.allclose(np.diff(edges), np.diff(edges)[0]))


def _cell_centres(axis: Any) -> np.ndarray:
    edges = np.asarray(axis.edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _offset_vectors(boxsize: float | None, axes: list[Any]) -> list[tuple[float, ...]]:
    """Every combination of per-axis periodic shifts that could matter.

    pynbody's renderer repeats particles by whole boxes when a snapshot is periodic
    (``_calculate_wrapping_repeat_array``); the same repeats are needed here, and
    ``boxsize is None`` means the single, unshifted case.
    """
    if boxsize is None:
        return [tuple(0.0 for _ in axes)]
    per_axis = []
    for axis in axes:
        span = float(axis.edges[-1] - axis.edges[0])
        repeats = int(round(span / (2.0 * boxsize))) + 1
        per_axis.append(np.linspace(-repeats * boxsize, repeats * boxsize, 2 * repeats + 1))
    return [tuple(float(component) for component in combination) for combination in itertools.product(*per_axis)]


def _scatter(
    position: np.ndarray,
    smoothing: np.ndarray,
    support: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    centres: list[np.ndarray],
    origins: list[float],
    widths: list[float],
    strides: list[int],
    low_row: int,
    high_row: int,
    kernel: Any,
    measure: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``(cell, value, weight)`` inside one slab of rows.

    Each particle's cell range comes from its own support (``2h``) and the uniform
    grid, so the block of cells one particle touches is a rectangular range;
    the ranges of all particles are then expanded into flat arrays at once
    (``np.repeat`` and a mixed-radix decomposition) rather than in a Python loop.
    The result is sorted cell-major and value-sorted within each cell, ready for
    :func:`~.statistics.weighted_percentiles` with a length per cell.
    """
    dimensions = len(centres)
    ranges: list[tuple[np.ndarray, np.ndarray]] = []
    for axis in range(dimensions):
        low = np.floor((position[:, axis] - support - origins[axis]) / widths[axis])
        high = np.ceil((position[:, axis] + support - origins[axis]) / widths[axis])
        # Clip to the *grid*, so the cell centres are the real ones; the slab is
        # accounted for when the row index is turned into a cell index below.
        lower_bound = low_row if axis == 0 else 0
        upper_bound = high_row if axis == 0 else len(centres[axis])
        low = np.clip(low, lower_bound, upper_bound)
        high = np.clip(high, lower_bound, upper_bound)
        ranges.append((low.astype(np.intp), high.astype(np.intp)))

    sizes = [high - low for low, high in ranges]
    block = np.prod(np.stack(sizes), axis=0)
    picked = np.flatnonzero(block > 0)
    if not len(picked):
        return (np.empty(0, dtype=np.intp), np.empty(0), np.empty(0))

    per_particle = block[picked]
    total = int(per_particle.sum())
    particle = np.repeat(picked, per_particle)
    within = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(per_particle) - per_particle, per_particle)

    offsets = []
    radix = np.ones(total, dtype=np.intp)
    for axis in reversed(range(dimensions)):
        size = sizes[axis][particle]
        offsets.append((axis, (within // radix) % size))
        radix = radix * size

    cell = np.zeros(total, dtype=np.intp)
    distance = np.zeros(total, dtype=float)
    for axis, offset in offsets:
        index = ranges[axis][0][particle] + offset
        if axis == 0:
            cell += (index - low_row) * strides[axis]
        else:
            cell += index * strides[axis]
        displacement = centres[axis][index] - position[particle, axis]
        distance += displacement**2

    entry_weight = _kernel_weights(np.sqrt(distance), smoothing[particle], kernel) * measure * weight[particle]
    entry_value = value[particle]
    order = np.lexsort((entry_value, cell))
    return cell[order], entry_value[order], entry_weight[order]


def _kernel_weights(distance: np.ndarray, smoothing: np.ndarray, kernel: Any) -> np.ndarray:
    r"""``W(distance, h)`` for a grid of separations and a matching grid of ``h``.

    A three-dimensional kernel evaluates its own vectorised ``value``; a projected
    one (``Kernel2D``, used for the two-dimensional render) only has the scalar
    numerical quadrature pynbody builds its lookup table from, so the table is
    indexed exactly as the renderer indexes it — the same table, the same lookup —
    so a quantile weighs its neighbours with the very weights the kernel sums use.
    """
    if getattr(kernel, "h_power", 3) != 2:
        return np.asarray(kernel.value(distance, smoothing), dtype=float)
    # pynbody caches the sample table per kernel and *not* per dtype, and its C
    # renderer asks for it as float32 — so ask for the same thing it will, or a
    # later kernel sum trips over a float64 table.
    samples = np.asarray(kernel.get_samples(), dtype=float)
    # ``_render.get_kernel``: index = num_samples * d**2 / (2h)**2, and zero beyond.
    index = ((distance / smoothing) ** 2 / 4.0 * len(samples)).astype(np.intp)
    inside = index < len(samples)
    return np.where(inside, samples[np.clip(index, 0, len(samples) - 1)], 0.0) / smoothing**2


def _position_array(sim: Any, axes: dict[str, Any]) -> np.ndarray:
    """The particle positions, in the axes' own units."""
    position = sim["pos"]
    units = getattr(position, "units", None)
    axis_units = getattr(axes["x"], "units", None)
    if units is not None and axis_units is not None and units != axis_units:
        position = position.in_units(axis_units)
    return np.asarray(position, dtype=float)


def _boxsize(sim: Any) -> float | None:
    """The periodic box, in position units, or ``None`` when there is none."""
    boxsize = sim.properties.get("boxsize") if hasattr(sim, "properties") else None
    if boxsize is None:
        return None
    position_units = getattr(sim["pos"], "units", None)
    if position_units is not None and getattr(boxsize, "units", None) is not None:
        boxsize = boxsize.in_units(position_units, **sim.conversion_context())
    return float(boxsize)
