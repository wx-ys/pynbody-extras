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

What a strict query cannot do, this cannot do either without a different engine:
``median`` and the percentiles ``pXX`` are not implemented yet (they need the
per-cell neighbour lists rather than a pair of kernel sums).

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
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .query import _wrap
from .statistics import RMS, Dispersion, Mean, Sum, apply_pipeline, parse_pipeline_key

if TYPE_CHECKING:
    from pynbody.sph.kernels import KernelBase

    from .arrays import BinsArray
    from .result import BinNDResult

__all__ = ["BinSphRenderMixin", "SphRender"]

#: The spatial axes a render understands, in the order the renderers index them.
_SPATIAL = ("x", "y", "z")

#: Statistics built from one or two kernel sums (no neighbour lists needed).
_SUM_STATISTICS = (Sum, Mean, RMS, Dispersion)

#: Particle-cell pairs in one render before we warn about the cost.
_ENTRY_WARNING = 20_000_000


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
        renderer does.  Turning it off restricts the render to the box.

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
    ) -> None:
        self._result = result
        self._kernel_spec = kernel
        self._smooth_floor = float(smooth_floor)
        self._wrap = bool(wrap)
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
            if not isinstance(statistic, _SUM_STATISTICS):
                raise NotImplementedError(
                    f"sph_render does not support {statistic.key!r} yet: a kernel-weighted quantile needs "
                    "the per-cell neighbour lists rather than a pair of kernel sums. "
                    "Use 'sum', 'mean', 'rms' or 'disp'."
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
        """One kernel sum for ``sum``/``count``, a ratio of two for the rest."""
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
                "sph_render needs SPH smoothing lengths, and this snapshot has no usable 'smooth' array. "
                "pynbody derives them from the particle distribution, so this usually means the snapshot "
                "has no particles to smooth over."
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
        plan = self._layout()
        if len(plan["present"]) == 2:
            rendered = self._render_projected(quantity, plan)
        else:
            rendered = self._render_volume(quantity, plan)
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
        r"""Render onto the 3-D cell grid ourselves.

        pynbody's :func:`~pynbody.sph._render.to_3d_grid` sizes the z pixels with
        the *y* resolution (``pixel_dz = (z2 - z1) / ny``), which misplaces — and
        for unequal ``ny``/``nz`` silently discards — every contribution, so the
        direct sum is done here instead.  The kernel is pynbody's, so the values
        agree with pynbody wherever its renderer is correct (``ny == nz``).
        """
        kernel = self._kernel(projected=False)
        position = plan["position"]
        smooth = self._smoothing(plan["smooth"])
        measure = plan["measure"]
        axes = plan["axes"]
        centres = {prop: 0.5 * (axes[prop].edges[:-1] + axes[prop].edges[1:]) for prop in _SPATIAL}
        shape = tuple(axes[prop].nbins for prop in _SPATIAL)
        out = np.zeros(shape, dtype=float)

        self._warn_if_expensive(smooth, plan)
        for index in np.nonzero(quantity)[0]:
            for offset in self._offset_vectors(plan["boxsize"], plan):
                atom = position[index] + offset
                ranges = [
                    _cell_range(centres[prop], axes[prop].edges[0], plan["widths"][prop], atom[i], 2.0 * smooth[index])
                    for i, prop in enumerate(_SPATIAL)
                ]
                if any(start >= stop for start, stop in ranges):
                    continue
                slices = tuple(slice(start, stop) for start, stop in ranges)
                delta = [
                    centres[prop][start:stop] - atom[i]
                    for i, (prop, (start, stop)) in enumerate(zip(_SPATIAL, ranges, strict=True))
                ]
                distance = np.sqrt(
                    delta[0][:, None, None] ** 2 + delta[1][None, :, None] ** 2 + delta[2][None, None, :] ** 2
                )
                out[slices] += quantity[index] * measure * kernel.value(distance, smooth[index])
        return out

    def _warn_if_expensive(self, smooth: np.ndarray, plan: dict[str, Any]) -> None:
        reach = np.prod([2.0 * smooth / plan["widths"][prop] + 1.0 for prop in _SPATIAL], axis=0)
        entries = float(np.sum(reach)) * self._offset_count(plan["boxsize"], plan)
        if entries > _ENTRY_WARNING:
            warnings.warn(
                f"this sph_render touches about {entries / 1e6:.0f} million particle-cell pairs "
                "(the smoothing lengths are large next to the cells); consider smoothing_floor, "
                "a coarser grid or smooth_floor to bound it.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ wrap
    def _offset_vectors(self, boxsize: float | None, plan: dict[str, Any]) -> list[tuple[float, float, float]]:
        per_axis = [self._wrap_offsets(plan["axes"][prop], boxsize) for prop in _SPATIAL]
        return [(x, y, z) for x in per_axis[0] for y in per_axis[1] for z in per_axis[2]]

    def _offset_count(self, boxsize: float | None, plan: dict[str, Any]) -> int:
        return int(np.prod([len(self._wrap_offsets(plan["axes"][prop], boxsize)) for prop in _SPATIAL]))

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
        sub-result.

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
            bins, or the snapshot has no smoothing lengths.  Raised on the first
            query, not here.

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


def _is_uniform(axis: Any) -> bool:
    edges = np.asarray(axis.edges, dtype=float)
    return bool(np.allclose(np.diff(edges), np.diff(edges)[0]))


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


def _cell_range(centres: np.ndarray, start: float, width: float, coordinate: float, reach: float) -> tuple[int, int]:
    """The cells of one axis whose centres are within *reach* of *coordinate*."""
    low = int(np.floor((coordinate - reach - start) / width))
    high = int(np.ceil((coordinate + reach - start) / width))
    return max(low, 0), min(high, len(centres))
