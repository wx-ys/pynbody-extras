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

The statistic picks the engine, so there is no mode to remember:

- **the kernel sums** — ``count``, ``sum`` and the mean-likes (``mean``, ``rms``,
  ``disp``) — accumulate over the whole kernel support, which is what pynbody's
  renderer does in C.  They are exact and cheap;
- **the quantiles** — ``median`` and the percentiles ``pXX`` — are weighted
  quantiles over **every particle the kernel reaches**, since a quantile is not
  linear in the weights and so cannot be a kernel sum.  They are built by
  *scattering*: each particle lays its kernel over the cells it touches — the same
  thing pynbody's renderer does internally, periodic images included — and the
  resulting ``(value, weight)`` pairs are reduced by
  :func:`~.statistics.bucketed_weighted_percentiles`, which is the strict query's
  own definition: it counts the weights into value bins, finds the bin the
  percentile falls in and orders only the entries there, so a cell's answer is what
  ``bins[...]`` would give for the same neighbours, smoothed.  A cell therefore
  sees exactly the neighbours its kernel sums see, and ``"vz.abs.p16@mass"`` is
  the mass-weighted 16th percentile of ``|vz|`` over all of them, while
  ``"vz.median"`` is the weighted median — which need not equal ``"vz.mean"``.
  (An earlier version looked at the *nearest* particles, capped at a fixed number;
  a real snapshot holds hundreds inside ``2h``, so that truncated the weight by up
  to 44% and moved medians by tens of km/s.  Scattering is why there is no such
  parameter now.)

  The pair loop itself is ``cpp/image/scatter.cpp`` — pynbody's ``result[...] +=
  ...`` walk writing ``(cell, value, weight)`` records rather than sums, the
  kernel lookup included — with a NumPy fallback that weighs the pairs the same
  way.  Building and weighing the million pairs of a 32² grid off 200k particles
  costs about what a kernel sum costs (23 ms against 12 ms); the reduction is what
  makes a quantile cost about a dozen kernel sums (142 ms).  That is the price of
  the definition rather than a mode: ``bins.sph_render["vz.median"]`` scatters
  because a median has to, and ``bins.sph_render["vz.mean"]`` does not.  A scatter
  that would touch an unreasonable number of pairs warns instead of being refused
  — see ``smooth_floor`` below, which is what bounds it.

Derived quantities
------------------
The view answers the result's *derived* properties too, wherever smoothing leaves
their meaning alone:

- ``<field>.density`` — the strict suffix with the smoothed numerator, so
  ``"mass.sum.density"`` is a kernel-smoothed mass per cell over that same cell
  measure, in the units ``bins["mass.sum.density"]`` carries (and ``"count.density"``,
  ``"mass.density"`` and ``"density"`` work as they do on the result);
- a property registered ``allow_sph=True`` — its own callback runs again, against
  the queries above, so there is no second definition of it to keep in step.
  ``gas_fraction`` and ``number_density`` ship that way.  The flag belongs on a
  per-cell function of what the callback reads: a ratio of two ``mass.sum`` maps
  becomes the ratio of two SPH maps, but a cumulative sum or a statistic of the
  strict particle census would mean something else once cells borrow from their
  neighbours;
- geometry (``measure``) — the grid's own, identical either way.

A derived property that is not declared sph-renderable is refused by name, rather
than answered with the strict map and left to look smoothed.

Relationship to pynbody's other renderers
-----------------------------------------
The kernel sums call pynbody's own ``_render.render_image`` and
``_render.to_3d_grid`` — the loops its ``ImageRenderer`` and ``Grid3dRenderer``
drive — with pynbody's kernels, and for a projection pynbody's
``KernelBase.projection`` (the ``∫W dz`` two-dimensional kernel).  The high-level
classes are deliberately *not* used: ``ImageRenderer.render`` reads ``mass``,
``rho``, ``x``, ``y``, ``z`` and ``smooth`` off the snapshot and accumulates
``Σ q_i W_i m_i/ρ_i``, whereas this view is defined to keep the strict query's
units — ``Σ q_i W_i V_cell``, which is what passing ``mass = rho = 1`` gives.  The
renderer API offers no way to override ``mass``/``rho``, and computing ``rho`` is
not free: a 32² image of the gas in ``testdata/gadget2`` costs 65 ms that way
against 2 ms here, most of it the density.  ``make_render_pipeline`` has the same
convention, and a mixed particle set fails in pynbody's renderer too — on ``rho``
rather than ``smooth`` (``KeyError: Block rho is not available for all
families``).

Requirements
------------
- **Two or three spatial axes.**  The bin axes must be ``x``, ``y`` and — for a
  three-dimensional result — ``z``, all distinct.  A radial or temperature axis
  has no SPH geometry to render onto.
- **Evenly spaced bins** (``mode="linear"``), because the renderer works on the
  cell grid: ``V_cell`` is a single number.  Anything else raises.
- **An SPH snapshot.**  Every particle needs a smoothing length.  By default it is
  read as ``sim["smooth"]``, which pynbody fills in from the particle distribution
  when a *single family* has none on disk — so ``bins.gas.sph_render[...]`` works
  on any snapshot — but a *mixed* set must carry one for every family it contains
  and says so when it cannot.  ``bins.sph_render(smooth="kdtree")`` derives one
  for the whole set instead, which is the only way to render a mixed set.  Zeros
  or non-finite values are rejected rather than rendered.

``pynbody.sph`` is imported on first use, so importing the bins package stays
light.
"""

from __future__ import annotations

import itertools
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .model import BinResultModel
from .query import DensityResolver, QueryCache, _wrap, evaluate_derived
from .statistics import (
    RMS,
    Dispersion,
    Mean,
    Median,
    Percentile,
    Sum,
    apply_pipeline,
    bucketed_weighted_percentiles,
    parse_pipeline_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pynbody.sph.kernels import KernelBase

    from .arrays import BinsArray
    from .axes import BinDerivedSpec
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

#: Where a render gets its smoothing lengths: the snapshot's own array, or a
#: derivation over the whole particle set.  See :meth:`SphRender._smooth_array`.
_SMOOTH_SOURCES = ("snapshot", "kdtree")

#: Sentinel for :meth:`SphRender.__call__`, meaning "keep this view's setting".
_INHERIT = object()

#: Buckets (cells x value bins) the quantile reduction may allocate at once.  The
#: bin count follows from the slab; it affects only speed, never the answer.
_BUCKET_BUDGET = 4_000_000

#: Bounds on that bin count, for a slab of very many or very few cells.
_MIN_BUCKETS = 64
_MAX_BUCKETS = 4096


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
        quantiles scatter the same periodic images, so both engines agree across
        a boundary.
    smooth : {"snapshot", "kdtree"}, default: "snapshot"
        Where the smoothing lengths come from.  ``"snapshot"`` reads
        ``sim["smooth"]``: on-disk values where the snapshot has them, and
        pynbody's k-d tree estimate for a single family that has none.  A *mixed*
        particle set (gas plus collisionless families) cannot be read that way,
        because pynbody refuses a block only some families carry; ``"kdtree"``
        derives a length for every particle with :func:`pynbody.sph.smooth`
        instead — the run pynbody's own ``rho`` uses, and the only way to render
        such a set.  It disregards on-disk values, so the numbers differ from
        ``"snapshot"`` wherever the snapshot has them.

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
        smooth: str = "snapshot",
    ) -> None:
        if smooth not in _SMOOTH_SOURCES:
            raise ValueError(f"smooth must be one of {list(_SMOOTH_SOURCES)}; got {smooth!r}.")
        self._result = result
        self._kernel_spec = kernel
        self._smooth_floor = float(smooth_floor)
        self._wrap = bool(wrap)
        self._smooth_source = smooth
        self._renders: dict[str, BinsArray] = {}
        self._plan: dict[str, Any] | None = None
        # The strict query's density definition, reading its numerator from here.
        self._density_resolver = DensityResolver(result, QueryCache(), self.__getitem__)

    def __repr__(self) -> str:
        return f"<SphRender {self._kernel_spec!r} of {self._result!r}>"

    def __call__(
        self, *, kernel: Any = _INHERIT, smooth_floor: Any = _INHERIT, wrap: Any = _INHERIT, smooth: Any = _INHERIT
    ) -> SphRender:
        """This view with some settings changed: ``bins.sph_render(smooth="kdtree")``.

        The view is what a result hands out, so this is how the constructor's
        options are reached from the query spelling.  Anything left out keeps this
        view's value, and the result caches the answer — so
        ``bins.sph_render(smooth="kdtree")`` is one object, whose renders are cached
        with it, however many times it is asked for.

        Parameters
        ----------
        kernel, smooth_floor, wrap, smooth : optional
            As in :class:`SphRender`.  ``smooth="kdtree"`` is the one that changes
            what a render can be answered at all: it derives a smoothing length for
            every particle, which a *mixed* particle set needs.

        Returns
        -------
        SphRender
            A view over the same result, configured as asked.

        Examples
        --------
        >>> bins.sph_render(smooth="kdtree")["gas_fraction"]  # doctest: +SKIP
        >>> bins.sph_render(smooth_floor=1.0, kernel="WendlandC2")["count"]  # doctest: +SKIP
        """
        return self._result._sph_render_view(
            kernel=self._kernel_spec if kernel is _INHERIT else kernel,
            smooth_floor=self._smooth_floor if smooth_floor is _INHERIT else smooth_floor,
            wrap=self._wrap if wrap is _INHERIT else wrap,
            smooth=self._smooth_source if smooth is _INHERIT else smooth,
        )

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
            ``.plot`` and the image layer work unchanged.  Besides the per-particle
            queries this answers ``"<field>.density"`` and any derived property
            registered ``allow_sph=True`` — the callback runs again against the
            smoothed queries, so e.g. ``"gas_fraction"`` is the ratio of the two
            *kernel-integrated* masses.  See the module docstring.
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
                return self._derived(key)
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
            raw = sim[field]
            units = getattr(raw, "units", None)
            values = apply_pipeline(np.asarray(raw, dtype=float), transforms)
            weights = None if weight is None else np.asarray(sim[weight], dtype=float)

        out = np.asarray(self._statistic(statistic, values, weights), dtype=float)
        result = _wrap(self._result, out, name=key, field=field, mode=statistic.key)
        if units is not None:
            result.units = units
        return result

    # ---------------------------------------------------------------- derived
    def _derived(self, key: str) -> BinsArray:
        """Answer a key that is not a per-particle query.

        Three kinds reach here.  ``<field>.density`` divides a smoothed map by the
        cell measure, because a density is a per-cell ratio and the grid is the
        one the strict query uses.  A property registered ``allow_sph=True`` runs
        its own callback — the very one the strict result runs — against a model
        whose queries come back smoothed.  Geometry (``measure``) is the grid's
        own, identical either way, so it is passed through unchanged.
        """
        base = DensityResolver.strip(key)
        if base is not None or key == "density":
            return self._density(key, key if base is None else base)

        spec = type(self._result)._extensions.get_derived_spec(self._result, key)
        if spec is None:
            raise KeyError(
                f"Unknown sph_render query {key!r}; use 'count', a '<field>.<stat>' query, a "
                "'<field>.density' query, or a derived property the result declares allow_sph=True."
            )
        if spec.allow_sph:
            return self._render_derived(spec)
        if spec.scope == "geometry":
            return self._result[key]
        raise KeyError(
            f"{key!r} is a derived property that is not declared sph-renderable.  Register it with "
            "allow_sph=True if it is a per-cell function of the quantities it reads; a smoothed "
            "version of it would otherwise mean something else."
        )

    def _density(self, key: str, base: str) -> BinsArray:
        """``<field>.density``: the smoothed map divided by the cell measure.

        Reuses :class:`~.query.DensityResolver` — the strict query's own definition
        of the suffix — with this view as its numerator source, so ``"mass.density"``,
        ``"mass.sum.density"`` and ``"count.density"`` behave exactly as they do on
        the result, only smoothed.
        """
        return self._density_resolver.resolve(key, base)

    def _render_derived(self, spec: BinDerivedSpec) -> BinsArray:
        """Run a derived callback against this view's model."""
        return evaluate_derived(self._result, spec, _SphModel(self._result.model, self))

    def _sibling(self, model: BinResultModel) -> SphRender:
        """A view of a sub-result, with this view's own settings."""
        owner = model.owner
        if owner is None:
            raise RuntimeError(f"sph_render reached a sub-result of {model!r} with no owning result.")
        return SphRender(
            owner,
            kernel=self._kernel_spec,
            smooth_floor=self._smooth_floor,
            wrap=self._wrap,
            smooth=self._smooth_source,
        )

    def _smooth_array(self, sim: Any) -> np.ndarray:
        """The smoothing length of every particle, in the position units.

        ``smooth="snapshot"`` (the default) reads ``sim["smooth"]`` — pynbody's
        on-disk values where they exist, and its k-d tree estimate for a single
        family that has none.  A *mixed* set can be refused this way; the error
        names ``smooth="kdtree"``, which derives the whole array with
        :func:`pynbody.sph.smooth` instead.  That is the same run pynbody's own
        ``rho`` uses, and the only way to render a particle set whose families do
        not all carry the block.
        """
        if self._smooth_source == "kdtree":
            from pynbody.sph import smooth as derive_smoothing

            try:
                return np.asarray(derive_smoothing(sim), dtype=float)
            except Exception as exc:  # noqa: BLE001 - report whatever it was, with the cause attached
                raise ValueError(
                    "sph_render could not derive smoothing lengths for this particle set with "
                    f"pynbody.sph.smooth: {type(exc).__name__}: {exc}"
                ) from exc
        try:
            return np.asarray(sim["smooth"], dtype=float)
        except Exception as exc:  # noqa: BLE001 - the spelling of "no such array" is the subclass's business
            # pynbody spells it KeyError, but a snapshot wrapper may raise anything
            # (a custom ``__getitem__`` often falls through to TypeError).  Type is
            # what the caller cannot guess, so it goes in the message; the original
            # exception stays attached as the cause.
            raise ValueError(
                "sph_render needs a smoothing length for every particle, and this particle set does not "
                "provide one: pynbody derives 'smooth' on demand for a single family, but a mixed set must "
                "carry one for each of them (usually only the gas does).  Render a family or sub-result "
                "that has it — bins.gas.sph_render[...] — give 'smooth' to every family first, or call "
                'bins.sph_render(smooth="kdtree") to derive one for the whole set.  '
                f"(reading sim['smooth'] raised {type(exc).__name__}.)"
            ) from exc

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
            slab_cells = (high - low) * row
            cell, entry_value, entry_weight = self._pairs(
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
            # Bucketing reduces the slab in O(pairs) instead of ordering it; the
            # answer is the same one :func:`weighted_percentiles` gives.
            bins = int(np.clip(_BUCKET_BUDGET // max(slab_cells, 1), _MIN_BUCKETS, _MAX_BUCKETS))
            reduced = bucketed_weighted_percentiles(
                entry_value, entry_weight, statistic.percentile, sample_of=cell, samples=slab_cells, bins=bins
            )
            out[low:high] = np.asarray(reduced, dtype=float).reshape((high - low, *shape[1:]))
        return out

    def _pairs(
        self,
        position: np.ndarray,
        smoothing: np.ndarray,
        support: np.ndarray,
        value: np.ndarray,
        extra: np.ndarray,
        centres: list[np.ndarray],
        origins: list[float],
        widths: list[float],
        strides: list[int],
        low: int,
        high: int,
        kernel: Any,
        measure: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(cell, value, weight)`` for every particle-cell pair in one slab.

        The compiled kernel does the work of :func:`_scatter_pairs` *and* pynbody's
        kernel lookup — the ``result[...] += ...`` loop of its renderer, writing
        pair records instead of sums — and is roughly an order of magnitude faster;
        the NumPy fallback keeps the layer working without the optional extension
        and weighs the pairs with the same lookup in :func:`_kernel_weights`.
        """
        builder = _native_pair_builder()
        if builder is not None:
            cell, values, weights = builder(
                np.ascontiguousarray(position, dtype=float),
                np.ascontiguousarray(smoothing, dtype=float),
                np.ascontiguousarray(support, dtype=float),
                np.ascontiguousarray(value, dtype=float),
                np.ascontiguousarray(extra, dtype=float),
                _kernel_table(kernel),
                int(getattr(kernel, "h_power", 3)),
                float(measure),
                list(origins),
                list(widths),
                [len(centre) for centre in centres],
                list(strides),
                int(low),
                int(high),
                0,
            )
            return (np.asarray(cell, dtype=np.intp), np.asarray(values, dtype=float), np.asarray(weights, dtype=float))
        cell, distance, found = _scatter_pairs(
            position, smoothing, support, centres, origins, widths, strides, low, high
        )
        weight = _kernel_weights(np.sqrt(distance), smoothing[found], kernel) * measure * extra[found]
        return cell, value[found], weight

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
        smooth = self._smooth_array(sim)
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
            # pynbody's own "min_smooth" clamp, left at 0 because ``_smoothing``
            # already applied the floor: ``to_3d_grid`` has no such parameter and
            # the quantile engine needs the floored values too, so the clamp lives
            # in one place for every path rather than in two here.
            0.0,
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
        keep the units of the strict query.
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
        sub-result.

        The statistic picks the engine.  The kernel sums (``count``, ``sum``,
        ``mean``, ``rms``, ``disp``) are accumulated by pynbody's C renderer and
        are exact and cheap; the quantiles (``median``, ``pXX``) are not kernel
        sums, so they scatter every particle over the cells its kernel reaches —
        exact, and about a dozen kernel sums.  Both answer ``bins[...]``'s own
        query grammar, so one view answers whatever the strict result would.  The
        result's derived properties join in too: ``"mass.sum.density"``, and any
        property registered ``allow_sph=True`` such as ``"gas_fraction"``, which
        is then the ratio of the kernel-integrated masses.

        Returns
        -------
        SphRender
            A view; index it with a query such as ``"mass.sum"``, ``"count"`` or
            ``"vz.median"``, ``"vz.mean@mass"``, or a derived key such as
            ``"mass.sum.density"``.  See :class:`SphRender` for what each part
            means and which results can be rendered.  The view is callable to
            change its settings — ``bins.sph_render(smooth="kdtree")`` — which is
            how a *mixed* particle set gets rendered.

        Raises
        ------
        ValueError
            If the result is not two or three spatial axes with evenly spaced
            bins, or the snapshot has no smoothing lengths.  Raised on the first
            query, not on this property.

        Examples
        --------
        >>> bins.sph_render["count"]  # doctest: +SKIP
        >>> bins.sph_render["vz.median"]  # doctest: +SKIP
        >>> bins.sph_render["mass.sum.density"]  # doctest: +SKIP
        >>> bins.sph_render(smooth="kdtree")["gas_fraction"]  # doctest: +SKIP
        >>> bins.s.sph_render["vz.mean"]  # doctest: +SKIP
        """
        return self._sph_render_view()

    def _sph_render_view(
        self,
        *,
        kernel: str | KernelBase | None = None,
        smooth_floor: float = 0.0,
        wrap: bool = True,
        smooth: str = "snapshot",
    ) -> SphRender:
        """A :class:`SphRender` over this result, cached by its settings.

        The default view lives in the same cache as the configured ones, so
        ``bins.sph_render is bins.sph_render`` and
        ``bins.sph_render(smooth="kdtree") is bins.sph_render(smooth="kdtree")`` —
        which matters because a render keeps the plan it built, and deriving
        smoothing lengths is not free.
        """
        views = self.__dict__.setdefault("_sph_render_views", {})
        key = (kernel, smooth_floor, wrap, smooth)
        render = views.get(key)
        if render is None:
            render = SphRender(
                cast("BinNDResult", self), kernel=kernel, smooth_floor=smooth_floor, wrap=wrap, smooth=smooth
            )
            views[key] = render
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


def _scatter_pairs(
    position: np.ndarray,
    smoothing: np.ndarray,
    support: np.ndarray,
    centres: list[np.ndarray],
    origins: list[float],
    widths: list[float],
    strides: list[int],
    low_row: int,
    high_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every ``(cell, distance², row)`` inside one slab of rows, in NumPy.

    Each particle's cell range comes from its own support (``2h``) and the uniform
    grid, so the block of cells one particle touches is a rectangular range;
    the ranges of all particles are then expanded into flat arrays at once
    (``np.repeat`` and a mixed-radix decomposition) rather than in a Python loop.
    Nothing is ordered: the caller reduces with a per-entry sample index, which is
    what :func:`~.statistics.bucketed_weighted_percentiles` takes.  The compiled
    kernel in ``cpp/image/scatter.cpp`` does the same arithmetic; this is the
    fallback for an install without it.
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
        return (np.empty(0, dtype=np.intp), np.empty(0), np.empty(0, dtype=np.intp))

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

    return cell, distance, particle


class _SphModel:
    """A :class:`~.model.BinResultModel` whose queries come back smoothed.

    A derived property is written once, against the query API — ``result["mass.sum"]``,
    ``result.gas["count"]`` — so handing the callback this wrapper instead of the
    model answers the smoothed version of the same property, with no second
    definition of it anywhere.  Sub-result access (``.gas``, ``.star``) recurses to
    that sub-result's own render, so a family ratio smooths each family with its own
    particles.  Anything that is not a query of this view (a model attribute, an
    axis) is passed through untouched.
    """

    def __init__(self, model: BinResultModel, render: SphRender) -> None:
        self._model = model
        self._render = render

    def __getitem__(self, key: str) -> BinsArray:
        return self._render[key]

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._model, name)
        if isinstance(value, BinResultModel):
            return _SphModel(value, self._render._sibling(value))
        return value


@lru_cache(maxsize=1)
def _native_pair_builder() -> Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """The compiled pair builder, or ``None`` without the optional extension."""
    try:
        from pynbodyext import _native
    except ImportError:
        return None
    builder = getattr(_native, "scatter_pairs", None)
    return cast("Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None", builder)


def _kernel_table(kernel: Any) -> np.ndarray:
    """pynbody's kernel lookup table as float64, for the compiled pair builder.

    Passing the table — rather than a kernel id — keeps the kernel definition on the
    Python side: the C loop indexes the same numbers pynbody's renderer indexes.
    """
    return np.ascontiguousarray(kernel.get_samples(), dtype=float)


def _kernel_weights(distance: np.ndarray, smoothing: np.ndarray, kernel: Any) -> np.ndarray:
    r"""``W(distance, h)`` for a grid of separations and a matching grid of ``h``.

    The kernel is indexed exactly as pynbody's renderer indexes it — its own
    lookup table, ``index = len(table) * d²/(2h)²``, zero beyond — so a quantile
    weighs its neighbours with the very weights the kernel sums use, whether the
    kernel is three-dimensional or the projected one.  (The C kernel in
    ``cpp/image/scatter.cpp`` does this same indexing.)
    """
    # pynbody caches the sample table per kernel and *not* per dtype, and its C
    # renderer asks for it as float32 — so ask for the same thing it will, or a
    # later kernel sum trips over a float64 table.
    samples = np.asarray(kernel.get_samples(), dtype=float)
    index = ((distance / smoothing) ** 2 / 4.0 * len(samples)).astype(np.intp)
    inside = index < len(samples)
    table = samples[np.clip(index, 0, len(samples) - 1)]
    return np.where(inside, table, 0.0) / smoothing ** int(getattr(kernel, "h_power", 3))


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
