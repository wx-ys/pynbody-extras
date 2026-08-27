from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np
from pynbody.family import get_family
from pynbody.snapshot import SimSnap

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.nodes.filters import FilterBase

from .accessors import BinParticlesAccessor
from .axes import BinAxisAccessor, BinDerivedCondition, BinDerivedFunc, BinMeasureResolver, axis_matches
from .extensions import BIN_RESULT_EXTENSIONS, BinExtensionRegistry
from .geometry import BinGeometry
from .model import BinResultModel
from .plot import BinPlotMixin
from .query import BinQueryService
from .selectors import is_int_sequence
from .statistics import BinNDStatAccessor
from .subresults import BinSubresultStore

if TYPE_CHECKING:
    from collections.abc import Callable

    from pynbody.family import Family
    from pynbody.filt import Filter

    from .arrays import BinsArray
    from .axes import BinAxis
    from .nodes import BinND


def _is_sim_like(value: Any) -> bool:
    return isinstance(value, SimSnap)


class BinsResultEngine:
    def __init__(self) -> None:
        self.diagnostics: list[dict[str, Any]] = []

    def record(self, event: str, **payload: Any) -> None:
        self.diagnostics.append({"event": event, **payload})


class _CSRBinsView:
    """Backward-compatible list-like view over CSR bin→particle storage.

    Allows existing ``for index, particles in enumerate(bins.bin_indices)``
    patterns to keep working while internal code uses the raw CSR arrays
    ``bin_data`` / ``bin_indptr`` directly.
    """

    __slots__ = ("_data", "_indptr")

    def __init__(self, data: np.ndarray, indptr: np.ndarray) -> None:
        self._data = data
        self._indptr = indptr

    def __len__(self) -> int:
        return len(self._indptr) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self._data[self._indptr[i] : self._indptr[i + 1]]

    def __iter__(self):
        for i in range(len(self)):
            yield self._data[self._indptr[i] : self._indptr[i + 1]]


class BinNDResult(BinPlotMixin):
    _SHARED_SCOPES: ClassVar[set[str]] = {"axis", "geometry"}
    _extensions: ClassVar[BinExtensionRegistry] = BIN_RESULT_EXTENSIONS
    _model: BinResultModel
    _query_service: BinQueryService
    _geometry: BinGeometry
    _measure_resolver: BinMeasureResolver
    _subresults: BinSubresultStore

    def __init__(
        self,
        *,
        model: BinResultModel | None = None,
        sim: Any = None,
        source_sim: Any = None,
        axes: tuple[BinAxis, ...] | None = None,
        bin_data: np.ndarray | None = None,
        bin_indptr: np.ndarray | None = None,
        particle_bin: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
        calculator: BinND | None = None,
        scope_signature: Any = None,
        parent: BinNDResult | None = None,
    ) -> None:
        if model is not None:
            # Spawn-layer model-ification: the executor builds the model first
            # and the façade wraps it (bind the owner back-reference here).
            self._model = model
            model._owner = self
            self.sim = model.sim
            self.source_sim = model.source_sim
            self._axes = model.axes
            self.shape_bins = model.shape_bins
            self.ndim = model.ndim
            self._bin_data = model.bin_data
            self._bin_indptr = model.bin_indptr
            self._particle_bin = model.particle_bin
            self._valid_mask = model.valid_mask
            self._calculator = model.calculator
            self._scope_signature = model.scope_signature
            self._parent = model.parent.owner if model.parent is not None else None
        else:
            assert sim is not None and axes is not None
            assert bin_data is not None and bin_indptr is not None
            assert particle_bin is not None and valid_mask is not None
            assert calculator is not None
            self.sim = sim
            self.source_sim = source_sim
            self._axes = axes
            self.shape_bins = tuple(axis.nbins for axis in axes)
            self.ndim = len(axes)
            self._bin_data = bin_data
            self._bin_indptr = bin_indptr
            self._particle_bin = particle_bin
            self._valid_mask = valid_mask
            self._calculator = calculator
            self._scope_signature = scope_signature
            self._parent = parent
            self._model = BinResultModel(
                sim=sim,
                source_sim=source_sim,
                axes=axes,
                bin_data=bin_data,
                bin_indptr=bin_indptr,
                particle_bin=particle_bin,
                valid_mask=valid_mask,
                calculator=calculator,
                scope_signature=scope_signature,
                parent=parent._model if parent is not None else None,
                owner=self,
            )

        self._diagnostics = BinsResultEngine()
        self._query_service = BinQueryService(self._model, self, self._diagnostics, extensions=type(self)._extensions)
        self._subresults = BinSubresultStore(self)
        self._measure_resolver = BinMeasureResolver()
        self._geometry = BinGeometry(self._model, self._measure_resolver)

        # Subscribe to global measure-type redefinitions (e.g.
        # BinAxis.register_measure_type(..., overwrite=True)) so that
        # cached density entries are invalidated when a named type changes.
        from .axes import _measure_change_subscribers

        _measure_change_subscribers.add(self)

    @property
    def root(self) -> BinNDResult:
        return self if self._parent is None else self._parent.root

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def model(self) -> BinResultModel:
        """The underlying :class:`BinResultModel` data container for this result."""
        return self._model

    @property
    def parent(self) -> BinNDResult | None:
        """The parent result this subresult was derived from, or ``None`` for a root."""
        return self._parent

    @property
    def nbins(self) -> int:
        return int(np.prod(self.shape_bins, dtype=int))

    @property
    def total_nbins(self) -> int:
        return self.nbins

    @property
    def unassigned_count(self) -> int:
        return self._model.unassigned_count

    @property
    def count(self) -> BinsArray:
        return self._resolve_query("count")

    @property
    def particles_at_bin(self) -> BinParticlesAccessor:
        """Accessor for retrieving the particles in each bin.

        Examples
        --------
        >>> bins.particles_at_bin[0]  # particles in the first bin
        >>> bins.particles_at_bin[1:5]  # particles in bins 1 through 4
        >>> bins.particles_at_bin[1, 3, 5]  # particles in bins 1, 3, and 5
        >>> bins.particles_at_bin[:, 0]  # particles in the first bin along the second axis (for 2D or higher)

        """
        return BinParticlesAccessor(self)

    @property
    def axes(self) -> tuple[BinAxis, ...]:
        """The bin axes (tuple). For individual lookups prefer ``bins.axis``."""
        return self._axes

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask marking particles that were successfully assigned to a bin."""
        return self._valid_mask

    @property
    def axis(self) -> BinAxisAccessor:
        """Accessor for individual axes.

        Examples
        --------
        >>> bins.axis.r  # axis with alias "r"
        >>> bins.axis["r"]  # same
        >>> bins.axis[0]  # first axis
        >>> bins.axis.set_axis_measure_type("r", "annulus")  # per-instance override
        """
        return BinAxisAccessor(self._axes, owner=self)

    def _resolve_axis_measure(self, axis: BinAxis) -> np.ndarray:
        """Return the effective per-bin measure for *axis*.

        Delegates to :attr:`_measure_resolver`, which checks per-instance
        overrides first, then the global :attr:`BinAxis.measure` property.
        """
        return self._geometry.axis_measure(axis)

    @property
    def bin_indices(self) -> _CSRBinsView:
        """Backward-compatible view over bin→particle indices (CSR format).

        Prefer the raw ``bin_data`` / ``bin_indptr`` arrays for
        performance-critical code: ``bin_data[bin_indptr[i]:bin_indptr[i+1]]``
        avoids creating a new Python object per access.
        """
        return _CSRBinsView(self._bin_data, self._bin_indptr)

    @property
    def num_cached_arr(self) -> int:
        return self._query_service.num_cached_arr

    @property
    def nsubs(self) -> int:
        return self._subresults.count()

    @property
    def total_cached_arr(self) -> int:
        return self._subresults.total_cached_arr()

    @property
    def edges(self) -> Any:
        return self._geometry.edges

    @property
    def mins(self) -> Any:
        return self._geometry.mins

    @property
    def maxs(self) -> Any:
        return self._geometry.maxs

    @property
    def centers(self) -> Any:
        return self._geometry.centers

    @property
    def widths(self) -> Any:
        return self._geometry.widths

    def _require_1d(self, name: str) -> None:
        self._geometry._require_1d(name)

    def families(self) -> Any:
        return self.sim.families()

    @property
    def stat(self) -> BinNDStatAccessor:
        """Accessor for pipeline statistic queries.

        Examples
        --------
        >>> bins.stat["mass.sum"]  # per-bin mass sum
        >>> bins.stat["vz.abs.mean"]  # per-bin mean of |vz|
        >>> bins.stat.keys()  # example keys for all registered statistics

        you can also specify a weight field for weighted statistics using the @ syntax:
        >>> bins.stat["age.median@mass"]  # per-bin mass-weighted median of age
        >>> bins.stat["age.p33@mass"]  # per-bin mass-weighted 33rd percentile of age
        >>> bins.stat["vz.abs.mean@mass"]  # per-bin mass-weighted mean of |vz|

        """
        return BinNDStatAccessor(self)

    def keys(self) -> list[str]:
        return self._query_service.keys()

    def property_keys(self) -> list[str]:
        return self._query_service.property_keys()

    def all_keys(self) -> list[str]:
        return self.keys()

    def _ipython_key_completions_(self) -> list[str]:
        return self.all_keys()

    def get_subresult(self, subset: Any, *, _cache_key: Any = None) -> SubBinNDResult:
        return self._subresults.get(subset, cache_key=_cache_key)

    @overload
    def __getitem__(self, key: str) -> BinsArray: ...
    @overload
    def __getitem__(self, key: FilterBase | Filter | Family) -> SubBinNDResult: ...
    @overload
    def __getitem__(self, key: CalculatorBase) -> BinsArray: ...
    def __getitem__(self, key: Any) -> SubBinNDResult | BinsArray:
        if isinstance(key, str):
            # Axis properties like "r.center" must be accessed via bins.axis("r").center
            # String queries only handle: geometry/derived properties and pipeline stat queries
            return self._resolve_query(key)
        if (isinstance(key, CalculatorBase) and not isinstance(key, FilterBase)) or (
            callable(key) and not isinstance(key, (str, bytes))
        ):
            return self.apply(key)
        if isinstance(key, tuple) or isinstance(key, (int, np.integer, slice)) or is_int_sequence(key):
            raise TypeError("Bin selectors must use bins.particles_at_bin[...], not BinNDResult.__getitem__.")
        return self._subresults.from_key(key)

    def __getattr__(self, name: str) -> Any:
        if name in {"center", "width", "min", "max", "edges"} and self.ndim == 1:
            # Convenience shortcuts for 1D results – delegate to the axis object
            return getattr(
                self._axes[0],
                {"center": "centers", "width": "widths", "min": "mins", "max": "maxs", "edges": "edges"}[name],
            )
        try:
            sub = getattr(self.sim, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if _is_sim_like(sub):
            return self.get_subresult(sub)
        raise AttributeError(name)

    @property
    def gas(self) -> SubBinNDResult:
        return self._family_subresult("gas")

    @property
    def g(self) -> SubBinNDResult:
        return self._family_subresult("gas")

    @property
    def dm(self) -> SubBinNDResult:
        return self._family_subresult("dm")

    @property
    def star(self) -> SubBinNDResult:
        return self._family_subresult("star")

    @property
    def s(self) -> SubBinNDResult:
        return self._family_subresult("star")

    def _family_subresult(self, family_name: str) -> SubBinNDResult:
        try:
            family = get_family(family_name)
        except Exception as exc:
            raise AttributeError(f"Family {family_name!r} not found.") from exc
        sub = self.sim[family]
        if not _is_sim_like(sub):
            raise AttributeError(f"Family {family_name!r} selector did not produce a SimSnap subset.")
        return self.get_subresult(sub, _cache_key=("family", family.name))

    def _resolve_query(self, key: str) -> BinsArray:
        # Shared-scope queries (axis/geometry) are computed once on the root and
        # reused by subresults; the service itself is model-pure.
        scope = type(self)._extensions.query_scope(self, key)
        if not self.is_root and scope in self._SHARED_SCOPES:
            return self.root._resolve_query(key)
        return self._query_service.resolve(key)

    def multi_index_array(self) -> np.ndarray:
        return self._geometry.multi_index_array()

    def find_axis(self, aliases: set[str]) -> Any:
        return self._geometry.find_axis(aliases)

    def _stat_pipeline(
        self,
        field: str,
        transforms: list[str],
        terminal_stat: Any,
        *,
        weight: str | Callable[[Any], Any] | Any | None = None,
        query_key: str | None = None,
    ) -> BinsArray:
        return self._query_service.stat_pipeline(field, transforms, terminal_stat, weight=weight, query_key=query_key)

    def stat_explicit(
        self,
        field: str,
        statistic: str,
        weight: str | Callable[[Any], Any] | Any | None = None,
        transforms: list[str] | None = None,
    ) -> BinsArray:
        """Explicitly compute a per-bin statistic.

        Equivalent to ``bins["field.stat"]`` but accepts optional transforms and
        weight.  Example::

            bins.stat_explicit("mass", "sum")
            bins.stat_explicit("vz", "mean", transforms=["abs"])
            bins.stat_explicit("mass", "mean", weight="mass")
        """
        return self._query_service.stat_explicit(field, statistic, weight=weight, transforms=transforms)

    def apply(
        self,
        query: Callable[[Any], Any] | CalculatorBase[Any, Any],
        *,
        name: str | None = None,
        empty: float = np.nan,
        vectorized: bool = False,
    ) -> BinsArray:
        """Evaluate *query* on each bin's particle subset and return a :class:`BinsArray`.

        Parameters
        ----------
        query:
            A callable ``(SimSnap) -> scalar`` or a :class:`CalculatorBase`.  The
            callable receives the particle sub-snapshot for each bin and must return
            a scalar value.
        name:
            Optional name for the resulting array.
        empty:
            Fill value for empty bins (default ``nan``).
        vectorized:
            If ``True``, *query* is expected to accept the full simulation snap and
            a flat ``particle_bin`` index array (shape ``(n_particles,)``, ``-1`` for
            unassigned particles) and must return a 1-D array of length ``nbins``.  This
            path bypasses the per-bin loop entirely and is much faster for large grids.

            Signature: ``query(sim, particle_bin) -> np.ndarray``
        """
        return self._query_service.apply(query, name=name, empty=empty, vectorized=vectorized)

    def _callable_cache_token(self, query: Any) -> Any:
        return self._query_service.callable_cache_token(query)

    def cache_report(self) -> dict[str, Any]:
        return self._query_service.cache_report()

    def query_report(self) -> list[dict[str, Any]]:
        return self._query_service.query_report()

    # ------------------------------------------------------------------
    # Axis measure type configuration
    # ------------------------------------------------------------------

    def set_axis_measure_type(self, alias: str, type_name: str | None) -> None:
        """Assign a registered measure type to an axis *alias* — **per-instance**.

        Only affects this :class:`BinNDResult` and its subresults.  Other
        instances are **not** affected.  Cached density entries that depend
        on the axis measure are invalidated for this instance only.

        Pass ``type_name=None`` to remove a previously set per-instance
        override, reverting to the global behaviour for *alias*.

        For a **global** change use :meth:`BinAxis.register_measure_type`
        with ``overwrite=True``.

        Parameters
        ----------
        alias:
            The axis alias (or prop name) to assign the measure type to.
        type_name:
            A measure type name previously registered via
            :meth:`BinAxis.register_measure_type`.  Built-in types include
            ``"spherical_shell"``, ``"annulus"``, and ``"linear"``.
            Pass ``None`` to clear a previously set override.
        """
        from .axes import _AXIS_MEASURE_TYPE_REGISTRY as _types

        if type_name is None:
            if self._measure_resolver.get(alias) is None:
                return
            self._measure_resolver.set(alias, None)
        else:
            if type_name not in _types:
                raise KeyError(f"Unknown measure type {type_name!r}. Known types: {sorted(_types)}.")
            old_type = self._measure_resolver.get(alias)
            if old_type == type_name:
                return  # no change — nothing to do
            self._measure_resolver.set(alias, type_name)

        # Invalidate measure-dependent cache entries on this instance + subresults
        n, names = self._query_service.invalidate_measure_dependent_cache()
        for sub in self._subresults.values():
            sn, snames = sub._query_service.invalidate_measure_dependent_cache()
            n += sn
            names.extend(snames)

        if n > 0:
            unique = sorted(set(names))
            entr = "y" if n == 1 else "ies"
            self._calculator.warning(
                f"Per-instance measure type for axis {alias!r} changed to "
                f"{type_name!r}. Cleared {n} cached entr{entr}: "
                f"{', '.join(unique)}."
            )

    def _on_axis_measure_change(self, alias: str, type_name: str) -> tuple[int, list[str]]:
        """Called by the global subscriber mechanism when a measure type is redefined.

        Invalidates measure-dependent cache entries on this instance and all
        subresults.  Returns ``(count, cleared_names)``.
        """
        n, names = self._query_service.invalidate_measure_dependent_cache()
        for sub in self._subresults.values():
            sn, snames = sub._query_service.invalidate_measure_dependent_cache()
            n += sn
            names.extend(snames)
        return n, names

    @classmethod
    def register_transform(
        cls, name: str, func: Callable[[np.ndarray], np.ndarray], *, overwrite: bool = False
    ) -> None:
        cls._extensions.register_transform(name, func, overwrite=overwrite)

    @overload
    @classmethod
    def register_derived(
        cls,
        fn: BinDerivedFunc,
        *,
        name: None = None,
        scope: str = "derived",
        condition: BinDerivedCondition | None = None,
        overwrite: bool = False,
    ) -> BinDerivedFunc: ...
    @overload
    @classmethod
    def register_derived(
        cls,
        fn: str,
        *,
        name: None = None,
        scope: str = "derived",
        condition: BinDerivedCondition | None = None,
        overwrite: bool = False,
    ) -> Callable[[BinDerivedFunc], BinDerivedFunc]: ...
    @overload
    @classmethod
    def register_derived(
        cls,
        fn: None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: BinDerivedCondition | None = None,
        overwrite: bool = False,
    ) -> Callable[[BinDerivedFunc], BinDerivedFunc]: ...
    @classmethod
    def register_derived(
        cls,
        fn: BinDerivedFunc | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: BinDerivedCondition | None = None,
        overwrite: bool = False,
    ) -> Any:
        return cls._extensions.register_derived(
            cls, fn, name=name, scope=scope, condition=condition, overwrite=overwrite
        )

    derived = register_derived
    derived_property = register_derived

    def __repr__(self) -> str:
        parent_flag = "root" if self.is_root else "sub"
        aliases = ",".join(axis.alias for axis in self._axes)
        return f"<{type(self).__name__} type={parent_flag} ndim={self.ndim} shape={self.shape_bins} axes={aliases} nsubs={self.nsubs} ncache={self.total_cached_arr}>"


class SubBinNDResult(BinNDResult):
    def get_subresult(self, subset: Any, *, _cache_key: Any = None) -> SubBinNDResult:
        return self.root.get_subresult(subset, _cache_key=_cache_key)


def _has_family(name: str) -> Callable[[Any], bool]:
    def condition(bins: Any) -> bool:
        try:
            return any(getattr(family, "name", str(family)) == name for family in bins.families())
        except Exception:
            return False

    return condition


@BinNDResult.derived("measure", scope="geometry")
def _bin_measure(bins: BinNDResult) -> np.ndarray:
    """Per-bin physical measure — product of each axis's effective measure.

    Respects any per-instance overrides set via
    :meth:`BinNDResult.set_axis_measure_type`.

    For a 1-D radial grid this is the shell volume; for a 1-D projected grid
    the annulus area; for a generic ND grid the product of per-axis measures.
    """
    axes = bins.axes
    if len(axes) == 1:
        return bins._resolve_axis_measure(axes[0])
    multi = bins.multi_index_array()
    # Start from the first axis so that units are inherited correctly
    # (e.g. kpc for a linear axis, kpc^2 for an annulus, kpc^3 for a shell).
    result = bins._resolve_axis_measure(axes[0])[multi[:, 0]].copy()
    for i in range(1, len(axes)):
        result *= bins._resolve_axis_measure(axes[i])[multi[:, i]]
    return result


@BinNDResult.derived("count", scope="particles")
def _count(bins: BinNDResult) -> np.ndarray:
    valid_mask = bins.valid_mask
    if not valid_mask.any():
        return np.zeros(bins.nbins, dtype=int)
    return np.bincount(bins.particle_bin[valid_mask], minlength=bins.nbins).astype(int)


@BinNDResult.derived(
    "vcirc", condition=lambda bins: bins.ndim == 1 and any(axis_matches(axis, {"r", "rxy"}) for axis in bins.axes)
)
def _vcirc(bins: BinNDResult) -> np.ndarray:
    """Per-bin circular velocity from the gravitational field.

    Available on 1-D radial (``r`` / ``rxy``) profiles only.  For each bin
    radius ``R`` the magnitude of the gravitational acceleration is evaluated
    at a set of points on the mid-plane circle of radius ``R`` (averaging a few
    azimuthal samples so a lumpy distribution does not bias one azimuth), then
    ``sqrt(<|a|> * R)`` is returned.

    The acceleration is computed by the gravity backend (``tree`` by default)
    from ``sim['pos']``, ``sim['mass']`` and ``sim['smooth']`` — i.e. the
    particles of :attr:`~BinNDResult.sim`.  For a spherically symmetric mass
    distribution this equals ``sqrt(G * M_enc(<R) / R)``, the classic
    circular (rotation) velocity.

    Returns a :class:`pynbody.array.SimArray` in ``km s**-1``.
    """
    from pynbody.array import SimArray

    from pynbodyext.gravity import KernelKind, calculate_acceleration

    sim = bins.sim
    centers = bins.centers

    # Bin-center radii as a unit-bearing array (kpc for a physical r/rxy
    # profile).  Keep the axis units if present so the kpc→km conversion is
    # exact; otherwise fall back to the snapshot's position units.
    if isinstance(centers, SimArray):
        radii_arr = centers
        radii = np.asarray(centers)
    else:
        radii = np.asarray(centers)
        radii_arr = SimArray(radii, units=sim["pos"].units)

    # Sample the mid-plane circle at each bin radius.
    n_azimuth = 8
    angles = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    rr = np.repeat(radii, n_azimuth)
    aa = np.tile(angles, radii.size)
    points = np.zeros((rr.size, 3), dtype=float)
    points[:, 0] = rr * np.cos(aa)
    points[:, 1] = rr * np.sin(aa)

    softening = sim["smooth"] if "smooth" in sim.keys() else None
    # A softening kernel must accompany any softening length (KernelKind.No is
    # only valid with no softenings).  Plummer with eps=0 reduces to Newtonian,
    # so it is exact whenever sim['smooth'] is present.
    kernel = KernelKind.Plummer if softening is not None else KernelKind.No
    acc = calculate_acceleration(sim, positions=points, softening=softening, method="tree", kernel=kernel, theta=0.7)
    # |a| in km s**-2, averaged over the azimuthal samples at each radius.
    acc_mag = np.linalg.norm(np.asarray(acc), axis=1)
    mean_a = acc_mag.reshape(radii.size, n_azimuth).mean(axis=1)

    # vcirc = sqrt(|a| * R): a is km/s**2, so R must be in km for the product
    # to yield km**2/s**2 (and sqrt → km/s).
    R_km = np.asarray(radii_arr.in_units("km"))
    with np.errstate(divide="ignore", invalid="ignore"):
        vcirc = np.sqrt(mean_a * R_km)

    out = SimArray(vcirc, units="km s**-1")
    out.sim = sim
    return out


@BinNDResult.derived("density", overwrite=True)
def _density(bins: BinNDResult) -> np.ndarray:
    """Per-bin mass density: total mass divided by physical measure.

    Equivalent to ``bins["mass.sum.density"]`` and ``bins["mass.density"]``
    — all three share the same cache entry under the canonical key
    ``"mass.sum.density"``.

    For a 1-D radial profile this is the volumetric mass density
    (mass / shell_volume).  For a 1-D projected profile this is the
    surface mass density (mass / annulus_area).
    """
    return bins["mass.sum.density"]


@BinNDResult.derived("number_density", overwrite=False)
def _number_density(bins: BinNDResult) -> np.ndarray:
    """Per-bin number density: particle count divided by physical measure."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return bins["count"] / bins["measure"]


@BinNDResult.derived("enclosed_mass")
def _enclosed_mass(bins: BinNDResult) -> np.ndarray:
    return bins["mass.sum"].cumsum()


@BinNDResult.derived("gas_fraction", condition=_has_family("gas"))
def gas_fraction(bins: BinNDResult) -> np.ndarray:
    gas_mass_sum = bins.gas["mass.sum"]
    total_mass_sum = bins["mass.sum"]
    numerator = np.nan_to_num(np.asarray(gas_mass_sum).ravel(), nan=0.0)
    denominator = np.asarray(total_mass_sum).ravel()
    values = np.full(bins.nbins, np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = numerator[valid] / denominator[valid]
    return values
