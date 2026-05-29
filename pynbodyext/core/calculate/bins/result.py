from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np
from pynbody.array import SimArray
from pynbody.family import get_family
from pynbody.snapshot import SimSnap

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.nodes.filters import FilterBase

from .accessors import BinParticlesAccessor
from .axes import BinAxisAccessor, BinDerivedCondition, BinDerivedFunc, axis_matches
from .extensions import BIN_RESULT_EXTENSIONS, BinExtensionRegistry
from .plot import BinPlotMixin
from .query_engine import BinQueryEngine
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

    def __init__(
        self,
        *,
        sim: Any,
        source_sim: Any,
        axes: tuple[BinAxis, ...],
        bin_data: np.ndarray,
        bin_indptr: np.ndarray,
        particle_bin: np.ndarray,
        valid_mask: np.ndarray,
        calculator: BinND,
        scope_signature: Any = None,
        parent: BinNDResult | None = None,
    ) -> None:
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

        self._multi_index_cache: np.ndarray | None = None

        self._diagnostics = BinsResultEngine()
        self._query_engine = BinQueryEngine(self, self._diagnostics)
        self._subresults = BinSubresultStore(self)


    @property
    def root(self) -> BinNDResult:
        return self if self._parent is None else self._parent.root

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def nbins(self) -> int:
        return int(np.prod(self.shape_bins, dtype=int))

    @property
    def total_nbins(self) -> int:
        return self.nbins

    @property
    def unassigned_count(self) -> int:
        return int(np.count_nonzero(~self._valid_mask))

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
        >>> bins.particles_at_bin[1,3,5] # particles in bins 1, 3, and 5
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
        >>> bins.axis.r          # axis with alias "r"
        >>> bins.axis["r"]       # same
        >>> bins.axis[0]         # first axis
        """
        return BinAxisAccessor(self._axes)

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
        return self._query_engine.num_cached_arr

    @property
    def nsubs(self) -> int:
        return self._subresults.count()

    @property
    def total_cached_arr(self) -> int:
        return self._subresults.total_cached_arr()

    @property
    def edges(self) -> Any:
        self._require_1d("edges")
        return self._axes[0].edges

    @property
    def mins(self) -> Any:
        self._require_1d("mins")
        return self._axes[0].mins

    @property
    def maxs(self) -> Any:
        self._require_1d("maxs")
        return self._axes[0].maxs

    @property
    def centers(self) -> Any:
        self._require_1d("centers")
        return self._axes[0].centers

    @property
    def widths(self) -> Any:
        self._require_1d("widths")
        return self._axes[0].widths

    def _require_1d(self, name: str) -> None:
        if self.ndim != 1:
            raise AttributeError(f"{name} is ambiguous for ND bins; use bins.axis[alias].{name}.")


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
        return self._query_engine.keys()

    def property_keys(self) -> list[str]:
        return self._query_engine.property_keys()

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
        if (isinstance(key, CalculatorBase) and not isinstance(key, FilterBase)) or (callable(key) and not isinstance(key, (str, bytes))):
            return self.apply(key)
        if isinstance(key, tuple) or isinstance(key, (int, np.integer, slice)) or is_int_sequence(key):
            raise TypeError("Bin selectors must use bins.particles_at_bin[...], not BinNDResult.__getitem__.")
        return self._subresults.from_key(key)

    def __getattr__(self, name: str) -> Any:
        if name in {"center", "width", "min", "max", "edges"} and self.ndim == 1:
            # Convenience shortcuts for 1D results – delegate to the axis object
            return getattr(self._axes[0], {"center": "centers", "width": "widths", "min": "mins", "max": "maxs", "edges": "edges"}[name])
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
        return self._query_engine.resolve(key)

    def multi_index_array(self) -> np.ndarray:
        if self._multi_index_cache is None:
            self._multi_index_cache = np.column_stack(
                np.unravel_index(np.arange(self.nbins), self.shape_bins, order="C")
            )
        return self._multi_index_cache

    def find_axis(self, aliases: set[str]) -> Any:
        for axis in self._axes:
            if axis_matches(axis, aliases):
                return axis
        raise KeyError(f"No axis matching {sorted(aliases)!r}.")

    def _stat_pipeline(
        self,
        field: str,
        transforms: list[str],
        terminal_stat: Any,
        *,
        weight: str | Callable[[Any], Any] | Any | None = None,
        query_key: str | None = None,
    ) -> BinsArray:
        return self._query_engine.stat_pipeline(
            field, transforms, terminal_stat, weight=weight, query_key=query_key
        )

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
        return self._query_engine.stat_explicit(
            field,
            statistic,
            weight = weight,
            transforms= transforms
        )

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
        return self._query_engine.apply(
            query,
            name=name,
            empty=empty,
            vectorized=vectorized,
        )

    def _callable_cache_token(self, query: Any) -> Any:
        return self._query_engine.callable_cache_token(query)

    def cache_report(self) -> dict[str, Any]:
        return self._query_engine.cache_report()

    def query_report(self) -> list[dict[str, Any]]:
        return self._query_engine.query_report()

    @classmethod
    def register_transform(
        cls,
        name: str,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        overwrite: bool = False,
    ) -> None:
        cls._extensions.register_transform(name, func, overwrite=overwrite)

    @overload
    @classmethod
    def register_derived(cls, fn: BinDerivedFunc, *, name: None = None, scope: str = "derived", condition: BinDerivedCondition | None = None, overwrite: bool = False) -> BinDerivedFunc: ...
    @overload
    @classmethod
    def register_derived(cls, fn: str, *, name: None = None, scope: str = "derived", condition: BinDerivedCondition | None = None, overwrite: bool = False) -> Callable[[BinDerivedFunc], BinDerivedFunc]: ...
    @overload
    @classmethod
    def register_derived(cls, fn: None = None, *, name: str | None = None, scope: str = "derived", condition: BinDerivedCondition | None = None, overwrite: bool = False) -> Callable[[BinDerivedFunc], BinDerivedFunc]: ...
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
            cls,
            fn,
            name=name,
            scope=scope,
            condition=condition,
            overwrite=overwrite,
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
    """Per-bin physical measure — product of each axis's :attr:`BinAxis.measure`.

    For a 1-D radial grid this is the shell volume; for a 1-D projected grid
    the annulus area; for a generic ND grid the product of per-axis measures.
    """
    axes = bins.axes
    if len(axes) == 1:
        return axes[0].measure
    multi = bins.multi_index_array()
    result = SimArray(np.ones(bins.nbins, dtype=float), units="1")
    for i, axis in enumerate(axes):
        result *= axis.measure[multi[:, i]]
    return result


@BinNDResult.derived("count", scope="particles")
def _count(bins: BinNDResult) -> np.ndarray:
    valid_mask = bins._valid_mask
    if not valid_mask.any():
        return np.zeros(bins.nbins, dtype=int)
    return np.bincount(bins._particle_bin[valid_mask], minlength=bins.nbins).astype(int)






@BinNDResult.derived("density")
def _density(bins: BinNDResult) -> np.ndarray:
    return np.asarray(bins["mass.sum"] / bins["volume"])


@BinNDResult.derived("surface_density")
def _surface_density(bins: BinNDResult) -> np.ndarray:
    return np.asarray(bins["mass.sum"] / bins["area"])


@BinNDResult.derived("enclosed_mass")
def _enclosed_mass(bins: BinNDResult) -> np.ndarray:
    return np.cumsum(bins["mass.sum"])


@BinNDResult.derived("gas_fraction", condition=_has_family("gas"))
def gas_fraction(bins: BinNDResult) -> np.ndarray:
    gas_mass_sum = bins.gas["mass.sum"]
    total_mass_sum = bins["mass.sum"]
    numerator = np.nan_to_num(np.asarray(gas_mass_sum), nan=0.0)
    denominator = np.asarray(total_mass_sum)
    values = np.full(bins.nbins, np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = numerator[valid] / denominator[valid]
    return values
