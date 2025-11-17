from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np
from pynbody import filt as _pyn_filt
from pynbody.family import Family
from pynbody.snapshot import SimSnap

from .bins import BinByFunc, BinsAlgorithmFunc, BinsAreaFunc, BinsSet, SimOrNpArray
from .proarray import ProfileArray

if TYPE_CHECKING:
    from pynbody.array import SimArray

# ------------------------------------------------------------------
# Assumptions / light shims (remove if already defined elsewhere)
# ------------------------------------------------------------------
FilterLike: TypeAlias = _pyn_filt.Filter | np.ndarray | Family | slice | int | np.int32 | np.int64

AllArray: TypeAlias = ProfileArray | SimOrNpArray

ProFunc: TypeAlias = Callable[["ProfileBase"], AllArray]
SimFunc: TypeAlias = Callable[[SimSnap], AllArray]

_BIN_PROPERTY_KEYS = ("rbins", "dr", "binsize", "npart_bins")  # per-bin properties


def _is_sim_like(x: Any) -> bool:
    return isinstance(x, SimSnap)

# ------------------------------------------------------------------
# Profile base
# ------------------------------------------------------------------
class ProfileBase:
    """
    Abstract base providing shared mechanics (bin set, caching, weight).
    Concrete classes: Profile (root) and SubProfile (subset view).
    """

    _profile_property_registry: defaultdict[type, dict[str, ProFunc]] = defaultdict(dict)


    def __init__(self,
                 sim: SimSnap,
                 bins_set: BinsSet,
                 weight: AllArray | None = None,
                 parent: Profile | None = None):
        self.sim: SimSnap = sim
        self._bins: BinsSet = bins_set if bins_set.is_defined() else bins_set(sim)
        self._parent: Profile | None = parent

        # Weight handling: if key provided, materialize array; else None
        self._weight: SimArray | np.ndarray | None = weight
        if self._weight is not None:
            assert len(self._weight) == len(sim), "Weight array length must match simulation length."

        # Caches:
        # _data_cache: field key -> base ProfileArray (per-bin or computed default)
        # _stats_cache: field key -> dict[mode/statistic] -> ProfileArray
        self._data_cache: dict[str, ProfileArray] = {}
        self._stats_cache: defaultdict[str, dict[str, ProfileArray]] = defaultdict(dict)

    # ---- Caching API required by ProfileArray ----
    def cache(self, pro_arr: ProfileArray) -> None:
        """Store a computed statistic ProfileArray in cache."""
        key = pro_arr._name
        mode = pro_arr._mode
        if key is None or mode is None:
            return
        bucket = self._stats_cache.setdefault(key, {})
        bucket[mode] = pro_arr

    def is_cached(self, key: str, item: str) -> bool:
        return key in self._stats_cache and item in self._stats_cache[key]

    def get_cached(self, key: str, item: str) -> ProfileArray:
        return self._stats_cache[key][item]

    # ---- Properties bridging to BinsSet ----

    @property
    def bins(self) -> BinsSet:
        return self._bins

    @property
    def nbins(self) -> int:
        return len(self._bins.rbins) if self._bins.rbins is not None else 0

    @property
    def binind(self) -> list[np.ndarray]:
        return self._bins.binind or []

    @property
    def rbins(self) -> SimArray | np.ndarray | None:
        return self._bins.rbins

    @property
    def bin_edges(self) -> SimArray | np.ndarray | None:
        return self._bins.bin_edges

    @property
    def binsize(self) -> SimArray | np.ndarray | None:
        return self._bins.binsize

    @property
    def dr(self) -> SimArray | np.ndarray | None:
        return self._bins.dr

    @property
    def npart_bins(self) -> np.ndarray | None:
        return self._bins.npart_bins

    def families(self):
        return self.sim.families()

    def keys(self) -> list[str]:
        # Return cached data keys + per-bin property keys (always exposed)
        data_keys = list(self._data_cache.keys()) + list(self._stats_cache.keys())
        return sorted(set(data_keys).union(_BIN_PROPERTY_KEYS))

    def property_keys(self) -> list[str]:
        par = self.parent
        reg_by_cls = par.__class__._profile_property_registry
        reg_keys: set[str] = set()
        for c in par.__class__.mro():
            reg_keys |= set(reg_by_cls.get(c, {}).keys())
        return sorted(reg_keys)
    def all_keys(self) -> list[str]:
        return sorted(set(self.keys()).union(set(self.property_keys())))

    def _ipython_key_completions_(self) -> list[str]:
        return self.all_keys()

    @property
    def parent(self) -> Profile:
        """Return the ultimate root (the original Profile)."""
        root: ProfileBase = self
        while root._parent is not None:
            root = root._parent
        # By construction, the root is a Profile
        return cast("Profile", root)

    @property
    def num_cached_arr(self) -> int:
        return len(self._data_cache) + sum(len(v) for v in self._stats_cache.values())

    def _spawn(self, sim_subset: SimSnap) -> SubProfile:
        """
        Create a SubProfile sharing the same bin edges. The parent is ALWAYS
        the root Profile (so all subprofiles reference the same parent).
        """
        child_bins = self._bins.spawn_with_same_edges(sim_subset)
        root_parent = self.parent
        sub_weight = (self._weight[sim_subset.get_index_list(self.sim)]
                      if self._weight is not None
                      else None)
        return SubProfile(
            sim_subset,
            bins_set=child_bins,
            weight=sub_weight,
            parent=root_parent,   # ensure single parent
        )

    # ---- Field resolution ----
    def _get_property_func(self, key: str) -> ProFunc | None:
        par = self.parent

        reg_by_cls = par.__class__._profile_property_registry
        for c in par.__class__.mro():
            bucket = reg_by_cls.get(c)
            if bucket and key in bucket:
                return bucket[key]
        return None

    def _resolve_field(self, key: str) -> ProfileArray:
        """
        Resolve a field key to a ProfileArray. If already cached, return it;
        otherwise build a base per-bin representation (per-particle → default statistic 'mean').
        """
        # Per-bin properties (rbins, etc.)
        if key in self._data_cache:
                return self._data_cache[key]

        if key in self._stats_cache:
            if "mean" in self._stats_cache[key]:
                return self._stats_cache[key]["mean"]

        func = self._get_property_func(key)
        if key in _BIN_PROPERTY_KEYS or func is not None:
            if key == "rbins":
                arr = self.rbins
            elif key == "dr":
                arr = self.dr
            elif key == "binsize":
                arr = self.binsize
            elif key == "npart_bins":
                arr = self.npart_bins

            if func is not None:
                arr = func(self)
            base = ProfileArray(self, name=key, array=arr, mode=None)
            self._data_cache[key] = base
            return base

        # Base creation: let ProfileArray decide source; default statistic mean
        base_arr = ProfileArray(self, name=key, mode="mean")
        self._stats_cache[key][cast("str", base_arr._mode)] = base_arr
        return base_arr


    def get_subprofile(self, subset: SimSnap) -> SubProfile:
        warnings.warn("get_subprofile is deprecated; use __getitem__ with a filter instead",
                      DeprecationWarning, stacklevel=2)
        subprof = self._spawn(subset)
        return subprof

     # ---- Main public indexing ----
    @overload
    def __getitem__(self, key: str) -> ProfileArray: ...
    @overload
    def __getitem__(self, key: FilterLike) -> SubProfile: ...

    def __getitem__(self, key: str | FilterLike) -> ProfileArray | SubProfile:

        # str -> ProfileArray
        if isinstance(key, str):
            return self._resolve_field(key)

        # FilterLike → subset
        subset = self.sim[key]
        return self.get_subprofile(subset)

    def __getattr__(self, name: str) -> SubProfile:
        # Mirror SimSnap pattern: attribute exposing subsnap returns SubProfile
        try:
            sub = getattr(self.sim, name)
        except AttributeError as e:
            raise AttributeError(name) from e
        if _is_sim_like(sub):
            return self.get_subprofile(sub)
        raise AttributeError(name)

    # ---- Representation ----
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        fams = [f.name for f in self.families()]
        parent_flag = "root" if self._parent is None else "sub"
        return f"<{cls} type={parent_flag} nbins={self.nbins} families={fams} ncached={self.num_cached_arr}>"

    @classmethod
    @overload
    def profile_property(
        cls,
        fn: ProFunc,
        name: str | None = None,
    ) -> ProFunc: ...
    @classmethod
    @overload
    def profile_property(
        cls,
        fn: None = None,
        name: str | None = None,
    ) -> Callable[[ProFunc], ProFunc]: ...
    @classmethod
    def profile_property(cls,
        fn: ProFunc | None = None,
        name: str | None = None
    ) -> ProFunc | Callable[[ProFunc], ProFunc]:
        def decorator(func: ProFunc) -> ProFunc:
            bucket = cls._profile_property_registry[cls]
            bucket[name or func.__name__] = func
            return func
        if fn is None:
            return decorator
        return decorator(fn)

# ------------------------------------------------------------------
# Root profile
# ------------------------------------------------------------------

class Profile(ProfileBase):
    """
    Root profile wrapping a full `SimSnap`. Creates bins on initialization.

    Usage
    -----
    >>> prof = Profile(sim, nbins=50, type='lin', weight_by='mass')
    >>> mprof = prof['mass']          # per-bin mean (default)
    >>> m50 = prof['mass']['p50']     # median via percentile syntax
    >>> disp = prof['vel']['dispersion']

    Indexing
    --------
    - Filtering with a pynbody filter returns a `SubProfile`.
    - String key returns a `ProfileArray`.
    - Chained string on returned `ProfileArray` computes statistics (see `StatisticBase` registry).
    """

    def __init__(
        self,
        sim: SimSnap,
        *,
        weight: str | AllArray | Callable[[SimSnap], AllArray] | None = None,
        bins_by: str | BinByFunc = "r",
        bins_area: str | BinsAreaFunc = "spherical_shell",
        bins_type: str | BinsAlgorithmFunc = "lin",
        nbins: int | AllArray = 100,
        bin_min: float | None = None,
        bin_max: float | None = None,
        bins_set: BinsSet | None = None,
        **kwargs: Any):
        # Build or reuse BinsSet
        if bins_set is not None:
            bset = bins_set
        else:
            bset = BinsSet(
                bins_by=bins_by, bins_area=bins_area, bins_type=bins_type,
                nbins=nbins, bin_min=bin_min, bin_max=bin_max,**kwargs)
        weight_arr: AllArray | None
        if weight is None:
            weight_arr = None
        elif isinstance(weight, str):
            weight_arr = sim[weight]
        elif callable(weight):
            weight_arr = weight(sim)
        else:
            weight_arr = weight

        super().__init__(sim, bset, weight=weight_arr, parent=None)
        self._subs_cache: dict[SimSnap, SubProfile] = {}


    def get_subprofile(self, subset: SimSnap) -> SubProfile:
        if subset in self._subs_cache:
            return self._subs_cache[subset]
        subprof = self._spawn(subset)
        self._subs_cache[subset] = subprof
        return subprof


    @property
    def nsubs(self) -> int:
        return len(self._subs_cache)

    @property
    def total_cached_arr(self) -> int:
        par = self.num_cached_arr
        sub = sum(subprof.num_cached_arr for subprof in self._subs_cache.values())
        return par + sub

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        fams = [f.name for f in self.families()]
        parent_flag = "root" if self._parent is None else "sub"
        return f"<{cls} type={parent_flag} nbins={self.nbins} families={fams} nsubs={len(self._subs_cache)} ncache={self.total_cached_arr}>"

# ------------------------------------------------------------------
# SubProfile (subset view)
# ------------------------------------------------------------------

class SubProfile(ProfileBase):
    """
    A subset view of a parent `Profile`, sharing bin edges but recalculating
    per-bin assignments for the subset's particles.
    """

    def get_subprofile(self, subset: SimSnap) -> SubProfile:
        return self.parent.get_subprofile(subset)

    def _resolve_field(self, key: str) -> ProfileArray:
        # avoid recalculating bin properties for subsets
        if key in _BIN_PROPERTY_KEYS:
            return self.parent._resolve_field(key)
        return super()._resolve_field(key)

@ProfileBase.profile_property
def density(pro: ProfileBase) -> AllArray:
    return pro["mass"]["sum"] / pro["binsize"]
