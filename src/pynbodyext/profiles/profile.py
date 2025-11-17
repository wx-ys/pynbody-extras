from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

import numpy as np
from pynbody import filt as _pyn_filt
from pynbody.family import Family
from pynbody.snapshot import SimSnap

from .bins import BinByFunc, BinsSet, SimOrNpArray
from .proarray import ProfileArray

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pynbody.array import SimArray

# ------------------------------------------------------------------
# Assumptions / light shims (remove if already defined elsewhere)
# ------------------------------------------------------------------
FilterLike: TypeAlias = _pyn_filt.Filter | np.ndarray | Family | slice | int | np.int32 | np.int64

ProFunc: TypeAlias = Callable[["ProfileBase"], ProfileArray | SimOrNpArray]

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

    _profile_property_registry: dict[str, ProFunc] = {}

    def __init__(self,
                 sim: SimSnap,
                 bins_set: BinsSet,
                 weight_key: str | None,
                 parent: Profile | None = None):
        self.sim: SimSnap = sim
        self._bins: BinsSet = bins_set
        self._parent: Profile | None = parent

        # Weight handling: if key provided, materialize array; else None
        self._weight_key = weight_key
        self._weight: SimArray | np.ndarray | None = (
            sim[weight_key] if (weight_key is not None) else None
        )

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
        return SubProfile(
            sim_subset,
            bins_set=child_bins,
            weight_key=self._weight_key,
            parent=root_parent,   # ensure single parent
        )

    # ---- Field resolution ----
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

        if key in _BIN_PROPERTY_KEYS or key in self._profile_property_registry:
            if key == "rbins":
                arr = self.rbins
            elif key == "dr":
                arr = self.dr
            elif key == "binsize":
                arr = self.binsize
            elif key == "npart_bins":
                arr = self.npart_bins

            if key in self._profile_property_registry:
                arr = self._profile_property_registry[key](self)
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
            cls._profile_property_registry[name or func.__name__] = func
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

    def __init__(self,
                 sim: SimSnap,
                 *,
                 weight_by: str | None = "mass",
                 bins_set: BinsSet | None = None,
                 type: str = "lin",
                 nbins: int = 100,
                 rmin: float | None = None,
                 rmax: float | None = None,
                 bins: ArrayLike | None = None,
                 calc_x: BinByFunc | None = None):
        # Build or reuse BinsSet
        if bins_set is not None:
            bset = bins_set
        else:
            if bins is not None:
                nbins_param: int | ArrayLike = bins
            else:
                nbins_param = nbins
            # If calc_x provided, wrap in callable respecting signature expected by BinsSet._resolve_x
            bins_by = (lambda s: calc_x(s)) if calc_x else "r"
            bset = BinsSet(
                bins_by=bins_by,
                bins_area="spherical_shell",     # choose a sensible default; adjust if 'auto' is implemented
                bins_type=type,
                nbins=nbins_param,
                bin_min=rmin,
                bin_max=rmax,
            ).__call__(sim)

        super().__init__(sim, bset, weight_key=weight_by, parent=None)
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

    def __init__(self,
                 sim_subset: SimSnap,
                 *,
                 bins_set: BinsSet,
                 weight_key: str | None,
                 parent: Profile):
        super().__init__(sim_subset, bins_set, weight_key=weight_key, parent=parent)

    def get_subprofile(self, subset: SimSnap) -> SubProfile:
        return self.parent.get_subprofile(subset)

    def _resolve_field(self, key: str) -> ProfileArray:
        if key in _BIN_PROPERTY_KEYS:
            return self.parent._resolve_field(key)
        return super()._resolve_field(key)

@ProfileBase.profile_property
def density(pro: ProfileBase) -> ProfileArray | SimOrNpArray:
    return pro["mass"]["sum"] / pro["binsize"]
