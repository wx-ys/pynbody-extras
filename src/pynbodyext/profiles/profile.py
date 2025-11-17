"""
High-level 1D profile interface for particle simulations.

This module provides:

- :class:`~pynbodyext.profiles.profile.Profile`: the root profile wrapping a full
  :class:`~pynbody.snapshot.SimSnap`. It constructs bins via
  :class:`~pynbodyext.profiles.bins.BinsSet` and exposes a dict-like interface to
  retrieve :class:`~pynbodyext.profiles.proarray.ProfileArray` fields.
- :class:`~pynbodyext.profiles.profile.SubProfile`: a subset view sharing the same bin edges.
- :class:`~pynbodyext.profiles.profile.ProfileBase`: shared mechanics (bin set, caching, weight).

Profiles support caching both base per-bin arrays and derived statistics. Fields are resolved
on demand: a string key retrieves a :class:`~pynbodyext.profiles.proarray.ProfileArray`, which
can itself be indexed with a statistic key (e.g., ``["median"]``, ``["p84"]``).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, overload

import numpy as np
from pynbody import filt as _pyn_filt
from pynbody.family import Family
from pynbody.snapshot import SimSnap

from .bins import (
    BinByFunc,
    BinsAlgorithmFunc,
    BinsAreaFunc,
    BinsSet,
    RegistBinAlgorithmString,
    RegistBinAreaString,
    RegistBinByString,
    SimOrNpArray,
)
from .proarray import ProfileArray

if TYPE_CHECKING:
    from pynbody.array import SimArray

# ------------------------------------------------------------------
# Assumptions / light shims (remove if already defined elsewhere)
# ------------------------------------------------------------------
FilterLike: TypeAlias = _pyn_filt.Filter | np.ndarray | Family | slice | int | np.int32 | np.int64

AllArray: TypeAlias = ProfileArray | SimOrNpArray

ProType = TypeVar("ProType",bound="ProfileBase")
ProFunc: TypeAlias = Callable[[ProType], AllArray]
SimFunc: TypeAlias = Callable[[SimSnap], AllArray]

_BIN_PROPERTY_KEYS = ("rbins", "dr", "binsize", "npart_bins")  # per-bin properties


def _is_sim_like(x: Any) -> bool:
    return isinstance(x, SimSnap)

# ------------------------------------------------------------------
# Profile base
# ------------------------------------------------------------------
class ProfileBase:
    """
    Abstract base providing shared mechanics (bin set, caching, weights).

    Concrete classes
    ----------------
    - :class:`~pynbodyext.profiles.profile.Profile` (root)
    - :class:`~pynbodyext.profiles.profile.SubProfile` (subset view)

    Attributes
    ----------
    sim : :class:`~pynbody.snapshot.SimSnap`
        The underlying simulation snapshot.
    bins : :class:`~pynbodyext.profiles.bins.BinsSet`
        The materialized bin definition (edges, centers, assignments).
    _weight : :class:`~pynbody.array.SimArray` or :class:`~numpy.ndarray` or None
        Optional per-particle weights aligned with ``sim``.
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
        """
        Store a computed statistic :class:`~pynbodyext.profiles.proarray.ProfileArray` in cache.

        Notes
        -----
        The cache is organized as ``_stats_cache[field_name][mode] -> ProfileArray``.
        """
        key = pro_arr._name
        mode = pro_arr._mode
        if key is None or mode is None:
            return
        bucket = self._stats_cache.setdefault(key, {})
        bucket[mode] = pro_arr

    def is_cached(self, key: str, item: str) -> bool:
        """
        Check whether a statistic array for ``key`` and ``item`` is available in cache.

        Parameters
        ----------
        key : str
            Field name (e.g., ``"mass"``).
        item : str
            Statistic key (e.g., ``"mean"``, ``"median"``, ``"p84"``).

        Returns
        -------
        bool
        """
        return key in self._stats_cache and item in self._stats_cache[key]

    def get_cached(self, key: str, item: str) -> ProfileArray:
        """
        Retrieve a cached statistic :class:`~pynbodyext.profiles.proarray.ProfileArray`.

        Raises
        ------
        KeyError
            If ``key`` or ``item`` is not found in the cache.
        """
        return self._stats_cache[key][item]

    # ---- Properties bridging to BinsSet ----

    @property
    def bins(self) -> BinsSet:
        """The :class:`~pynbodyext.profiles.bins.BinsSet` associated with this profile."""
        return self._bins

    @property
    def nbins(self) -> int:
        """Number of bins (:func:`len` of :attr:`~pynbodyext.profiles.bins.BinsSet.rbins`)."""
        return len(self._bins.rbins) if self._bins.rbins is not None else 0

    @property
    def binind(self) -> list[np.ndarray]:
        """List of particle index arrays per bin."""
        return self._bins.binind or []

    @property
    def rbins(self) -> SimArray | np.ndarray | None:
        """Bin centers from :class:`~pynbodyext.profiles.bins.BinsSet`."""
        return self._bins.rbins

    @property
    def bin_edges(self) -> SimArray | np.ndarray | None:
        """Bin edges from :class:`~pynbodyext.profiles.bins.BinsSet`."""
        return self._bins.bin_edges

    @property
    def binsize(self) -> SimArray | np.ndarray | None:
        """Area/volume per bin from :class:`~pynbodyext.profiles.bins.BinsSet`."""
        return self._bins.binsize

    @property
    def dr(self) -> SimArray | np.ndarray | None:
        """Approximate half-widths computed from :attr:`rbins`."""
        return self._bins.dr

    @property
    def npart_bins(self) -> np.ndarray | None:
        """Number of particles per bin."""
        return self._bins.npart_bins

    def families(self):
        """Proxy for :meth:`~pynbody.snapshot.SimSnap.families`."""
        return self.sim.families()

    def keys(self) -> list[str]:
        """
        Return available data/statistic keys for this profile (including per-bin properties).

        Returns
        -------
        list[str]
            Base data keys + statistic keys + ``("rbins", "dr", "binsize", "npart_bins")``.
        """
        # Return cached data keys + per-bin property keys (always exposed)
        data_keys = list(self._data_cache.keys()) + list(self._stats_cache.keys())
        return sorted(set(data_keys).union(_BIN_PROPERTY_KEYS))

    def property_keys(self) -> list[str]:
        """
        Return registered computed profile-property keys.

        These are functions registered via :meth:`ProfileBase.profile_property`.
        """
        par = self.parent
        reg_by_cls = par.__class__._profile_property_registry
        reg_keys: set[str] = set()
        for c in par.__class__.mro():
            reg_keys |= set(reg_by_cls.get(c, {}).keys())
        return sorted(reg_keys)
    def all_keys(self) -> list[str]:
        """Union of :meth:`keys` and :meth:`property_keys`."""
        return sorted(set(self.keys()).union(set(self.property_keys())))

    def _ipython_key_completions_(self) -> list[str]:
        return self.all_keys()

    @property
    def parent(self) -> Profile:
        """
        Return the ultimate root :class:`~pynbodyext.profiles.profile.Profile`.

        All subprofiles share a single root parent.
        """
        root: ProfileBase = self
        while root._parent is not None:
            root = root._parent
        # By construction, the root is a Profile
        return cast("Profile", root)

    @property
    def num_cached_arr(self) -> int:
        """Number of cached arrays held by this profile (base + statistics)."""
        return len(self._data_cache) + sum(len(v) for v in self._stats_cache.values())

    def _spawn(self, sim_subset: SimSnap) -> SubProfile:
        """
        Create a :class:`~pynbodyext.profiles.profile.SubProfile` for ``sim_subset``,
        sharing the same bin edges but recomputing particle assignments.

        Parameters
        ----------
        sim_subset : :class:`~pynbody.snapshot.SimSnap`
            The subset snapshot.
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
        """
        Look up a registered profile-property function for ``key`` on the root class hierarchy.
        """
        par = self.parent

        reg_by_cls = par.__class__._profile_property_registry
        for c in par.__class__.mro():
            bucket = reg_by_cls.get(c)
            if bucket and key in bucket:
                return bucket[key]
        return None

    def _resolve_field(self, key: str) -> ProfileArray:
        """
        Resolve a field key to a :class:`~pynbodyext.profiles.proarray.ProfileArray`.

        Behavior
        --------
        - If a cached base/statistic is present, return it.
        - If ``key`` is a per-bin property (e.g., ``"rbins"``), materialize it.
        - Else, create a base per-bin array from the per-particle field using the default
          statistic ``"mean"``.
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
        """
        Deprecated alias for creating a :class:`~pynbodyext.profiles.profile.SubProfile`.

        Use slicing/filtering on the profile instead, e.g. ``prof[pynbody.filt.Disc(...)]``.
        """
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
        """
        Main indexing entrypoint.

        - String key → return :class:`~pynbodyext.profiles.proarray.ProfileArray`
          (base representation; chain another string to compute a statistic).
        - Filter-like key (pynbody filter, family, boolean mask, slice, etc.) →
          return :class:`~pynbodyext.profiles.profile.SubProfile`.

        Examples
        --------
        >>> mprof = prof["mass"]           # base per-bin array (mean by default)
        >>> m50 = prof["mass"]["p50"]      # median via percentile syntax
        >>> disp = prof["vel"]["disp"]     # dispersion
        """

        # str -> ProfileArray
        if isinstance(key, str):
            return self._resolve_field(key)

        # FilterLike → subset
        subset = self.sim[key]
        return self.get_subprofile(subset)

    def __getattr__(self, name: str) -> SubProfile:
        """
        Mirror :class:`~pynbody.snapshot.SimSnap` pattern: attribute exposing a subsnap
        returns a :class:`~pynbodyext.profiles.profile.SubProfile`.
        """
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
    Root profile wrapping a full :class:`~pynbody.snapshot.SimSnap`. Creates bins on initialization.

    Usage
    -----
    >>> prof = Profile(sim, nbins=50, bins_type="lin", weight="mass")
    >>> zprof = prof["z"]          # per-bin mean (default)
    >>> z50 = prof["z"]["p50"]     # median via percentile syntax
    >>> disp = prof["z"]["disp"]

    Indexing
    --------
    - Filtering with a pynbody filter returns a :class:`~pynbodyext.profiles.profile.SubProfile`.
    - String key returns a :class:`~pynbodyext.profiles.proarray.ProfileArray`.
    - Chained string on the returned :class:`~pynbodyext.profiles.proarray.ProfileArray`
      computes statistics (see :class:`~pynbodyext.profiles.proarray.StatisticBase` registry).
    """

    def __init__(
        self,
        sim: SimSnap,
        *,
        weight: str | AllArray | Callable[[SimSnap], AllArray] | None = None,
        bins_by: RegistBinByString | BinByFunc = "r",
        bins_area: RegistBinAreaString | BinsAreaFunc = "spherical_shell",
        bins_type: RegistBinAlgorithmString | BinsAlgorithmFunc = "lin",
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
        """Return a cached or newly spawned :class:`~pynbodyext.profiles.profile.SubProfile`."""
        if subset in self._subs_cache:
            return self._subs_cache[subset]
        subprof = self._spawn(subset)
        self._subs_cache[subset] = subprof
        return subprof


    @property
    def nsubs(self) -> int:
        """Number of cached subprofiles."""
        return len(self._subs_cache)

    @property
    def total_cached_arr(self) -> int:
        """Total number of cached arrays across root and all cached subprofiles."""
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
    A subset view of a parent :class:`~pynbodyext.profiles.profile.Profile`, sharing bin edges
    but recalculating per-bin assignments for the subset's particles.
    """

    def get_subprofile(self, subset: SimSnap) -> SubProfile:
        """Delegate to the root :class:`~pynbodyext.profiles.profile.Profile` to reuse cache."""
        return self.parent.get_subprofile(subset)

    def _resolve_field(self, key: str) -> ProfileArray:
        """
        Resolve a field for the subset. Per-bin properties are forwarded to the root profile
        to avoid recalculation; other fields are resolved locally.
        """
        # avoid recalculating bin properties for subsets
        if key in _BIN_PROPERTY_KEYS:
            return self.parent._resolve_field(key)
        return super()._resolve_field(key)
