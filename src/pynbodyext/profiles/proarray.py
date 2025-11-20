"""
Profile-array utilities for binned profiles built on top of :class:`~pynbody.array.SimArray`.

This module provides:

- :class:`~pynbodyext.profiles.proarray.ProfileArray`: a lightweight wrapper around
  :class:`~pynbody.array.SimArray` representing a 1D profile sampled over profile bins.
  It understands how to compute per-bin statistics (mean, median, percentiles, dispersion, etc.)
  from a per-particle source field.
- A tiny statistic plug-in system based on :class:`~pynbodyext.profiles.proarray.StatisticBase`,
  with several built-ins:
  :class:`~pynbodyext.profiles.proarray.Mean`,
  :class:`~pynbodyext.profiles.proarray.Median`,
  :class:`~pynbodyext.profiles.proarray.Percentile`,
  :class:`~pynbodyext.profiles.proarray.RMS`,
  :class:`~pynbodyext.profiles.proarray.Dispersion`,
  and helpers like :class:`~pynbodyext.profiles.proarray.Abs`, :class:`~pynbodyext.profiles.proarray.Sum`,
  and :class:`~pynbodyext.profiles.proarray.Sum_w`.

Concepts
--------

Two "sources" exist for a :class:`~pynbodyext.profiles.proarray.ProfileArray`:

- "per_particle": created from a per-particle field (length = number of particles).
  In this case, the default profile stored in the array is the mean per bin,
  and additional statistics can be requested on demand, e.g. ``profile_array["median"]``
  or ``profile_array["p84"]``.
- "per_bin": created from an already aggregated per-bin array (length = number of bins).
  In this case, further statistics based on the original per-particle data are disabled,
  because per-particle inputs are not known. If a per-bin array is provided together with a
  ``mode`` (e.g., ``"mean"``), the instance is tagged as if it originated from a per-particle
  field so the label is preserved for representation; computation still refers back to the named
  field in the simulation.

Units
-----

If a :class:`~pynbody.array.SimArray` per-particle field is used as input,
the computed per-bin profile inherits its units and simulation context.

Caching
-------

Statistics may be cached by the owning profile object (see
:meth:`~pynbodyext.profiles.profile.ProfileBase.is_cached` and
:meth:`~pynbodyext.profiles.profile.ProfileBase.cache`), so repeated queries like
``profile_array["median"]`` are fast.

Quick example
-------------

>>> # Given a :class:`~pynbodyext.profiles.profile.ProfileBase` instance `prof`
>>> # with bins and access to `prof.sim[...]`:
>>> v_rad = ProfileArray(prof, name="vr")            # mean radial velocity per bin
>>> v_rad = prof["vr"]                               # mean radial velocity per bin
>>> v_med = v_rad["median"]                          # median per bin, recomputed
>>> v_p16 = v_rad["p16"]                             # weighted/unweighted 16th percentile
>>> v_disp = v_rad["dispersion"]                     # dispersion (sqrt(E[v^2] - E[v]^2))

New statistics
--------------

Subclass :class:`~pynbodyext.profiles.proarray.StatisticBase` and implement ``__call__``; optionally
override ``valid`` to control the matching key. See :class:`~pynbodyext.profiles.proarray.Percentile`
for a pattern-based matcher.

"""

import re
import warnings
from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

import numpy as np
from pynbody.array import IndexedSimArray, SimArray

from pynbodyext.util._type import SimNpArray

if TYPE_CHECKING:
    from .profile import ProfileBase

__all__ = [
    "ProfileArray",
    "StatisticBase",
    "Mean",
    "Percentile",
    "RMS",
    "Median",
    "Dispersion",
]

class ProfileArray(SimArray):
    """
    A profile-valued 1D array bound to a specific profile definition and source field.

    This class extends :class:`~pynbody.array.SimArray` and behaves like a standard
    NumPy/pynbody array. It also carries enough context to compute per-bin statistics
    from a per-particle source field on demand.

    Construction rules (array shape)
    --------------------------------
    - If ``array`` is ``None``, the per-particle field ``profile.sim[name]`` is used.
    - If the array length equals the number of particles in ``profile.sim``,
      the instance is considered "per_particle": the base values stored are,
      by default, the mean per bin unless ``mode`` specifies a different statistic
      to precompute.
    - If the array length equals the number of bins in ``profile``, the array
      is treated as already per-bin ("per_bin") and used as-is. If ``mode`` is not
      ``None``, it serves as a label describing which statistic was used.

    Indexing rules
    --------------
    - Numeric/slice/boolean indexing behaves like on :class:`~pynbody.array.SimArray`.
      Views created by slicing are marked as views in the representation and do not
      recompute statistics.
    - String indexing is overloaded to request a statistic by name, e.g.,
      ``arr["median"]``, ``arr["mean"]``, ``arr["rms"]``, or ``arr["p16"]``.
      This returns a new :class:`~pynbodyext.profiles.proarray.ProfileArray` with per-bin
      values computed from the per-particle source field.

    Parameters
    ----------
    profile : :class:`~pynbodyext.profiles.profile.ProfileBase`
        The owning profile that defines bins, weights, and access to the per-particle
        simulation arrays (via ``profile.sim[name]``).
    name : str
        The name of the per-particle field in ``profile.sim`` (e.g., ``"vr"``,
        ``"rho"``, etc.). Also used as the label for per-bin arrays.
    array : :data:`~pynbodyext.profiles.bins.SimNpArray`, optional
        Input array. If ``None``, uses ``profile.sim[name]``. If provided, it must be
        either a per-particle array (length = number of particles) or a per-bin array
        (length = ``profile.nbins``).
    mode : str, optional
        The statistic key to precompute when ``array`` is per-particle (default: ``"mean"``).
        When ``array`` is per-bin, ``mode`` acts as a label describing which statistic produced
        the per-bin values.

    Notes
    -----
    - Units and simulation binding: when the per-particle input is a
      :class:`~pynbody.array.SimArray`, the computed per-bin values inherit
      the units and simulation binding.
    - Weighting: weights are taken from the profile via ``profile._weight``.
      A weight sum of zero in a bin results in ``nan`` for that bin.
    - Caching: If the owning profile supports caching, repeated requests for
      the same statistic are answered from cache.

    Examples
    --------
    - Compute basic statistics per bin:

      >>> # you can also use prof["vr"] to access the radial velocity profile
      >>> v_r = ProfileArray(prof, name="vr")     # mean per bin
      >>> vr_med = v_r["median"]             # median per bin
      >>> vr_p84 = v_r["p84"]                # 84th percentile per bin

    - Per-bin array as input:

      >>> precomputed = some_np_array_of_length_nbins
      >>> vr_mean = ProfileArray(prof, name="vr", array=precomputed, mode="mean")
    """

    __slots__ = ["_profile","_name","_source","_arr", "_mode","_is_view"]
    _registry: list[type["StatisticBase"]] = [] # Registered statistic calculators

    _profile: Optional["ProfileBase"]   # The owning profile that defines bins, weights, and access to per-particle simulation arrays
    _name: str                          # The name of the per-particle field in the profile's simulation
    _source: Literal["per_particle", "per_bin"] # Indicates whether the array is per-particle or per-bin
    _arr: SimNpArray | None           # The underlying array data, either per-particle or per-bin
    _mode: str | None                   # The statistic key used for computation or labeling
    _is_view: bool                      # Indicates if this is a view of another ProfileArray


    def __new__(cls,
        profile: "ProfileBase",
        *,
        name: str,
        array: SimNpArray | None = None,
        mode: str | None = None,
    ) -> "ProfileArray":
        """
        Allocate and construct a new :class:`~pynbodyext.profiles.proarray.ProfileArray` instance.

        Raises
        ------
        ValueError
            If ``name`` is not a string, or ``array`` is neither a numpy array
            nor a :class:`~pynbody.array.SimArray`-like object, or if the length
            of ``array`` does not match the particle count or bin count.
        """
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        array = profile.sim[name] if array is None else array

        if not (isinstance(array, (np.ndarray, SimArray, IndexedSimArray))):
            raise ValueError("array must be a numpy ndarray or SimArray, got " + str(type(array)))

        n_particles = len(profile.sim)
        n_bins = profile.nbins
        arr_len = len(array)


        if arr_len == n_particles:
            # per-particle → compute mean as default (or `mode` if provided)
            arr_pp = array
            compute_mode = mode if mode is not None else "mean"
            base, mode = cls._compute(profile, arr_pp, compute_mode=compute_mode)
            obj = super().__new__(cls, base)
            obj._source = "per_particle"
            obj._arr = arr_pp
            obj._name = name
            obj._mode = mode
        elif arr_len == n_bins:
            # per-bin → use as-is, but source depends on mode
            base = array
            obj = super().__new__(cls, base)
            obj._source = "per_bin" if mode is None else "per_particle"
            obj._arr = array
            obj._name = name
            obj._mode = mode
        else:
            raise ValueError(
                f"array length {arr_len} not compatible: expected number of particles ({n_particles}) "
                f"or number of bins ({n_bins})"
            )

        # attach context
        obj._profile = profile
        obj._is_view = False
        return obj


    def __init__(
        self,
        profile: "ProfileBase",
        *,
        name: str,
        array: SimNpArray | None = None,
        mode: str | None = None,
        ):
        """Initialize the instance (no-op; all logic lives in __new__)."""
        # Intentionally empty: initialization handled in __new__.

    @classmethod
    def _compute(
        cls,
        profile: "ProfileBase",
        arr: SimNpArray | str,
        compute_mode: str,
        ) -> tuple[SimArray, str]:
        """
        Compute a per-bin statistic array for the given source.

        Parameters
        ----------
        profile : :class:`~pynbodyext.profiles.profile.ProfileBase`
            The owning profile providing bin membership and optional weights.
        arr : :data:`~pynbodyext.profiles.bins.SimNpArray` or str
            Source array. If a string, it is interpreted as ``profile.sim[arr]``.
            If array-like, it is treated as a per-particle field and binned.
        compute_mode : str
            Statistic key to compute (e.g. ``"mean"``, ``"median"``, ``"p16"``,
            ``"rms"``, ``"sum"``, ``"disp"``).

        Returns
        -------
        (out, mode) : tuple[:class:`~pynbody.array.SimArray`, str]
            The computed per-bin array and the resolved statistic key (which may be
            normalized or canonicalized compared to the input).

        Notes
        -----
        Implementations should respect the profile weights (``profile._weight``)
        and return ``nan`` where the bin has zero effective weight.
        """
        calculator = cls.get_statistic(compute_mode)
        if calculator is None:
            raise ValueError(f"Statistic '{compute_mode}' not found")

        vals = np.zeros(profile.nbins)
        weights = profile._weight

        if isinstance(arr, str):
            arr_pp = profile.sim[arr]
        else:
            arr_pp = arr

        for i,ind in enumerate(profile.binind):
            if len(ind) == 0:
                vals[i] = np.nan
                continue
            sub = arr_pp[ind]
            w = None if weights is None else weights[ind]
            vals[i] = calculator(sub, w)

        res_val = vals.view(SimArray)
        if isinstance(arr_pp, SimArray):
            res_val.units = arr_pp.units
            res_val.sim = arr_pp.sim
        return res_val, calculator.key


    @classmethod
    def get_statistic(cls, key: str) -> Union["StatisticBase", None]:
        """
        Resolve a statistic plug-in by key.

        This scans the registry populated by :meth:`StatisticBase.__init_subclass__`
        and returns an instance configured for the provided key, or ``None`` if no
        match is found.

        Parameters
        ----------
        key : str
            Statistic selector (e.g., ``"mean"``, ``"median"``, ``"p84"``, ``"abs_mean"``).

        Returns
        -------
        :class:`~pynbodyext.profiles.proarray.StatisticBase` or None
            The statistic instance if recognized, else ``None``.
        """
        for stat in cls._registry:
            res = stat.valid(key)
            if res is not None:
                return res
        return None

    def __array_finalize__(self, obj: Union["ProfileArray", None]) -> None:
        """
        NumPy/pynbody subclass hook for view creation.

        Ensures that when a view is created (e.g., slicing), the profile context
        is carried over, and the instance is marked as a view. Units and sim
        binding are handled by :class:`~pynbody.array.SimArray`.

        Parameters
        ----------
        obj : :class:`~pynbodyext.profiles.proarray.ProfileArray` or None
            The source object from which the view is created.
        """
        # Let SimArray handle its own finalize first (units/sim)
        super().__array_finalize__(obj)
        if obj is None:
            return
        else:
            # This is a view/slice of an existing ProfileData
            self._profile = getattr(obj, "_profile", None)
            self._name = getattr(obj, "_name", "None")
            self._source = getattr(obj, "_source", "per_bin")
            self._arr = getattr(obj,"_arr", None)
            self._mode = getattr(obj, "_mode", None)
            self._is_view = True

    # ---- public API ----

    def _check_base_for_stats(self) -> bool:
        """
        Guard that the instance can compute further statistics.

        Raises
        ------
        TypeError
            If the instance is detached from any profile.
        RuntimeError
            If the instance was built from per-bin values (no per-particle source),
            or if the name is not set.

        Returns
        -------
        bool
            ``True`` if additional statistics can be computed.
        """
        if self._profile is None:
            warnings.warn(
                "Statistics are only available on the base ProfileData.",
                stacklevel=2,
            )
            return False
        if self._source == "per_bin":
            warnings.warn(
                "Statistics are only available on the base ProfileData.",
                stacklevel=2,
            )
            return False
        return True
    @property
    def profile(self) -> Union["ProfileBase", None]:
        """
        The owning :class:`~pynbodyext.profiles.profile.ProfileBase` or ``None`` if detached.
        """
        return self._profile

    @profile.setter
    def profile(self, value: "ProfileBase") -> None:
        """Set the owning profile."""
        self._profile = value


    @overload
    def __getitem__(self, item: str | slice | np.ndarray) -> "ProfileArray":
        ...
    @overload
    def __getitem__(self, item: int) -> np.dtype:
        ...
    def __getitem__(self, item: str | int | slice | np.ndarray) -> Union["ProfileArray", np.dtype]:
        """
        Overloaded indexing.

        - If ``item`` is a string, treat it as a statistic key and compute the
          corresponding per-bin values from the per-particle field (e.g.
          ``arr["median"]``, ``arr["p84"]``). A new :class:`~pynbodyext.profiles.proarray.ProfileArray`
          is returned, possibly served from cache.
        - Otherwise, forward to standard array indexing and return a view with
          the appropriate type.

        Parameters
        ----------
        item : str or int or slice or :class:`~numpy.ndarray`
            Statistic key or standard indexer.

        Returns
        -------
        :class:`~pynbodyext.profiles.proarray.ProfileArray` or dtype
            A new per-bin statistic array (when string key is used), or a view
            of the underlying array for standard indexing.

        Raises
        ------
        TypeError, RuntimeError
            If a statistic is requested on an instance that cannot compute
            further statistics (see :meth:`_check_base_for_stats`).
        """
        # dispatch on stats name vs normal indexing
        if isinstance(item, str):
            if not self._check_base_for_stats():
                raise RuntimeError("Cannot compute statistics on this ProfileArray")
            profile = cast("ProfileBase", self._profile)
            name = self._name
            if profile.is_cached(name,item):
                return profile.get_cached(name,item)
            out,mode = self._compute(profile, name, compute_mode=item)
            res = ProfileArray(profile, name=name, array=out, mode=mode)
            profile.cache(res)
            return res
        # normal ndarray-like indexing
        return super().__getitem__(item)

    def keys(self) -> list[str]:
        """
        Return available statistic keys for this array instance.

        Returns
        -------
        list[str]
            The list of statistic keys recognized by the registry for this instance.
        """
        key = []
        if self._check_base_for_stats():
            for i in ProfileArray._registry:
                if i.example_name is not None:
                    key.append(i.example_name)
        return key

    def _ipython_key_completions_(self) -> list[str]:
        return self.keys()

    def __repr__(self):
        """
        Return a representation including provenance.

        The representation appends a tag of the form
        ``'source::name[::mode][::View]'``, where:

        - ``source`` is either ``per_particle`` or ``per_bin``,
        - ``name`` is the source field name, and
        - ``mode`` is the statistic used for the stored values (if any).
        - ``View`` indicates that the object is a view/slice.

        Returns
        -------
        str
        """
        x = SimArray.__repr__(self)
        flag = ", '" + str(self._source)+"::" + str(self._name)
        if self._mode is not None:
            flag += "::" + str(self._mode)
        if self._is_view:
            flag += "::View"
        return x[:-1] + flag + "')"


class StatisticBase:
    """
    Abstract base class for per-bin statistics.

    Each subclass automatically registers itself with :class:`~pynbodyext.profiles.proarray.ProfileArray`
    so that :meth:`~pynbodyext.profiles.proarray.ProfileArray.get_statistic` can discover and dispatch to it.

    Subclassing
    -----------
    Implement ``__call__(arr, weight)`` to return a scalar value given the values and optional
    weights for one bin. Optionally override ``valid`` to match one or more string keys,
    returning an instance of your statistic when matched, or ``None`` otherwise.

    Notes
    -----
    - ``arr`` and ``weight`` are arrays corresponding to values and weights of the particles
      that fall into a single bin.
    - Implementations should return :data:`numpy.nan` when an empty bin is given.
    """
    example_name: str | None = None
    def __init_subclass__(cls) -> None:
        """Register subclasses for discovery."""
        if getattr(cls,"example_name", None) is None:
            warnings.warn(
                f"StatisticBase subclass {cls.__name__} missing example_name attribute, would be good to add one for clarity.",
                DeprecationWarning,
                stacklevel=2,
            )
        ProfileArray._registry.append(cls)

    def __init__(self, key:str):
        """
        Parameters
        ----------
        key : str
            The key that matched this statistic (useful for display or reuse).
        """
        self.key = key

    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:
        """
        Evaluate the statistic for a single bin.

        Parameters
        ----------
        arr : :data:`~pynbodyext.profiles.bins.SimNpArray`
            The per-particle values in this bin.
        weight : :data:`~pynbodyext.profiles.bins.SimNpArray` or None
            Optional per-particle weights aligned with ``arr``.

        Returns
        -------
        float or int or :class:`~numpy.floating`
            The per-bin statistic value.
        """
        raise NotImplementedError
    @classmethod
    def valid(cls, key:str)-> Union["StatisticBase", None]:
        """
        Return an instance of this statistic if ``key`` matches, else ``None``.

        Default behavior only matches the class name. Subclasses typically
        override this to match more user-friendly keys.

        Parameters
        ----------
        key : str
            The lookup key.

        Returns
        -------
        StatisticBase or None
        """
        if key == cls.__name__:
            return cls(key)
        else:
            return None

class Mean(StatisticBase):
    """Weighted or unweighted arithmetic mean per bin."""
    example_name: str = "mean"
    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:
        if weight is not None:
            return (arr * weight).sum() / weight.sum()
        else:
            return arr.mean()

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() == "mean":
            return cls("mean")
        else:
            return None

class Sum(StatisticBase):
    """Unweighted sum of per-particle values per bin."""
    example_name: str = "sum"
    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:
        return arr.sum()

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() == "sum":
            return cls("sum")
        else:
            return None

class Sum_w(StatisticBase):
    """Weighted sum of per-particle values per bin."""
    example_name: str = "sum_w"
    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:
        if weight is not None:
            return (arr * weight).sum()
        else:
            return arr.sum()

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() == "sum_w":
            return cls("sum_w")
        else:
            return None

class Percentile(StatisticBase):
    """
    Weighted or unweighted percentile per bin.

    Keys of the form ``"pXX"`` (e.g., ``"p16"``, ``"p50"``, ``"p84"``) are recognized.
    """
    example_name: str = "p16"
    def __init__(self, key: str, percentile: int):
        super().__init__(key)
        self.percentile = percentile


    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:

        if len(arr) == 0:
            return np.nan
        else:
            indices = np.argsort(arr)
            arr_sorted = arr[indices]
            if weight is None:
                weight_cumsum = np.linspace(0,1,len(arr_sorted))
            else:
                weight_sorted = weight[indices]

                weight_cumsum = weight_sorted.cumsum()
                # normalize to (0,1)
                weight_cumsum -= weight_cumsum[0]
                weight_cumsum /= weight_cumsum[-1]

            return np.interp(self.percentile/100, weight_cumsum, arr_sorted)

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        k = key.lower()
        if not k.startswith("p"):
            return None
        m = re.match(r"^p(\d{1,3})$", k)
        if not m:
            return None
        pct = int(m.group(1))
        if 0 <= pct <= 100:
            inst = cls(k, pct)
            return inst
        return None

    @classmethod
    def name(cls):
        return "p**"


class RMS(StatisticBase):
    """Root-mean-square (quadratic mean) of values per bin."""
    example_name: str = "rms"
    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:
        if len(arr) == 0:
            return np.nan
        if weight is not None:
            return np.sqrt((arr**2 * weight).sum() / weight.sum())
        else:
            return np.sqrt((arr**2).mean())

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() == "rms":
            return cls("rms")
        else:
            return None

class Median(StatisticBase):
    """Median per bin. Equivalent to ``p50``."""
    example_name: str = "median"
    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:

        return Percentile(self.key, 50)(arr, weight)

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() in ["med", "median"]:
            return cls("median")
        else:
            return None

class Abs(StatisticBase):
    """
    A wrapper statistic that applies :func:`abs` before delegating to another statistic.

    Examples
    --------
    - ``"abs"`` or ``"abs_mean"`` → mean of absolute values (weighted if provided)
    - ``"abs_p16"`` → 16th percentile of absolute values
    - ``"abs_median"`` → median of absolute values
    - ``"abs_sum"`` → sum of absolute values
    - ``"abs_sum_w"`` → weighted sum of absolute values
    """
    example_name: str = "abs"

    def __init__(self, key: str, substat: "StatisticBase"):
        super().__init__(key)
        self._substat = substat

    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    ) -> float | int | np.floating:
        return self._substat(np.abs(arr), weight)

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        k = key.lower()
        # accept "abs" and alias to "abs_mean"
        if k in ["abs","abs_"]:
            subkey = "mean"
        elif k.startswith("abs_"):
            subkey = k[4:]
        else:
            return None

        # Delegate to the underlying statistic (e.g., "mean", "p16", "median", "sum_w", etc.)
        sub = ProfileArray.get_statistic(subkey)
        if sub is None:
            # Don't claim keys we can't actually compute
            return None
        standard_key = "abs_"+sub.key
        return cls(standard_key, sub)

class Dispersion(StatisticBase):
    """
    Velocity-like dispersion per bin, computed as ``sqrt(E[v^2] - E[v]^2)`` (weighted if provided).
    """
    example_name: str = "disp"
    def __call__(
        self,
        arr: SimNpArray,
        weight: SimNpArray | None,
    )-> float | int | np.floating:
        if len(arr) == 0:
            return np.nan
        if weight is not None:
            wsum = float(np.asarray(weight.sum()))
            if wsum == 0:
                return np.nan
            sq_mean = float(np.asarray((arr**2 * weight).sum() / wsum))
            mean_sq = float(np.asarray((arr * weight).sum() / wsum)) ** 2
        else:
            sq_mean = float(np.asarray((arr**2).mean()))
            mean_sq = float(np.asarray(arr.mean())) ** 2

        # numerical safety: clamp small negative due to fp errors
        diff = sq_mean - mean_sq
        if diff < 0 and diff > -1e-12:
            diff = 0.0
        return float(np.sqrt(diff)) if diff >= 0 else np.nan

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() in ["dispersion","disp"]:
            return cls("disp")
        else:
            return None
