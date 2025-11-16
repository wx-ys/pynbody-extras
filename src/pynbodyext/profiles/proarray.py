
import re
from typing import Literal, Union, cast

import numpy as np
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

#if TYPE_CHECKING:
#    from .profile import Profile


class Profile:

    nbins: int
    _weight: np.ndarray | None
    binind: list[np.ndarray]
    sim: SimSnap

    def cache(self,pro_arr: "ProfileArray") -> None:
        pass

    def is_cached(self, key: str, item: str) -> bool:
        return True

    def get_cached(self, key: str, item: str) -> "ProfileArray":
        return ProfileArray(self, key=key, mode=item)


class ProfileArray(SimArray):

    __slots__ = ["_profile","_key","_source","_arr", "_mode"]
    _registry: list[type["StatisticBase"]] = []

    _profile: Profile | None
    _key: str | None
    _source: Literal["per_particle", "per_bin"]
    _arr: SimArray | np.ndarray | None
    _mode: str | None

    def __new__(cls,
                profile: "Profile",
                *,
                key: str,
                array: np.ndarray | SimArray | None = None,
                mode: str | None = None,
                ) -> "ProfileArray":
        if not isinstance(key, str):
            raise ValueError("key must be a string")

        array = profile.sim[key] if array is None else array

        if not (isinstance(array, (np.ndarray, SimArray))):
            raise ValueError("array must be a numpy ndarray or SimArray")

        n_particles = len(profile.sim)
        n_bins = profile.nbins
        arr_len = len(array)


        if arr_len == n_particles:
            # per-particle → compute mean as default
            arr_pp = array
            compute_mode = mode if mode is not None else "mean"
            base = cls._compute(profile, arr_pp, compute_mode=compute_mode)
            obj = super().__new__(cls, base)
            obj._source = "per_particle"
            obj._arr = arr_pp
            obj._key = key
            obj._mode = compute_mode
        elif arr_len == n_bins:
            # per-bin → use as-is, but source depends on mode
            base = array
            obj = super().__new__(cls, base)
            obj._source = "per_bin" if mode is None else "per_particle"
            obj._arr = array
            obj._key = key
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
        profile: "Profile",
        *,
        key: str,
        array: np.ndarray | SimArray | None = None,
        mode: str | None = None,
        ):
        """ """

    @classmethod
    def _compute(
        cls,
        profile: "Profile",
        arr: SimArray | np.ndarray | str,
        compute_mode: str,
        ) -> SimArray:

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
        return res_val


    @classmethod
    def get_statistic(cls, key: str) -> Union["StatisticBase", None]:
        for stat in cls._registry:
            res = stat.valid(key)
            if res is not None:
                return res
        return None

    def __array_finalize__(self, obj: Union["ProfileArray", None]) -> None:
        # Let SimArray handle its own finalize first (units/sim)
        super().__array_finalize__(obj)
        if obj is None:
            # Default init when created from buffer – set safe defaults
            self._profile = None
            self._key = None
            self._source = "per_bin"
            self._arr = None
            self._mode = None
        else:
            # This is a view/slice of an existing ProfileData – disable stats on views
            self._profile = getattr(obj, "_profile", None)
            self._key = getattr(obj, "_key", None)
            self._source = getattr(obj, "_source", "per_bin")
            self._arr = getattr(obj, "_arr", None)
            self._mode = getattr(obj, "_mode", None)

    # ---- public API ----

    def _ensure_base_for_stats(self):
        if self._profile is None:
            raise TypeError("Statistics are only available on the base ProfileData.")
        if self._source == "per_bin":
            raise RuntimeError("Per-bin array cannot be used for per-particle statistics")

    def __getitem__(self, item: str | int | slice | np.ndarray) -> Union["ProfileArray", SimArray, np.ndarray]:
        # dispatch on stats name vs normal indexing
        if isinstance(item, str):
            self._ensure_base_for_stats()
            profile = cast("Profile", self._profile)
            key = cast("str", self._key)
            if profile.is_cached(key,item):
                return profile.get_cached(key,item)
            out = self._compute(profile, key, compute_mode=item)
            res = ProfileArray(profile, key=key, array=out, mode=item)
            profile.cache(res)
            return res
        # normal ndarray-like indexing
        return super().__getitem__(item)


    def __repr__(self):
        x = SimArray.__repr__(self)
        return x[:-1] + ", '" + str(self._source)+"::" + str(self._key) + "')"



class StatisticBase:

    def __init_subclass__(cls) -> None:
        ProfileArray._registry.append(cls)

    def __init__(self, key:str):
        self.key = key

    def __call__(
        self,
        arr: SimArray | np.ndarray,
        weight: SimArray | np.ndarray | None,
    )-> float | int | np.floating:
        raise NotImplementedError
    @classmethod
    def valid(cls, key:str)-> Union["StatisticBase", None]:
        if key == cls.__name__:
            return cls(key)
        else:
            return None

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError

class Mean(StatisticBase):

    def __call__(
        self,
        arr: SimArray | np.ndarray,
        weight: SimArray | np.ndarray | None,
    )-> float | int | np.floating:
        if weight is not None:
            return (arr * weight).sum() / weight.sum()
        else:
            return arr.mean()

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() == "mean":
            return cls(key)
        else:
            return None

    @classmethod
    def name(cls):
        return "mean"


class Percentile(StatisticBase):

    def __init__(self, key: str, percentile: int):
        super().__init__(key)
        self.percentile = percentile


    def __call__(
        self,
        arr: SimArray | np.ndarray,
        weight: SimArray | np.ndarray | None,
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
            inst = cls(key, pct)
            return inst
        return None

    @classmethod
    def name(cls):
        return "p**"


class RMS(StatisticBase):

    def __call__(
        self,
        arr: SimArray | np.ndarray,
        weight: SimArray | np.ndarray | None,
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
            return cls(key)
        else:
            return None

    @classmethod
    def name(cls):
        return "rms"

class Median(StatisticBase):

    def __call__(
        self,
        arr: SimArray | np.ndarray,
        weight: SimArray | np.ndarray | None,
    )-> float | int | np.floating:

        return Percentile(self.key, 50)(arr, weight)

    @classmethod
    def valid(cls, key: str) -> Union["StatisticBase", None]:
        if key.lower() in ["med", "median"]:
            return cls(key)
        else:
            return None

    @classmethod
    def name(cls):
        return "med"

class Dispersion(StatisticBase):

    def __call__(
        self,
        arr: SimArray | np.ndarray,
        weight: SimArray | np.ndarray | None,
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
            return cls(key)
        else:
            return None

    @classmethod
    def name(cls):
        return "disp"
