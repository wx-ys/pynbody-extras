"""Cache and spawning of :class:`SubBinNDResult` sub-results."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.family import Family
from pynbody.filt import Filter
from pynbody.snapshot import SimSnap

from pynbodyext.core.calculate.nodes.filters import FilterBase

from .selectors import is_bool_array

if TYPE_CHECKING:
    from .result import BinNDResult, SubBinNDResult


def _is_sim_like(value: Any) -> bool:
    return isinstance(value, SimSnap)


class BinSubresultStore:
    def __init__(self, owner: BinNDResult) -> None:
        self._owner = owner
        self._cache: dict[Any, SubBinNDResult] = {}

    def values(self):
        return self._cache.values()

    def count(self) -> int:
        root = self._owner.root
        return len(root._subresults._cache)

    def total_cached_arr(self) -> int:
        root = self._owner.root
        total = root.num_cached_arr
        for subresult in root._subresults.values():
            total += subresult.num_cached_arr
        return total

    def get(self, subset: Any, *, cache_key: Any = None) -> SubBinNDResult:
        owner = self._owner
        if not owner.is_root:
            return owner.root._subresults.get(subset, cache_key=cache_key)

        key = cache_key if cache_key is not None else self.subset_cache_key(subset)
        if key in self._cache:
            return self._cache[key]

        sub = self.spawn(subset)
        self._cache[key] = sub
        return sub

    def spawn(self, subset: Any) -> SubBinNDResult:
        owner = self._owner
        return owner._calculator._executor().spawn_result(owner.root, subset)

    def subset_cache_key(self, subset: Any) -> Any:
        root_sim = self._owner.root.sim
        if subset is root_sim:
            return "__root__"
        if hasattr(subset, "get_index_list"):
            try:
                indices = np.asarray(subset.get_index_list(root_sim), dtype=np.int64)
            except Exception as exc:
                raise TypeError("SimSnap subset cannot be mapped to the root BinNDResult sim.") from exc
            # Hash bytes to keep the dict key O(1) in memory; SHA-1 collision risk is negligible.
            digest = hashlib.sha1(indices.tobytes()).hexdigest()
            return ("indices", int(indices.shape[0]), digest)
        raise TypeError("SubBinNDResult requires a SimSnap subset that can be mapped to the root sim.")

    def from_key(self, key: Any) -> SubBinNDResult:
        """Resolve *key* to a SubBinNDResult, choosing a readable cache key where possible."""
        owner = self._owner

        if isinstance(key, FilterBase):
            sub = owner.sim[key]
            if not _is_sim_like(sub):
                raise TypeError("FilterBase selector did not produce a SimSnap subset.")
            return self.get(sub, cache_key=("filter", key.to_signature().pretty()))

        if isinstance(key, Filter):
            sub = owner.sim[key]
            if not _is_sim_like(sub):
                raise TypeError("Filter selector did not produce a SimSnap subset.")
            return self.get(sub, cache_key=("pynfilter", repr(key)))

        if isinstance(key, Family):
            sub = owner.sim[key]
            if not _is_sim_like(sub):
                raise TypeError("Family selector did not produce a SimSnap subset.")
            return self.get(sub, cache_key=("family", key.name))

        # Generic: bool mask, arbitrary sim subscript, …
        if is_bool_array(key):
            mask = np.asarray(key, dtype=bool)
            if len(mask) == len(owner.sim):
                return self.get(owner.sim[mask])
            if len(mask) == owner.nbins:
                raise TypeError("Bin boolean masks must use bins.particles_at_bin[mask].")
            raise ValueError("Boolean selector length must match the current sim length for particle selection.")

        try:
            subset = owner.sim[key]
        except Exception as exc:
            raise TypeError(f"Unsupported BinNDResult selector: {key!r}.") from exc

        if _is_sim_like(subset):
            return self.get(subset)

        raise TypeError(f"Selector did not produce a SimSnap subset: {type(subset)!r}.")
