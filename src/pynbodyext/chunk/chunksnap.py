
from typing import Any, Union

import dask
import dask.array as da
import numpy as np
from pynbody.family import Family, _registry as family_registry
from pynbody.filt import Filter
from pynbody.snapshot import SimSnap

from .chunk import ChunkManager
from .simdaskarray import SimDaskArray, sim_from_dask
from .snapview import MiniSimSnap, SimSnapView


class ChunkDaskArrayLoader:

    simsnap: SimSnap
    chunks: ChunkManager
    minisnap: MiniSimSnap
    def __init__(self, simsnap: SimSnap, chunk_size: int = 1000_000, **kwargs: Any):

        self.simsnap = simsnap
        self.chunks = kwargs.get("chunks", ChunkManager(simsnap, chunk_size))
        self.minisnap = kwargs.get("minisnap", MiniSimSnap(simsnap))


    def __getitem__(self, key: str) -> SimDaskArray | Any:

        # preserve family order from the snapshot’s internal layout
        fam_order = self.chunks.families
        all_fam_das: list[da.Array] = []

        for fam in fam_order:
            chunk_list = self.chunks.chunks.get(fam, [])
            if not chunk_list:
                continue

            # Probe metadata (shape for a single particle in this family, dtype, units)
            arr_info = self.minisnap.get_array_info(key, fam)

            fam_chunk_das: list[da.Array] = []
            for ch in chunk_list:
                shape = (ch.num_particles,) + arr_info.shape[1:]
                d = dask.delayed(ch.get_np_array,pure=False,traverse=False)(key)
                fam_chunk_das.append(da.from_delayed(d, shape=shape, dtype=arr_info.dtype))

            if not fam_chunk_das:
                continue
            fam_da = da.concatenate(fam_chunk_das, axis=0) if len(fam_chunk_das) > 1 else fam_chunk_das[0]
            all_fam_das.append(fam_da)

        if not all_fam_das:
            raise KeyError(f"Field {key!r} not found in any family for this snapshot")

        all_da = da.concatenate(all_fam_das, axis=0) if len(all_fam_das) > 1 else all_fam_das[0]
        # Wrap in SimDaskArray to track units/sim/name
        return sim_from_dask(all_da, units=arr_info.units, sim=self.simsnap, name=arr_info.name, family=arr_info.family)


    def clear_cache(self) -> None:
        self.chunks.clear_cache()

class ChunkSimSnap(ChunkDaskArrayLoader, SimSnapView):
    def __init__(self, simsnap: SimSnap, chunk_size: int = 1000_000, **kwargs: Any):
        ChunkDaskArrayLoader.__init__(self, simsnap, chunk_size, **kwargs)
        SimSnapView.__init__(self, simsnap)

    def __getitem__(self, key: slice | np.ndarray | int | str | Family | Filter) -> Union[SimDaskArray, "ChunkSimSnap"]:
        if isinstance(key, str):
            return self._get_array_with_lazy_actions(key)
        else:
            return ChunkSubSnap(self, key)

    def __delitem__(self, key: str) -> None:
        self.minisnap.base.__delitem__(key)
        super().__delitem__(key)

    def _load_array(self, key: str, fam: Family | None = None) -> None:

        base = self.minisnap.base
        def fk():
            return {fami: {k for k in list(base._family_arrays.keys()) if fami in base._family_arrays[k]}
                           for fami in family_registry}

        pre_keys = set(base.keys())
        pre_fam_keys = fk()

        base[key]

        new_keys = set(base.keys()) - pre_keys
        new_fam_keys = fk()
        for fami in new_fam_keys:
            new_fam_keys[fami] = new_fam_keys[fami] - pre_fam_keys[fami]

        for i in new_keys:
            arr = ChunkDaskArrayLoader.__getitem__(self, i)

            arr.sim = self.simsnap
            arr.family = fam
            self._arrays[i] = arr


    def family_keys(self, fam=None):
        return SimSnap.family_keys(self, fam)

    def keys(self):
        return SimSnap.keys(self)

    @property
    def ancestor(self) -> "ChunkSimSnap":
        return self

    def is_ancestor(self, other: "ChunkSimSnap") -> bool:
        return self is other.ancestor


class ChunkSubSnap(ChunkSimSnap):


    chunk_ancestor: ChunkSimSnap
    def __init__(self, base: ChunkSimSnap, slice_: slice | np.ndarray | int | Family | Filter):
        self.chunk_ancestor = base.ancestor
        if isinstance(slice_, Filter):
            indices = slice_.where(base)[0]
        else:
            indices = slice_
        new_chunks = base.chunks.select(indices)
        ChunkSimSnap.__init__(self, new_chunks.simsnap, chunk_size=base.chunks.chunk_size, chunks=new_chunks)

    @property
    def ancestor(self):
        return self.chunk_ancestor
