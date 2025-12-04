
from typing import Any, Union

import dask
import dask.array as da
import numpy as np
from pynbody.family import Family
from pynbody.filt import Filter
from pynbody.snapshot import SimSnap

from pynbodyext.log import logger

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
        if simsnap.keys():
            logger.warning("Creating ChunkSimSnap from a SimSnap with loaded arrays. Loaded arrays will be ignored.")
        ChunkDaskArrayLoader.__init__(self, simsnap, chunk_size, **kwargs)
        SimSnapView.__init__(self, simsnap)

    def __getitem__(self, key: slice | np.ndarray | int | str | Family | Filter) -> Union[SimDaskArray, "ChunkSimSnap"]:
        if isinstance(key, str):
            return self._get_array_with_lazy_actions(key)
        else:
            return ChunkSubSnap(self, key)

    def __delitem__(self, key: str) -> None:
        self.minisnap.ancestor.__delitem__(key)
        super().__delitem__(key)


    def _load_array(self, key: str, fam: Family | None = None) -> None:
        self.minisnap._load_array(key, fam)

        NDname = self._array_name_1D_to_ND(key)
        if NDname:
            self._load_array(NDname, fam)
            return

        arr = ChunkDaskArrayLoader.__getitem__(self, key)

        self._make_array(arr, key, fam=fam)

    def _make_array(self,arr: SimDaskArray, name: str, fam: Family | None = None, derived: bool = False) -> None:

        arr.sim = self
        arr.family = fam
        self._arrays[name] = arr

        if derived:
            if name not in self._derived_array_names:
                self._derived_array_names.append(name)

        if arr.ndim > 1 and arr.shape[-1] == 3:
            array_name_1D = self._array_name_ND_to_1D(name)
            for i, a in enumerate(array_name_1D):
                self._arrays[a] = arr[:, i]
                self._arrays[a].pb_name = a

    def _derive_array(self, name, fam=None):
        """Calculate and store, for this SnapShot, the derivable array 'name'.
        If *fam* is not None, derive only for the specified family.

        This searches the registry of @X.derived_array functions
        for all X in the inheritance path of the current class.
        """
        fn = self.find_deriving_function(name)
        if fn:
            with self.auto_propagate_off:
                if fam is None:
                    result = fn(self)
                    self._make_array(result, name, derived=not fn.__stable__)
                else:
                    result = fn(self[fam])
                    self[fam]._make_array(result, name, fam=fam, derived=not fn.__stable__)

    def family_keys(self, fam=None):
        return SimSnap.family_keys(self, fam)

    def keys(self):
        return SimSnap.keys(self)

    @property
    def ancestor(self) -> "ChunkSimSnap":
        return self

    def is_ancestor(self, other: "ChunkSimSnap") -> bool:
        return self is other.ancestor

    def apply_transformation_to_array(self, array_name, family = None):
        pass


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
