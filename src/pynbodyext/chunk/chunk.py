

import gc
import math
import threading
from typing import TYPE_CHECKING, cast

import numpy as np
from pynbody.array import SimArray
from pynbody.family import Family, _registry as family_registry
from pynbody.filt import Filter
from pynbody.snapshot import SimSnap

if TYPE_CHECKING:
    from pynbody.snapshot.subsnap import IndexedSubSnap


class _ChunkDescriptor:
    """
    Descriptor for a chunk of a SimSnap.
    Holds optional numbering: global_index (global) and family_index (per-family).
    """
    def __init__(
        self,
        ancestor: SimSnap,
        family: Family,
        chunk_slice: slice | np.ndarray,
        global_index: int | None = None,
        family_index: int | None = None,
    ):
        self.ancestor = ancestor
        self.family = family
        self.chunk_slice = chunk_slice
        self.global_index = global_index
        self.family_index = family_index

    @property
    def num_particles(self) -> int:
        if isinstance(self.chunk_slice, slice):
            return self.chunk_slice.stop - self.chunk_slice.start
        else:
            return len(self.chunk_slice)

    def __repr__(self) -> str:
        ancestor_class_name = self.ancestor.__class__.__name__
        filename = getattr(self.ancestor, "_filename", "")

        prefix = self.chunk_label()

        if filename != "":
            return f'<{prefix}:{ancestor_class_name} "{filename}" len={self.num_particles}>'
        else:
            return f"<{prefix}:{ancestor_class_name} len={self.num_particles}>"

    def chunk_label(self) -> str:
        if self.global_index is not None and self.family_index is not None:
            return f"Chunk_{self.global_index}:{self.family}_{self.family_index}"
        elif self.global_index is not None:
            return f"Chunk_{self.global_index}:{self.family}"
        elif self.family_index is not None:
            return f"Chunk:{self.family}_{self.family_index}"
        else:
            return f"Chunk:{self.family}"

class _ChunkSnapshotCache:
    _global_sem = threading.Semaphore(1)  # cap simultaneously loaded chunk snaps

    @classmethod
    def set_max_active_chunks(cls, n: int) -> None:
        # Replace the semaphore with a fresh one; use carefully (set once at startup)
        cls._global_sem = threading.Semaphore(max(1, int(n)))

    def __init__(self, ancestor: SimSnap, chunk_slice: slice | np.ndarray):
        self.ancestor = ancestor
        self.chunk_slice = chunk_slice
        self._chunk_snap: SimSnap | None = None

        self._lock = threading.RLock()      # TODO Rlock or Lock?
        self._active = 0

        self._holds_global = False

    @property
    def chunk_snap(self) -> SimSnap:
        if self._chunk_snap is None:
            with self._lock:
                if self._chunk_snap is None:
                    type(self)._global_sem.acquire()
                    self._holds_global = True
                    sel = cast("IndexedSubSnap", self.ancestor[self.chunk_slice])
                    self._chunk_snap = sel.load_copy()
        return self._chunk_snap

    def _acquire(self):
        with self._lock:
            self._active += 1

    def _release(self):
        with self._lock:
            self._active -= 1
            if self._active <= 0:
                self._active = 0
                self.clear_cache()

    def get_sim_array(self, key: str) -> SimArray:
        arr = self.chunk_snap[key]
        return arr

    def get_np_array(self, key: str) -> np.ndarray:
        self._acquire()
        try:
            arr = self.get_sim_array(key)
            return arr.view(np.ndarray)
        finally:
            self._release()

    def clear_cache(self) -> None:
        with self._lock:
            self._chunk_snap = None
            if self._holds_global:
                type(self)._global_sem.release()
                self._holds_global = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable entries
        state["_lock"] = None
        state["_active"] = 0
        state["_chunk_snap"] = None
        state["_holds_global"] = False
        try:
            fname = getattr(self.ancestor, "_filename", None)
        except Exception:
            fname = None
        state["_ancestor_filename"] = fname
        state["ancestor"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._active = 0
        fname = self.__dict__.get("_ancestor_filename")
        if self.ancestor is None and fname:
            try:
                import pynbody
                self.ancestor = pynbody.load(fname)
            except Exception:
                pass


class FamilyChunk(_ChunkDescriptor, _ChunkSnapshotCache):
    def __init__(
        self,
        ancestor: SimSnap,
        family: Family,
        chunk_slice: slice | np.ndarray,
        global_index: int | None = None,
        family_index: int | None = None,
    ):
        _ChunkDescriptor.__init__(self, ancestor, family, chunk_slice, global_index, family_index)
        _ChunkSnapshotCache.__init__(self, ancestor, chunk_slice)

    def __repr__(self):
        return _ChunkDescriptor.__repr__(self)


class ChunkManager:

    simsnap: SimSnap
    chunk_size: int
    chunks: dict[Family, list[FamilyChunk]]
    def __init__(self, simsnap: SimSnap, chunk_size: int):
        self.simsnap = simsnap
        self.chunk_size = chunk_size
        self.chunks = self.build_chunks(simsnap, chunk_size)


    # --------- initialization helpers ---------
    @classmethod
    def make_family_chunk_slices(cls, simsnap: SimSnap, chunk_size: int) -> dict[Family, list[slice]]:
        family_chunk_slice = {}

        for family, fam_slice in cast("dict[Family, slice]", simsnap._family_slice).items():
            N = fam_slice.stop - fam_slice.start
            num_chunks = math.ceil(N / chunk_size)
            avg_chunk_size = N // num_chunks if num_chunks > 0 else N
            chunks = []
            for i in range(num_chunks):
                start = fam_slice.start + i * avg_chunk_size

                if i == num_chunks - 1:
                    end = fam_slice.stop
                else:
                    end = min(fam_slice.start + (i + 1) * avg_chunk_size, fam_slice.stop)
                new_slice = slice(start, end)
                chunks.append(new_slice)
            family_chunk_slice[family] = chunks
        return family_chunk_slice

    @classmethod
    def build_chunks(cls, simsnap: SimSnap, chunk_size: int) -> dict[Family, list[FamilyChunk]]:
        family_chunk_slice = cls.make_family_chunk_slices(simsnap, chunk_size)
        family_chunks = {}
        ancestor = simsnap.ancestor
        use_ancestor_indices = ancestor is not simsnap

        global_index = 0
        for family, chunk_slices in family_chunk_slice.items():
            chunks = []
            for chunk_idx, sl in enumerate(chunk_slices):
                sli = sl if not use_ancestor_indices else cast("IndexedSubSnap", simsnap[sl]).get_index_list(ancestor)

                chunks.append(FamilyChunk(ancestor, family, sli, global_index=global_index, family_index=chunk_idx))
                global_index += 1
            family_chunks[family] = chunks
        return family_chunks


    # --------- building new ChunkManager from selection ---------
    def select(self, indices: slice | np.ndarray | int | Family | Filter) -> "ChunkManager":
        # Implement this method to create new ChunkManager from given indices

        fam_regis = family_registry
        new_sim = cast("SimSnap", self.simsnap[indices])
        if not new_sim._family_slice and len(new_sim)>0:
            # Rebuild family_slice if missing
            fm_sl = {}
            for fam in fam_regis:
                sl = new_sim._get_family_slice(fam)
                if sl.start != sl.stop:
                    fm_sl[fam] = sl
            new_sim._family_slice = dict(sorted(fm_sl.items(), key=lambda item: item[1].start))

        return ChunkManager(new_sim, self.chunk_size)


    # --------- cache management ---------
    def clear_cache(self) -> None:
        for chunks in self.chunks.values():
            for chunk in chunks:
                chunk.clear_cache()
        gc.collect()


    # --------- properties ---------
    @property
    def num_chunks(self):
        return sum(len(chunk_list) for chunk_list in self.chunks.values())

    @property
    def num_particles(self):
        return sum(chunk.num_particles for chunk_list in self.chunks.values() for chunk in chunk_list)

    @property
    def families(self):
        return list(self.chunks.keys())


    # --------- representation ---------
    def __repr__(self):
        base = repr(self.simsnap)[1:-1]
        first_line = f"<Chunks:{self.num_chunks}:{base}>"

        fam_parts = []
        for fa, chunk_list in self.chunks.items():
            num_particles = sum(chunk.num_particles for chunk in chunk_list)
            num_chunks = len(chunk_list)
            fam_parts.append(f"{fa}:{num_particles}p:{num_chunks}c")

        second_line = " ".join(fam_parts) if fam_parts else "<no families>"
        second_line = f"[ChunkSize={self.chunk_size} | {second_line}]"
        width = max(len(first_line), len(second_line))
        return first_line.center(width) + "\n" + second_line.center(width)
