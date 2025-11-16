

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import SimCallable

BoolMask = NDArray[np.bool_]

@runtime_checkable
class FilterLike(SimCallable[BoolMask], Protocol):
    def where(self, sim: SimSnap) -> tuple[np.ndarray, ...]: ...

    def __and__(self, other: "FilterLike") -> "FilterLike": ...
    def __or__(self, other: "FilterLike") -> "FilterLike": ...
    def __invert__(self) -> "FilterLike": ...

    def cubic_cell_intersection(self, centroids: np.ndarray) -> BoolMask: ...

    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...



