from typing import TYPE_CHECKING

from pynbody.filt import (
    And as _And,
    Annulus as _Annulus,
    BandPass as _BandPass,
    Cuboid as _Cuboid,
    Disc as _Disc,
    FamilyFilter as _FamilyFilter,
    Filter as _Filter,
    HighPass as _HighPass,
    LowPass as _LowPass,
    Not as _Not,
    Or as _Or,
    SolarNeighborhood as _SolarNeighborhood,
    Sphere as _Sphere,
)

if TYPE_CHECKING:

    import numpy as np
    from pynbody.snapshot import SimSnap

    from .base import BoolMask, FilterLike
    class Filter(FilterLike,_Filter):
        # make return types explicit for type checkers
        def __call__(self, sim: SimSnap) -> BoolMask: ...
        def where(self, sim: SimSnap) -> tuple[np.ndarray, ...]: ...
        def __and__(self, other: "FilterLike") -> "And": ...
        def __or__(self, other: "FilterLike") -> "Or": ...
        def __invert__(self) -> "Not": ...
        def cubic_cell_intersection(self, centroids: np.ndarray) -> BoolMask: ...

    class Sphere(Filter,_Sphere): ...
    class Cuboid(Filter,_Cuboid): ...
    class Disc(Filter,_Disc): ...
    class BandPass(Filter, _BandPass): ...
    class HighPass(Filter, _HighPass): ...
    class LowPass(Filter, _LowPass): ...
    class Annulus(Filter, _Annulus): ...
    class SolarNeighborhood(Filter, _SolarNeighborhood): ...
    class And(Filter, _And): ...
    class Or(Filter, _Or): ...
    class Not(Filter, _Not): ...
    class FamilyFilter(Filter,_FamilyFilter): ...
else:
    Filter = _Filter
    Sphere = _Sphere
    Cuboid = _Cuboid
    Disc = _Disc
    BandPass = _BandPass
    HighPass = _HighPass
    LowPass = _LowPass
    Annulus = _Annulus
    SolarNeighborhood = _SolarNeighborhood
    And = _And
    Or = _Or
    Not = _Not
    FamilyFilter = _FamilyFilter

__all__ = [
    "Filter",
    "Sphere",
    "Cuboid",
    "Disc",
    "BandPass",
    "HighPass",
    "LowPass",
    "Annulus",
    "SolarNeighborhood",
    "And",
    "Or",
    "Not",
    "FamilyFilter",
]
