
from abc import abstractmethod
from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from pynbody.array import SimArray
from pynbody.family import Family, get_family
from pynbody.filt import geometry_selection
from pynbody.snapshot import SimSnap
from pynbody.units import Unit, UnitBase

from pynbodyext.chunk import DASK_AVAILABLE
from pynbodyext.util._type import get_signature_safe

from .base import FilterBase
from .pynfilt import (
    _Annulus,
    _BandPass,
    _Cuboid,
    _Disc,
    _FamilyFilter,
    _HighPass,
    _LowPass,
    _SolarNeighborhood,
    _Sphere,
)

__all__ = ["Sphere","FamilyFilter","Cuboid","Disc","Annulus","BandPass","HighPass","LowPass","SolarNeighborhood"]

ValueLike: TypeAlias = float | str | UnitBase
ValueLikeFunc: TypeAlias = Callable[[SimSnap], ValueLike]

ArrayLike: TypeAlias = tuple[float, float, float] | SimArray | np.ndarray
ArrayLikeFunc: TypeAlias = Callable[[SimSnap], ArrayLike]

FamilyLike: TypeAlias = Family | str
FamilyLikeFunc: TypeAlias = Callable[[SimSnap], FamilyLike]


class VolumeFilter(FilterBase):
    """ A filter that defines a volume in the simulation space."""
    @abstractmethod
    def volume(self, sim: SimSnap | None = None) -> float | UnitBase:
        """ Total volume of the filter """


class Sphere(VolumeFilter,_Sphere):

    def __init__(self,
                radius: ValueLike | ValueLikeFunc,
                cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        self.radius = radius
        self.cen = cen

    def instance_signature(self):
        radius = get_signature_safe(self.radius, fallback_to_id=True)
        cen = get_signature_safe(self.cen, fallback_to_id=True)
        return (self.__class__.__name__, radius, cen)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        radius, cen = self._get_lazy_params(sim)
        pos = sim["pos"]

        if DASK_AVAILABLE and not isinstance(pos, SimArray):
            import dask.array as da
            cen = np.asarray(cen)
            r = float(radius)

            # compute wrap in the units of pos if available, else -1.0
            try:
                wrap = self._get_wrap_in_position_units(sim)
            except Exception:
                wrap = -1.0

            def _block_sphere(block, block_info=None, cen=cen, r=r, wrap=wrap):
                # block is a numpy ndarray with shape (n,3)
                if block.size == 0:
                    return np.empty((0,), dtype=bool)
                if not block.flags["C_CONTIGUOUS"]:
                    block = np.ascontiguousarray(block)
                out = geometry_selection.selection(np.asarray(block), "sphere",
                                                (float(cen[0]), float(cen[1]), float(cen[2]), float(r)),
                                                float(wrap))
                return out.view(np.bool_)

            # map_blocks: input is 2D -> output is 1D, so drop_axis=1 (drop second axis)
            # chunks for output should be a 1-tuple (first-dim chunks)
            out_chunks = (pos.chunks[0],) if isinstance(pos.chunks, tuple) else (pos.shape[0],)
            result = da.map_blocks(_block_sphere, pos,
                                dtype=bool,
                                chunks=out_chunks,
                                drop_axis=1)
            return result

        # fallback: non-dask path (or SimArray) - use upstream implementation
        return _Sphere(radius, cen)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> tuple[float, ArrayLike]:
        radius = self.radius(sim) if callable(self.radius) else self.radius
        radius = self._in_sim_units(radius, "pos",sim)
        cen = self.cen(sim) if callable(self.cen) else self.cen
        return radius, cen

    def volume(self, sim: SimSnap | None = None) -> float | UnitBase:
        if callable(self.radius) and sim is None:
            raise ValueError("Simulation snapshot must be provided when radius is callable.")
        radius = self.radius(sim) if callable(self.radius) else self.radius
        if sim is not None:
            radius = self._in_sim_units(radius, "pos", sim)
        elif isinstance(radius, str):
            radius = Unit(radius)
        volume = 4/3 * np.pi * radius**3
        return volume


class FamilyFilter(FilterBase, _FamilyFilter):

    def __init__(self, family : Family | str | FamilyLikeFunc):
        if callable(family):
            self.family = family
        else:
            self.family = get_family(family, False)

    def instance_signature(self):
        family = get_signature_safe(self.family, fallback_to_id=True)
        return (self.__class__.__name__, family)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        family = self._get_lazy_params(sim)
        return _FamilyFilter(family)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> Family:
        family = self.family(sim) if callable(self.family) else self.family
        family = get_family(family, False)
        return family

class Cuboid(VolumeFilter, _Cuboid):
    def __init__(self, x1: ValueLike | ValueLikeFunc, y1: ValueLike | ValueLikeFunc | None = None,
                 z1: ValueLike | ValueLikeFunc | None = None, x2: ValueLike | ValueLikeFunc | None = None,
                 y2: ValueLike | ValueLikeFunc | None = None, z2: ValueLike | ValueLikeFunc | None = None):
        """Create a cuboid filter.

        If any of the cube coordinates ``x1``, ``y1``, ``z1``, ``x2``, ``y2``, ``z2`` are not specified they are
        determined as ``y1=x1``; ``z1=x1``; ``x2=-x1``; ``y2=-y1``; ``z2=-z1``.
        """
        if y1 is None:
            y1 = x1
        if z1 is None:
            z1 = x1
        self.x1, self.y1, self.z1, self.x2, self.y2, self.z2 = x1, y1, z1, x2, y2, z2

    def instance_signature(self):
        x1 = get_signature_safe(self.x1, fallback_to_id=True)
        y1 = get_signature_safe(self.y1, fallback_to_id=True)
        z1 = get_signature_safe(self.z1, fallback_to_id=True)
        x2 = get_signature_safe(self.x2, fallback_to_id=True)
        y2 = get_signature_safe(self.y2, fallback_to_id=True)
        z2 = get_signature_safe(self.z2, fallback_to_id=True)
        return (self.__class__.__name__, x1, y1, z1, x2, y2, z2)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        params = self._get_lazy_params(sim)
        return _Cuboid(*params)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> tuple[ValueLike,ValueLike,ValueLike,ValueLike,ValueLike,ValueLike]:
        x1 = self.x1(sim) if callable(self.x1) else self.x1
        y1 = self.y1(sim) if callable(self.y1) else self.y1
        z1 = self.z1(sim) if callable(self.z1) else self.z1
        x2 = self.x2(sim) if callable(self.x2) else self.x2
        y2 = self.y2(sim) if callable(self.y2) else self.y2
        z2 = self.z2(sim) if callable(self.z2) else self.z2
        return x1, y1, z1, x2, y2, z2

    def volume(self, sim: SimSnap | None = None) -> float | UnitBase:
        if callable(self.x1) or callable(self.y1) or callable(self.z1) or callable(self.x2) or callable(self.y2) or callable(self.z2):
            if sim is None:
                raise ValueError("Simulation snapshot must be provided when any coordinate is callable.")
            x1, y1, z1, x2, y2, z2 = self._get_lazy_params(sim)
        else:
            x1, y1, z1, x2, y2, z2 = self.x1, self.y1, self.z1, self.x2, self.y2, self.z2
        x1 = Unit(x1) if isinstance(x1, str) else x1
        y1 = Unit(y1) if isinstance(y1, str) else y1
        z1 = Unit(z1) if isinstance(z1, str) else z1
        x2 = Unit(x2) if isinstance(x2, str) else x2
        y2 = Unit(y2) if isinstance(y2, str) else y2
        z2 = Unit(z2) if isinstance(z2, str) else z2
        l1 = (x2 - x1) if x2 is not None else 2*x1
        l2 = (y2 - y1) if y2 is not None else 2*y1
        l3 = (z2 - z1) if z2 is not None else 2*z1
        volume = l1 * l2 * l3
        return volume

class Disc(VolumeFilter, _Disc):
    def __init__(self, radius: ValueLike | ValueLikeFunc, height: ValueLike | ValueLikeFunc,
                 cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        self.radius = radius
        self.height = height
        self.cen = cen

    def instance_signature(self):
        radius = get_signature_safe(self.radius, fallback_to_id=True)
        height = get_signature_safe(self.height, fallback_to_id=True)
        cen = get_signature_safe(self.cen, fallback_to_id=True)
        return (self.__class__.__name__, radius, height, cen)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        radius, height, cen = self._get_lazy_params(sim)
        return _Disc(radius, height, cen)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> tuple[float, float, ArrayLike]:
        radius = self.radius(sim) if callable(self.radius) else self.radius
        radius = self._in_sim_units(radius, "pos", sim)
        height = self.height(sim) if callable(self.height) else self.height
        height = self._in_sim_units(height, "pos", sim)
        cen = self.cen(sim) if callable(self.cen) else self.cen
        return radius, height, cen

    def volume(self, sim: SimSnap | None = None) -> float | UnitBase:
        radius: ValueLike
        height: ValueLike
        if callable(self.radius) or callable(self.height) or callable(self.cen):
            if sim is None:
                raise ValueError("Simulation snapshot must be provided when any parameter is callable.")
            radius, height, _ = self._get_lazy_params(sim)
        else:
            radius, height = self.radius, self.height
        radius = Unit(radius) if isinstance(radius, str) else radius
        height = Unit(height) if isinstance(height, str) else height
        volume = 2 * np.pi * radius**2 * height
        return volume


class BandPass(FilterBase, _BandPass):

    def __init__(self, prop: str, min: ValueLike | ValueLikeFunc, max: ValueLike | ValueLikeFunc):
        self._prop = prop
        self._min = min
        self._max = max

    def instance_signature(self):
        prop = get_signature_safe(self._prop, fallback_to_id=True)
        _min = get_signature_safe(self._min, fallback_to_id=True)
        _max = get_signature_safe(self._max, fallback_to_id=True)
        return (self.__class__.__name__, prop, _min, _max)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        prop, _min, _max = self._get_lazy_params(sim)
        return _BandPass(prop, _min, _max)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> tuple[str, ValueLike, ValueLike]:
        prop = self._prop
        _min = self._min(sim) if callable(self._min) else self._min
        _min = self._in_sim_units(_min, self._prop, sim)
        _max = self._max(sim) if callable(self._max) else self._max
        _max = self._in_sim_units(_max, self._prop, sim)
        return prop, _min, _max

class HighPass(FilterBase, _HighPass):
    def __init__(self, prop: str, min: ValueLike | ValueLikeFunc):
        self._prop = prop
        self._min = min

    def instance_signature(self):
        prop = get_signature_safe(self._prop, fallback_to_id=True)
        _min = get_signature_safe(self._min, fallback_to_id=True)
        return (self.__class__.__name__, prop, _min)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        prop, _min = self._get_lazy_params(sim)
        return _HighPass(prop, _min)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> tuple[str, ValueLike]:
        prop = self._prop
        _min = self._min(sim) if callable(self._min) else self._min
        _min = self._in_sim_units(_min, self._prop, sim)
        return prop, _min

class LowPass(FilterBase, _LowPass):
    def __init__(self, prop: str, max: ValueLike | ValueLikeFunc):
        self._prop = prop
        self._max = max

    def instance_signature(self):
        prop = get_signature_safe(self._prop, fallback_to_id=True)
        _max = get_signature_safe(self._max, fallback_to_id=True)
        return (self.__class__.__name__, prop, _max)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        prop, _max = self._get_lazy_params(sim)
        return _LowPass(prop, _max)(sim)
    def _get_lazy_params(self, sim: SimSnap) -> tuple[str, ValueLike]:
        prop = self._prop
        _max = self._max(sim) if callable(self._max) else self._max
        _max = self._in_sim_units(_max, self._prop, sim)
        return prop, _max

class Annulus(VolumeFilter, _Annulus):

    def __init__(self, r1: ValueLike | ValueLikeFunc, r2: ValueLike | ValueLikeFunc, cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        self.r1 = r1
        self.r2 = r2
        self.cen = cen

    def instance_signature(self):
        r1 = get_signature_safe(self.r1, fallback_to_id=True)
        r2 = get_signature_safe(self.r2, fallback_to_id=True)
        cen = get_signature_safe(self.cen, fallback_to_id=True)
        return (self.__class__.__name__, r1, r2, cen)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        r1, r2, cen = self._get_lazy_params(sim)
        return _Annulus(r1, r2, cen)(sim)
    def _get_lazy_params(self, sim: SimSnap) -> tuple[float, float, ArrayLike]:
        r1 = self.r1(sim) if callable(self.r1) else self.r1
        r1 = self._in_sim_units(r1, "pos", sim)
        r2 = self.r2(sim) if callable(self.r2) else self.r2
        r2 = self._in_sim_units(r2, "pos", sim)
        cen = self.cen(sim) if callable(self.cen) else self.cen
        return r1, r2, cen

    def volume(self, sim: SimSnap | None = None) -> float | UnitBase:
        r1: ValueLike
        r2: ValueLike
        if callable(self.r1) or callable(self.r2) or callable(self.cen):
            if sim is None:
                raise ValueError("Simulation snapshot must be provided when any parameter is callable.")
            r1, r2, _ = self._get_lazy_params(sim)
        else:
            r1, r2 = self.r1, self.r2
        r1 = Unit(r1) if isinstance(r1, str) else r1
        r2 = Unit(r2) if isinstance(r2, str) else r2
        volume = 4/3 * np.pi * (r2**3 - r1**3)
        return volume


class SolarNeighborhood(VolumeFilter, _SolarNeighborhood):

    def __init__(self, r1: ValueLike | ValueLikeFunc = "5 kpc",
                 r2: ValueLike | ValueLikeFunc = "10 kpc",
                 height: ValueLike | ValueLikeFunc = "2 kpc",
                 cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        self.r1 = r1
        self.r2 = r2
        self.height = height
        self.cen = cen

    def instance_signature(self):
        r1 = get_signature_safe(self.r1, fallback_to_id=True)
        r2 = get_signature_safe(self.r2, fallback_to_id=True)
        height = get_signature_safe(self.height, fallback_to_id=True)
        cen = get_signature_safe(self.cen, fallback_to_id=True)
        return (self.__class__.__name__, r1, r2, height, cen)

    def calculate(self, sim: SimSnap) -> np.ndarray:
        r1, r2, height, cen = self._get_lazy_params(sim)
        return _SolarNeighborhood(r1, r2, height, cen)(sim)

    def _get_lazy_params(self, sim: SimSnap) -> tuple[float, float, float, ArrayLike]:
        r1 = self.r1(sim) if callable(self.r1) else self.r1
        r1 = self._in_sim_units(r1, "pos", sim)
        r2 = self.r2(sim) if callable(self.r2) else self.r2
        r2 = self._in_sim_units(r2, "pos", sim)
        height = self.height(sim) if callable(self.height) else self.height
        height = self._in_sim_units(height, "pos", sim)
        cen = self.cen(sim) if callable(self.cen) else self.cen
        return r1, r2, height, cen

    def volume(self, sim: SimSnap | None = None) -> float | UnitBase:
        r1: ValueLike
        r2: ValueLike
        height: ValueLike
        if callable(self.r1) or callable(self.r2) or callable(self.height) or callable(self.cen):
            if sim is None:
                raise ValueError("Simulation snapshot must be provided when any parameter is callable.")
            r1, r2, height, _ = self._get_lazy_params(sim)
        else:
            r1, r2, height = self.r1, self.r2, self.height
        r1 = Unit(r1) if isinstance(r1, str) else r1
        r2 = Unit(r2) if isinstance(r2, str) else r2
        height = Unit(height) if isinstance(height, str) else height
        volume = 2 * np.pi * height * (r2**2 - r1**2)
        return volume
