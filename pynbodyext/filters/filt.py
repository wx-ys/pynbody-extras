
from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

import numpy as np
from pynbody.array import SimArray
from pynbody.family import Family, get_family
from pynbody.snapshot import SimSnap
from pynbody.units import UnitBase

from pynbodyext.core.calculate import FilterBase

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
    """A filter that defines a volume in the simulation space."""

    def volume(self, sim: SimSnap | None = None, *, options: Any | None = None) -> float | UnitBase:
        """Return the volume of the filter."""
        raise NotImplementedError


class Sphere(VolumeFilter,_Sphere):
    """Spherical selection filter.

    Parameters
    ----------
    radius : float, unit-like, callable, or CalculatorBase
        Sphere radius.  Calculator-valued radii are evaluated in the current
        scope before the mask is built.
    cen : array-like, callable, or CalculatorBase, default: (0, 0, 0)
        Sphere center in position units.
    """
    dynamic_param_specs = {"radius": "pos", "cen": "pos"}
    def __init__(self,
                radius: ValueLike | ValueLikeFunc,
                cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        super().__init__()
        self.radius = radius
        self.cen = cen

    def instance_signature(self) -> tuple[Any, ...]:
        return ("Sphere", )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> tuple[float, Any]:
        return values["radius"], values["cen"]

    def build_mask(self, sim: SimSnap, params: tuple[float, Any]) -> Any:
        radius, cen = params
        return _Sphere(radius, cen)(sim)

    def volume(self, sim: SimSnap | None = None, *, options: Any | None = None) -> float | UnitBase:
        radius = self.resolve_param_for_sim(
            sim,
            "radius",
            options=options,
            coerce_unit_string=True,
        )
        return 4.0 / 3.0 * np.pi * radius**3


class FamilyFilter(FilterBase, _FamilyFilter):

    dynamic_param_specs = {"family": None}
    def __init__(self, family : Family | str | FamilyLikeFunc):
        super().__init__()
        if callable(family):
            self.family = family
        else:
            self.family = get_family(family, False)

    def instance_signature(self):
        return (self.__class__.__name__, )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> Family:
        return get_family(values["family"], False)

    def build_mask(self, sim: SimSnap, params: Family) -> Any:
        return _FamilyFilter(params)(sim)

class Cuboid(VolumeFilter, _Cuboid):
    """Cuboid selection filter.

    Parameters
    ----------
    x1, y1, z1, x2, y2, z2 : float, unit-like, callable, or CalculatorBase
        Coordinates of the cuboid corners.  Calculator-valued coordinates are
        evaluated in the current scope before the mask is built.
    """
    dynamic_param_specs = {"x1": "pos", "y1": "pos", "z1": "pos", "x2": "pos", "y2": "pos", "z2": "pos"}

    def __init__(self, x1: ValueLike | ValueLikeFunc, y1: ValueLike | ValueLikeFunc | None = None,
                 z1: ValueLike | ValueLikeFunc | None = None, x2: ValueLike | ValueLikeFunc | None = None,
                 y2: ValueLike | ValueLikeFunc | None = None, z2: ValueLike | ValueLikeFunc | None = None):
        """Create a cuboid filter.

        If any of the cube coordinates ``x1``, ``y1``, ``z1``, ``x2``, ``y2``, ``z2`` are not specified they are
        determined as ``y1=x1``; ``z1=x1``; ``x2=-x1``; ``y2=-y1``; ``z2=-z1``.
        """
        super().__init__()
        if y1 is None:
            y1 = x1
        if z1 is None:
            z1 = x1
        self.x1, self.y1, self.z1, self.x2, self.y2, self.z2 = x1, y1, z1, x2, y2, z2

    def instance_signature(self):
        return (self.__class__.__name__, )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
        return (
            values["x1"],
            values["y1"],
            values["z1"],
            values["x2"],
            values["y2"],
            values["z2"],
        )

    def build_mask(self, sim: SimSnap, params: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        return _Cuboid(*params)(sim)

    def volume(self, sim: SimSnap | None = None, *, options: Any | None = None) -> float | UnitBase:
        x1 = self.resolve_param_for_sim(sim, "x1", options=options, coerce_unit_string=True)
        y1 = self.resolve_param_for_sim(sim, "y1", options=options, coerce_unit_string=True)
        z1 = self.resolve_param_for_sim(sim, "z1", options=options, coerce_unit_string=True)

        x2 = self.resolve_param_for_sim(sim, "x2", options=options, coerce_unit_string=True)
        y2 = self.resolve_param_for_sim(sim, "y2", options=options, coerce_unit_string=True)
        z2 = self.resolve_param_for_sim(sim, "z2", options=options, coerce_unit_string=True)

        l1 = (x2 - x1) if x2 is not None else 2 * x1
        l2 = (y2 - y1) if y2 is not None else 2 * y1
        l3 = (z2 - z1) if z2 is not None else 2 * z1
        return l1 * l2 * l3

class Disc(VolumeFilter, _Disc):

    dynamic_param_specs = {"radius": "pos", "height": "pos", "cen": "pos"}
    def __init__(self, radius: ValueLike | ValueLikeFunc, height: ValueLike | ValueLikeFunc,
                 cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        super().__init__()
        self.radius = radius
        self.height = height
        self.cen = cen

    def instance_signature(self):
        return (self.__class__.__name__,)

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return values["radius"], values["height"], values["cen"]

    def build_mask(self, sim: SimSnap, params: tuple[Any, Any, Any]) -> Any:
        radius, height, cen = params
        return _Disc(radius, height, cen)(sim)

    def volume(self, sim: SimSnap | None = None, *, options: Any | None = None) -> float | UnitBase:
        radius = self.resolve_param_for_sim(
            sim,
            "radius",
            options=options,
            coerce_unit_string=True,
        )
        height = self.resolve_param_for_sim(
            sim,
            "height",
            options=options,
            coerce_unit_string=True,
        )
        return 2 * np.pi * radius**2 * height


class BandPass(FilterBase, _BandPass):
    dynamic_param_specs = {"min": None, "max": None}

    def __init__(self, prop: str, min: ValueLike | ValueLikeFunc, max: ValueLike | ValueLikeFunc):
        super().__init__()
        self.prop = prop
        self.min = min
        self.max = max

    def instance_signature(self) -> tuple[Any, ...]:
        return ("BandPass", self.prop, )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> tuple[Any, Any]:
        return values["min"], values["max"]

    def build_mask(self, sim: SimSnap, params: tuple[Any, Any]) -> Any:
        lower, upper = params
        return _BandPass(self.prop, lower, upper)(sim)

class HighPass(FilterBase, _HighPass):
    dynamic_param_specs = {"min": None}

    def __init__(self, prop: str, min: ValueLike | ValueLikeFunc):
        super().__init__()
        self.prop = prop
        self.min = min

    def instance_signature(self):
        return (self.__class__.__name__, self.prop)


    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> Any:
        return values["min"]

    def build_mask(self, sim: SimSnap, params: Any) -> Any:
        return _HighPass(self.prop, params)(sim)

class LowPass(FilterBase, _LowPass):
    dynamic_param_specs = {"max": None}

    def __init__(self, prop: str, max: ValueLike | ValueLikeFunc):
        super().__init__()
        self.prop = prop
        self.max = max

    def instance_signature(self):
        return (self.__class__.__name__, self.prop)

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> Any:
        return values["max"]

    def build_mask(self, sim: SimSnap, params: Any) -> Any:
        return _LowPass(self.prop, params)(sim)


class Annulus(VolumeFilter, _Annulus):
    dynamic_param_specs = {"r1": "pos", "r2": "pos", "cen": "pos"}

    def __init__(self, r1: ValueLike | ValueLikeFunc, r2: ValueLike | ValueLikeFunc, cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.cen = cen

    def instance_signature(self):
        return (self.__class__.__name__, )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> tuple[Any, Any, Any]:
        return values["r1"], values["r2"], values["cen"]

    def build_mask(self, sim: SimSnap, params: tuple[Any, Any, Any]) -> Any:
        r1, r2, cen = params
        return _Annulus(r1, r2, cen)(sim)

    def volume(self, sim: SimSnap | None = None, *, options: Any | None = None) -> float | UnitBase:
        r1 = self.resolve_param_for_sim(
            sim,
            "r1",
            options=options,
            coerce_unit_string=True,
        )
        r2 = self.resolve_param_for_sim(
            sim,
            "r2",
            options=options,
            coerce_unit_string=True,
        )
        return 4.0 / 3.0 * np.pi * (r2**3 - r1**3)


class SolarNeighborhood(VolumeFilter, _SolarNeighborhood):

    dynamic_param_specs = {"r1": "pos", "r2": "pos", "height": "pos", "cen": "pos"}
    def __init__(self, r1: ValueLike | ValueLikeFunc = "5 kpc",
                 r2: ValueLike | ValueLikeFunc = "10 kpc",
                 height: ValueLike | ValueLikeFunc = "2 kpc",
                 cen: ArrayLike | ArrayLikeFunc = (0, 0, 0)):
        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.height = height
        self.cen = cen

    def instance_signature(self):
        return (self.__class__.__name__, )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
        return values["r1"], values["r2"], values["height"], values["cen"]

    def build_mask(self, sim: SimSnap, params: tuple[Any, Any, Any, Any]) -> Any:
        r1, r2, height, cen = params
        return _SolarNeighborhood(r1, r2, height, cen)(sim)

    def volume(self, sim: SimSnap | None = None, *, options: Any | None = None) -> float | UnitBase:
        r1 = self.resolve_param_for_sim(
            sim,
            "r1",
            options=options,
            coerce_unit_string=True,
        )
        r2 = self.resolve_param_for_sim(
            sim,
            "r2",
            options=options,
            coerce_unit_string=True,
        )
        height = self.resolve_param_for_sim(
            sim,
            "height",
            options=options,
            coerce_unit_string=True,
        )
        return 2 * np.pi * height * (r2**2 - r1**2)
