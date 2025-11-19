
from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
from pynbody.analysis.halo import hybrid_center, shrink_sphere_center
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap
from pynbody.transformation import GenericTranslation, Transformation

from .base import TransformBase

__all__ = ["PosToCenter", "VelToCenter"]

_CenArr: TypeAlias = np.ndarray | SimArray
_CenFunc: TypeAlias = Callable[[SimSnap], _CenArr]


class PosToCenter(TransformBase[GenericTranslation]):

    def __init__(self, mode: Literal["ssc", "com", "pot", "hyb"] | _CenFunc | _CenArr= "ssc", move_all: bool = True):
        if isinstance(mode, str):
            if mode not in ["ssc", "com", "pot","hyb"]:
                raise ValueError(f"Invalid mode: {mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")
        elif not (callable(mode) or isinstance(mode, (np.ndarray, SimArray))):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected str, callable, or array.")

        self.mode = mode
        self.move_all = move_all

    def calculate(self, sim: SimSnap, previous: Transformation | None = None) -> GenericTranslation:

        description: str
        if isinstance(self.mode, str):
            cen = self.get_center(sim, mode=self.mode)  # type: ignore
            description = self.mode
        elif isinstance(self.mode, (np.ndarray, SimArray)):
            cen = self.mode
            description = "given"
        elif callable(self.mode):
            cen = self.mode(sim)
            description = "call"
        else:
            raise ValueError(f"Invalid mode type: {type(self.mode)}. Expected str, callable, or array.")

        target = self.get_target(sim, previous)

        return GenericTranslation(target, "pos", -cen, description=f"PosToCenter_{description}")

    @classmethod
    def get_center(cls, sim: SimSnap, mode: Literal["ssc", "com", "pot", "hyb"]) -> _CenArr:
        if mode == "com":
            cen = sim.mean_by_mass("pos")
        elif mode == "pot":
            i = sim["phi"].argmin()
            cen = sim["pos"][i].copy()
        elif mode == "ssc":
            cen = shrink_sphere_center(sim)
        elif mode == "hyb":
            cen = hybrid_center(sim)
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")

        if isinstance(cen, SimArray):
            cen.sim = sim
        return cen


class VelToCenter(TransformBase[GenericTranslation]):



    def __init__(self,mode: Literal["com"] | _CenFunc | _CenArr = "com", move_all: bool = True):
        self.mode = mode
        self.move_all = move_all

    def calculate(self, sim: SimSnap, previous: Transformation | None = None) -> GenericTranslation:
        if isinstance(self.mode, str):
            vcen = self.get_center(sim, mode=self.mode)  # type: ignore
        elif isinstance(self.mode, (np.ndarray, SimArray)):
            vcen = self.mode
        elif callable(self.mode):
            vcen = self.mode(sim)
        else:
            raise ValueError(f"Invalid mode type: {type(self.mode)}. Expected str, callable, or array.")

        target = self.get_target(sim, previous)
        return GenericTranslation(target, "vel", -vcen, description="VelToCenter")

    @classmethod
    def get_center(cls, sim: SimSnap, mode: Literal["com"]) -> _CenArr:
        vcen = (sim["vel"].transpose() * sim["mass"]).sum(axis=1) / sim["mass"].sum()
        vcen.units = sim["vel"].units
        vcen.sim = sim
        return vcen
