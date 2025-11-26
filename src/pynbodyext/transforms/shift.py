
from typing import Literal

import numpy as np
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap
from pynbody.transformation import GenericTranslation, Transformation

from pynbodyext.properties import CenPos, CenVel
from pynbodyext.util._type import SimNpArray, SimNpArrayFunc, get_signature_safe

from .base import TransformBase

__all__ = ["PosToCenter", "VelToCenter"]


class PosToCenter(TransformBase[GenericTranslation]):

    def __init__(self, mode: Literal["ssc", "com", "pot", "hyb"] | SimNpArray | SimNpArrayFunc= "ssc", move_all: bool = True):
        if isinstance(mode, str):
            if mode not in ("ssc", "com", "pot","hyb"):
                raise ValueError(f"Invalid mode: {mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")
        elif not (callable(mode) or isinstance(mode, (np.ndarray, SimArray))):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected str, callable, or array.")

        self.mode = mode
        self.move_all = move_all

    def instance_signature(self):
        mode_sig = get_signature_safe(self.mode, fallback_to_id=True)
        move_all_sig = get_signature_safe(self.move_all, fallback_to_id=True)
        return (self.__class__.__name__, mode_sig, move_all_sig)

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
    def get_center(cls, sim: SimSnap, mode: Literal["ssc", "com", "pot", "hyb"]) -> SimNpArray:
        return CenPos(mode=mode).calculate(sim)


class VelToCenter(TransformBase[GenericTranslation]):



    def __init__(self,mode: Literal["com"] | SimNpArrayFunc | SimNpArray = "com", move_all: bool = True):
        self.mode = mode
        self.move_all = move_all

    def instance_signature(self):
        mode_sig = get_signature_safe(self.mode, fallback_to_id=True)
        move_all_sig = get_signature_safe(self.move_all, fallback_to_id=True)
        return (self.__class__.__name__, mode_sig, move_all_sig)

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
    def get_center(cls, sim: SimSnap, mode: Literal["com"]) -> SimNpArray:
        return CenVel(mode=mode).calculate(sim)
