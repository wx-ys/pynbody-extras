
from typing import Literal

import numpy as np
from pynbody.array import SimArray
from pynbody.transformation import GenericTranslation

from pynbodyext.calculate import TransformBase
from pynbodyext.properties import CenPos, CenVel
from pynbodyext.util._type import SimNpArray, SimNpArrayFunc

__all__ = ["ShiftPosTo", "ShiftVelTo"]


class ShiftPosTo(TransformBase[GenericTranslation]):
    dynamic_param_specs = {"mode": "pos"}
    def __init__(self, mode: Literal["ssc", "com", "pot", "hyb"] | SimNpArray | SimNpArrayFunc= "ssc", move_all: bool = True):
        super().__init__(move_all=move_all)
        self.description = "given"
        if isinstance(mode, str):
            if mode not in ("ssc", "com", "pot","hyb"):
                raise ValueError(f"Invalid mode: {mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")
            self.description = mode
            mode = CenPos(mode=mode)        # type: ignore[arg-type]
        elif not (callable(mode) or isinstance(mode, (np.ndarray, SimArray))):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected str, callable, or array.")
        self.mode = mode

    def instance_signature(self):
        return (self.__class__.__name__, )


    def build_handle(self, sim, target, params = None):
        cen = params["mode"]
        return GenericTranslation(target, "pos", -cen, description=f"PosToCenter_{self.description}")


class ShiftVelTo(TransformBase[GenericTranslation]):

    dynamic_param_specs = {"mode": "vel"}
    def __init__(self,mode: Literal["com"] | SimNpArrayFunc | SimNpArray = "com", move_all: bool = True):
        super().__init__(move_all=move_all)
        self.description = "given"
        if isinstance(mode, str):
            if mode != "com":
                raise ValueError(f"Invalid mode: {mode}. Expected 'com'.")
            self.description = mode
            mode = CenVel(mode=mode)        # type: ignore[arg-type]
        elif not (callable(mode) or isinstance(mode, (np.ndarray, SimArray))):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected str, callable, or array.")
        self.mode = mode

    def instance_signature(self):
        return (self.__class__.__name__, )

    def build_handle(self, sim, target, params = None):
        vcen = params["mode"]
        return GenericTranslation(target, "vel", -vcen, description=f"VelToCenter_{self.description}")
