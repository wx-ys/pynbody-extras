
from typing import Any, cast

import numpy as np
from pynbody.array import SimArray
from pynbody.transformation import GenericTranslation

from pynbodyext.calculate import Param, TransformBase
from pynbodyext.properties import CenPos, CenVel
from pynbodyext.util._type import SimNpArray

__all__ = ["ShiftPosTo", "ShiftVelTo"]


@TransformBase.dataclass
class ShiftPosTo(TransformBase[GenericTranslation]):
    mode: Param[SimNpArray | str] = Param(default="ssc", field_name="pos")
    move_all: bool = True

    def __post_init__(self) -> None:
        TransformBase.__init__(self, move_all=self.move_all)
        self.description = "given"
        mode = self.mode
        if isinstance(mode, str):
            if mode not in ("ssc", "com", "pot","hyb"):
                raise ValueError(f"Invalid mode: {mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")
            self.description = mode
            mode = CenPos(mode) # type: ignore
        elif not (callable(mode) or isinstance(mode, (np.ndarray, SimArray))):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected str, callable, or array.")
        self.mode = mode


    def build_handle(self, sim, target, params = None):
        cen = params.mode
        return GenericTranslation(target, "pos", -cen, description=f"PosToCenter_{self.description}")


@TransformBase.dataclass
class ShiftVelTo(TransformBase[GenericTranslation]):
    mode: Param[SimNpArray | str] = Param(default="com", field_name="vel")
    move_all: bool = True

    def __post_init__(self) -> None:
        TransformBase.__init__(self, move_all=self.move_all)
        self.description = "given"
        mode = self.mode
        if isinstance(mode, str):
            if mode != "com":
                raise ValueError(f"Invalid mode: {mode}. Expected 'com'.")
            self.description = mode
            mode = cast("Any", CenVel)(mode)
        elif not (callable(mode) or isinstance(mode, (np.ndarray, SimArray))):
            raise ValueError(f"Invalid mode type: {type(mode)}. Expected str, callable, or array.")
        self.mode = mode

    def build_handle(self, sim, target, params = None):
        vcen = params.mode
        return GenericTranslation(target, "vel", -vcen, description=f"VelToCenter_{self.description}")
