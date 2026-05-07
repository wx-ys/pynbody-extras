
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
    """
    Shift the positions of particles to a specified center.

    Parameters
    ----------
    mode : str, callable, or array-like, default: "ssc"
        Method to determine the center. Options are:
        - "ssc": Shrinking sphere center
        - "com": Center of mass
        - "pot": Potential minimum
        - "hyb": Hybrid method (initially "pot" or "com", then "ssc")
        - callable: A function that returns the center
        - array-like: Directly specify the center coordinates
    move_all : bool, default: True
        Whether to move all particles or only a subset.
    """

    mode: Param[SimNpArray | str] = Param(default="ssc", field_name="pos")
    move_all: bool = True

    def __post_init__(self) -> None:
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
    """
    Shift the velocities of particles to a specified center.

    Parameters
    ----------
    mode : str, callable, or array-like, default: "com"
        Method to determine the velocity center. Options are:
        - "com": Center of mass velocity
        - callable: A function that returns the velocity center
        - array-like: Directly specify the velocity center coordinates
     move_all : bool, default: True
        Whether to move all particles or only a subset.

    """
    mode: Param[SimNpArray | str] = Param(default="com", field_name="vel")
    move_all: bool = True

    def __post_init__(self) -> None:
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
