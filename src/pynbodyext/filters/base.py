

import numpy as np
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import CalculatorBase

from .pynfilt import _And, _Filter, _Not, _Or

__all__ = ["FilterBase", "And", "Or", "Not"]

class FilterBase(CalculatorBase[np.ndarray],_Filter):
    """
    Base class for filters.

    Connect pynbody filters with CalculatorBase.
    """


    def calculate(self, sim: SimSnap)-> np.ndarray:
        return np.ones(len(sim), dtype=bool)

    def __and__(self, f2):
        return And(self, f2)

    def __invert__(self):
        return Not(self)

    def __or__(self, f2):
        return Or(self, f2)

class And(FilterBase,_And):
    calculate = _And.__call__

class Or(FilterBase,_Or):
    calculate = _Or.__call__

class Not(FilterBase,_Not):
    calculate = _Not.__call__

