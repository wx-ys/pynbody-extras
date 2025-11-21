

from typing import Any

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
    def children(self) -> list[CalculatorBase[Any]]:
        return super().children()

    def calculate(self, sim: SimSnap)-> np.ndarray:
        return np.ones(len(sim), dtype=bool)

    def __and__(self, f2):
        return And(self, f2)

    def __invert__(self):
        return Not(self)

    def __repr__(self):
        return f"<Filter {self.__class__.__name__}>"

    def __or__(self, f2):
        return Or(self, f2)


class And(FilterBase,_And):
    calculate = _And.__call__
    def children(self) -> list[CalculatorBase[Any]]:
        # Combine child nodes + own filter/transformation
        return [self.f1,self.f2,*super().children()]
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        return [self.f1,self.f2]
class Or(FilterBase,_Or):
    calculate = _Or.__call__
    def children(self) -> list[CalculatorBase[Any]]:
        return [self.f1,self.f2,*super().children()]
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        return [self.f1,self.f2]

class Not(FilterBase,_Not):
    calculate = _Not.__call__

    def children(self) -> list[CalculatorBase[Any]]:
        return [self.f,*super().children()]
