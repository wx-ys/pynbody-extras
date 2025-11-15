
from typing import Generic, TypeVar

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation

from pynbodyext.calculate import CalculatorBase

TTrans = TypeVar("TTrans",bound=Transformation, covariant=True)

class TransformBase(CalculatorBase[TTrans], Generic[TTrans]):
    """ Simsnap -> TTrans """

    def __call__(self, sim: SimSnap) -> TTrans:
        raise NotImplementedError

