
import warnings
from typing import Generic, TypeVar

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformable, Transformation

from pynbodyext.calculate import CalculatorBase

TTrans = TypeVar("TTrans",bound=Transformation, covariant=True)

class TransformBase(CalculatorBase[TTrans], Generic[TTrans]):
    """ Simsnap -> TTrans """

    move_all: bool = True

    def __call__(self, sim: SimSnap) -> TTrans:
        trans_obj = None
        if self._transformation is not None:
            trans_obj = self._transformation(sim)
        if self._filter is not None:
            sim = sim[self._filter]
        try:
            result = self.calculate(sim, trans_obj)
        except Exception as e:
            if trans_obj is not None and self._revert_transformation:
                trans_obj.revert()
            raise e
        if trans_obj is not None and self._revert_transformation:
            warnings.warn("Previous transformation merged, not reverting", stacklevel=2)
        return result

    def get_target(self, sim: SimSnap, previous: Transformation | None = None) -> Transformable:
        if previous is not None:
            target = previous
        elif self.move_all:
            target = sim.ancestor
        else:
            target = sim
        return target

    def calculate(self, sim: SimSnap, previous: Transformation | None = None) -> TTrans:
        raise NotImplementedError

