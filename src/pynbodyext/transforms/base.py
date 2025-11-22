
from typing import Generic, TypeVar

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformable, Transformation

from pynbodyext.calculate import CalculatorBase
from pynbodyext.log import logger
from pynbodyext.util.perf import PerfStats

TTrans = TypeVar("TTrans",bound=Transformation, covariant=True)

class TransformBase(CalculatorBase[TTrans], Generic[TTrans]):
    """ Simsnap -> TTrans """

    move_all: bool = True


    def _do_calculate(self, sim: SimSnap, trans_obj: Transformation | None, stats: PerfStats) -> TTrans:
        try:
            with stats.step("calculate"):
                logger.debug("[%s] performing calculation ...", self)
                result = self.calculate(sim, trans_obj)
        except Exception as e:
            if trans_obj is not None and self._revert_transformation:
                    trans_obj.revert()
            raise e
        if trans_obj is not None and self._revert_transformation:
            logger.debug("[%s] merging pre-transformation: %s", self, self._transformation)
        return result

    def __repr__(self):
        return f"<Transform {self.__class__.__name__}>"

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

