
from typing import Any, Generic, TypeVar, cast

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation

from pynbodyext.calculate import CalculatorBase
from pynbodyext.log import logger
from pynbodyext.util.perf import PerfStats
from pynbodyext.util.tracecache import TraceManager

TTrans = TypeVar("TTrans",bound=Transformation, covariant=True)

_LastT = TypeVar("_LastT", bound="TransformBase[Any]")

class TransformBase(CalculatorBase[TTrans], Generic[TTrans]):
    """ Simsnap -> TTrans """

    move_all: bool = True


    def _perform_calculation(self, sim: SimSnap, trans_obj: Transformation | None, stats: PerfStats) -> TTrans:
        try:
            with TraceManager.trace_phase(self, "calculate"):
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

    def get_target(self, sim: SimSnap, previous: Transformation | None = None) -> Transformation | SimSnap:
        target: Transformation | SimSnap
        if previous is not None:
            target = previous
        elif self.move_all:
            target = sim.ancestor
        else:
            target = sim
        return target

    def calculate(self, sim: SimSnap, previous: Transformation | None = None) -> TTrans:
        raise NotImplementedError

    @classmethod
    def chain(self, *transforms: *tuple[*tuple["TransformBase[Any]", ...], _LastT]) -> _LastT:
        return chain_transforms(*transforms)

# ---------- typed TransformChain helper ----------

def chain_transforms(
    *transforms: *tuple[*tuple[TransformBase[Any], ...], _LastT]
) -> _LastT:
    """ Chain multiple TransformBase instances together.

    Parameters
    ----------
    *transforms : TransformBase
        The transformations to chain together.

    Returns
    -------
    TransformBase
        The final transformation in the chain.

    Notes
    -----
    Each transformation will be applied in the order they are provided.
    """
    if not transforms:
        raise ValueError("At least one transform must be provided.")
    prev: TransformBase[Any] | _LastT | None = None
    for transform in transforms:
        cur = transform
        if prev is not None:
            cur = cur.with_transformation(prev)
        prev = cur

    assert prev is not None
    return cast("_LastT", prev)
