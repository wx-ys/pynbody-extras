
import time
from typing import Generic, TypeVar

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformable, Transformation

from pynbodyext.calculate import CalculatorBase
from pynbodyext.log import logger

TTrans = TypeVar("TTrans",bound=Transformation, covariant=True)

class TransformBase(CalculatorBase[TTrans], Generic[TTrans]):
    """ Simsnap -> TTrans """

    move_all: bool = True

    def __call__(self, sim: SimSnap) -> TTrans:

        logger.debug(f"[{self}] Starting: {sim}")
        t0 = time.perf_counter()
        trans_obj = None

        # pre-transform
        t_pre_transform = None
        if self._transformation is not None:
            t_pre_transform_start = time.perf_counter()
            try:
                logger.debug(f"[{self}] Pre-transform starting")
                trans_obj = self._transformation(sim)
            except Exception as e:
                logger.error(f"Error applying pre-transformation: {e}, when applying {self}.")
                raise RuntimeError(f"Error applying pre-transformation {self._transformation}") from e
            t_pre_transform = time.perf_counter() - t_pre_transform_start

        # filter
        t_filter = None
        if self._filter is not None:
            t_filter_start = time.perf_counter()
            try:
                logger.debug(f"[{self}] filter: {self._filter}")
                sim = sim[self._filter]
            except Exception as e:
                logger.error(f"Error applying pre-filter: {e}, when applying {self}")
                raise RuntimeError(f"Error applying pre-filter {self._filter}") from e
            t_filter = time.perf_counter() - t_filter_start
        # self apply
        t_cal_start = time.perf_counter()
        try:
            logger.debug(f"[{self}] calculating ...")
            result = self.calculate(sim, trans_obj)
        except Exception as e:
            if trans_obj is not None and self._revert_transformation:
                trans_obj.revert()
            raise e

        if trans_obj is not None and self._revert_transformation:
            logger.debug(f"[{self}] merging pre-transformation: {self._transformation}")

        t_cal = time.perf_counter() - t_cal_start
        t_total = time.perf_counter() - t0

        # stats
        stats_lines = []
        if t_pre_transform is not None:
            stats_lines.append(f"pre_transform : {t_pre_transform:.3f}s")
        if t_filter is not None:
            stats_lines.append(f"filter : {t_filter:.3f}s")
        stats_lines.append(f"calculate : {t_cal:.3f}s")

        stats_table = " | ".join(stats_lines)

        if self._time_enabled:
            old_level = logger.level
            logger.setLevel(20)
            logger.info(f"[{self}] \n Total : {t_total:.3f}s  <| {stats_table} |>")
            logger.setLevel(old_level)
        else:
            logger.debug(f"[{self}] \n Total : {t_total:.3f}s  <| {stats_table} |>")
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

