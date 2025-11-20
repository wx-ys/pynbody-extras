"""
Generic calculation interface for pynbody snapshots.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.log import logger

from .util._type import (
    FilterLike,
    Self,
    SimCallable,
    SingleElementArray,
    TransformLike,
    TypeVarTuple,
    UnitLike,
    Unpack,
)

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap
    from pynbody.transformation import Transformation


ReturnT = TypeVar("ReturnT",covariant=True)

Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
U = TypeVar("U")
T = TypeVar("T")

class CalculatorBase(SimCallable[ReturnT], Generic[ReturnT], ABC):
    """An abstract base class for performing calculations on particle data."""

    _filter: FilterLike | None
    _transformation: TransformLike | None
    _revert_transformation: bool
    _time_enabled: bool

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        # initialize per-instance filter and transformation settings
        self = super().__new__(cls)
        self._filter = None
        self._transformation = None
        self._revert_transformation = True
        self._time_enabled = False
        return self

    def _do_pre_transform(self, sim: SimSnap) -> tuple[float | None, Transformation | None]:
        t_pre_transform = None
        trans_obj = None
        if self._transformation is not None:
            t_pre_transform_start = time.perf_counter()
            try:
                logger.debug(f"[{self}] transformation: {self._transformation}")
                trans_obj = self._transformation(sim)
            except Exception as e:
                logger.error(f"Error applying pre-transformation: {e}, when calculating {self}")
                raise RuntimeError(f"Error applying pre-transformation {self._transformation}") from e
            t_pre_transform = time.perf_counter() - t_pre_transform_start
        return t_pre_transform, trans_obj

    def _do_pre_filter(self, sim: SimSnap) -> tuple[float | None, SimSnap]:
        t_filter = None
        if self._filter is not None:
            t_filter_start = time.perf_counter()
            try:
                logger.debug(f"[{self}] filter: {self._filter}")
                sim = sim[self._filter]
            except Exception as e:
                logger.error(f"Error applying pre-filter: {e}, when calculating {self}")
                raise RuntimeError(f"Error applying pre-filter {self._filter}") from e
            t_filter = time.perf_counter() - t_filter_start
        return t_filter, sim

    def _time_stats(
        self,
        t_pre_transform: float | None,
        t_filter: float | None,
        t_cal: float,
        t_revert: float | None,
        t_total: float
        ) -> None:
        stats_lines = []
        if t_pre_transform is not None:
            stats_lines.append(f"pre_transform : {t_pre_transform:.3f}s")
        if t_filter is not None:
            stats_lines.append(f"filter : {t_filter:.3f}s")
        stats_lines.append(f"calculate : {t_cal:.3f}s")
        if t_revert is not None:
            stats_lines.append(f"revert : {t_revert:.3f}s")

        stats_table = " | ".join(stats_lines)

        if self._time_enabled:
            old_level = logger.level
            logger.setLevel(20)
            logger.info(f"[{self}] \n Total : {t_total:.3f}s  <| {stats_table} |>")
            logger.setLevel(old_level)
        else:
            logger.debug(f"[{self}] \n Total : {t_total:.3f}s  <| {stats_table} |>")

    def __call__(self, sim: SimSnap) -> ReturnT:
        """Executes the calculation on a given simulation snapshot."""

        logger.debug(f"[{self}] Starting: {sim}")
        t0 = time.perf_counter()
        trans_obj = None

        # pre-transform
        t_pre_transform, trans_obj = self._do_pre_transform(sim)

        # filter
        t_filter, sim = self._do_pre_filter(sim)

        # calculate
        t_cal_start = time.perf_counter()
        try:
            logger.debug(f"[{self}] calculating ...")
            result = self.calculate(sim)
        finally:
            t_cal = time.perf_counter() - t_cal_start
            # revert
            t_revert = None
            if trans_obj is not None and self._revert_transformation:
                t_revert_start = time.perf_counter()
                logger.debug(f"[{self}] reverting transformation: {self._transformation}")
                trans_obj.revert()
                t_revert = time.perf_counter() - t_revert_start

        t_total = time.perf_counter() - t0

        # stats
        self._time_stats(t_pre_transform, t_filter, t_cal, t_revert, t_total)
        return result

    @abstractmethod
    def calculate(self, sim: SimSnap, *args: Any, **kwargs: Any) -> ReturnT:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate"
        )

    def enable_timing(self: Self, enabled: bool = True) -> Self:
        """Enable or disable timing for this calculator."""
        self._time_enabled = enabled
        return self

    def with_filter(self: Self, filt: FilterLike) -> Self:
        """Return a new CalculatorBase instance with the given filter applied."""
        logger.debug(f"Applying filter: {filt} to {self}")
        self._filter = filt
        return self

    def __getitem__(self, key: FilterLike) -> Self:
        return self.with_filter(key)

    def with_transformation(self: Self, transformation: TransformLike, revert: bool = True) -> Self:
        """Return a new CalculatorBase instance with the given transformation applied."""
        logger.debug(f"Applying transformation: {transformation} with revert={revert} to {self}")
        self._transformation = transformation
        self._revert_transformation = revert
        return self

    def _in_sim_units(
        self,
        value:  UnitLike | float | int | SingleElementArray,
        sim_parameter: str,
        sim: SimSnap,
    ) -> float:
        """
        Convert a value into the simulation's native units for a given array.

        This converts `value` - which may be a string with units (e.g., "10 kpc"),
        a pynbody Unit object, or a raw number assumed to already be in native units -
        to a float expressed in the units of `sim[sim_parameter]`.

        Parameters
        ----------
        value : str or pynbody.units.UnitBase or float or int or single-element SimArray or NDArray
            The value to be converted.
            - str: parsed by `pynbody.units.Unit`, e.g., "10 kpc", "200 km s**-1".
            - UnitBase: converted using `.in_units(target_units, **context)`.
            - number: treated as already in the target native units.
        sim_parameter : str
            The name of the simulation array (e.g., 'r', 'rxy', 'mass') whose units
            define the conversion target.
        sim : pynbody.snapshot.SimSnap
            The simulation snapshot.

        Returns
        -------
        float
            The numeric value in the native units of `sim[sim_parameter]`.
        """
        if isinstance(value, str):
            value = units.Unit(value)
        if isinstance(value, units.UnitBase):
            value = float(value.in_units(sim[sim_parameter].units,
                                          **sim[sim_parameter].conversion_context()))
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 1:
                if isinstance(value, SimArray):
                    value = value.in_units(sim[sim_parameter].units,
                                       **sim[sim_parameter].conversion_context()).item()
                else:
                    value = value.item()
            else:
                raise TypeError(
                    "value must be a single-element array if it is an ndarray, got array with shape "
                    f"{value.shape}"
                )
        if isinstance(value, (int, float)):
            return float(value)

        raise TypeError(
            "value must be str, pynbody.units.UnitBase (or unit-like), or a number, or a single-element SimArray or NDArray, got "
            f"{type(value)}"
        )

    # ------- combinators --------
    def __and__(self: CalculatorBase[T], other: CalculatorBase[U]) -> CombinedCalculator[T, U]:
        return CombinedCalculator(self, other)

    def __repr__(self):
        if hasattr(self,"name"):
            return f"<Calculator {self.name}()>"
        return f"<Calculator {self.__class__.__name__}()>"


class CombinedCalculator(CalculatorBase[tuple[Unpack[Ts]]], Generic[Unpack[Ts]]):
    props: tuple[CalculatorBase[Any], ...]
    def __init__(self, *props: CalculatorBase[Unpack[Ts]]) -> None: # type: ignore
        self.props = props

    def calculate(self, sim: SimSnap) -> tuple[Unpack[Ts]]:
        return tuple(p(sim) for p in self.props)

    @overload   # type: ignore
    def __and__(self: CombinedCalculator[Unpack[Ts]],
                other: CalculatorBase[U]) -> CombinedCalculator[Unpack[Ts], U]: ...
    @overload
    def __and__(self: CombinedCalculator[Unpack[Ts]],     # type: ignore
                other: CombinedCalculator[Unpack[Us]]) -> CombinedCalculator[Unpack[Ts], Unpack[Us]]: ...   # type: ignore

    def __and__(self, other: Any) -> Any:
        if isinstance(other, CombinedCalculator):
            return CombinedCalculator(*self.props, *other.props)
        elif isinstance(other, CalculatorBase):
            return CombinedCalculator(*self.props, other)
        else:
            raise TypeError(f"Unsupported operand type(s) for &: 'CombinedCalculator' and '{type(other).__name__}'")

    def __repr__(self) -> str:
        return f"CombinedCalculator({', '.join(repr(p) for p in self.props)})"
