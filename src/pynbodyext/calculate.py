"""
Generic calculation interface for pynbody snapshots.
"""
from __future__ import annotations

import itertools
import json
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.log import logger
from pynbodyext.util.perf import PerfStats

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
    from collections.abc import Iterator

    from pynbody.snapshot import SimSnap
    from pynbody.transformation import Transformation


ReturnT = TypeVar("ReturnT",covariant=True)

Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
U = TypeVar("U")
T = TypeVar("T")

# --- runtime trace context (run-level id + call path) ---
_CALC_PATH: ContextVar[tuple[str, ...]] = ContextVar("_CALC_PATH", default=())
_CALC_RUN_ID: ContextVar[int | None] = ContextVar("_CALC_RUN_ID", default=None)
_RUN_COUNTER = itertools.count(1)
_TRACE_JSON = bool(int(os.getenv("PYNBODYEXT_TRACE_JSON", "0")))

def _node_label(node: Any) -> str:
    # Prefer an explicit `.name` if provided, fall back to class name
    return getattr(node, "name", node.__class__.__name__)

@contextmanager
def _trace_phase(node: Any, phase: str) -> Iterator[None]:
    """
    Push node into the calc-path, log enter/leave with run-id and full path.
    If PYNBODYEXT_TRACE_JSON=1, also emit JSON lines for machine parsing.
    """
    path = _CALC_PATH.get()
    run_id = _CALC_RUN_ID.get()
    owner = False
    if run_id is None:
        run_id = next(_RUN_COUNTER)
        _CALC_RUN_ID.set(run_id)
        owner = True

    new_path = (*path, _node_label(node))
    token = _CALC_PATH.set(new_path)
    prefix = f"calc#{run_id} {' > '.join(new_path)}"

    if _TRACE_JSON:
        logger.debug("TRACE %s", json.dumps({
            "event": "enter",
            "run": run_id,
            "phase": phase,
            "node": _node_label(node),
            "class": node.__class__.__name__,
            "id": id(node),
            "depth": len(new_path)-1
        }, ensure_ascii=False))
    else:
        logger.debug("%s | enter: %s", prefix, phase)

    try:
        yield
    finally:
        if _TRACE_JSON:
            logger.debug("TRACE %s", json.dumps({
                "event": "leave",
                "run": run_id,
                "phase": phase,
                "node": _node_label(node),
                "class": node.__class__.__name__,
                "id": id(node),
                "depth": len(new_path)-1
            }, ensure_ascii=False))
        else:
            logger.debug("%s | leave: %s", prefix, phase)

        _CALC_PATH.reset(token)
        if owner:
            _CALC_RUN_ID.set(None)

class CalculatorBase(SimCallable[ReturnT], Generic[ReturnT], ABC):
    """An abstract base class for performing calculations on particle data."""

    _filter: FilterLike | None
    _transformation: TransformLike | None
    _revert_transformation: bool
    _perf_stats: PerfStats

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        # initialize per-instance filter and transformation settings
        self = super().__new__(cls)
        self._filter = None
        self._transformation = None
        self._revert_transformation = True
        self._perf_stats = PerfStats(time=False, memory=False)
        return self

    # Calculation phase "child calculation" hook: default is none, subclasses may override
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        return []

    def format_flow(self, indent: int = 0, short: bool = True) -> str:
        """
        Semantic flow rendering:
        Node
        ├─ 1) transform: <transform-node-or-repr>
        ├─ 2) filter:    <filter-node-or-repr>
        └─ 3) calculate:
             └─ <child-calculation(s)...>
        Note: If transform/filter is CalculatorBase, recursively expand with format_flow.
        """
        pad = "  " * indent
        lines: list[str] = []
        title = f"{pad}{_node_label(self)}"
        lines.append(title)

        # 1) transform
        if getattr(self, "_transformation", None) is not None:
            tobj = self._transformation
            if isinstance(tobj, CalculatorBase):
                lines.append(f"{pad}├─ 1) transform:")
                lines.append(tobj.format_flow(indent + 2, short=short))
            else:
                lines.append(f"{pad}├─ 1) transform: {repr(tobj)}")
        elif not short:
            lines.append(f"{pad}├─ 1) transform: <none>")

        # 2) filter
        if getattr(self, "_filter", None) is not None:
            fobj = self._filter
            if isinstance(fobj, CalculatorBase):
                lines.append(f"{pad}├─ 2) filter:")
                lines.append(fobj.format_flow(indent + 2))
            else:
                lines.append(f"{pad}├─ 2) filter: {repr(fobj)}")
        elif not short:
            lines.append(f"{pad}├─ 2) filter: <none>")

        # 3) calculate
        kids = self.calculate_children()
        if kids:
            lines.append(f"{pad}└─ 3) calculate:")
            last = len(kids) - 1
            for i, ch in enumerate(kids):
                # Child calculation also uses format_flow (they may have their own three steps)
                rendered = ch.format_flow(indent + 2)
                # Visual alignment: add a branch line before the first child block
                if i < last:
                    # Middle block
                    lines.append(f"{pad}   ├─")
                else:
                    # Last block
                    lines.append(f"{pad}   └─")
                lines.append(rendered)
        else:
            lines.append(f"{pad}└─ 3) calculate: {self.__class__.__name__}")

        return "\n".join(lines)

    # children hook for tree formatting and optional traversal
    def children(self) -> list[CalculatorBase[Any]]:
        kids: list[CalculatorBase[Any]] = []
        if isinstance(getattr(self, "_filter", None), CalculatorBase):
            kids.append(self._filter)   # type: ignore[arg-type]
        if isinstance(getattr(self, "_transformation", None), CalculatorBase):
            kids.append(self._transformation)  # type: ignore[arg-type]
        return kids

    def format_tree(self, indent: int = 0) -> str:
        """
        Pretty-print the calculator tree using children() to discover subnodes.
        Falls back to showing filter/transformation if they don't support children().
        """
        # box-drawing ASCII
        def render(node: CalculatorBase[Any], prefix: str, is_last: bool) -> list[str]:
            branch = "└─" if is_last else "├─"
            name = _node_label(node)
            line = f"{prefix}{branch} {name}"
            out = [line]
            new_prefix = f"{prefix}{'  ' if is_last else '│ '}"
            # Use children() to traverse deeper
            kids = []
            if hasattr(node, "children"):
                try:
                    kids = list(node.children())
                except Exception:
                    kids = []
            # For non-Calculator filter/transformation, still display a one-liner
            fobj = getattr(node, "_filter", None)
            tobj = getattr(node, "_transformation", None)
            if fobj is not None and not isinstance(fobj, CalculatorBase):
                out.append(f"{new_prefix}├─ filter: {repr(fobj)}")
            if tobj is not None and not isinstance(tobj, CalculatorBase):
                out.append(f"{new_prefix}├─ transformation: {repr(tobj)}")
            # Recurse into calculator-like children
            for i, ch in enumerate(kids):
                out.extend(render(ch, new_prefix, i == len(kids)-1))
            return out

        lines = render(self, "" if indent == 0 else "  " * indent, True)
        return "\n".join(lines)


    def _do_pre_transform(self, sim: SimSnap) -> Transformation | None:
        if self._transformation is not None:

            with _trace_phase(self, "transform"):
                try:
                    logger.debug(f"[{self}] transformation: {self._transformation}")
                    trans_obj = self._transformation(sim)
                except Exception as e:
                    logger.error(f"Error applying pre-transformation: {e}, when calculating {self}")
                    raise RuntimeError(f"Error applying pre-transformation {self._transformation}") from e
                return trans_obj
        return None

    def _do_pre_filter(self, sim: SimSnap) -> SimSnap:
        if self._filter is not None:
            with _trace_phase(self, "filter"):
                try:
                    logger.debug(f"[{self}] filter: {self._filter}")
                    sim = sim[self._filter]
                except Exception as e:
                    logger.error(f"Error applying pre-filter: {e}, when calculating {self}")
                    raise RuntimeError(f"Error applying pre-filter {self._filter}") from e
        return sim

    def _do_calculate(self, sim: SimSnap, trans_obj: Transformation | None, stats: PerfStats) -> ReturnT:
        try:
            with _trace_phase(self, "calculate"):
                with stats.step("calculate"):
                    logger.debug(f"[{self}] performing calculation ...")
                    result = self.calculate(sim)
        finally:
            if trans_obj is not None and self._revert_transformation:
                with _trace_phase(self, "revert"):
                    with stats.step("revert"):
                        logger.debug(f"[{self}] reverting transformation: {self._transformation}")
                        trans_obj.revert()
        return result


    def __call__(self, sim: SimSnap) -> ReturnT:
        """Executes the calculation on a given simulation snapshot."""

        logger.debug(f"[{self}] Starting: {sim}")

        with self._perf_stats as stats:
            # pre-transform
            with stats.step("transform"):
                trans_obj = self._do_pre_transform(sim)

            # filter
            with stats.step("filter"):
                sim = self._do_pre_filter(sim)

            # calculate
            result = self._do_calculate(sim, trans_obj, stats)
        # stats
        self._perf_stats.report(logger)
        logger.debug(f"[{self}] Finished: {sim} \n")
        return result

    @abstractmethod
    def calculate(self, sim: SimSnap, *args: Any, **kwargs: Any) -> ReturnT:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate"
        )

    def enable_perf(self: Self, time: bool = True, memory: bool = True) -> Self:
        """Enable or disable timing and memory profiling for this calculator."""
        self._perf_stats.time_enabled = time
        self._perf_stats.memory_enabled = memory
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

    def children(self) -> list[CalculatorBase[Any]]:
        # Expand own props and also keep parent filter/transformation expansion
        return [*self.props, *super().children()]

    def calculate_children(self) -> list[CalculatorBase[Any]]:
        # Expand all child calculations under "3) calculate"
        return list(self.props)

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
