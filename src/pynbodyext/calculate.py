"""
pynbodyext.calculate
====================

Generic calculation interface for pynbody snapshots.

This module provides an abstract base class `CalculatorBase` for defining
calculations on particle data in pynbody simulations, supporting composable
filters, transformations, performance profiling, and caching. It also provides
`CombinedCalculator` for combining multiple calculators.

Classes
-------
CalculatorBase : Abstract base class for calculations on simulation data.
CombinedCalculator : Combines multiple CalculatorBase instances for joint evaluation.

Usage
-----
Subclass `CalculatorBase` to implement custom calculations, and use
`CombinedCalculator` to combine multiple calculators.


>>> class MyCalc(CalculatorBase[float]):
...     def calculate(self, sim: SimSnap) -> float:
...         return np.mean(sim['mass'])
...
>>> calc = MyCalc()
>>> result = calc.enable_perf()(sim)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.log import logger
from pynbodyext.util.perf import PerfStats
from pynbodyext.util.tracecache import EvalCacheManager, TraceManager

from .util._type import (
    FilterLike,
    Self,
    SimCallable,
    SingleElementArray,
    TransformLike,
    TypeVarTuple,
    UnitLike,
    Unpack,
    get_signature_safe,
)

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap
    from pynbody.transformation import Transformation


ReturnT = TypeVar("ReturnT",covariant=True)

Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
U = TypeVar("U")
T = TypeVar("T")
TCalculator = TypeVar("TCalculator", bound="CalculatorBase[Any]")

class CalculatorBase(SimCallable[ReturnT], Generic[ReturnT], ABC):
    """
    Abstract base class for performing calculations on particle data.

    This class provides a unified interface for calculations on pynbody
    simulation snapshots, supporting composable filters, transformations,
    performance profiling, and evaluation caching.

    Subclasses should implement the `calculate` method.

    Attributes
    ----------
    _filter : FilterLike or None
        Optional filter to apply before calculation.
    _transformation : TransformLike or None
        Optional transformation to apply before calculation.
    _revert_transformation : bool
        Whether to revert the transformation after calculation.
    _perf_stats : PerfStats
        Performance statistics collector.
    _enable_eval_cache : bool
        Whether to enable evaluation caching for this instance.

    Methods
    -------
    calculate(sim, *args, **kwargs)
        Abstract method to perform the calculation.
    with_filter(filt)
        Returns a new instance with the given filter applied.
    with_transformation(transformation, revert=True)
        Returns a new instance with the given transformation applied.
    enable_perf(time=True, memory=True)
        Enables/disables timing and memory profiling.
    enable_cache(enable=True)
        Enables/disables evaluation caching.
    format_flow(indent=0, short=True)
        Returns a semantic flow rendering of the calculation.
    format_tree(indent=0)
        Returns a pretty-printed tree of the calculator and its children.
    """

    _filter: FilterLike | None
    _transformation: TransformLike | None
    _revert_transformation: bool
    _perf_stats: PerfStats
    _enable_eval_cache: bool = True
    _enable_chunk: bool = False
    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """
        Create a new instance, initializing filter, transformation, and performance stats.
        """
        # initialize per-instance filter and transformation settings
        self = super().__new__(cls)
        self._filter = None
        self._transformation = None
        self._revert_transformation = True
        self._perf_stats = PerfStats(time=False, memory=False)
        self._enable_eval_cache = True
        self._enable_chunk = False
        return self

    # Calculation phase "child calculation" hook: default is none, subclasses may override
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        """
        Returns a list of child calculations for this node.

        Default is an empty list. Subclasses may override to provide child calculators.
        """
        return []


    def instance_signature(self) -> tuple:
        """
        Returns a signature describing how this calculator was constructed.

        Default is identity-based. Subclasses should override to return a stable
        representation of constructor arguments if possible.
        """
        return ("id", id(self))

    def signature(self) -> tuple:
        """
        Returns the full structural signature for this calculator node.

        Includes instance signature, filter signature, transformation signature,
        and revert flag. Used for run-level caching and common subexpression elimination.
        """
        # instance/init signature
        try:
            init_sig = self.instance_signature()
        except Exception:
            init_sig = ("id", id(self))

        # filter signature
        flt = getattr(self, "_filter", None)
        if flt is None:
            filter_sig = None
        else:
            filter_sig = get_signature_safe(flt, fallback_to_id=True)

        # transformation signature
        trans = getattr(self, "_transformation", None)
        if trans is None:
            trans_sig = None
        else:
            trans_sig = get_signature_safe(trans, fallback_to_id=True)

        return (
            self.__class__.__name__,
            ("init", init_sig),
            ("filter", filter_sig),
            ("trans", trans_sig),
            ("rev", bool(getattr(self, "_revert_transformation", True))),
        )

    def format_flow(self, indent: int = 0, short: bool = True) -> str:
        """
        Returns a semantic flow rendering of the calculation node.

        Parameters
        ----------
        indent : int, optional
            Indentation level for pretty-printing, by default 0
        short : bool, optional
            Whether to use a short form omitting empty transform/filter, by default True

        Rendering
        ---------
        Node
        ├─ 1) transform: <transform-node-or-repr>
        ├─ 2) filter:    <filter-node-or-repr>
        └─ 3) calculate:
             └─ <child-calculation(s)...>

        Notes
        -----
        If transform/filter is CalculatorBase, recursively expand with format_flow.
        """
        pad = "  " * indent
        lines: list[str] = []
        title = f"{pad}{TraceManager._node_label(self)}"
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
        """
        Returns a list of calculator-like children (filter and transformation if present).

        Used for tree formatting and traversal.
        """
        kids: list[CalculatorBase[Any]] = []
        if isinstance(getattr(self, "_filter", None), CalculatorBase):
            kids.append(self._filter)   # type: ignore[arg-type]
        if isinstance(getattr(self, "_transformation", None), CalculatorBase):
            kids.append(self._transformation)  # type: ignore[arg-type]
        return kids

    def format_tree(self, indent: int = 0) -> str:
        """
        Returns a pretty-printed string representation of the calculator tree.

        Parameters
        ----------
        indent : int, optional
            Indentation level for pretty-printing, by default 0

        Pretty-print the calculator tree using children() to discover subnodes.
        Falls back to showing filter/transformation if they don't support children().
        """
        # box-drawing ASCII
        def render(node: CalculatorBase[Any], prefix: str, is_last: bool) -> list[str]:
            branch = "└─" if is_last else "├─"
            name = TraceManager._node_label(node)
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
        """
        Applies the transformation to the simulation snapshot if present.

        Returns the transformation object, or None if no transformation is set.
        """
        if self._transformation is not None:

            with TraceManager.trace_phase(self, "transform"):
                try:
                    logger.debug("[%s] transformation: %s", self, self._transformation)
                    trans_obj = self._transformation(sim)
                except Exception as e:
                    logger.error("Error applying pre-transformation: %s, when calculating %s", e, self)
                    raise RuntimeError(f"Error applying pre-transformation {self._transformation}") from e
                return trans_obj
        return None

    def _do_pre_filter(self, sim: SimSnap) -> SimSnap:
        """
        Applies the filter to the simulation snapshot if present.

        Returns the filtered simulation snapshot.
        """
        if self._filter is not None:
            with TraceManager.trace_phase(self, "filter"):
                try:
                    logger.debug("[%s] filter: %s", self, self._filter)
                    sim = sim[self._filter]
                except Exception as e:
                    logger.error("Error applying pre-filter: %s, when calculating %s", e, self)
                    raise RuntimeError(f"Error applying pre-filter {self._filter}") from e
        return sim

    def _do_calculate(self, sim: SimSnap, trans_obj: Transformation | None, stats: PerfStats) -> ReturnT:
        """
        Performs the calculation and reverts the transformation if needed.

        Returns the calculation result.
        """
        try:
            with TraceManager.trace_phase(self, "calculate"):
                with stats.step("calculate"):
                    logger.debug("[%s] performing calculation ...", self)
                    result = self.calculate(sim)
        finally:
            if trans_obj is not None and self._revert_transformation:
                with TraceManager.trace_phase(self, "revert"):
                    with stats.step("revert"):
                        logger.debug("[%s] reverting transformation: %s", self, self._transformation)
                        trans_obj.revert()
        return result


    def __call__(self, sim: SimSnap) -> ReturnT:
        """
        Unified top-level call for calculator evaluation.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot to evaluate the calculator on.

        Examples
        --------
        >>> # If enable performance profiling
        >>> calculator.enable_perf()(sim)
        >>>
        >>> # If enable evaluation caching
        >>> calculator.enable_cache()(sim)
        """
        logger.debug("")
        token = id(sim)

        # detect existing ctx for this sim (fast path)
        prev_ctx = EvalCacheManager._CTX.get()
        prev_tok = EvalCacheManager._SIM_TOKEN.get()
        is_top = not (prev_ctx is not None and prev_tok == token)

        title = repr(self) + " : " + repr(sim)

        # Convenience local refs
        eval_mgr = EvalCacheManager
        trace_mgr = TraceManager

        # Helper to handle cached lookup/store when cache is enabled
        def _run_with_cache() -> ReturnT:
            # compute node signature once (fallback to struct_signature on error)
            node_sig = self.signature()
            # build key only when needed
            key = (token, ("calc", ("sig", node_sig)))
            cached = eval_mgr.get(key)
            if cached is not None:
                trace_mgr.trace_cache_event(self, "hit", {"sig": node_sig})
                return cached
            res = self._invoke(sim)
            eval_mgr.set(key, res)
            return res

        # Top-level: create trace run and evaluation context (decide enable_cache once)
        if is_top:
            with trace_mgr.trace_run(self):
                # pass instance preference once to use(); returns to previous ctx on exit
                with eval_mgr.use(sim, enable_cache=bool(getattr(self, "_enable_eval_cache", True))):
                    if eval_mgr.cache_enabled():
                        result =  _run_with_cache()
                    else:
                        # cache disabled for this run -> direct call
                        result = self._invoke(sim)
        else:
            # nested call: reuse existing ctx and its cache_enabled decision
            if eval_mgr.cache_enabled():
                return _run_with_cache()
            result = self._invoke(sim)

        self._perf_stats.report(logger, title=title)
        return result

    def _invoke(self, sim: SimSnap) -> ReturnT:
        """
        Performs the actual transform/filter/calculate under PerfStats.

        Note: TraceManager.run and EvalCacheManager.use(sim) should already be active.
        """
        # At this point TraceManager.run and EvalCacheManager.use(sim) should already be active.
        with self._perf_stats as stats:
            use_sim = sim
            if self._enable_chunk:
                # TODO, new chunking implementation
                use_sim = sim
            with stats.step("transform"):
                trans_obj = self._do_pre_transform(use_sim)
            with stats.step("filter"):
                sim2 = self._do_pre_filter(use_sim)
            # calculate (and possibly revert)
            result = self._do_calculate(sim2, trans_obj, stats)
            # TODO for new dask result use compute()
        return result

    @abstractmethod
    def calculate(self, sim: SimSnap, *args: Any, **kwargs: Any) -> ReturnT:
        """
        Abstract method to perform the calculation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate"
        )

    def enable_perf(self: Self, time: bool = True, memory: bool = True) -> Self:
        """
        Enable or disable timing and memory profiling for this calculator.

        Parameters
        ----------
        time : bool
            Enable timing profiling.
        memory : bool
            Enable memory profiling.

        Returns
        -------
        Self
            The calculator instance.
        """
        self._perf_stats.time_enabled = time
        self._perf_stats.memory_enabled = memory
        return self

    def enable_cache(self: Self, enable: bool = True) -> Self:
        """
        Enable or disable evaluation caching for this calculator instance.

        Parameters
        ----------
        enable : bool
            Whether to enable caching.

        Returns
        -------
        Self
            The calculator instance.
        """
        self._enable_eval_cache = bool(enable)
        return self

    def enable_chunk(self: Self, enable: bool = True) -> Self:
        """
        Enable or disable chunked evaluation for this calculator instance.

        Parameters
        ----------
        enable : bool
            Whether to enable chunked evaluation.

        Returns
        -------
        Self
            The calculator instance.
        """
        self._enable_chunk = bool(enable)
        return self

    def with_filter(self: Self, filt: FilterLike) -> Self:
        """
        Return a new CalculatorBase instance with the given filter applied.

        Parameters
        ----------
        filt : FilterLike
            The filter to apply.

        Returns
        -------
        Self
            The calculator instance.
        """
        logger.debug("Applying filter: %s to %s", filt, self)
        self._filter = filt
        return self

    def with_transformation(self: Self, transformation: TransformLike, revert: bool = True) -> Self:
        """
        Return a new CalculatorBase instance with the given transformation applied.

        Parameters
        ----------
        transformation : TransformLike
            The transformation to apply.
        revert : bool
            Whether to revert the transformation after calculation.

        Returns
        -------
        Self
            The calculator instance.
        """
        logger.debug("Applying transformation: %s with revert=%s to %s", transformation, revert, self)
        self._transformation = transformation
        self._revert_transformation = revert
        return self

    def __getitem__(self, key: FilterLike) -> Self:
        """
        Shorthand for applying a filter using indexing syntax.

        Returns
        -------
        Self
            The calculator instance with the filter applied.
        """
        return self.with_filter(key)

    def _in_sim_units(
        self,
        value:  UnitLike | float | int | SingleElementArray,
        sim_parameter: str,
        sim: SimSnap,
    ) -> float:
        """
        Convert a value into the simulation's native units for a given array.

        Converts `value` (str, Unit, number, or single-element array) to a float
        in the units of `sim[sim_parameter]`.

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

    @overload
    def __and__(                # type: ignore
        self: CombinedCalculator[Unpack[Ts]],
        other: CombinedCalculator[Unpack[Us]],
    ) -> CombinedCalculator[Unpack[Ts], Unpack[Us]]: ...  # type: ignore

    @overload
    def __and__(                # type: ignore
        self: CombinedCalculator[Unpack[Ts]],
        other: CalculatorBase[U],
    ) -> CombinedCalculator[Unpack[Ts], U]: ...

    @overload
    def __and__(
        self: CalculatorBase[T],
        other: CombinedCalculator[Unpack[Us]],
    ) -> CombinedCalculator[T, Unpack[Us]]: ...
    @overload
    def __and__(self: CalculatorBase[T], other: CalculatorBase[U]) -> CombinedCalculator[T, U]: ...

    def __and__(self: TCalculator, other: object) -> CombinedCalculator[Unpack[tuple[Any, ...]]]:
        """
        Combine this calculator with another using the & operator.

        Returns a CombinedCalculator containing both calculators.

        Parameters
        ----------
        other : CalculatorBase or CombinedCalculator
            The other calculator to combine.

        Returns
        -------
        CombinedCalculator
            A new CombinedCalculator instance.
        """
        if not isinstance(other, CalculatorBase):
            raise TypeError(f"Unsupported operand type(s) for &: '{type(self).__name__}' and '{type(other).__name__}'")

        left_props: tuple[CalculatorBase[Any], ...]
        if isinstance(self, CombinedCalculator):
            left_props = self.props
        else:
            left_props = (self,)

        right_props: tuple[CalculatorBase[Any], ...]
        if isinstance(other, CombinedCalculator):
            right_props = other.props
        else:
            right_props = (other,)
        return CombinedCalculator(*left_props, *right_props)

    def __repr__(self):
        if hasattr(self,"name"):
            return f"<Calculator {self.name}()>"
        return f"<Calculator {self.__class__.__name__}()>"


class CombinedCalculator(CalculatorBase[tuple[Unpack[Ts]]], Generic[Unpack[Ts]]):
    """
    Combines multiple CalculatorBase instances for joint evaluation.

    Evaluates all contained calculators and returns a tuple of their results.

    Attributes
    ----------
    props : tuple of CalculatorBase
        The calculators to combine.

    Methods
    -------
    calculate(sim)
        Evaluates all calculators and returns a tuple of results.
    children()
        Returns the list of child calculators.
    calculate_children()
        Returns the list of calculators for semantic flow rendering.
    """

    props: tuple[CalculatorBase[Any], ...]
    def __init__(self, *props: CalculatorBase[Unpack[Ts]]) -> None: # type: ignore
        """
        Initialize with the given calculators.

        Parameters
        ----------
        *props : CalculatorBase
            The calculators to combine.
        """
        self.props = props


    def instance_signature(self):
        """
        Returns a signature describing how this CombinedCalculator was constructed.
        """
        return (self.__class__.__name__, tuple(p.instance_signature() for p in self.props))

    def calculate(self, sim: SimSnap) -> tuple[Unpack[Ts]]:
        """
        Evaluates all contained calculators and returns a tuple of their results.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot.

        Returns
        -------
        tuple
            Tuple of results from each calculator.
        """
        return tuple(p(sim) for p in self.props)

    def children(self) -> list[CalculatorBase[Any]]:
        """
        Returns the list of child calculators, including parent filter/transformation.
        """
        # Expand own props and also keep parent filter/transformation expansion
        return [*self.props, *super().children()]

    def calculate_children(self) -> list[CalculatorBase[Any]]:
        """
        Returns the list of calculators for semantic flow rendering.
        """
        # Expand all child calculations under "3) calculate"
        return list(self.props)


    def __repr__(self) -> str:
        return f"CombinedCalculator({', '.join(repr(p) for p in self.props)})"
