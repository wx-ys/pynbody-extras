"""Evaluation engine for calculator graphs.

The engine is the runtime core of :mod:`pynbodyext.core.calculate`.  It owns
graph evaluation, dependency traversal, per-run cache interaction, trace and
performance recording, error collection, and result assembly.

Most users do not call :class:`EvalEngine` directly.  The normal entry point is
still ``calculator.run(sim)`` or ``calculator(sim)``.  Direct engine use is
mainly helpful for advanced integration, tests, and low-level debugging.

What The Engine Does
--------------------
For each run, the engine is responsible for:

- creating an :class:`ExecutionContext`
- traversing the calculator dependency graph
- detecting cycles
- recording per-node execution status and phases
- storing and reusing runtime cache entries
- assembling the final :class:`Result`

This means the engine is the layer where the abstract calculator graph becomes
a concrete execution trace.

Normal Usage
------------
In everyday analysis code, you should usually let the calculator object invoke
the engine for you::

    result = calc.run(sim)
    print(result.value)

This keeps the call site compact and ensures default options attached to the
calculator are respected.

Direct Engine Usage
-------------------
Direct engine use is occasionally useful when writing framework-level tests or
embedding the execution model into a larger workflow::

    from pynbodyext.core.calculate import EvalEngine, PropertyBase

    class StellarMass(PropertyBase[float]):
        def instance_signature(self):
            return ("stellar_mass",)

        def calculate(self, sim):
            return float(sim["mass"].sum())

    engine = EvalEngine()
    result = engine.run(StellarMass(), sim)
    print(result.value)

Execution Model Overview
------------------------
A typical engine-managed run follows this sequence:

1. create a fresh :class:`ExecutionContext`
2. estimate graph size for progress reporting
3. evaluate the root node and its dependencies
4. reuse cached runtime values when available
5. store node summaries, reports, and diagnostics
6. assemble and return a :class:`Result`

Each node is evaluated in the same runtime context, which is why cache,
diagnostics, and provenance are coherent across the full graph.

When To Read This Module
------------------------
This module is most relevant when you are:

- debugging why a node ran more than once
- understanding how result trees are assembled
- checking how cache hits and misses are recorded
- writing low-level tests for new calculator behaviors

Notes
-----
If you are documenting or teaching the public API, prefer showing
``calc.run(sim)`` first and introduce :class:`EvalEngine` only as the advanced
runtime layer underneath.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from pynbody.array import SimArray

from .context import ExecutionContext, FilterResult, NodeInput, NodeProgressEvent, RunOptions, RunProgressEvent
from .enums import CachePolicy, ErrorPolicy, NodeStatus, RecordPolicy
from .exceptions import CycleError
from .result import ErrorInfo, ProvenanceInfo, Result, ResultNode, ValueSummary

if TYPE_CHECKING:
    from .base import CalculatorBase

T = TypeVar("T")
TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")

@dataclass(slots=True)
class _EvaluationPlan:
    work: NodeInput
    stack_key: tuple[Any, ...]
    cache_key: tuple[Any, ...]
    cacheable: bool
    cache_policy: CachePolicy

@dataclass(slots=True)
class _NodeExecutionState:
    raw_value: Any = None
    public_value: Any = None


class _NodeExecutionFailure(Exception):
    def __init__(self, cause: Exception, state: _NodeExecutionState) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.state = state


class EvalEngine:
    """Evaluate calculator DAGs in a single run context."""

    def run(
        self,
        node: CalculatorBase[TRaw, TPublic],
        sim: Any,
        options: RunOptions | None = None,
    ) -> Result[TPublic]:
        """Run ``node`` on ``sim`` and return a :class:`Result`.

        Parameters
        ----------
        node : CalculatorBase
            Root calculator node.
        sim : object
            Simulation object passed to the root calculator.
        options : RunOptions, optional
            Execution options.
        """
        opts = options or RunOptions()
        started = time.perf_counter()
        estimated_total_nodes = self._estimate_total_nodes(node)

        ctx = ExecutionContext(
            sim=sim,
            sim_signature=self.make_sim_signature(sim),
            run_id=str(uuid.uuid4()),
            options=opts,
            engine=self,
        )

        status = "ok"
        run_label = node.log_label
        root: ResultNode | None = None

        ctx._progress_sink.on_run_start(
            RunProgressEvent(
                run_id=ctx.run_id,
                root_name=run_label,
                started_at=started,
                total_nodes=estimated_total_nodes,
            )
        )
        ctx.log("debug", f"run start: {run_label}")

        try:
            root = self.evaluate(node, ctx, NodeInput(sim_raw=sim, sim_current=sim))
        except Exception:
            status = "error"
            if opts.errors == ErrorPolicy.RAISE:
                raise
            root = self._resolve_failed_root(ctx)
        finally:
            finished_for_event = time.perf_counter()
            ctx._progress_sink.on_run_end(
                RunProgressEvent(
                    run_id=ctx.run_id,
                    root_name=run_label,
                    status=status,
                    started_at=started,
                    finished_at=finished_for_event,
                    elapsed_s=finished_for_event - started,
                    total_nodes=estimated_total_nodes,
                    node_count=len(ctx.node_registry),
                    warning_count=len(ctx.warnings),
                    error_count=len(ctx.errors),
                )
            )
            ctx.log("debug", f"run end: {run_label} status={status}")

        return self._assemble_result(
            node=node,
            ctx=ctx,
            root=root,
            run_label=run_label,
            started=started,
        )

    def evaluate(
        self,
        node: CalculatorBase[TRaw, TPublic],
        ctx: ExecutionContext,
        input: NodeInput | None = None
    ) -> ResultNode:
        """Evaluate one node within an existing :class:`ExecutionContext`."""
        plan = self._make_evaluation_plan(node, ctx, input)
        if plan.stack_key in ctx._evaluation_stack:
            raise CycleError(f"Cycle detected while evaluating {node.log_label!r}")

        cached_node = self._try_cache_hit(node, ctx, plan)
        if cached_node is not None:
            return cached_node

        node_result = self._create_node_result(node, ctx)
        ctx.register_node(node_result)
        is_root = ctx.current_node is None

        ctx._evaluation_stack.append(plan.stack_key)
        try:
            state = self._execute_node_body(node, ctx, node_result, plan.work)
        except _NodeExecutionFailure as failure:
            self._record_node_error(
                ctx=ctx,
                node_result=node_result,
                exc=failure.cause,
                raw_value=failure.state.raw_value,
                public_value=failure.state.public_value,
                is_root=is_root,
            )
            raise failure.cause from None

        finally:
            ctx._evaluation_stack.pop()

        return self._finalize_successful_node(
            node=node,
            ctx=ctx,
            node_result=node_result,
            state=state,
            plan=plan,
            is_root=is_root,
        )

    def _make_evaluation_plan(
        self,
        node: CalculatorBase[Any, Any],
        ctx: ExecutionContext,
        input: NodeInput | None,
    ) -> _EvaluationPlan:
        work = input or NodeInput(sim_raw=ctx.sim, sim_current=ctx.sim)
        work = work.with_mutation_generation(ctx.mutation_generation)

        cache_policy = getattr(node, "cache_policy", CachePolicy.AUTO)
        cacheable = bool(getattr(node, "cacheable", True)) and cache_policy != CachePolicy.NONE

        return _EvaluationPlan(
            work=work,
            stack_key=(id(node), work.cache_token),
            cache_key=(ctx.sim_signature, work.cache_token, node.signature()),
            cacheable=cacheable,
            cache_policy=cache_policy,
        )

    def _estimate_total_nodes(self, node: CalculatorBase[Any, Any]) -> int | None:
        """Best-effort static estimate of unique nodes in the dependency DAG."""
        seen: set[int] = set()

        def visit(current: CalculatorBase[Any, Any]) -> int:
            key = id(current)
            if key in seen:
                return 0
            seen.add(key)

            total = 1
            for child in current.dependencies():
                total += visit(child)
            return total

        try:
            return visit(node)
        except Exception:
            return None

    def _try_cache_hit(
        self,
        node: CalculatorBase[Any, Any],
        ctx: ExecutionContext,
        plan: _EvaluationPlan,
    ) -> ResultNode | None:
        cached_runtime = ctx.cache.get(plan.cache_key) if plan.cacheable else None
        if cached_runtime is None:
            ctx.trace.cache(
                node_id="pending",
                node_name=node.log_label,
                event="miss" if plan.cacheable else "skip",
                key=repr(plan.cache_key),
            )
            return None

        cached_node = ctx.node_registry[cached_runtime.node_id]
        parent = ctx.current_node
        if parent is not None and cached_node.node_id not in parent.children:
            parent.children.append(cached_node.node_id)

        node_name = cached_node.label or node.log_label
        ctx.trace.cache(
            node_id=cached_node.node_id,
            node_name=node_name,
            event="hit",
            key=repr(plan.cache_key),
        )
        ctx.log("debug", f"cache hit: {node_name}", node_id=cached_node.node_id)

        now = time.perf_counter()
        ctx._progress_sink.on_node_end(
            NodeProgressEvent(
                run_id=ctx.run_id,
                node_id=cached_node.node_id,
                node_name=node_name,
                kind=str(cached_node.kind),
                depth=len(ctx._node_stack),
                status="ok",
                started_at=now,
                finished_at=now,
                elapsed_s=0.0,
                from_cache=True,
            )
        )
        return cached_node

    def _create_node_result(
        self,
        node: CalculatorBase[Any, Any],
        ctx: ExecutionContext,
    ) -> ResultNode:
        return ResultNode(
            node_id=ctx.new_node_id(),
            kind=node.kind,
            signature=node.signature(),
            name=node.name,
            display_name=node.log_label,
            calculator_type=node.__class__.__name__,
            record_policy=node.record_policy or ctx.options.default_record_policy,
        )

    def _execute_node_body(
        self,
        node: CalculatorBase[T, Any],
        ctx: ExecutionContext,
        node_result: ResultNode,
        work: NodeInput,
    ) -> _NodeExecutionState:
        state = _NodeExecutionState()
        with ctx.node_scope(node_result, node):
            try:
                state.raw_value = node.execute(ctx, work)
                state.raw_value = node.materialize(ctx, state.raw_value)
                state.public_value = node.public_value(state.raw_value)
                state.public_value = node.materialize_public(ctx, state.public_value)
            except Exception as exc:
                raise _NodeExecutionFailure(exc, state) from exc
        return state

    def _record_node_error(
        self,
        *,
        ctx: ExecutionContext,
        node_result: ResultNode,
        exc: Exception,
        raw_value: Any,
        public_value: Any,
        is_root: bool,
    ) -> None:
        node_result.status = NodeStatus.ERROR
        node_result.error = ErrorInfo(
            error_type=exc.__class__.__name__,
            message=str(exc),
            phase=node_result.phases[-1].phase if node_result.phases else None,
        )
        ctx.last_error_node_id = node_result.node_id
        ctx.errors.append(node_result.error)

        summary_source = public_value if public_value is not None else raw_value
        if summary_source is not None:
            node_result.value_summary = self.summarize_value(summary_source)

        self._store_recorded_values(
            node_result,
            raw_value=raw_value,
            public_value=public_value,
            is_root=is_root,
            had_error=True,
        )

    def _finalize_successful_node(
        self,
        *,
        node: CalculatorBase[Any, Any],
        ctx: ExecutionContext,
        node_result: ResultNode,
        state: _NodeExecutionState,
        plan: _EvaluationPlan,
        is_root: bool,
    ) -> ResultNode:
        ctx.register_runtime_value(node_result.node_id, state.raw_value, state.public_value)

        if plan.cacheable and self._should_store_runtime_cache(
            node,
            state.raw_value,
            state.public_value,
            plan.cache_policy,
            ctx.options,
        ):
            ctx.cache.set(plan.cache_key, ctx.runtime_store[node_result.node_id])

        node_result.value_summary = self.summarize_value(state.public_value)
        node_result.status = NodeStatus.OK
        self._store_recorded_values(
            node_result=node_result,
            raw_value=state.raw_value,
            public_value=state.public_value,
            is_root=is_root,
            had_error=False,
        )
        return node_result

    def _resolve_failed_root(self, ctx: ExecutionContext) -> ResultNode:
        if ctx.last_error_node_id is not None:
            return ctx.node_registry[ctx.last_error_node_id]
        if ctx.node_registry:
            return next(reversed(ctx.node_registry.values()))

        raise RuntimeError("run failed before any result node was registered")

    def _assemble_result(
        self,
        *,
        node: CalculatorBase[TRaw, TPublic],
        ctx: ExecutionContext,
        root: ResultNode,
        run_label: str,
        started: float,
    ) -> Result[TPublic]:
        finished = time.perf_counter()
        provenance = ProvenanceInfo(
            calculator_signature=node.signature(),
            calculator_signature_text=node.signature_text(),
            calculator_signature_hash=node.signature_hash(),
            sim_signature=ctx.sim_signature,
            started_at=started,
            finished_at=finished,
        )

        named = {
            name: ctx.node_registry[node_id]
            for name, node_id in ctx.named_registry.items()
            if node_id in ctx.node_registry
        }
        named_values = {
            name: ctx.runtime_store[node_result.node_id].public_value
            for name, node_result in named.items()
            if node_result.node_id in ctx.runtime_store
        }

        perf_report = ctx.perf.report_text(ctx.node_registry, title=run_label)
        cache_report = ctx.cache.report_text()
        trace_timeline = ctx.trace.render_timeline()
        trace_tree = ctx.trace.render_tree(ctx.node_registry, root.node_id)

        perf_summary = ctx.perf.summary(ctx.node_registry, cache_hit_count=ctx.cache.hit_count)
        cache_summary = ctx.cache.summary()
        perf_summary.cache_miss_count = int(cache_summary["misses"])
        perf_summary.cache_store_count = int(cache_summary["stores"])

        root_value = cast(
            "TPublic",
            ctx.runtime_store[root.node_id].public_value if root.node_id in ctx.runtime_store else None,
        )

        root.artifacts["perf_report"] = perf_report
        root.artifacts["cache_report"] = cache_report
        root.artifacts["trace_timeline"] = trace_timeline
        root.artifacts["trace_tree"] = trace_tree
        root.artifacts["log_events"] = list(ctx.log_events)

        return Result(
            value=root_value,
            root=root,
            nodes=dict(ctx.node_registry),
            named=named,
            provenance=provenance,
            perf_summary=perf_summary,
            warnings=list(ctx.warnings),
            errors=list(ctx.errors),
            reports={
                "perf": perf_report,
                "cache": cache_report,
                "trace_timeline": trace_timeline,
                "trace_tree": trace_tree,
            },
            diagnostics={
                "trace_events": list(ctx.trace.events),
                "cache_events": list(ctx.cache.events),
                "log_events": list(ctx.log_events),
                "named_values": named_values,
            },
        )

    def _should_store_runtime_cache(
        self,
        node: CalculatorBase[Any, Any],
        raw_value: Any,
        public_value: Any,
        policy: CachePolicy,
        options: RunOptions,
    ) -> bool:
        if policy == CachePolicy.NONE:
            return False
        if policy == CachePolicy.FULL:
            return True
        if getattr(node, "effect", None) is not None and str(node.effect) == "mutating":
            return False
        size = self._estimate_cache_bytes(public_value)
        if isinstance(raw_value, FilterResult):
            size += self._estimate_cache_bytes(raw_value.mask)
        elif raw_value is not public_value:
            size += self._estimate_cache_bytes(raw_value)
        return size <= options.cache_small_value_bytes

    def _estimate_cache_bytes(self, value: Any) -> int:
        size = 2 * 1_000_000

        if value is None or isinstance(value, (bool, int, float, np.generic)):
            size = 64
        elif isinstance(value, str):
            size = len(value.encode("utf-8"))
        elif isinstance(value, np.ndarray):
            size = int(value.nbytes)
        elif isinstance(value, FilterResult):
            size = self._estimate_cache_bytes(value.mask)
        elif isinstance(value, dict):
            size = sum(
                self._estimate_cache_bytes(key) + self._estimate_cache_bytes(item)
                for key, item in value.items()
            )
        elif isinstance(value, (tuple, list)):
            size = sum(self._estimate_cache_bytes(item) for item in value)

        return size

    def _store_recorded_values(
        self,
        node_result: ResultNode,
        raw_value: Any,
        public_value: Any,
        *,
        is_root: bool,
        had_error: bool = False,
    ) -> None:
        policy = node_result.record_policy or RecordPolicy.SUMMARY

        node_result.raw_value = None
        node_result.value = None
        node_result.stored_raw = False
        node_result.stored_value = False

        if is_root:
            if public_value is not None:
                node_result.value = public_value
                node_result.stored_value = True
            if policy == RecordPolicy.FULL or (
                had_error and policy == RecordPolicy.ERROR_ONLY and raw_value is not None):
                node_result.raw_value = raw_value
                node_result.stored_raw = True
            return

        if policy == RecordPolicy.FULL:
            node_result.raw_value = raw_value
            node_result.value = public_value
            node_result.stored_raw = True
            node_result.stored_value = True

        elif policy == RecordPolicy.ERROR_ONLY:
            if had_error:
                if raw_value is not None:
                    node_result.raw_value = raw_value
                    node_result.stored_raw = True
                if public_value is not None:
                    node_result.value = public_value
                    node_result.stored_value = True

        elif policy in (RecordPolicy.SUMMARY, RecordPolicy.NONE):
            node_result.raw_value = None
            node_result.value = None


    def summarize_value(self, value: Any) -> ValueSummary | None:
        """Create a compact summary used in reports and result nodes."""
        if value is None:
            return ValueSummary(python_type="NoneType", preview="None")

        units = None
        if hasattr(value, "units"):
            try:
                units = str(value.units)
            except Exception:
                units = None

        shape = None
        if hasattr(value, "shape"):
            try:
                shape = tuple(value.shape)
            except Exception:
                shape = None

        dtype = None
        if hasattr(value, "dtype"):
            try:
                dtype = str(value.dtype)
            except Exception:
                dtype = None

        if isinstance(value, (int, float, bool, str)):
            preview = repr(value)
        elif isinstance(value, np.ndarray):
            preview = f"ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, SimArray):
            preview = f"SimArray(shape={value.shape}, units={units})"
        else:
            preview = value.__class__.__name__

        return ValueSummary(
            python_type=value.__class__.__name__,
            shape=shape,
            dtype=dtype,
            units=units,
            preview=preview,
        )

    def make_sim_signature(self, sim: Any) -> tuple[Any, ...]:
        """Return the simulation identity fragment used in cache keys."""
        return ("sim", id(sim))
