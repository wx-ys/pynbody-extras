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
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from pynbody.array import SimArray

from pynbodyext.core.calculate.diagnostics.observer import observation_phase, render_observer_report
from pynbodyext.core.calculate.result.enums import CachePolicy, ErrorPolicy, NodeStatus, RecordPolicy
from pynbodyext.core.calculate.result.exceptions import CycleError
from pynbodyext.core.calculate.result.result import ErrorInfo, ProvenanceInfo, Result, ResultNode, ValueSummary

from ._engine_types import (
    _EvaluationPlan,
    _MinimalBatchContext,
    _NodeExecutionFailure,
    _NodeExecutionState,
)
from .context import ExecutionContext
from .input import FilterResult, NodeInput
from .options import RunOptions
from .progress import NodeProgressEvent, RunProgressEvent
from .sim_identity import SimIdentityProvider, id_based_sim_identity

if TYPE_CHECKING:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.store.base import ResultStore

T = TypeVar("T")
TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")
class EvalEngine:
    """Evaluate calculator DAGs in a single run context.

    Parameters
    ----------
    sim_identity : SimIdentityProvider, optional
        Callable mapping a simulation object to the identity tuple used in
        provenance and in-run cache keys.  Defaults to :func:`id_based_sim_identity`,
        which keys on object ``id``.
    """

    def __init__(self, *, sim_identity: SimIdentityProvider = id_based_sim_identity) -> None:
        self._sim_identity = sim_identity

    def _class_path(self, value: Any) -> str:
        cls = value if isinstance(value, type) else type(value)
        return f"{cls.__module__}.{cls.__qualname__}"

    def run(
        self,
        node: CalculatorBase[TRaw, TPublic],
        sim: Any,
        options: RunOptions | None = None,
        *,
        store: ResultStore | None = None,
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
        store : ResultStore, optional
            When given, the assembled result is persisted on completion keyed by
            ``(engine.make_sim_signature(sim), root CalculatorSignature)``.  The
            store is only written to when the run completes cleanly (no errors),
            so a failed run never silently persists a partial value.

        Notes
        -----
        This is the hook that connects the result-store seam (see
        :mod:`pynbodyext.core.calculate.store`) into the real execution path: the
        simulation identity comes from the engine's pluggable ``sim_identity``
        provider, and the calculator identity is the content-addressed root
        signature.
        """
        opts = options or RunOptions()
        started = time.perf_counter()
        estimated_total_nodes = self._estimate_total_nodes(node)

        ctx = ExecutionContext(
            sim=sim, sim_signature=self.make_sim_signature(sim), run_id=str(uuid.uuid4()), options=opts, engine=self
        )

        status = "ok"
        run_label = node.log_label
        root: ResultNode | None = None

        ctx._progress_sink.on_run_start(
            RunProgressEvent(
                run_id=ctx.run_id, root_name=run_label, started_at=started, total_nodes=estimated_total_nodes
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

        result = self._assemble_result(node=node, ctx=ctx, root=root, run_label=run_label, started=started)

        if store is not None and not result.errors:
            calculator_signature = getattr(ctx, "root_signature", None)
            if calculator_signature is None:
                calculator_signature = node.to_signature()
            store.store(result, sim_signature=ctx.sim_signature, calculator_signature=calculator_signature)

        return result

    # ------------------------------------------------------------------
    # Batch / light-weight execution path
    # ------------------------------------------------------------------

    def run_light(
        self,
        node: CalculatorBase[TRaw, TPublic],
        sim: Any,
        options: RunOptions,
        *,
        _precomputed_node_sig: tuple[Any, ...] | None = None,
        _precomputed_structured_sig: Any | None = None,
    ) -> TPublic:
        """Run *node* on *sim* and return only the public value, with minimal overhead.

        Compared with :meth:`run`, this method:

        - uses an empty ``sim_signature`` and ``run_id`` to avoid ``uuid.uuid4()``
          and ``make_sim_signature`` overhead,
        - skips ``_estimate_total_nodes``, ``on_run_start``, and ``on_run_end``,
        - uses :meth:`_evaluate_light` which avoids calling ``node.to_signature()``
          when pre-computed values are supplied,
        - skips ``_assemble_result`` and all report/diagnostics collection.

        For nodes with **no** child :class:`CalculatorBase` dependencies
        (no nested calculators in ``Param`` fields, no filter/transform wrappers),
        dispatches to :meth:`_run_minimal` which avoids creating
        :class:`ExecutionContext`, :class:`ResultNode`, and all collector
        objects entirely.

        Intended for tight loops (e.g. per-bin evaluation) where the same node
        is applied to many sub-snapshots.  Use :meth:`CalculatorBase.batch` to
        obtain a caller that pre-computes the node signature once.
        """
        # Ultra-minimal path: no child CalculatorBase dependencies
        if not node.dependencies():
            return self._run_minimal(node, sim, options)

        # Full light path: nested-calculator Param deps, etc.
        ctx = ExecutionContext(sim=sim, sim_signature=(), run_id="", options=options, engine=self)
        work = NodeInput(sim_raw=sim, sim_current=sim)
        try:
            root = self._evaluate_light(
                node,
                ctx,
                work,
                node_sig=_precomputed_node_sig if _precomputed_node_sig is not None else node.signature(),
                structured_sig=_precomputed_structured_sig,
            )
        except Exception:
            if options.errors == ErrorPolicy.RAISE:
                raise
            return None  # type: ignore[return-value]
        store = ctx.runtime_store.get(root.node_id)
        if store is None:
            return None  # type: ignore[return-value]
        return store.public_value

    def _run_minimal(self, node: CalculatorBase[TRaw, TPublic], sim: Any, options: RunOptions) -> TPublic:
        """Absolute minimal execution path for nodes with no child CalculatorBase dependencies.

        Uses :class:`_MinimalBatchContext` instead of :class:`ExecutionContext` to avoid
        allocating :class:`RuntimeCache`, :class:`TraceCollector`, :class:`PerfCollector`,
        :class:`ResultNode`, ``log_events``, ``node_registry``, and ``runtime_store``
        on every invocation.

        Only called by :meth:`run_light` when ``node.dependencies()`` is empty.
        """
        ctx = _MinimalBatchContext(sim, options, self)
        work = NodeInput(sim_raw=sim, sim_current=sim)
        try:
            raw = node.execute(ctx, work)  # type: ignore[arg-type]
            raw = node.materialize(ctx, raw)  # type: ignore[arg-type]
            public = node.public_value(raw)
            return node.materialize_public(ctx, public)  # type: ignore[arg-type]
        except Exception:
            if options.errors == ErrorPolicy.RAISE:
                raise
            return None  # type: ignore[return-value]

    def _evaluate_light(
        self,
        node: CalculatorBase[TRaw, TPublic],
        ctx: ExecutionContext,
        work: NodeInput,
        *,
        node_sig: tuple[Any, ...],
        structured_sig: Any | None,
    ) -> ResultNode:
        """Like :meth:`evaluate` but uses pre-computed signatures.

        Skips ``node.to_signature()`` / ``node.signature()`` calls (saves ~2
        serialisation round-trips per invocation).  Also uses
        :meth:`_execute_node_body_light` which bypasses ``node_scope``
        (skips per-node progress events and log-event appends).

        Dependencies of *node* (e.g. scope filter/transform or child nodes) are
        still evaluated through the normal :meth:`evaluate` path so that
        filters and transforms work correctly.
        """
        cache_key = (ctx.sim_signature, work.cache_token, node_sig)
        stack_key = (id(node), work.cache_token)

        if stack_key in ctx._evaluation_stack:
            raise CycleError(f"Cycle detected while evaluating {node.log_label!r}")

        # Cache check (only when cache is enabled in options)
        if ctx.cache.enabled:
            cached_runtime = ctx.cache.get(cache_key)
            if cached_runtime is not None:
                return ctx.node_registry[cached_runtime.node_id]

        # Build a ResultNode using the pre-computed or lightly-computed signature
        sig_tuple = node_sig
        if structured_sig is not None:
            display_sig = structured_sig.cache_key()
        else:
            display_sig = sig_tuple

        node_result = ResultNode(
            node_id=ctx.new_node_id(),
            kind=node.kind,
            signature=display_sig,
            name=node.name,
            display_name=node.log_label,
            calculator_type=node.__class__.__name__,
            calculator_class_path=self._class_path(node),
            semantic_calculator_class_path=None,
            record_policy=node.record_policy or ctx.options.default_record_policy,
        )
        ctx.register_node(node_result)

        ctx._evaluation_stack.append(stack_key)
        try:
            state = self._execute_node_body_light(node, ctx, node_result, work)
        except _NodeExecutionFailure as failure:
            raise failure.cause from None
        finally:
            ctx._evaluation_stack.pop()

        ctx.register_runtime_value(node_result.node_id, state.raw_value, state.public_value)
        node_result.status = NodeStatus.OK
        return node_result

    def _execute_node_body_light(
        self, node: CalculatorBase[T, Any], ctx: ExecutionContext, node_result: ResultNode, work: NodeInput
    ) -> _NodeExecutionState:
        """Like :meth:`_execute_node_body` but without progress events or log entries.

        Pushes *node_result* onto the node stack so that ``ctx.phase()`` inside
        ``node.execute()`` still has a valid current node.  Skips
        ``ctx.node_scope`` (which emits ``on_node_start``/``on_node_end`` and
        two ``ctx.log`` calls) and ``ctx.observe_node_access`` (which is
        irrelevant when ``options.observe=False``).
        """
        state = _NodeExecutionState()
        ctx._node_stack.append(node_result)
        try:
            work = self._apply_node_scope(node, ctx, work)
            state.raw_value = node.execute(ctx, work)
            state.raw_value = node.materialize(ctx, state.raw_value)
            state.public_value = node.public_value(state.raw_value)
            state.public_value = node.materialize_public(ctx, state.public_value)
        except Exception as exc:
            raise _NodeExecutionFailure(exc, state) from exc
        finally:
            ctx._node_stack.pop()
        return state

    def _apply_node_scope(self, node: CalculatorBase[Any, Any], ctx: ExecutionContext, work: NodeInput) -> NodeInput:
        """Apply ``node.scope`` (transforms then filter) to ``work`` before execution.

        Scoped composition is expressed by cloning the concrete calculator with a
        non-empty :class:`~.runtime.scopes.ScopeSpec`; the engine applies that scope
        right before running ``node.execute`` so the same filtering/transform logic
        works for every calculator type without a wrapper node.
        """
        scope = getattr(node, "scope", None)
        if scope is None or scope.is_empty:
            return work
        transform = scope.as_transform()
        if transform is not None:
            from pynbodyext.core.calculate.runtime.input import TransformResult

            with ctx.phase(node, "transform"):
                transform_result = ctx.raw_value(transform, work)
                if not isinstance(transform_result, TransformResult):
                    raise TypeError("transform nodes must return TransformResult")
                work = work.with_transform(transform_result)

        if scope.filter is not None:
            from pynbodyext.core.calculate.runtime.input import FilterResult

            with ctx.phase(node, "filter"):
                filter_result = ctx.raw_value(scope.filter, work)
                if not isinstance(filter_result, FilterResult):
                    raise TypeError("filter nodes must return FilterResult")
                work = work.with_selection(filter_result)

        return work

    # ------------------------------------------------------------------

    def evaluate(
        self, node: CalculatorBase[TRaw, TPublic], ctx: ExecutionContext, input: NodeInput | None = None
    ) -> ResultNode:
        """Evaluate one node within an existing :class:`ExecutionContext`."""
        plan = self._make_evaluation_plan(node, ctx, input)
        if plan.stack_key in ctx._evaluation_stack:
            raise CycleError(f"Cycle detected while evaluating {node.log_label!r}")

        cached_node = self._try_cache_hit(node, ctx, plan)
        if cached_node is not None:
            return cached_node

        structured_signature = (
            plan.structured_signature if plan.structured_signature is not None else node.to_signature()
        )
        node_result = self._create_node_result(node, ctx, structured_signature)
        ctx.register_node(node_result)
        is_root = ctx.current_node is None
        if is_root:
            ctx.root_signature = structured_signature

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
            node=node, ctx=ctx, node_result=node_result, state=state, plan=plan, is_root=is_root
        )

    def _make_evaluation_plan(
        self, node: CalculatorBase[Any, Any], ctx: ExecutionContext, input: NodeInput | None
    ) -> _EvaluationPlan:
        work = input or NodeInput(sim_raw=ctx.sim, sim_current=ctx.sim)
        work = work.with_mutation_generation(ctx.mutation_generation)

        cache_policy = getattr(node, "cache_policy", CachePolicy.AUTO)
        cacheable = bool(getattr(node, "cacheable", True)) and cache_policy != CachePolicy.NONE
        compiled_signature = node.to_signature()
        node_signature = compiled_signature.cache_key()
        observed_cache_prefix = None
        if ctx.options.observe:
            observed_scope = work.observed_scope_cache_token
            stack_scope = observed_scope
            observed_cache_prefix = (ctx.sim_signature, observed_scope, node_signature)
            cache_key = observed_cache_prefix
        else:
            cache_token = work.cache_token
            stack_scope = cache_token  # type: ignore[assignment]
            cache_key = (ctx.sim_signature, cache_token, node_signature)  # type: ignore[assignment]

        return _EvaluationPlan(
            work=work,
            stack_key=(id(node), stack_scope),
            cache_key=cache_key,
            observed_cache_prefix=observed_cache_prefix,
            cacheable=cacheable,
            cache_policy=cache_policy,
            structured_signature=compiled_signature,
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

    def _semantic_calculator_class_path(self, payload: Any) -> str | None:
        current = payload
        while isinstance(current, dict):
            if current.get("node") == "bound":
                current = current.get("base")
                continue

            class_path = current.get("class")
            if isinstance(class_path, str):
                return class_path

            if current.get("node") == "calculator_value_property":
                current = current.get("calculator")
                continue

            return None
        return None

    def _try_cache_hit(
        self, node: CalculatorBase[Any, Any], ctx: ExecutionContext, plan: _EvaluationPlan
    ) -> ResultNode | None:
        cached_runtime = None
        cache_key_for_trace = plan.cache_key
        if plan.cacheable:
            if plan.observed_cache_prefix is not None:
                cached_runtime = ctx.cache.get_compatible(
                    plan.observed_cache_prefix, ctx.observed_cache_token_is_current
                )
                cache_key_for_trace = plan.observed_cache_prefix
            else:
                cached_runtime = ctx.cache.get(plan.cache_key)
        if cached_runtime is None:
            ctx.trace.cache(
                node_id="pending",
                node_name=node.log_label,
                event="miss" if plan.cacheable else "skip",
                key=repr(cache_key_for_trace),
            )
            return None

        cached_node = ctx.node_registry[cached_runtime.node_id]
        parent = ctx.current_node
        if parent is not None:
            if cached_node.node_id not in parent.children:
                parent.children.append(cached_node.node_id)
            if parent.node_id not in cached_node.parent_ids:
                cached_node.parent_ids.append(parent.node_id)

        node_name = cached_node.label or node.log_label
        ctx.trace.cache(node_id=cached_node.node_id, node_name=node_name, event="hit", key=repr(cache_key_for_trace))
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
        self, node: CalculatorBase[Any, Any], ctx: ExecutionContext, structured_signature: Any
    ) -> ResultNode:
        return ResultNode(
            node_id=ctx.new_node_id(),
            kind=node.kind,
            signature=structured_signature.cache_key(),
            name=node.name,
            display_name=node.log_label,
            calculator_type=node.__class__.__name__,
            calculator_class_path=self._class_path(node),
            semantic_calculator_class_path=self._semantic_calculator_class_path(structured_signature.payload),
            record_policy=node.record_policy or ctx.options.default_record_policy,
        )

    def _execute_node_body(
        self, node: CalculatorBase[T, Any], ctx: ExecutionContext, node_result: ResultNode, work: NodeInput
    ) -> _NodeExecutionState:
        state = _NodeExecutionState()
        with ctx.node_scope(node_result, node):
            with ctx.observe_node_access(node_result, node):
                try:
                    work = self._apply_node_scope(node, ctx, work)
                    state.raw_value = node.execute(ctx, work)
                    with observation_phase("materialize"):
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
            node_result, raw_value=raw_value, public_value=public_value, is_root=is_root, had_error=True
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

        stored_in_runtime_cache = False
        if plan.cacheable and self._should_store_runtime_cache(
            node, state.raw_value, state.public_value, plan.cache_policy, ctx.options
        ):
            cache_key = plan.cache_key
            if plan.observed_cache_prefix is not None:
                observed_fields = ctx.observed_cache_fields(node_result)
                observed_token = ctx.observed_cache_token(observed_fields)
                node_result.artifacts["observed_cache_fields"] = tuple(sorted(observed_fields))
                node_result.artifacts["observed_cache_token"] = observed_token
                cache_key = (*plan.observed_cache_prefix, observed_token)
            ctx.cache.set(
                cache_key,
                ctx.runtime_store[node_result.node_id],
                index_compatible=plan.observed_cache_prefix is not None,
            )
            stored_in_runtime_cache = True

        node_result.value_summary = self.summarize_value(state.public_value)
        node_result.status = NodeStatus.OK
        self._store_recorded_values(
            node_result=node_result,
            raw_value=state.raw_value,
            public_value=state.public_value,
            is_root=is_root,
            had_error=False,
            auto_record_public_value=stored_in_runtime_cache
            and self._should_auto_record_public_value(
                public_value=state.public_value, record_policy=node_result.record_policy, options=ctx.options
            ),
        )
        return node_result

    def _resolve_failed_root(self, ctx: ExecutionContext) -> ResultNode:
        if ctx.last_error_node_id is not None:
            return ctx.node_registry[ctx.last_error_node_id]
        if ctx.node_registry:
            return next(reversed(ctx.node_registry.values()))

        raise RuntimeError("run failed before any result node was registered")

    def _build_provenance(
        self, node: CalculatorBase[Any, Any], ctx: ExecutionContext, started: float, finished: float
    ) -> ProvenanceInfo:
        root_signature = getattr(ctx, "root_signature", None) or node.to_signature()
        return ProvenanceInfo(
            calculator_signature=root_signature.cache_key(),
            calculator_signature_text=root_signature.to_json(),
            calculator_signature_hash=root_signature.short_hash(),
            sim_signature=ctx.sim_signature,
            started_at=started,
            finished_at=finished,
        )

    def _collect_reports(self, ctx: ExecutionContext, root: ResultNode, run_label: str) -> dict[str, str]:
        return {
            "perf": ctx.perf.report_text(ctx.node_registry, title=run_label),
            "cache": ctx.cache.report_text(),
            "observer": render_observer_report(ctx.access_observations.values()),
            "trace_timeline": ctx.trace.render_timeline(),
            "trace_tree": ctx.trace.render_tree(ctx.node_registry, root.node_id),
        }

    def _collect_diagnostics(self, ctx: ExecutionContext, named: dict[str, ResultNode]) -> dict[str, Any]:
        named_values = {
            name: ctx.runtime_store[node_result.node_id].public_value
            for name, node_result in named.items()
            if node_result.node_id in ctx.runtime_store
        }
        return {
            "trace_events": list(ctx.trace.events),
            "cache_events": list(ctx.cache.events),
            "observations": {
                node_id: observation.as_dict() for node_id, observation in ctx.access_observations.items()
            },
            "observer_events": [
                event.as_dict() for observation in ctx.access_observations.values() for event in observation.events
            ],
            "log_events": list(ctx.log_events),
            "named_values": named_values,
        }

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

        provenance = self._build_provenance(node, ctx, started, finished)

        named = {
            name: ctx.node_registry[node_id]
            for name, node_id in ctx.named_registry.items()
            if node_id in ctx.node_registry
        }

        reports = self._collect_reports(ctx, root, run_label)
        diagnostics = self._collect_diagnostics(ctx, named)

        perf_summary = ctx.perf.summary(ctx.node_registry, cache_hit_count=ctx.cache.hit_count)
        cache_summary = ctx.cache.summary()
        perf_summary.cache_miss_count = int(cache_summary["misses"])
        perf_summary.cache_store_count = int(cache_summary["stores"])

        root_value = cast(
            "TPublic", ctx.runtime_store[root.node_id].public_value if root.node_id in ctx.runtime_store else None
        )

        result = Result(
            value=root_value,
            root=root,
            nodes=dict(ctx.node_registry),
            named=named,
            calculator=node,
            observations=dict(ctx.access_observations),
            provenance=provenance,
            perf_summary=perf_summary,
            warnings=list(ctx.warnings),
            errors=list(ctx.errors),
            reports=reports,
            diagnostics=diagnostics,
        )
        execution_tree_report = result.report_execution_tree()

        root.artifacts.update(reports)
        root.artifacts["execution_tree_report"] = execution_tree_report
        root.artifacts["log_events"] = list(ctx.log_events)
        result.reports["execution_tree"] = execution_tree_report

        return result

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
            if raw_value.mask is not public_value:
                size += self._estimate_cache_bytes(raw_value.mask)
        elif raw_value is not public_value:
            size += self._estimate_cache_bytes(raw_value)
        return size <= options.cache_small_value_bytes

    def _should_auto_record_public_value(
        self, *, public_value: Any, record_policy: RecordPolicy | None, options: RunOptions
    ) -> bool:
        if not options.auto_record_cached_values:
            return False
        if record_policy != RecordPolicy.SUMMARY:
            return False
        limit = options.auto_record_small_value_bytes
        if limit is None:
            return False
        return self._estimate_cache_bytes(public_value) <= limit

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
                self._estimate_cache_bytes(key) + self._estimate_cache_bytes(item) for key, item in value.items()
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
        auto_record_public_value: bool = False,
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
                had_error and policy == RecordPolicy.ERROR_ONLY and raw_value is not None
            ):
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

        elif policy == RecordPolicy.SUMMARY:
            if auto_record_public_value and public_value is not None:
                node_result.value = public_value
                node_result.stored_value = True

        elif policy == RecordPolicy.NONE:
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
            python_type=value.__class__.__name__, shape=shape, dtype=dtype, units=units, preview=preview
        )

    def make_sim_signature(self, sim: Any) -> tuple[Any, ...]:
        """Return the simulation identity fragment used in cache keys and provenance."""
        return self._sim_identity(sim)
