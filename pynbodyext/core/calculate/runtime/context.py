"""Runtime context, run options, and scoped node input objects.

This module defines the per-run execution state used by the calculator engine.
Most users primarily touch :class:`RunOptions`; framework authors and advanced
users may also interact with :class:`ExecutionContext` and :class:`NodeInput`.

Core Types
----------
The main exported runtime types are:

- :class:`RunOptions` for run-time configuration
- :class:`ExecutionContext` for per-run orchestration state
- :class:`NodeInput` for active simulation view and scope at one node
- :class:`FilterResult` as raw filter output
- :class:`TransformResult` as raw transform output

RunOptions
----------
Use :class:`RunOptions` (or equivalent keyword arguments on ``run``) to control
execution behavior such as:

- progress verbosity
- field-access observation
- error handling policy
- value recording policy
- timing and memory collection
- runtime cache behavior

Example::

    result = calc.run(sim, progress="phase", perf_time=True, perf_memory=True)

ExecutionContext
----------------
:class:`ExecutionContext` is engine-owned runtime state for one run. It
coordinates:

- node evaluation
- phase recording
- cache access
- trace and perf collectors
- named node registration
- diagnostic logs

Custom subclasses usually interact with it only inside runtime-level hooks.

NodeInput And Scope
-------------------
:class:`NodeInput` carries the currently active simulation object for one node
evaluation, including filtered/transformed scope state.

This is the key object that makes scoped execution deterministic across the
graph.

FilterResult And TransformResult
--------------------------------
Filters and transforms keep richer raw values than their public values:

- :class:`FilterResult` holds mask + filtered simulation view
- :class:`TransformResult` holds handle + post-transform view + revert metadata

This is why filters/transforms should usually use their specialized base
classes rather than plain :class:`CalculatorBase`.

Notes
-----
For most end-user analysis code, :class:`RunOptions` is the only type from this
module that needs direct use. The rest form the public runtime model for
extension and debugging.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pynbodyext.core.calculate.diagnostics.perf import PerfCollector
from pynbodyext.core.calculate.diagnostics.trace import TraceCollector
from pynbodyext.core.calculate.runtime.cache import ExecutionValue, RuntimeCache
from pynbodyext.log import logger

from .progress import NodeProgressEvent, PhaseProgressEvent, ProgressSink, resolve_progress_sink

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.result.result import ErrorInfo, ResultNode

    from .engine import EvalEngine
    from .input import NodeInput
    from .options import RunOptions


TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")


@dataclass(slots=True)
class LogEvent:
    """Runtime log entry captured in :attr:`Result.diagnostics`."""

    timestamp: float
    level: str
    node_id: str | None
    phase: str | None
    message: str


@dataclass(slots=True)
class _NodeBookkeeping:
    """Per-run node registry, value store, and naming state."""

    registry: dict[str, ResultNode] = field(default_factory=dict)
    runtime: dict[str, ExecutionValue] = field(default_factory=dict)
    named: dict[str, str] = field(default_factory=dict)
    counter: int = 0
    last_error_id: str | None = None
    #: Structured signature of the root calculator, computed once during
    #: evaluation and reused for provenance assembly (avoids a redundant
    #: re-serialization of the root node).
    root_signature: Any | None = None


@dataclass(slots=True)
class _MutationState:
    """State-mutation generations used to invalidate caches safely."""

    generation: int = 0
    unknown_generation: int = 0
    field_generations: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class _RunRecords:
    """Warnings, errors, logs, and field-access observations for one run."""

    warnings: list[str] = field(default_factory=list)
    errors: list[ErrorInfo] = field(default_factory=list)
    log_events: list[LogEvent] = field(default_factory=list)
    access_observations: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionContext:
    """Mutable state shared by all nodes in one calculator run."""

    sim: Any
    sim_signature: tuple[Any, ...]
    run_id: str
    options: RunOptions
    engine: EvalEngine

    cache: RuntimeCache = field(default_factory=RuntimeCache)
    trace: TraceCollector = field(default_factory=TraceCollector)
    perf: PerfCollector = field(default_factory=PerfCollector)

    nodes: _NodeBookkeeping = field(default_factory=_NodeBookkeeping)
    mutation: _MutationState = field(default_factory=_MutationState)
    records: _RunRecords = field(default_factory=_RunRecords)

    _node_stack: list[ResultNode] = field(default_factory=list)
    _evaluation_stack: list[tuple[int, tuple[Any, ...]]] = field(default_factory=list)
    _progress_sink: ProgressSink = field(init=False)

    def __post_init__(self) -> None:
        self.cache.enabled = self.options.cache
        self._progress_sink = resolve_progress_sink(self.options.progress)

    def new_node_id(self) -> str:
        """Allocate a new node id for this run."""
        self.nodes.counter += 1
        return f"{self.run_id}:{self.nodes.counter}"

    @property
    def current_node(self) -> ResultNode | None:
        """Currently active result node, if any."""
        return self._node_stack[-1] if self._node_stack else None

    def evaluate(self, node: CalculatorBase[Any, Any], input: NodeInput | None = None) -> ResultNode:
        """Evaluate a dependency node in this context."""
        return self.engine.evaluate(node, self, input)

    def public_value(self, node: CalculatorBase[TRaw, TPublic], input: NodeInput | None = None) -> TPublic:
        node_result = self.evaluate(node, input)
        return self.nodes.runtime[node_result.node_id].public_value

    def raw_value(self, node: CalculatorBase[TRaw, TPublic], input: NodeInput | None = None) -> TRaw:
        """Evaluate a dependency and return its raw value."""
        node_result = self.evaluate(node, input)
        return self.nodes.runtime[node_result.node_id].raw_value

    def register_node(self, node_result: ResultNode) -> None:
        """Register a newly created result node."""
        self.nodes.registry[node_result.node_id] = node_result

        if node_result.name:
            existing = self.nodes.named.get(node_result.name)
            if existing is not None and existing != node_result.node_id:
                existing_node = self.nodes.registry.get(existing)
                if existing_node is not None and existing_node.signature == node_result.signature:
                    node_result.run.artifacts["duplicate_named_node"] = existing
                    self.log("debug", f"duplicate named calculator {node_result.name!r}; keeping first registration")
                    return

                raise ValueError(f"Duplicate named calculator node {node_result.name!r}.")

            self.nodes.named[node_result.name] = node_result.node_id

    def register_runtime_value(self, node_id: str, raw_value: Any, public_value: Any) -> None:
        """Store raw and public runtime values for a node."""
        self.nodes.runtime[node_id] = ExecutionValue(node_id=node_id, raw_value=raw_value, public_value=public_value)

    def advance_mutation_generation(self, reason: str, *, observed_phase: str | None | object = Ellipsis) -> int:
        """Advance the mutation generation after a transform changes state."""
        self.mutation.generation += 1
        fields = self._current_observed_mutation_fields(observed_phase) if self.options.observe else set()
        if fields:
            for field_name in fields:
                self.mutation.field_generations[field_name] = self.mutation.generation
            rendered_fields = ", ".join(sorted(fields))
            self.log("debug", f"mutation generation {self.mutation.generation}: {reason}; fields={rendered_fields}")
        else:
            self.mutation.unknown_generation = self.mutation.generation
            self.log("debug", f"mutation generation {self.mutation.generation}: {reason}; fields=*unknown*")
        return self.mutation.generation

    def _current_observed_mutation_fields(self, phase: str | None | object = Ellipsis) -> set[str]:
        try:
            from pynbodyext.core.calculate.diagnostics.observer import current_observation, current_observation_phase
        except Exception:
            return set()

        observation = current_observation()
        if observation is None:
            return set()
        if phase is Ellipsis:
            phase = current_observation_phase()
        return set(observation.fields_for("dirty", phase=phase)) | set(observation.fields_for("delete", phase=phase))

    def observed_cache_fields(self, node_result: ResultNode) -> frozenset[str]:
        """Return direct and child observed fields that affect this node value."""
        fields: set[str] = set()
        if node_result.run.observation is not None:
            fields.update(node_result.run.observation.reads)
        for child_id in node_result.children:
            child = self.nodes.registry.get(child_id)
            if child is None:
                continue
            fields.update(child.run.artifacts.get("observed_cache_fields", ()))
        return frozenset(fields)

    def observed_cache_token(self, fields: frozenset[str]) -> tuple[int, tuple[tuple[str, int], ...]]:
        """Return the current generation token for observed cache fields."""
        return (
            self.mutation.unknown_generation,
            tuple((field_name, self.mutation.field_generations.get(field_name, 0)) for field_name in sorted(fields)),
        )

    def observed_cache_token_is_current(self, token: Any) -> bool:
        """Whether an observed cache token is still valid for current generations."""
        try:
            unknown_generation, field_items = token
        except Exception:
            return False
        if unknown_generation != self.mutation.unknown_generation:
            return False
        try:
            return all(
                self.mutation.field_generations.get(field_name, 0) == generation for field_name, generation in field_items
            )
        except Exception:
            return False

    def log(self, level: str, message: str, *, node_id: str | None = None, phase: str | None = None) -> None:
        """Record and emit a runtime log message."""
        self.records.log_events.append(
            LogEvent(timestamp=time.perf_counter(), level=level, node_id=node_id, phase=phase, message=message)
        )
        log_fn = getattr(logger, level, logger.debug)
        log_fn(message)

    @contextmanager
    def observe_node_access(self, node_result: ResultNode, node: CalculatorBase[Any, Any]) -> Iterator[Any]:
        """Observe pynbody snapshot field access for one executing node."""
        if not self.options.observe:
            yield None
            return

        from pynbodyext.core.calculate.diagnostics.observer import observe_sim_access

        with observe_sim_access(node_result.node_id, node.log_label) as observation:
            try:
                yield observation
            finally:
                node_result.run.observation = observation
                node_result.run.artifacts["observer"] = observation.as_dict()
                self.records.access_observations[node_result.node_id] = observation

    @contextmanager
    def node_scope(self, node_result: ResultNode, node: CalculatorBase[Any, Any]) -> Iterator[None]:
        """Context manager for entering an evaluated node."""
        parent = self.current_node
        if parent is not None:
            if node_result.node_id not in parent.children:
                parent.children.append(node_result.node_id)
            if parent.node_id not in node_result.parent_ids:
                node_result.parent_ids.append(parent.node_id)

        node_name = node.log_label
        depth = len(self._node_stack)
        started_at = time.perf_counter()

        self._progress_sink.on_node_start(
            NodeProgressEvent(
                run_id=self.run_id,
                node_id=node_result.node_id,
                node_name=node_name,
                kind=str(node_result.kind),
                depth=depth,
                started_at=started_at,
            )
        )
        self.log("debug", f"node start: {node_name}", node_id=node_result.node_id)

        status = "ok"
        self._node_stack.append(node_result)
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self._node_stack.pop()
            finished_at = time.perf_counter()
            access_summary = None
            try:
                from pynbodyext.core.calculate.diagnostics.observer import format_observation_access

                access_summary = format_observation_access(node_result.run.observation, include_reads=False)
            except Exception:
                access_summary = None
            self._progress_sink.on_node_end(
                NodeProgressEvent(
                    run_id=self.run_id,
                    node_id=node_result.node_id,
                    node_name=node_name,
                    kind=str(node_result.kind),
                    depth=depth,
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_s=finished_at - started_at,
                    access_summary=access_summary,
                )
            )
            self.log("debug", f"node end: {node_name} status={status}", node_id=node_result.node_id)

    @contextmanager
    def phase(self, node: CalculatorBase[Any, Any], phase_name: str) -> Iterator[None]:
        """Context manager for tracing and timing a node phase."""
        current = self.current_node
        if current is None:
            raise RuntimeError("phase() requires an active node scope")

        node_name = node.log_label
        depth = len(self._node_stack) - 1

        self._progress_sink.on_phase_start(
            PhaseProgressEvent(
                run_id=self.run_id,
                node_id=current.node_id,
                node_name=node_name,
                phase=phase_name,
                depth=depth,
                started_at=time.perf_counter(),
            )
        )
        self.log("debug", f"{node_name}:{phase_name} start", node_id=current.node_id, phase=phase_name)

        status = "ok"
        record: Any = None

        from pynbodyext.core.calculate.diagnostics.observer import observation_phase

        with self.trace.phase(current.node_id, node_name, phase_name):
            try:
                with observation_phase(phase_name):
                    with self.perf.phase(
                        phase_name, measure_time=self.options.perf_time, measure_memory=self.options.perf_memory
                    ) as phase_record:
                        record = phase_record
                        yield
            except Exception as exc:
                status = "error"
                self.log(
                    "error",
                    f"{node_name}:{phase_name} error: {exc.__class__.__name__}: {exc}",
                    node_id=current.node_id,
                    phase=phase_name,
                )
                raise
            finally:
                if record is not None:
                    current.run.phases.append(record)
                    access_summary = None
                    try:
                        from pynbodyext.core.calculate.diagnostics.observer import (
                            current_observation,
                            format_observation_access,
                        )

                        access_summary = format_observation_access(
                            current_observation(), phase=phase_name, include_reads=True
                        )
                    except Exception:
                        access_summary = None
                    self._progress_sink.on_phase_end(
                        PhaseProgressEvent(
                            run_id=self.run_id,
                            node_id=current.node_id,
                            node_name=node_name,
                            phase=phase_name,
                            depth=depth,
                            status=status,
                            started_at=record.started_at,
                            finished_at=record.finished_at,
                            elapsed_s=record.elapsed_s,
                            access_summary=access_summary,
                        )
                    )
                    self.log("debug", f"{node_name}:{phase_name} {status}", node_id=current.node_id, phase=phase_name)


def resolve_value(
    ctx: ExecutionContext,
    input: NodeInput,
    value: Any,
    *,
    field_name: str | None = None,
    target_units: Any | None = None,
) -> Any:
    """Resolve constants, callables, and calculator-valued parameters."""
    from pynbodyext.core.calculate.params import resolve_dynamic_value

    return resolve_dynamic_value(ctx, input, value, field_name=field_name, target_units=target_units)
