"""Runtime context, run options, and scoped input objects.

This module defines the objects that flow through the calculator engine while a
graph is evaluated.  Most users interact directly with :class:`RunOptions`.
Framework authors and advanced users may also encounter
:class:`ExecutionContext`, :class:`NodeInput`, :class:`FilterResult`, and
:class:`TransformResult`.

What This Module Provides
-------------------------
The main concepts exported here are:

- :class:`RunOptions`: per-run configuration such as progress, caching, error
  handling, and performance collection
- :class:`ExecutionContext`: the runtime object that owns cache, trace, perf,
  logs, and node registries during one run
- :class:`NodeInput`: the active simulation view and scope passed into one node
- :class:`FilterResult`: raw output of a filter node
- :class:`TransformResult`: raw output of a transform node

Basic RunOptions Example
------------------------
Most users only need :class:`RunOptions`::

    from pynbodyext.core.calculate import RunOptions

    options = RunOptions(
        progress="phase",
        perf_time=True,
        perf_memory=True,
    )

    result = calc.run(sim, options=options)
    print(result.perf_summary)

Using Keyword Overrides
-----------------------
The same options can usually be passed directly to :meth:`run` or
:meth:`__call__`::

    result = calc.run(sim, progress="phase", perf_memory=True)

This is convenient for one-off runs, while :class:`RunOptions` is better for
reusable presets.

What ExecutionContext Does
--------------------------
:class:`ExecutionContext` is the engine-owned runtime object passed to custom
calculator subclasses.  It provides methods for evaluating dependencies,
recording phases, accessing public or raw child values, and emitting logs.

Custom subclasses usually touch it only inside runtime hooks such as
:meth:`CalculatorBase.execute`,
:meth:`PropertyBase._calculate_runtime`,
:meth:`FilterBase._build_mask_runtime`, or
:meth:`TransformBase._build_handle_runtime`.

FilterResult And TransformResult
--------------------------------
Filter and transform nodes return richer raw values than their public value.

A :class:`FilterResult` stores both:

- the computed mask
- the filtered simulation view used by downstream scoped calculators

A :class:`TransformResult` stores both:

- the transform handle
- the active simulation view after mutation
- revertibility metadata and artifacts

This is why filters and transforms should usually use their own specialized
base classes rather than plain :class:`CalculatorBase`.

NodeInput
---------
:class:`NodeInput` represents the currently active simulation state for one node
evaluation.  It tracks the raw simulation object, the current filtered or
transformed view, and any active transform/filter scope.  Most users do not
construct it directly.

When This Module Matters
------------------------
This module is most important when you are:

- configuring runs with :class:`RunOptions`
- writing new calculator subclasses
- debugging how filter and transform scope is propagated
- inspecting advanced runtime behavior

Notes
-----
For everyday analysis code, :class:`RunOptions` is usually the only public type
from this module that needs direct use.  The other classes are part of the
public execution model, but are primarily intended for extension and debugging.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, cast

import numpy as np

from pynbodyext.log import logger

from .cache import ExecutionValue, RuntimeCache
from .display import display_value, format_time, html_card, mimebundle
from .enums import ErrorPolicy, RecordPolicy, normalize_error_policy
from .perf import PerfCollector
from .trace import TraceCollector

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterator

    from .base import CalculatorBase
    from .engine import EvalEngine
    from .result import ErrorInfo, ResultNode, ValueSummary


HandleT = TypeVar("HandleT")
ProgressVerbosity = Literal["run", "node", "phase", "debug"]

TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")

def _node_ref(node_id: str) -> str:
    return f"n{node_id.rsplit(':', 1)[-1]}"

@dataclass(slots=True)
class RunProgressEvent:
    """Progress event emitted at the start and end of a calculator run."""

    run_id: str
    root_name: str
    status: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    total_nodes: int | None = None
    node_count: int | None = None
    warning_count: int | None = None
    error_count: int | None = None

@dataclass(slots=True)
class NodeProgressEvent:
    """Progress event emitted for one calculator node."""

    run_id: str
    node_id: str
    node_name: str
    kind: str
    depth: int
    status: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    from_cache: bool = False

@dataclass(slots=True)
class PhaseProgressEvent:
    """Progress event emitted for one named node phase."""

    run_id: str
    node_id: str
    node_name: str
    phase: str
    depth: int
    status: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None

def _format_count(value: int | None) -> str:
    return str(value) if value is not None else "-"

def _status_text(value: str | None) -> str:
    return value or "-"

progress_logger = logger.getChild("progress")

class ProgressSink(Protocol):
    """Protocol for receiving run, node, and phase progress events."""

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Handle the start of a calculator run."""
        ...

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Handle the end of a calculator run."""
        ...

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Handle the start of a node evaluation."""
        ...

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Handle the end of a node evaluation."""
        ...

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Handle the start of a node phase."""
        ...

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Handle the end of a node phase."""
        ...

@dataclass(slots=True)
class LoggerProgressSink:
    """Progress sink that writes compact events to the package logger."""

    verbosity: ProgressVerbosity = "node"
    indent: str = "  "
    show_ids: bool = False
    log: logging.Logger = field(default_factory=lambda: progress_logger)

    def _prefix(self, depth: int) -> str:
        return self.indent * max(depth, 0)

    def _shows_node_summary(self) -> bool:
        return self.verbosity in {"node", "phase", "debug"}

    def _shows_node_start(self) -> bool:
        return self.verbosity in {"node", "phase", "debug"}

    def _shows_phase(self) -> bool:
        return self.verbosity in {"phase", "debug"}

    def _shows_phase_start(self) -> bool:
        return self.verbosity == "debug"

    def _node_prefix(self, depth: int) -> str:
        return ("│  " * max(depth, 0)) + "├─ "

    def _phase_prefix(self, depth: int) -> str:
        return ("│  " * max(depth + 1, 0)) + "├─ "

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Log a run start event."""
        id_suffix = f" [{event.run_id}]" if self.show_ids or self.verbosity == "debug" else ""
        self.log.info("run start %s%s", event.root_name, id_suffix)

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Log a run end event."""
        self.log.info(
            "run end %s status=%s total=%s nodes=%s warnings=%s errors=%s",
            event.root_name,
            _status_text(event.status),
            format_time(event.elapsed_s),
            _format_count(event.node_count),
            _format_count(event.warning_count),
            _format_count(event.error_count),
        )

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Log a node start event when verbosity includes debug starts."""
        if not self._shows_node_start():
            return
        self.log.info(
            "%s[%s] %s <%s> start",
            self._node_prefix(event.depth),
            _node_ref(event.node_id),
            event.node_name,
            event.kind,
        )

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Log a node end event when verbosity includes node summaries."""
        if not self._shows_node_summary():
            return

        cache_suffix = " cache-hit" if event.from_cache else ""
        self.log.info(
            "%s[%s] %s <%s> %s %s%s",
            self._node_prefix(event.depth),
            _node_ref(event.node_id),
            event.node_name,
            event.kind,
            _status_text(event.status),
            format_time(event.elapsed_s),
            cache_suffix,
        )

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Log a phase start event when verbosity includes debug starts."""
        if not self._shows_phase_start():
            return
        self.log.info(
            "%s[%s] phase %s start",
            self._phase_prefix(event.depth),
            _node_ref(event.node_id),
            event.phase,
        )

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Log a phase summary event."""
        if not self._shows_phase():
            return
        self.log.info(
            "%s[%s] phase %s %s %s",
            self._phase_prefix(event.depth),
            _node_ref(event.node_id),
            event.phase,
            _status_text(event.status),
            format_time(event.elapsed_s),
        )

@dataclass(slots=True)
class TqdmProgressSink:
    """Optional tqdm-backed progress bar sink.

    Falls back to LoggerProgressSink when tqdm is unavailable.
    """

    verbosity: ProgressVerbosity = "node"
    leave: bool = True
    mininterval: float = 0.1
    show_current: bool = True

    _bar: Any = field(init=False, default=None)
    _fallback: LoggerProgressSink | None = field(init=False, default=None)
    _warned_fallback: bool = field(init=False, default=False)

    def _ensure_backend(self, event: RunProgressEvent) -> None:
        if self._bar is not None or self._fallback is not None:
            return

        try:
            from tqdm.auto import tqdm
        except Exception:
            self._fallback = LoggerProgressSink(verbosity=self.verbosity)
            if not self._warned_fallback:
                progress_logger.warning("tqdm unavailable, falling back to LoggerProgressSink")
                self._warned_fallback = True
            return

        self._bar = tqdm(
            total=event.total_nodes if event.total_nodes and event.total_nodes > 0 else None,
            desc=event.root_name,
            leave=self.leave,
            dynamic_ncols=True,
            mininterval=self.mininterval,
        )

    def _set_postfix(self, text: str) -> None:
        if self._bar is not None and self.show_current:
            self._bar.set_postfix_str(text[:96], refresh=False)

    def on_run_start(self, event: RunProgressEvent) -> None:
        self._ensure_backend(event)
        if self._fallback is not None:
            self._fallback.on_run_start(event)
            return

        if self._bar is not None and self._bar.total is None and event.total_nodes:
            self._bar.total = event.total_nodes
            self._bar.refresh()

    def on_run_end(self, event: RunProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_run_end(event)
            return

        if self._bar is None:
            return

        if event.node_count is not None and self._bar.n < event.node_count:
            self._bar.n = event.node_count
            self._bar.refresh()

        status = _status_text(event.status)
        total = format_time(event.elapsed_s)
        self._set_postfix(f"status={status} total={total}")

        self._bar.close()
        self._bar = None

    def on_node_start(self, event: NodeProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_node_start(event)
            return
        if self.verbosity == "debug":
            self._set_postfix(f"{event.node_name} start")

    def on_node_end(self, event: NodeProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_node_end(event)
            return

        if self._bar is None:
            return

        if not event.from_cache:
            self._bar.update(1)

        if self.verbosity in {"node", "phase", "debug"}:
            cache_suffix = " cache-hit" if event.from_cache else ""
            self._set_postfix(
                f"{event.node_name} {_status_text(event.status)} {format_time(event.elapsed_s)}{cache_suffix}"
            )

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_phase_start(event)
            return
        if self.verbosity == "debug":
            self._set_postfix(f"{event.node_name}:{event.phase} start")

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_phase_end(event)
            return
        if self.verbosity in {"phase", "debug"}:
            self._set_postfix(
                f"{event.node_name}:{event.phase} {_status_text(event.status)} {format_time(event.elapsed_s)}"
            )

BasicProgressSink = LoggerProgressSink


@dataclass(slots=True)
class CompositeProgressSink:
    """Progress sink that forwards events to multiple sinks."""

    sinks: tuple[ProgressSink, ...]

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Forward a run start event to all sinks."""
        for sink in self.sinks:
            sink.on_run_start(event)

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Forward a run end event to all sinks."""
        for sink in self.sinks:
            sink.on_run_end(event)

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Forward a node start event to all sinks."""
        for sink in self.sinks:
            sink.on_node_start(event)

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Forward a node end event to all sinks."""
        for sink in self.sinks:
            sink.on_node_end(event)

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Forward a phase start event to all sinks."""
        for sink in self.sinks:
            sink.on_phase_start(event)

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Forward a phase end event to all sinks."""
        for sink in self.sinks:
            sink.on_phase_end(event)

@dataclass(slots=True)
class NullProgressSink:
    """Progress sink that ignores all events."""

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Ignore a run start event."""
        return None

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Ignore a run end event."""
        return None

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Ignore a node start event."""
        return None

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Ignore a node end event."""
        return None

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Ignore a phase start event."""
        return None

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Ignore a phase end event."""
        return None



@dataclass(slots=True)
class LogEvent:
    """Runtime log entry captured in :attr:`Result.diagnostics`."""

    timestamp: float
    level: str
    node_id: str | None
    phase: str | None
    message: str


@dataclass(slots=True)
class RunOptions:
    """Execution options for a calculator run.

    Parameters
    ----------
    cache : bool, default: True
        Enable the per-run runtime cache.
    progress : bool, str, ProgressSink, or sequence, optional
        Progress reporting configuration. Strings may be "run", "node",
        "phase", "debug", "bar", or "bar:<verbosity>", or "bar-only".
    perf_time, perf_memory : bool
        Enable time and memory collection.
    backend : str, default: "serial"
        Backend label reserved for future execution backends.
    default_record_policy : RecordPolicy, default: RecordPolicy.SUMMARY
        Recording policy for nodes without an explicit policy.
    errors : ErrorPolicy or str, default: ErrorPolicy.RAISE
        Error handling policy.
    cache_small_value_bytes : int, default: 1000000
        Maximum public-value size for automatic cache storage.
    """

    cache: bool = True
    progress: bool | str | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None
    perf_time: bool = True
    perf_memory: bool = False
    backend: str = "serial"
    default_record_policy: RecordPolicy = RecordPolicy.SUMMARY
    errors: ErrorPolicy | str = ErrorPolicy.RAISE
    cache_small_value_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        self.errors = normalize_error_policy(self.errors)

    def __repr__(self) -> str:
        return (
            "RunOptions("
            f"cache={self.cache!r}, progress={self.progress!r}, "
            f"perf_time={self.perf_time!r}, perf_memory={self.perf_memory!r}, "
            f"backend={self.backend!r}, errors={display_value(self.errors)!r}"
            ")"
        )

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("RunOptions(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        return html_card(
            "RunOptions",
            [
                ("cache", self.cache),
                ("progress", self.progress),
                ("perf time", self.perf_time),
                ("perf memory", self.perf_memory),
                ("backend", self.backend),
                ("record policy", display_value(self.default_record_policy)),
                ("errors", display_value(self.errors)),
                ("small cache bytes", self.cache_small_value_bytes),
            ],
        )

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())


def resolve_progress_sink(
    progress: bool | str | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None,
) -> ProgressSink:
    """Normalize a progress option into a ProgressSink."""
    sink: ProgressSink

    if progress is None or progress is False:
        sink = NullProgressSink()
    elif progress is True:
        sink = LoggerProgressSink(verbosity="node")
    elif isinstance(progress, str):
        progress_name = progress.lower()
        if progress_name == "bar":
            sink = CompositeProgressSink(
                (
                    LoggerProgressSink(verbosity="node"),
                    TqdmProgressSink(verbosity="node"),
                )
            )
        elif progress_name.startswith("bar:"):
            verbosity = cast("ProgressVerbosity", progress_name.split(":", 1)[1])
            sink = CompositeProgressSink(
                (
                    LoggerProgressSink(verbosity=verbosity),
                    TqdmProgressSink(verbosity=verbosity),
                )
            )
        elif progress_name == "bar-only":
            sink = TqdmProgressSink(verbosity="node")
        else:
            sink = LoggerProgressSink(verbosity=cast("ProgressVerbosity", progress_name))
    elif isinstance(progress, list):
        sink = CompositeProgressSink(tuple(progress))
    elif isinstance(progress, tuple):
        sink = CompositeProgressSink(progress)
    else:
        sink = progress

    return sink

@dataclass(slots=True)
class FilterResult:
    """Raw result of a filter calculation.

    The filtered simulation view is derived lazily from source_sim and mask
    so runtime cache entries do not keep a strong reference to a subsnap.
    """

    mask: Any
    source_sim: Any
    mask_summary: ValueSummary | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def filtered_sim(self) -> Any:
        """Build the filtered simulation view on demand."""
        mask = self.mask
        shape = getattr(mask, "shape", None)
        dtype = getattr(mask, "dtype", None)

        if (
            shape is not None
            and dtype is not None
            and len(shape) == 1
            and shape[0] == len(self.source_sim)
            and np.dtype(dtype) != np.dtype(np.bool_)
        ):
            try:
                mask = mask.astype(np.bool_, copy=False)
            except TypeError:
                mask = mask.astype(np.bool_)

        return self.source_sim[mask]

    @property
    def cache_token(self) -> tuple[int, int]:
        """Stable token for this selection within one run."""
        return (id(self.source_sim), id(self.mask))

@dataclass(slots=True)
class TransformResult(Generic[HandleT]):
    """Raw result of a transform calculation."""

    handle: HandleT
    target: Any
    sim_after: Any
    revertible: bool = True
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeInput:
    """Active simulation view and scope state for node evaluation."""

    sim_raw: Any
    sim_current: Any
    selection: FilterResult | None = None
    transform: TransformResult[Any] | None = None
    mutation_generation: int = 0
    upstream: dict[str, ResultNode] = field(default_factory=dict)

    @property
    def active_sim(self) -> Any:
        """Simulation object visible to the current node."""
        if self.selection is not None:
            return self.selection.filtered_sim
        return self.sim_current

    @property
    def cache_token(self) -> tuple[int, int, tuple[int, int] | None, int | None, int]:
        """Input-state token used as part of runtime cache keys."""
        transform_id = id(self.transform.handle) if self.transform is not None else None
        selection_token = self.selection.cache_token if self.selection is not None else None
        return (id(self.sim_raw), id(self.sim_current), selection_token, transform_id, self.mutation_generation)


    def with_transform(self, result: TransformResult[Any]) -> NodeInput:
        """Return a copy after applying a transform result."""
        generation = result.artifacts.get("mutation_generation", self.mutation_generation)
        return replace(
            self,
            transform=result,
            sim_current=result.sim_after,
            mutation_generation=generation,
        )

    def with_selection(self, result: FilterResult) -> NodeInput:
        """Return a copy with an active filter selection."""
        return replace(self, selection=result)

    def with_mutation_generation(self, generation: int) -> NodeInput:
        """Return a copy with an updated mutation generation."""
        if self.mutation_generation == generation:
            return self
        return replace(self, mutation_generation=generation)


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

    node_registry: dict[str, ResultNode] = field(default_factory=dict)
    runtime_store: dict[str, ExecutionValue] = field(default_factory=dict)
    named_registry: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[ErrorInfo] = field(default_factory=list)
    log_events: list[LogEvent] = field(default_factory=list)

    _node_counter: int = 0
    mutation_generation: int = 0
    last_error_node_id: str | None = None
    _node_stack: list[ResultNode] = field(default_factory=list)
    _evaluation_stack: list[tuple[int, tuple[Any, ...]]] = field(default_factory=list)
    _progress_sink: ProgressSink = field(init=False)

    def __post_init__(self) -> None:
        self.cache.enabled = self.options.cache
        self._progress_sink = resolve_progress_sink(self.options.progress)

    def new_node_id(self) -> str:
        """Allocate a new node id for this run."""
        self._node_counter += 1
        return f"{self.run_id}:{self._node_counter}"

    @property
    def current_node(self) -> ResultNode | None:
        """Currently active result node, if any."""
        return self._node_stack[-1] if self._node_stack else None

    def evaluate(self, node: CalculatorBase[Any, Any], input: NodeInput | None = None) -> ResultNode:
        """Evaluate a dependency node in this context."""
        return self.engine.evaluate(node, self, input)

    def public_value(self, node: CalculatorBase[TRaw, TPublic], input: NodeInput | None = None) -> TPublic:
        node_result = self.evaluate(node, input)
        return self.runtime_store[node_result.node_id].public_value

    def raw_value(self, node: CalculatorBase[TRaw, TPublic], input: NodeInput | None = None) -> TRaw:
        """Evaluate a dependency and return its raw value."""
        node_result = self.evaluate(node, input)
        return self.runtime_store[node_result.node_id].raw_value

    def register_node(self, node_result: ResultNode) -> None:
        """Register a newly created result node."""
        self.node_registry[node_result.node_id] = node_result

        if node_result.name:
            existing = self.named_registry.get(node_result.name)
            if existing is not None and existing != node_result.node_id:
                existing_node = self.node_registry.get(existing)
                if existing_node is not None and existing_node.signature == node_result.signature:
                    node_result.artifacts["duplicate_named_node"] = existing
                    self.log(
                        "debug",
                        f"duplicate named calculator {node_result.name!r}; keeping first registration",
                    )
                    return

                raise ValueError(f"Duplicate named calculator node {node_result.name!r}.")

            self.named_registry[node_result.name] = node_result.node_id

    def register_runtime_value(self, node_id: str, raw_value: Any, public_value: Any) -> None:
        """Store raw and public runtime values for a node."""
        self.runtime_store[node_id] = ExecutionValue(
            node_id=node_id,
            raw_value=raw_value,
            public_value=public_value,
        )

    def advance_mutation_generation(self, reason: str) -> int:
        """Advance the mutation generation after a transform changes state."""
        self.mutation_generation += 1
        self.log("debug", f"mutation generation {self.mutation_generation}: {reason}")
        return self.mutation_generation

    def log(self, level: str, message: str, *, node_id: str | None = None, phase: str | None = None) -> None:
        """Record and emit a runtime log message."""
        self.log_events.append(
            LogEvent(
                timestamp=time.perf_counter(),
                level=level,
                node_id=node_id,
                phase=phase,
                message=message,
            )
        )
        log_fn = getattr(logger, level, logger.debug)
        log_fn(message)

    @contextmanager
    def node_scope(self, node_result: ResultNode, node: CalculatorBase[Any, Any]) -> Iterator[None]:
        """Context manager for entering an evaluated node."""
        parent = self.current_node
        if parent is not None and node_result.node_id not in parent.children:
            parent.children.append(node_result.node_id)

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
        if node.__class__.__name__ == "BoundCalculator":
            if phase_name == "filter":
                node_name = f"{node.log_label}.scope"
            elif phase_name == "transform":
                node_name = f"{node.log_label}.scope"
            elif phase_name == "calculate":
                node_name = f"{node.log_label}.scope"
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

        with self.trace.phase(current.node_id, node_name, phase_name):
            try:
                with self.perf.phase(
                    phase_name,
                    measure_time=self.options.perf_time,
                    measure_memory=self.options.perf_memory,
                ) as phase_record:
                    record = phase_record
                    yield
            except Exception:
                status = "error"
                self.log("error", f"{node_name}:{phase_name} error", node_id=current.node_id, phase=phase_name)
                raise
            finally:
                if record is not None:
                    current.phases.append(record)
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
                        )
                    )
                    self.log(
                        "debug",
                        f"{node_name}:{phase_name} {status}",
                        node_id=current.node_id,
                        phase=phase_name,
                    )


def resolve_value(
    ctx: ExecutionContext,
    input: NodeInput,
    value: Any,
    *,
    field_name: str | None = None,
    target_units: Any | None = None,
) -> Any:
    """Resolve constants, callables, and calculator-valued parameters."""
    from .params import resolve_dynamic_value

    return resolve_dynamic_value(
        ctx,
        input,
        value,
        field_name=field_name,
        target_units=target_units,
    )
