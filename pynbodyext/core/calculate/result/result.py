"""Result objects and reporting surface for calculator execution.

Every calculator run returns a :class:`Result`. A result stores the public root
value together with node-level execution records, named outputs, summaries,
reports, and raw diagnostics.

Most users interact with this module after execution has finished.

What A Result Contains
----------------------
A :class:`Result` exposes several layers of output:

- ``value``: public root value
- ``root``: root :class:`ResultNode`
- ``nodes``: all recorded nodes keyed by node id
- ``named``: nodes registered by ``named()``, ``keep()``, or pipeline outputs
- ``perf_summary``: aggregate runtime counters
- ``reports``: formatted trace/cache/performance reports
- ``diagnostics``: raw event payloads
- ``observations``: per-node pynbody field read/dirty/delete observations

Common Usage
------------
Typical post-run inspection::

    result = calc.run(sim, progress="phase")
    print(result.value)
    print(result.get("mass"))
    print(result.perf_summary)
    print(result.reports["trace_tree"])

ResultNode
----------
Each :class:`ResultNode` stores one node's execution outcome, including status,
value summaries, phase records, children, and optional error information.

This is useful when debugging graph-level behavior or partial failures under
collect-style error policies.

Reports Versus Diagnostics
--------------------------
Use ``result.reports`` for human-readable summaries.

Use ``result.diagnostics`` for raw machine-friendly event streams, such as:

- trace events
- cache events
- emitted logs
- observed pynbody field access events

When To Read This Module Directly
---------------------------------
This module matters most when you are:

- debugging a new calculator class
- inspecting why a node recomputed or reused cache
- analyzing partial failures in pipelines
- building custom notebook or reporting helpers on top of run outputs

Notes
-----
If you care about execution provenance, trace order, or cache behavior, prefer
``result = calc.run(sim)`` over extracting only the scalar value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pynbodyext.core.calculate.diagnostics.observer import (
    AccessObservation,
    format_observation_access,
    render_observer_report,
)
from pynbodyext.core.calculate.display import ViewObject, compact_repr, mimebundle

from .enums import NodeKind, NodeStatus, RecordPolicy
from .views import ErrorListView, NamedView, WarningListView

if TYPE_CHECKING:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

T = TypeVar("T")


@dataclass(slots=True)
class ValueSummary:
    """Compact description of a runtime value."""

    python_type: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    units: str | None = None
    preview: str | None = None

    def __repr__(self) -> str:
        parts = [self.python_type]
        if self.shape is not None:
            parts.append(f"shape={self.shape!r}")
        if self.dtype is not None:
            parts.append(f"dtype={self.dtype!r}")
        if self.units is not None:
            parts.append(f"units={self.units!r}")
        if self.preview is not None:
            parts.append(f"preview={compact_repr(self.preview, max_length=40)}")
        return f"ValueSummary({', '.join(parts)})"


@dataclass(slots=True)
class PhaseRecord:
    """Timing and memory information for one node phase."""

    phase: str
    started_at: float
    finished_at: float | None = None
    elapsed_s: float | None = None
    memory_used: int | None = None
    memory_peak: int | None = None
    rss_used: int | None = None
    status: str = "ok"


@dataclass(slots=True)
class ErrorInfo:
    """Structured error information captured during evaluation."""

    error_type: str
    message: str
    phase: str | None = None
    traceback_text: str | None = None


@dataclass(slots=True)
class ResultNode:
    """Execution record for one evaluated calculator node."""

    node_id: str
    kind: NodeKind
    signature: tuple[Any, ...]
    status: NodeStatus = NodeStatus.PENDING
    name: str | None = None
    display_name: str | None = None
    calculator_type: str | None = None
    calculator_class_path: str | None = None
    semantic_calculator_class_path: str | None = None

    record_policy: RecordPolicy | None = None
    raw_value: Any = None
    value: Any = None
    value_summary: ValueSummary | None = None
    stored_raw: bool = False
    stored_value: bool = False

    parent_ids: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    phases: list[PhaseRecord] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    observation: AccessObservation | None = None
    error: ErrorInfo | None = None

    @property
    def label(self) -> str:
        """Return a human-readable label for reports and display."""
        return self.display_name or self.name or str(self.kind)

    @property
    def ref(self) -> str:
        return f"n{self.node_id.rsplit(':', 1)[-1]}"

    def __repr__(self) -> str:
        from .repr import ResultRepr

        return ResultRepr.result_node_repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("ResultNode(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        from .repr import ResultRepr

        return ResultRepr.result_node_html(self)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())


@dataclass(slots=True)
class ProvenanceInfo:
    """Provenance metadata describing the calculator and simulation input."""

    calculator_signature: tuple[Any, ...]
    sim_signature: tuple[Any, ...]
    started_at: float
    finished_at: float | None = None
    calculator_signature_text: str | None = None
    calculator_signature_hash: str | None = None


@dataclass(slots=True)
class PerfSummary:
    """Aggregate performance counters for a calculator run."""

    total_time_s: float | None = None
    node_count: int = 0
    phase_count: int = 0
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_store_count: int = 0


@dataclass(slots=True)
class Result(Generic[T]):
    """Public result returned by :meth:`CalculatorBase.run`.

    Parameters
    ----------
    value : object
        Public value of the root calculator.
    root : ResultNode
        Execution record for the root node.
    nodes : dict
        All nodes evaluated during the run, keyed by node id.
    named : dict, optional
        Named nodes registered by ``keep()``, ``named()``, or pipeline outputs.

    Notes
    -----
    Node ids are available for debugging but are hidden from user-facing reports
    unless ``show_ids=True`` is requested.
    """

    value: T
    root: ResultNode
    nodes: dict[str, ResultNode]
    named: dict[str, ResultNode] | NamedView = field(default_factory=NamedView)
    #: The live calculator that produced this result, when known.  Set by the
    #: engine to the root node on a live run, and reconstructed from the stored
    #: ``provenance.calculator_signature_text`` for a result loaded from a store.
    #: Excluded from equality/repr so it does not change result identity.
    calculator: CalculatorBase | None = field(default=None, compare=False, repr=False)
    observations: dict[str, AccessObservation] = field(default_factory=dict)
    provenance: ProvenanceInfo | None = None
    perf_summary: PerfSummary = field(default_factory=PerfSummary)
    warnings: list[str] | WarningListView = field(default_factory=WarningListView)
    errors: list[ErrorInfo] | ErrorListView = field(default_factory=ErrorListView)

    reports: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.named, NamedView):
            self.named = NamedView(self.named) if self.named else NamedView({})
        if not isinstance(self.errors, ErrorListView):
            self.errors = ErrorListView(self.errors) if self.errors else ErrorListView([])
        if not isinstance(self.warnings, WarningListView):
            self.warnings = WarningListView(self.warnings) if self.warnings else WarningListView([])

    def __repr__(self) -> str:
        from .repr import ResultRepr

        return ResultRepr.result_repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("Result(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        from .repr import ResultRepr

        return ResultRepr.result_html(self)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())

    @property
    def ok(self) -> bool:
        """Whether the run completed without collected errors."""
        return not self.has_errors()

    @property
    def node_count(self) -> int:
        """Number of result nodes evaluated during the run."""
        return len(self.nodes)

    @property
    def named_values(self) -> dict[str, Any]:
        """Return public values for named nodes that were materialized."""
        return dict(self.diagnostic("named_values", {}))

    @property
    def execution_tree(self) -> ViewObject:
        """A view object exposing the runtime execution tree report."""
        text = self.report_execution_tree()

        class _ExecutionTreeView(ViewObject):
            def _title(self) -> str:
                return "Execution tree"

            def _summary(self) -> str:
                first = text.strip().splitlines()
                return first[0] if first else text.strip()

            def _sections(self) -> list[tuple[str, str]]:
                return [("execution_tree", text)]

        return _ExecutionTreeView()

    @property
    def performance(self) -> ViewObject:
        """A view object exposing the formatted performance report."""
        text = self.report_perf()

        class _PerformanceView(ViewObject):
            def _title(self) -> str:
                return "Performance"

            def _summary(self) -> str:
                first = text.strip().splitlines()
                return first[0] if first else text.strip()

            def _sections(self) -> list[tuple[str, str]]:
                return [("performance", text)]

        return _PerformanceView()

    @property
    def cache(self) -> ViewObject:
        """A view object exposing the runtime cache report."""
        text = self.cache_report()

        class _CacheView(ViewObject):
            def _title(self) -> str:
                return "Cache"

            def _summary(self) -> str:
                first = text.strip().splitlines()
                return first[0] if first else text.strip()

            def _sections(self) -> list[tuple[str, str]]:
                return [("cache", text)]

        return _CacheView()

    def get_node(self, node_id: str) -> ResultNode:
        """Return a result node by internal node id."""
        return self.nodes[node_id]

    def get_named(self, name: str) -> ResultNode:
        """Return a result node by registered name."""
        return self.named[name]

    def get(self, name: str, default: Any = None) -> Any:
        """Return a named public value."""
        node = self.named.get(name)
        if node is None:
            return default
        if node.stored_value:
            return node.value
        named_values = self.diagnostic("named_values", {})
        return named_values.get(name, default)

    def value_of(self, name: str, default: Any = None) -> Any:
        """Alias for :meth:`get`."""
        return self.get(name, default)

    def node(self, id_or_name: str) -> ResultNode:
        """Return a node by node id or registered name."""
        from .query import ResultQuery

        return ResultQuery.resolve_node(self, id_or_name)

    def has_errors(self) -> bool:
        """Whether any errors were collected during the run."""
        return bool(self.errors)

    def has_warnings(self) -> bool:
        """Whether any warnings were collected during the run."""
        return bool(self.warnings)

    def raise_if_errors(self) -> None:
        """Raise a :class:`RuntimeError` for the first collected error."""
        if not self.errors:
            return
        first = self.errors[0]
        raise RuntimeError(f"{first.error_type}: {first.message}")

    def available_reports(self) -> tuple[str, ...]:
        """Return names of available text reports."""
        return tuple(self.reports.keys())

    def available_diagnostics(self) -> tuple[str, ...]:
        """Return names of available diagnostic payloads."""
        return tuple(self.diagnostics.keys())

    def report(self, name: str, default: str = "") -> str:
        """Return a named text report."""
        return self.reports.get(name, default)

    def diagnostic(self, name: str, default: Any = None) -> Any:
        """Return a named diagnostic payload."""
        return self.diagnostics.get(name, default)

    def report_cache(self) -> str:
        """Return the runtime cache report."""
        return self.report("cache")

    def cache_report(self) -> str:
        """Alias for :meth:`report_cache`."""
        return self.report_cache()

    def report_trace_timeline(self, *, show_ids: bool = False, include_observer: bool = True) -> str:
        """Return a trace timeline for the run."""
        lines: list[str] = []
        for event in self.trace_events():
            node_suffix = f" [{event.node_id}]" if show_ids else ""
            access_suffix = ""
            if include_observer and event.event == "leave":
                observation = self.observations.get(event.node_id)
                access_text = format_observation_access(observation, phase=event.phase, include_reads=True)
                if access_text:
                    access_suffix = f" {access_text}"
            lines.append(
                f"{'  ' * event.depth}{event.node_name}{node_suffix} {event.phase}:{event.event}{access_suffix}"
            )
        return "\n".join(lines)

    def trace_timeline(self, *, show_ids: bool = False, include_observer: bool = True) -> str:
        """Alias for :meth:`report_trace_timeline`."""
        return self.report_trace_timeline(show_ids=show_ids, include_observer=include_observer)

    def report_trace_tree(self, *, show_ids: bool = False) -> str:
        """Return the stored execution trace tree."""
        text = self.report("trace_tree")
        if show_ids:
            return text
        lines: list[str] = []
        for line in text.splitlines():
            if " [" in line and line.endswith("]"):
                lines.append(line.rsplit(" [", 1)[0])
            else:
                lines.append(line)
        return "\n".join(lines)

    def trace_tree(self, *, show_ids: bool = False) -> str:
        """Alias for :meth:`report_trace_tree`."""
        return self.report_trace_tree(show_ids=show_ids)

    def trace_events(self) -> list[Any]:
        """Return raw trace events."""
        return list(self.diagnostic("trace_events", []))

    def cache_events(self) -> list[Any]:
        """Return raw cache events."""
        return list(self.diagnostic("cache_events", []))

    def log_events(self) -> list[Any]:
        """Return runtime log events captured during evaluation."""
        return list(self.diagnostic("log_events", []))

    def observer_events(self) -> list[Any]:
        """Return raw pynbody field observer events."""
        return list(self.diagnostic("observer_events", []))

    def access_observations(self) -> dict[str, AccessObservation]:
        """Return per-node pynbody field access observations."""
        if self.observations:
            return dict(self.observations)
        return {node_id: node.observation for node_id, node in self.nodes.items() if node.observation is not None}

    def observation_of(self, node: str | ResultNode) -> AccessObservation | None:
        """Return the access observation for one node, if available."""
        from .query import ResultQuery

        resolved = ResultQuery.resolve_node(self, node)
        if resolved.observation is not None:
            return resolved.observation
        return self.observations.get(resolved.node_id)

    def report_observer(self, *, include_empty: bool = False, show_ids: bool = False) -> str:
        """Return a formatted pynbody field observer report."""
        if not include_empty and not show_ids:
            return self.report("observer")
        return render_observer_report(
            self.access_observations().values(), include_empty=include_empty, show_ids=show_ids
        )

    def iter_nodes(self) -> list[ResultNode]:
        """Return result nodes in registry order."""
        return list(self.nodes.values())

    def root_children(self) -> list[ResultNode]:
        """Return direct children of the root node."""
        return self.children_of(self.root)

    def find(self, query: Any) -> list[ResultNode]:
        """Return nodes matching a calculator class, instance, signature, or predicate."""
        from .query import ResultQuery

        return ResultQuery.find(self, query)

    def parents_of(self, node: str | ResultNode) -> list[ResultNode]:
        """Return parent nodes for a node id, name, or node object."""
        from .query import ResultQuery

        return ResultQuery.parents_of(self, node)

    def parent_of(self, node: str | ResultNode) -> ResultNode | None:
        """Return the unique parent node, or ``None`` for the root."""
        from .query import ResultQuery

        return ResultQuery.parent_of(self, node)

    def children_of(self, node: str | ResultNode) -> list[ResultNode]:
        """Return child nodes for a node id, name, or node object."""
        from .query import ResultQuery

        return ResultQuery.children_of(self, node)

    def ancestors_of(self, node: str | ResultNode) -> list[ResultNode]:
        """Return ancestor nodes in nearest-first order."""
        from .query import ResultQuery

        return ResultQuery.ancestors_of(self, node)

    def descendants_of(self, node: str | ResultNode) -> list[ResultNode]:
        """Return descendant nodes in depth-first order."""
        from .query import ResultQuery

        return ResultQuery.descendants_of(self, node)

    def phases_of(self, node: str | ResultNode) -> list[PhaseRecord]:
        """Return phase records for a node id, name, or node object."""
        from .query import ResultQuery

        return ResultQuery.phases_of(self, node)

    def walk_depth_first(self) -> list[ResultNode]:
        """Return nodes in depth-first order starting at the root."""
        from .query import ResultQuery

        return ResultQuery.walk_depth_first(self)

    def find_by_kind(self, kind: str) -> list[ResultNode]:
        """Return nodes whose kind matches ``kind``."""
        from .query import ResultQuery

        return ResultQuery.find_by_kind(self, kind)

    def find_error_nodes(self) -> list[ResultNode]:
        """Return nodes that captured an exception."""
        from .query import ResultQuery

        return ResultQuery.find_error_nodes(self)

    def describe_node(self, node: str | ResultNode) -> str:
        """Return a detailed text description of one result node."""
        from .query import ResultQuery

        return ResultQuery.describe_node(self, node)

    def report_node_tree(
        self,
        node: str | ResultNode | None = None,
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> str:
        """Return a tree view of evaluated nodes."""
        from .query import ResultQuery

        return ResultQuery.node_tree(self, node=node, show_ids=show_ids, max_depth=max_depth, max_children=max_children)

    def report_execution_tree(
        self,
        node: str | ResultNode | None = None,
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
        include_perf: bool = True,
        include_cache: bool = True,
        include_observer: bool = True,
        include_values: bool = False,
    ) -> str:
        """Return a tree report annotated with runtime diagnostics."""
        from .query import ResultQuery

        return ResultQuery.execution_tree(
            self,
            node=node,
            show_ids=show_ids,
            max_depth=max_depth,
            max_children=max_children,
            include_perf=include_perf,
            include_cache=include_cache,
            include_observer=include_observer,
            include_values=include_values,
        )

    def report_perf(
        self, *, show_ids: bool = False, max_depth: int | None = None, max_children: int | None = None
    ) -> str:
        """Return a formatted performance report."""
        from .repr import ResultRepr

        return ResultRepr.perf_table(self, show_ids=show_ids, max_depth=max_depth, max_children=max_children)

    def report_summary(self) -> str:
        """Return a compact text summary of the run."""
        from .repr import ResultRepr

        return ResultRepr.summary(self)

    def report_pipeline(
        self,
        *,
        include_perf: bool = True,
        include_trace: bool = False,
        include_cache: bool = False,
        include_errors: bool = True,
        include_execution_tree: bool = False,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> str:
        """Return a multi-section text report for the run."""
        from .repr import ResultRepr

        return ResultRepr.pipeline_report(
            self,
            include_perf=include_perf,
            include_trace=include_trace,
            include_cache=include_cache,
            include_errors=include_errors,
            include_execution_tree=include_execution_tree,
            show_ids=show_ids,
            max_depth=max_depth,
            max_children=max_children,
        )
