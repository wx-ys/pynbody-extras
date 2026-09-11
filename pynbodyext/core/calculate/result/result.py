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
from .views import (
    DiagnosticsView,
    ErrorListView,
    NamedView,
    ObservationsView,
    ReportsView,
    WarningListView,
    as_view,
)

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


def _text_view(title: str, text: str) -> ViewObject:
    """Build a single-section ``ViewObject`` that renders *text* under *title*."""

    class _TextView(ViewObject):
        def _title(self) -> str:
            return title

        def _summary(self) -> str:
            first = text.strip().splitlines()
            return first[0] if first else text.strip()

        def _sections(self) -> list[tuple[str | None, str]]:
            return [(None, text)]

    return _TextView()


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
    observations: ObservationsView = field(default_factory=ObservationsView)
    provenance: ProvenanceInfo | None = None
    perf_summary: PerfSummary = field(default_factory=PerfSummary)
    warnings: list[str] | WarningListView = field(default_factory=WarningListView)
    errors: list[ErrorInfo] | ErrorListView = field(default_factory=ErrorListView)

    reports: ReportsView = field(default_factory=ReportsView)
    diagnostics: DiagnosticsView = field(default_factory=DiagnosticsView)

    def __post_init__(self) -> None:
        if not isinstance(self.named, NamedView):
            self.named = NamedView(self.named) if self.named else NamedView({})
        if not isinstance(self.errors, ErrorListView):
            self.errors = ErrorListView(self.errors) if self.errors else ErrorListView([])
        if not isinstance(self.warnings, WarningListView):
            self.warnings = WarningListView(self.warnings) if self.warnings else WarningListView([])
        self.reports = as_view(self.reports, ReportsView)
        self.diagnostics = as_view(self.diagnostics, DiagnosticsView)
        self.observations = as_view(self.observations, ObservationsView, owner=self)
        self.observations._owner = self

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
        return dict(self.diagnostics.get("named_values", {}))

    @property
    def execution_tree(self) -> ViewObject:
        """A view object exposing the runtime execution tree report."""
        return _text_view("Execution tree", self.report_execution_tree())

    @property
    def performance(self) -> ViewObject:
        """A view object exposing the formatted performance report."""
        return _text_view("Performance", self.report_perf())

    @property
    def cache(self) -> ViewObject:
        """A view object exposing the runtime cache report."""
        return _text_view("Cache", self.report_cache())

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
        named_values = self.diagnostics.get("named_values", {})
        return named_values.get(name, default)

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

    def report_cache(self) -> str:
        """Return the runtime cache report."""
        return self.reports.get("cache", "")

    def report_trace_timeline(self, *, show_ids: bool = False, include_observer: bool = True) -> str:
        """Return a trace timeline for the run."""
        lines: list[str] = []
        for event in self.diagnostics.trace():
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

    def report_trace_tree(self, *, show_ids: bool = False) -> str:
        """Return the stored execution trace tree."""
        text = self.reports.get("trace_tree", "")
        if show_ids:
            return text
        lines: list[str] = []
        for line in text.splitlines():
            if " [" in line and line.endswith("]"):
                lines.append(line.rsplit(" [", 1)[0])
            else:
                lines.append(line)
        return "\n".join(lines)

    def report_observer(self, *, include_empty: bool = False, show_ids: bool = False) -> str:
        """Return a formatted pynbody field observer report."""
        if not include_empty and not show_ids:
            return self.reports.get("observer", "")
        return render_observer_report(
            self.observations.all().values(), include_empty=include_empty, show_ids=show_ids
        )

    def iter_nodes(self) -> list[ResultNode]:
        """Return result nodes in registry order."""
        return list(self.nodes.values())

    def root_children(self) -> list[ResultNode]:
        """Return direct children of the root node."""
        from .query import ResultQuery

        return ResultQuery.children_of(self, self.root)

    def find(self, query: Any, relative_to: str | ResultNode | None = None) -> list[ResultNode]:
        """Return nodes matching a query.

        With ``relative_to=None`` the query selects across all nodes (a node
        reference, name substring, calculator class/instance, signature, a
        predicate callable, or the string constants ``"errors"`` / ``"all"``).
        With ``relative_to`` set to a node (id/name/object) the query is a
        relation name (``"parents"`` / ``"children"`` / ``"ancestors"`` /
        ``"descendants"`` / ``"root"``) returning the related nodes.
        """
        from .query import ResultQuery

        return ResultQuery.find(self, query, relative_to=relative_to)

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
