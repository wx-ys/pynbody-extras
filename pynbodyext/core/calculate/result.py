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
from typing import Any, Generic, TypeVar

from .display import (
    compact_repr,
    display_value,
    format_mem,
    format_time,
    html_card,
    html_pre,
    html_table,
    mimebundle,
)
from .enums import NodeKind, NodeStatus, RecordPolicy
from .observer import AccessObservation, format_observation_access, render_observer_report

T = TypeVar("T")


@dataclass(slots=True)
class ValueSummary:
    """Compact description of a runtime value.

    Parameters
    ----------
    python_type : str
        Name of the Python value type.
    shape, dtype, units, preview : optional
        Optional array and unit metadata captured for reports.
    """

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

    record_policy: RecordPolicy | None = None
    raw_value: Any = None
    value: Any = None
    value_summary: ValueSummary | None = None
    stored_raw: bool = False
    stored_value: bool = False

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
        return ResultRepr.result_node_repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("ResultNode(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
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
    named: dict[str, ResultNode] = field(default_factory=dict)
    observations: dict[str, AccessObservation] = field(default_factory=dict)
    provenance: ProvenanceInfo | None = None
    perf_summary: PerfSummary = field(default_factory=PerfSummary)
    warnings: list[str] = field(default_factory=list)
    errors: list[ErrorInfo] = field(default_factory=list)

    reports: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return ResultRepr.result_repr(self)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("Result(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
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

    def get_node(self, node_id: str) -> ResultNode:
        """Return a result node by internal node id."""
        return self.nodes[node_id]

    def get_named(self, name: str) -> ResultNode:
        """Return a result node by registered name."""
        return self.named[name]

    def get(self, name: str, default: Any = None) -> Any:
        """Return a named public value.

        Parameters
        ----------
        name : str
            Named node or pipeline output name.
        default : object, optional
            Value returned when the name is absent or unavailable.
        """
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
                access_text = format_observation_access(
                    observation,
                    phase=event.phase,
                    include_reads=True,
                )
                if access_text:
                    access_suffix = f" {access_text}"
            lines.append(
                f"{'  ' * event.depth}{event.node_name}{node_suffix} "
                f"{event.phase}:{event.event}{access_suffix}"
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
        return {
            node_id: node.observation
            for node_id, node in self.nodes.items()
            if node.observation is not None
        }

    def observation_of(self, node: str | ResultNode) -> AccessObservation | None:
        """Return the access observation for one node, if available."""
        resolved = ResultQuery.resolve_node(self, node)
        if resolved.observation is not None:
            return resolved.observation
        return self.observations.get(resolved.node_id)

    def report_observer(self, *, include_empty: bool = False, show_ids: bool = False) -> str:
        """Return a formatted pynbody field observer report."""
        if not include_empty and not show_ids:
            return self.report("observer")
        return render_observer_report(
            self.access_observations().values(),
            include_empty=include_empty,
            show_ids=show_ids,
        )

    def iter_nodes(self) -> list[ResultNode]:
        """Return result nodes in registry order."""
        return list(self.nodes.values())

    def root_children(self) -> list[ResultNode]:
        """Return direct children of the root node."""
        return self.children_of(self.root)


    def children_of(self, node: str | ResultNode) -> list[ResultNode]:
        """Return child nodes for a node id, name, or node object."""
        return ResultQuery.children_of(self, node)

    def phases_of(self, node: str | ResultNode) -> list[PhaseRecord]:
        """Return phase records for a node id, name, or node object."""
        return ResultQuery.phases_of(self, node)

    def walk_depth_first(self) -> list[ResultNode]:
        """Return nodes in depth-first order starting at the root."""
        return ResultQuery.walk_depth_first(self)

    def find_by_kind(self, kind: str) -> list[ResultNode]:
        """Return nodes whose kind matches ``kind``."""
        return ResultQuery.find_by_kind(self, kind)

    def find_error_nodes(self) -> list[ResultNode]:
        """Return nodes that captured an exception."""
        return ResultQuery.find_error_nodes(self)


    def describe_node(self, node: str | ResultNode) -> str:
        """Return a detailed text description of one result node."""
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
        return ResultQuery.node_tree(
            self,
            node=node,
            show_ids=show_ids,
            max_depth=max_depth,
            max_children=max_children,
        )

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
        self,
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> str:
        """Return a formatted performance report."""
        return ResultRepr.perf_table(
            self,
            show_ids=show_ids,
            max_depth=max_depth,
            max_children=max_children,
        )

    def report_summary(self) -> str:
        """Return a compact text summary of the run."""
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
        """Return a multi-section text report for the run.

        Parameters
        ----------
        include_perf, include_trace, include_cache, include_errors : bool
            Select optional report sections.
        show_ids : bool, default: False
            Include internal node ids when true.
        """
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


class ResultQuery:
    @staticmethod
    def resolve_node(result: Result[Any], node: str | ResultNode) -> ResultNode:
        if isinstance(node, ResultNode):
            return node
        if node in result.nodes:
            return result.nodes[node]
        if node in result.named:
            return result.named[node]
        raise KeyError(node)

    @staticmethod
    def children_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        return [result.nodes[node_id] for node_id in resolved.children if node_id in result.nodes]


    @staticmethod
    def display_children_of(result: Result[Any], node: str | ResultNode) -> list[ResultNode]:
        resolved = ResultQuery.resolve_node(result, node)
        children = ResultQuery.children_of(result, resolved)

        if (
            resolved.calculator_type == "BoundCalculator"
            and children
            and children[0].label == resolved.label
            and children[0].kind == resolved.kind
        ):
            base_node = children[0]
            return ResultQuery.children_of(result, base_node) + children[1:]

        return children

    @staticmethod
    def phases_of(result: Result[Any], node: str | ResultNode) -> list[PhaseRecord]:
        return list(ResultQuery.resolve_node(result, node).phases)

    @staticmethod
    def walk_depth_first(result: Result[Any]) -> list[ResultNode]:
        out: list[ResultNode] = []

        def visit(current: ResultNode) -> None:
            out.append(current)
            for child in ResultQuery.children_of(result, current):
                visit(child)

        visit(result.root)
        return out

    @staticmethod
    def find_by_kind(result: Result[Any], kind: str) -> list[ResultNode]:
        return [node for node in result.nodes.values() if str(node.kind) == kind]

    @staticmethod
    def find_error_nodes(result: Result[Any]) -> list[ResultNode]:
        return [node for node in result.nodes.values() if node.error is not None]

    @staticmethod
    def describe_node(result: Result[Any], node: str | ResultNode) -> str:
        resolved = ResultQuery.resolve_node(result, node)
        lines = [
            f"node_id: {resolved.node_id}",
            f"name: {resolved.name}",
            f"kind: {resolved.kind}",
            f"status: {resolved.status}",
            f"record_policy: {resolved.record_policy}",
            f"stored_value: {resolved.stored_value}",
            f"stored_raw: {resolved.stored_raw}",
            f"children: {len(resolved.children)}",
            f"phases: {len(resolved.phases)}",
        ]
        if resolved.value_summary is not None:
            lines.append(f"value_type: {resolved.value_summary.python_type}")
            if resolved.value_summary.preview:
                lines.append(f"preview: {resolved.value_summary.preview}")
        if resolved.error is not None:
            lines.append(f"error: {resolved.error.error_type}: {resolved.error.message}")
        if resolved.observation is not None:
            lines.append(f"observer_events: {resolved.observation.event_count}")
            lines.append(f"observer_reads: {len(resolved.observation.reads)}")
            lines.append(f"observer_dirty: {len(resolved.observation.dirty_fields)}")
            lines.append(f"observer_deletes: {len(resolved.observation.deletes)}")
        return "\n".join(lines)

    @staticmethod
    def node_label(
        node: ResultNode,
        *,
        show_ids: bool = False,
        show_ref: bool = False,
        show_kind: bool = True,
        max_width: int | None = None,
    ) -> str:
        label = node.label
        if show_kind:
            label = f"{label}<{node.kind}>"
        if show_ref:
            label = f"[{node.ref}] {label}"
        if show_ids:
            label = f"{label} [{node.node_id}]"
        if max_width is not None and len(label) > max_width:
            return label[: max_width - 3] + "..."
        return label

    @staticmethod
    def _validate_tree_limits(max_depth: int | None, max_children: int | None) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if max_children is not None and max_children < 0:
            raise ValueError("max_children must be non-negative or None")

    @staticmethod
    def _subtree_nodes(result: Result[Any], roots: list[ResultNode]) -> list[ResultNode]:
        out: list[ResultNode] = []
        seen: set[str] = set()
        stack = list(reversed(roots))
        while stack:
            node = stack.pop()
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            out.append(node)
            stack.extend(reversed(ResultQuery.display_children_of(result, node)))
        return out

    @staticmethod
    def _hidden_count_label(result: Result[Any], roots: list[ResultNode]) -> str:
        count = len(ResultQuery._subtree_nodes(result, roots))
        suffix = "node" if count == 1 else "nodes"
        return f"... {count} {suffix} hidden"

    @staticmethod
    def _visible_tree_nodes(
        result: Result[Any],
        start: ResultNode,
        *,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> tuple[list[ResultNode], int]:
        visible: list[ResultNode] = []
        hidden_ids: set[str] = set()
        visible_ids: set[str] = set()

        def hide(roots: list[ResultNode]) -> None:
            for node in ResultQuery._subtree_nodes(result, roots):
                if node.node_id not in visible_ids:
                    hidden_ids.add(node.node_id)

        def visit(node: ResultNode, depth: int) -> None:
            visible.append(node)
            visible_ids.add(node.node_id)
            children = ResultQuery.display_children_of(result, node)
            if max_depth is not None and depth >= max_depth and children:
                hide(children)
                return

            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for child in visible_children:
                visit(child, depth + 1)
            hide(hidden_children)

        visit(start, 0)
        return visible, len(hidden_ids)

    @staticmethod
    def _node_elapsed_s(node: ResultNode) -> float | None:
        values = [phase.elapsed_s for phase in node.phases if phase.elapsed_s is not None]
        if not values:
            return None
        return sum(values)

    @staticmethod
    def _cache_events_by_node(result: Result[Any]) -> dict[str, dict[str, int]]:
        grouped: dict[str, dict[str, int]] = {}
        for event in result.cache_events():
            node_id = getattr(event, "node_id", None)
            if not node_id:
                continue
            node_events = grouped.setdefault(node_id, {})
            event_name = str(getattr(event, "event", ""))
            node_events[event_name] = node_events.get(event_name, 0) + 1
        return grouped

    @staticmethod
    def _cache_suffix(
        node: ResultNode,
        cache_events: dict[str, dict[str, int]],
        *,
        hit_occurrence: bool = False,
    ) -> str:
        if hit_occurrence:
            return "hit"

        events = cache_events.get(node.node_id, {})
        parts: list[str] = []
        hits = events.get("hit", 0)
        stores = events.get("store", 0)
        if stores:
            parts.append("store" if stores == 1 else f"{stores} stores")
            if hits:
                parts.append(f"nhit={hits}")
        elif hits:
            parts.append("hit" if hits == 1 else f"{hits} hits")
        return "; ".join(parts)

    @staticmethod
    def _format_field_set(values: set[str], *, max_items: int = 4) -> str:
        if not values:
            return ""
        items = sorted(values)
        if len(items) <= max_items:
            return ",".join(items)
        hidden = len(items) - max_items
        return f"{','.join(items[:max_items])},+{hidden}"

    @staticmethod
    def _value_suffix(node: ResultNode) -> str:
        if node.value_summary is None:
            return ""
        summary = node.value_summary
        parts = [summary.python_type]
        if summary.shape is not None:
            parts.append(f"shape={summary.shape!r}")
        if summary.units is not None:
            parts.append(f"units={summary.units}")
        return " ".join(parts)

    @staticmethod
    def _execution_label(
        result: Result[Any],
        node: ResultNode,
        *,
        show_ids: bool,
        include_perf: bool,
        include_cache: bool,
        include_observer: bool,
        include_values: bool,
        cache_events: dict[str, dict[str, int]],
        hit_occurrence: bool = False,
    ) -> str:
        label = ResultQuery.node_label(node, show_ids=show_ids)
        parts: list[str] = []

        if hit_occurrence:
            return f"{label}  [cache=hit]"

        if node.error is not None:
            parts.append(f"error={node.error.error_type}")
        elif node.status != NodeStatus.OK:
            parts.append(f"status={display_value(node.status)}")

        if include_perf:
            elapsed = ResultQuery._node_elapsed_s(node)
            if elapsed is not None:
                parts.append(format_time(elapsed))

        if include_cache:
            cache_text = ResultQuery._cache_suffix(node, cache_events)
            if cache_text:
                parts.append(f"cache={cache_text}")

        if include_observer:
            access_text = format_observation_access(
                result.observation_of(node),
                read_items=3,
                dirty_items=3,
                delete_items=2,
            )
            if access_text:
                parts.append(access_text)

        if include_values:
            value_text = ResultQuery._value_suffix(node)
            if value_text:
                parts.append(f"value={value_text}")

        if not parts:
            return label
        return f"{label}  [{'; '.join(parts)}]"

    @staticmethod
    def _hidden_execution_summary(
        result: Result[Any],
        roots: list[ResultNode],
        *,
        include_perf: bool,
        include_cache: bool,
        include_observer: bool,
        cache_events: dict[str, dict[str, int]],
    ) -> str:
        nodes = ResultQuery._subtree_nodes(result, roots)
        suffix = "node" if len(nodes) == 1 else "nodes"
        parts = [f"... {len(nodes)} {suffix} hidden"]

        if include_perf:
            elapsed_values = [ResultQuery._node_elapsed_s(node) for node in nodes]
            elapsed = sum(value for value in elapsed_values if value is not None)
            if elapsed:
                parts.append(format_time(elapsed))

        if include_cache:
            hits = sum(cache_events.get(node.node_id, {}).get("hit", 0) for node in nodes)
            stores = sum(cache_events.get(node.node_id, {}).get("store", 0) for node in nodes)
            cache_parts: list[str] = []
            if stores:
                cache_parts.append(f"{stores} store")
            if hits:
                cache_parts.append(f"nhit={hits}")
            if cache_parts:
                parts.append(f"cache={','.join(cache_parts)}")

        if include_observer:
            reads: set[str] = set()
            dirty_fields: set[str] = set()
            deletes: set[str] = set()
            for node in nodes:
                observation = result.observation_of(node)
                if observation is None:
                    continue
                reads.update(observation.reads)
                dirty_fields.update(observation.dirty_fields)
                deletes.update(observation.deletes)
            read_text = ResultQuery._format_field_set(reads, max_items=3)
            dirty_text = ResultQuery._format_field_set(dirty_fields, max_items=3)
            delete_text = ResultQuery._format_field_set(deletes, max_items=2)
            if read_text:
                parts.append(f"read={read_text}")
            if dirty_text:
                parts.append(f"dirty={dirty_text}")
            if delete_text:
                parts.append(f"del={delete_text}")

        return "; ".join(parts)

    @staticmethod
    def node_tree(
        result: Result[Any],
        node: str | ResultNode | None = None,
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> str:
        ResultQuery._validate_tree_limits(max_depth, max_children)
        start = result.root if node is None else ResultQuery.resolve_node(result, node)

        def render(current: ResultNode, prefix: str, is_last: bool, depth: int) -> list[str]:
            branch = "└─" if is_last else "├─"
            lines = [f"{prefix}{branch} {ResultQuery.node_label(current, show_ids=show_ids)}"]
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = ResultQuery.display_children_of(result, current)
            if max_depth is not None and depth >= max_depth and children:
                lines.append(f"{child_prefix}└─ {ResultQuery._hidden_count_label(result, children)}")
                return lines

            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, child_prefix, is_child_last, depth + 1))
            if hidden_children:
                lines.append(f"{child_prefix}└─ {ResultQuery._hidden_count_label(result, hidden_children)}")
            return lines

        lines = [ResultQuery.node_label(start, show_ids=show_ids)]
        children = ResultQuery.display_children_of(result, start)
        if max_depth == 0 and children:
            lines.append(f"└─ {ResultQuery._hidden_count_label(result, children)}")
        else:
            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, "", is_child_last, 1))
            if hidden_children:
                lines.append(f"└─ {ResultQuery._hidden_count_label(result, hidden_children)}")
        return "\n".join(lines)

    @staticmethod
    def execution_tree(
        result: Result[Any],
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
        ResultQuery._validate_tree_limits(max_depth, max_children)
        start = result.root if node is None else ResultQuery.resolve_node(result, node)
        cache_events = ResultQuery._cache_events_by_node(result)
        rendered_nodes: set[str] = set()

        def label(current: ResultNode, *, hit_occurrence: bool = False) -> str:
            return ResultQuery._execution_label(
                result,
                current,
                show_ids=show_ids,
                include_perf=include_perf,
                include_cache=include_cache,
                include_observer=include_observer,
                include_values=include_values,
                cache_events=cache_events,
                hit_occurrence=hit_occurrence,
            )

        def hidden_summary(children: list[ResultNode]) -> str:
            return ResultQuery._hidden_execution_summary(
                result,
                children,
                include_perf=include_perf,
                include_cache=include_cache,
                include_observer=include_observer,
                cache_events=cache_events,
            )

        def render(current: ResultNode, prefix: str, is_last: bool, depth: int) -> list[str]:
            branch = "└─" if is_last else "├─"
            hit_occurrence = current.node_id in rendered_nodes
            lines = [f"{prefix}{branch} {label(current, hit_occurrence=hit_occurrence)}"]
            if hit_occurrence:
                return lines
            rendered_nodes.add(current.node_id)
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = ResultQuery.display_children_of(result, current)
            if max_depth is not None and depth >= max_depth and children:
                lines.append(f"{child_prefix}└─ {hidden_summary(children)}")
                return lines

            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, child_prefix, is_child_last, depth + 1))
            if hidden_children:
                lines.append(f"{child_prefix}└─ {hidden_summary(hidden_children)}")
            return lines

        lines = [label(start)]
        rendered_nodes.add(start.node_id)
        children = ResultQuery.display_children_of(result, start)
        if max_depth == 0 and children:
            lines.append(f"└─ {hidden_summary(children)}")
        else:
            visible_children = children if max_children is None else children[:max_children]
            hidden_children = [] if max_children is None else children[max_children:]
            for index, child in enumerate(visible_children):
                is_child_last = index == len(visible_children) - 1 and not hidden_children
                lines.extend(render(child, "", is_child_last, 1))
            if hidden_children:
                lines.append(f"└─ {hidden_summary(hidden_children)}")
        return "\n".join(lines)


class ResultRepr:

    @staticmethod
    def result_node_repr(node: ResultNode) -> str:
        parts = [
            f"label={node.label!r}",
            f"kind={display_value(node.kind)!r}",
            f"status={display_value(node.status)!r}",
        ]
        if node.stored_value:
            parts.append(f"value={compact_repr(node.value, max_length=60)}")
        elif node.value_summary is not None:
            parts.append(f"summary={compact_repr(node.value_summary, max_length=80)}")
        if node.children:
            parts.append(f"children={len(node.children)}")
        if node.error is not None:
            parts.append(f"error={node.error.error_type!r}")
        if node.observation is not None and node.observation.event_count:
            parts.append(f"observer_events={node.observation.event_count}")
        return f"ResultNode({', '.join(parts)})"

    @staticmethod
    def result_node_html(node: ResultNode) -> str:
        rows: list[tuple[str, Any]] = [
            ("label", node.label),
            ("kind", display_value(node.kind)),
            ("status", display_value(node.status)),
            ("stored value", node.stored_value),
            ("stored raw", node.stored_raw),
            ("children", len(node.children)),
            ("phases", len(node.phases)),
        ]
        if node.value_summary is not None:
            rows.append(("value", node.value_summary))
        if node.error is not None:
            rows.append(("error", f"{node.error.error_type}: {node.error.message}"))
        if node.observation is not None:
            rows.append(("observer events", node.observation.event_count))
            rows.append(("observer reads", len(node.observation.reads)))
            rows.append(("observer dirty", len(node.observation.dirty_fields)))
            rows.append(("observer deletes", len(node.observation.deletes)))
        return html_card("ResultNode", rows)

    @staticmethod
    def result_repr(result: Result[Any]) -> str:
        parts = [
            f"value={type(result.value).__name__}",
            f"ok={result.ok}",
            f"nodes={len(result.nodes)}",
        ]
        if result.named:
            parts.append(f"named={tuple(result.named.keys())!r}")
        if result.warnings:
            parts.append(f"warnings={len(result.warnings)}")
        if result.errors:
            parts.append(f"errors={len(result.errors)}")
        if result.observations:
            parts.append(f"observations={len(result.observations)}")
        return f"Result({', '.join(parts)})"

    @staticmethod
    def result_html(result: Result[Any]) -> str:
        rows: list[tuple[str, Any]] = [
            ("root", result.root.label),
            ("value", type(result.value).__name__),
            ("ok", result.ok),
            ("nodes", len(result.nodes)),
            ("named", ", ".join(result.named) if result.named else "-"),
            ("warnings", len(result.warnings)),
            ("errors", len(result.errors)),
            ("total time", format_time(result.perf_summary.total_time_s)),
            ("cache", f"{result.perf_summary.cache_hit_count} hit / {result.perf_summary.cache_miss_count} miss"),
        ]
        named_values = result.named_values
        named_table = ""
        if named_values:
            named_table = (
                "<div class='pynbodyext-calc-section-title'>Named values</div>"
                + html_table(
                    [(key, compact_repr(value, max_length=80)) for key, value in named_values.items()]
                )
            )
        return html_card(
            "Result",
            rows,
            body=named_table + html_pre(result.report_execution_tree(max_depth=4)),
        )

    @staticmethod
    def perf_table(
        result: Result[Any],
        *,
        show_ids: bool = False,
        max_depth: int | None = None,
        max_children: int | None = None,
    ) -> str:
        ResultQuery._validate_tree_limits(max_depth, max_children)
        title = result.root.name or str(result.root.kind)
        lines: list[str] = [title] if title else []
        header = "Node                           | Phase           | Time         | Mem Used       | Peak Mem       | RSS Delta"
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))

        if max_depth is None and max_children is None:
            nodes = list(result.nodes.values())
            hidden_count = 0
        else:
            visible_nodes, hidden_count = ResultQuery._visible_tree_nodes(
                result,
                result.root,
                max_depth=max_depth,
                max_children=max_children,
            )
            seen_node_ids: set[str] = set()
            nodes = []
            for node in visible_nodes:
                if node.node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node.node_id)
                nodes.append(node)

        for node in nodes:
            node_label = ResultQuery.node_label(
                node, show_ids=show_ids, show_ref=True, show_kind=True, max_width=30
            )
            for phase in node.phases:
                lines.append(
                    f"{node_label:<30} | "
                    f"{phase.phase[:15]:<15} | "
                    f"{format_time(phase.elapsed_s):>12} | "
                    f"{format_mem(phase.memory_used):>14} | "
                    f"{format_mem(phase.memory_peak):>14} | "
                    f"{format_mem(phase.rss_used):>10}"
                )

        if hidden_count:
            hidden_suffix = "node" if hidden_count == 1 else "nodes"
            hidden_label = f"... {hidden_count} {hidden_suffix} hidden"
            lines.append(
                f"{hidden_label[:30]:<30} | "
                f"{'-':<15} | "
                f"{'-':>12} | "
                f"{'-':>14} | "
                f"{'-':>14} | "
                f"{'-':>10}"
            )

        lines.append("-" * len(header))
        lines.append(
            f"{'Total':<30} | {'-':<15} | "
            f"{format_time(result.perf_summary.total_time_s):>12} | "
            f"{'-':>14} | {'-':>14} | {'-':>10}"
        )
        lines.append("-" * len(header))
        return "\n".join(lines)

    @staticmethod
    def cache_section(result: Result[Any], *, max_events: int = 12) -> str:
        lines = [
            "Runtime Cache",
            f"entries: {result.perf_summary.cache_store_count}",
            f"hits: {result.perf_summary.cache_hit_count}",
            f"misses: {result.perf_summary.cache_miss_count}",
            f"stores: {result.perf_summary.cache_store_count}",
        ]

        events = result.cache_events()
        if not events:
            return "\n".join(lines)

        lines.append("")
        lines.append("Recent events")
        for event in events[-max_events:]:
            if event.node_id and event.node_id in result.nodes:
                node = result.nodes[event.node_id]
                label = ResultQuery.node_label(
                    node,
                    show_ref=True,
                    show_kind=True,
                    max_width=48,
                )
            else:
                label = "-"
            lines.append(f"- {event.event}: {label}")

        return "\n".join(lines)

    @staticmethod
    def summary(
        result: Result[Any],
        *,
        include_cache_counts: bool = True
    ) -> str:
        root_label = result.root.label
        lines = [
            f"root: {root_label}",
            f"value_type: {type(result.value).__name__}",
            f"nodes: {len(result.nodes)}",
            f"warnings: {len(result.warnings)}",
            f"errors: {len(result.errors)}",
        ]

        if result.perf_summary.total_time_s is not None:
            lines.append(f"total_time_s: {result.perf_summary.total_time_s:.6f}")
        else:
            lines.append("total_time_s: -")

        if include_cache_counts:
            lines.append(f"cache_hits: {result.perf_summary.cache_hit_count}")
            lines.append(f"cache_misses: {result.perf_summary.cache_miss_count}")
            lines.append(f"cache_stores: {result.perf_summary.cache_store_count}")
        return "\n".join(lines)

    @staticmethod
    def pipeline_report(
        result: Result[Any],
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
        sections: list[str] = [
            "Summary",
            ResultRepr.summary(result),
            "Pipeline",
            ResultQuery.node_tree(
                result,
                show_ids=show_ids,
                max_depth=max_depth,
                max_children=max_children,
            ),
        ]

        if include_execution_tree:
            execution_text = ResultQuery.execution_tree(
                result,
                show_ids=show_ids,
                max_depth=max_depth,
                max_children=max_children,
            ).strip()
            if execution_text:
                sections.extend(["Execution", execution_text])

        if include_perf:
            perf_text = ResultRepr.perf_table(
                result,
                show_ids=show_ids,
                max_depth=max_depth,
                max_children=max_children,
            ).strip()
            if perf_text:
                sections.extend(["Performance", perf_text])

        if include_trace:
            trace_text = result.report_trace_timeline(show_ids=show_ids).strip()
            if trace_text:
                sections.extend(["Trace Timeline", trace_text])

        if include_cache:
            cache_text = ResultRepr.cache_section(result).strip()
            if cache_text:
                sections.extend(["Cache", cache_text])

        if include_errors and (result.errors or ResultQuery.find_error_nodes(result)):
            error_section: list[str] = []
            error_nodes = ResultQuery.find_error_nodes(result)

            if error_nodes:
                error_section.append("nodes:")
                for node in error_nodes:
                    phase = node.error.phase if node.error is not None else None
                    phase_suffix = f" phase={phase}" if phase else ""
                    label = ResultQuery.node_label(
                        node,
                        show_ids=show_ids,
                        show_ref=True,
                        show_kind=True,
                    )
                    error_section.append(f"- {label}{phase_suffix}")

            if result.errors:
                if error_section:
                    error_section.append("")
                    error_section.append("messages:")
                for err in result.errors:
                    phase_suffix = f" (phase={err.phase})" if err.phase else ""
                    error_section.append(f"- {err.error_type}: {err.message}{phase_suffix}")

            sections.extend(["Errors", "\n".join(error_section)])

        return "\n\n".join(section for section in sections if section)
