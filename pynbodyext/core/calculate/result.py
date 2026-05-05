"""Result objects and display helpers for calculator execution.

Every calculator run returns a :class:`Result`.  Results contain the public root
value, execution records for all visited nodes, named outputs, provenance,
reports, warnings, errors, and raw diagnostics.

This module is the main place to look after a run has finished.  It is designed
to support both programmatic inspection and interactive notebook use.

What A Result Contains
----------------------
A :class:`Result` gives access to several layers of information:

- ``value``: the public root value
- ``root``: the root :class:`ResultNode`
- ``nodes``: all evaluated nodes keyed by node id
- ``named``: nodes registered by ``keep()``, ``named()``, or pipeline outputs
- ``perf_summary``: aggregate runtime counters
- ``reports``: formatted text reports
- ``diagnostics``: raw trace/cache/log payloads

Basic Example
-------------
Inspect a result after running a simple calculator::

    result = calc.run(sim)

    print(result.value)
    print(result.root)
    print(result.perf_summary)

Named Output Example
--------------------
Named nodes and pipeline outputs can be retrieved after the run::

    result = pipe.run(sim)
    print(result.get("mass"))
    print(result.get("temp_mean"))

This is often the cleanest way to expose reusable intermediate products from a
larger graph.

Reports And Diagnostics
-----------------------
Results also provide human-readable reports for debugging and profiling::

    result = calc.run(sim, progress="phase")

    print(result.reports["trace_tree"])
    print(result.reports["trace_timeline"])
    print(result.reports["cache"])
    print(result.reports["perf"])

For deeper debugging, the raw event streams are available through
``result.diagnostics``.

Working With ResultNode
-----------------------
Each :class:`ResultNode` records what happened for one calculator node during
the run.  It includes status, summaries, phase timing, stored values, children,
and any error information.

This is useful when you want to inspect the graph in more detail::

    node = result.root
    print(node.label)
    print(node.status)
    print(node.value_summary)

Notebook-Friendly Display
-------------------------
The classes in this module provide compact text ``repr`` output as well as rich
HTML representations.  In notebooks, simply evaluating ``result`` or
``result.root`` is often enough to get a readable summary.

When To Read This Module Directly
---------------------------------
Most users interact with :class:`Result` but do not need to construct these
classes directly.  They become especially useful when:

- debugging a new calculator subclass
- understanding why a graph recomputed or reused cache
- checking which nodes failed under ``errors="collect"``
- building custom reports or notebook helpers on top of run outputs

Notes
-----
A :class:`Result` is intentionally richer than just the root value.  If you are
writing examples or tutorials for this framework, prefer showing
``result = calc.run(sim)`` rather than only ``calc(sim)`` whenever diagnostics
or provenance matter.
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

    def cache_report(self) -> str:
        """Return the runtime cache report."""
        return self.report("cache")

    def trace_timeline(self, *, show_ids: bool = False) -> str:
        """Return a trace timeline for the run."""
        if show_ids:
            return self.report("trace_timeline")
        return "\n".join(
            f"{'  ' * event.depth}{event.node_name} {event.phase}:{event.event}"
            for event in self.trace_events()
        )

    def trace_tree(self, *, show_ids: bool = False) -> str:
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

    def trace_events(self) -> list[Any]:
        """Return raw trace events."""
        return list(self.diagnostic("trace_events", []))

    def cache_events(self) -> list[Any]:
        """Return raw cache events."""
        return list(self.diagnostic("cache_events", []))

    def log_events(self) -> list[Any]:
        """Return runtime log events captured during evaluation."""
        return list(self.diagnostic("log_events", []))

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

    def node_tree(self, node: str | ResultNode | None = None, *, show_ids: bool = False) -> str:
        """Return a tree view of evaluated nodes."""
        return ResultQuery.node_tree(self, node=node, show_ids=show_ids)

    def perf_report(self, *, show_ids: bool = False) -> str:
        """Return a formatted performance report."""
        return ResultRepr.perf_table(self, show_ids=show_ids)

    def summary(self) -> str:
        """Return a compact text summary of the run."""
        return ResultRepr.summary(self)

    def pipeline_report(
        self,
        *,
        include_perf: bool = True,
        include_trace: bool = False,
        include_cache: bool = False,
        include_errors: bool = True,
        show_ids: bool = False,
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
            show_ids=show_ids,
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
    def node_tree(
        result: Result[Any],
        node: str | ResultNode | None = None,
        *,
        show_ids: bool = False,
    ) -> str:
        start = result.root if node is None else ResultQuery.resolve_node(result, node)

        def render(current: ResultNode, prefix: str, is_last: bool) -> list[str]:
            branch = "└─" if is_last else "├─"
            lines = [f"{prefix}{branch} {ResultQuery.node_label(current, show_ids=show_ids)}"]
            child_prefix = prefix + ("   " if is_last else "│  ")
            children = ResultQuery.display_children_of(result, current)
            for index, child in enumerate(children):
                lines.extend(render(child, child_prefix, index == len(children) - 1))
            return lines

        return "\n".join(render(start, "", True))


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
            body=named_table + html_pre(result.node_tree()),
        )

    @staticmethod
    def perf_table(result: Result[Any], *, show_ids: bool = False) -> str:
        title = result.root.name or str(result.root.kind)
        lines: list[str] = [title] if title else []
        header = "Node                           | Phase           | Time         | Mem Used       | Peak Mem       | RSS Delta"
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))

        for node in result.nodes.values():
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
        show_ids: bool = False,
    ) -> str:
        sections: list[str] = [
            "Summary",
            ResultRepr.summary(result),
            "Pipeline",
            ResultQuery.node_tree(result, show_ids=show_ids),
        ]

        if include_perf:
            perf_text = ResultRepr.perf_table(result, show_ids=show_ids).strip()
            if perf_text:
                sections.extend(["Performance", perf_text])

        if include_trace:
            trace_text = result.trace_timeline(show_ids=show_ids).strip()
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
