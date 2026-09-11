"""HTML and text rendering for :class:`Result` and :class:`ResultNode`.

:class:`ResultRepr` is a stateless helper that produces human-readable and
notebook-renderable representations of result objects. All methods are static.

Long HTML builders are split into focused section helpers:

- ``_node_value_section``    – stored value / value summary block
- ``_node_phases_section``   – phase timing table
- ``_node_observer_section`` – access event counts + field lists
- ``_node_error_section``    – error / traceback block
- ``_result_metrics_section``  – top-level metric strip
- ``_result_named_section``    – named-value table
- ``_result_provenance_section`` – provenance fold-out
- ``_result_errors_section``   – errors table
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.display import (
    _style,
    compact_repr,
    display_value,
    format_mem,
    format_time,
    html_badge,
    html_card,
    html_data_table,
    html_details,
    html_metric_grid,
    html_metric_strip,
    html_pre,
    html_scroll_x,
    html_section,
    html_table,
)

from .query import ResultQuery

if TYPE_CHECKING:
    from .result import Result, ResultNode


class ResultRepr:
    # ── Status helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _tone_for_status(status: Any) -> str:
        text = str(display_value(status)).lower()
        if text in {"ok", "success", "ready", "true"}:
            return "ok"
        if text in {"pending", "running", "partial", "warning"}:
            return "warn"
        if text in {"error", "failed", "false"}:
            return "error"
        return "neutral"

    @staticmethod
    def _short_class_name(path: str | None) -> str:
        if not path:
            return "-"
        return path.rsplit(".", 1)[-1]

    @staticmethod
    def _format_field_set(values: set[str], *, limit: int = 6) -> str:
        if not values:
            return "-"
        items = sorted(values)
        if len(items) <= limit:
            return ", ".join(items)
        hidden = len(items) - limit
        return f"{', '.join(items[:limit])}, +{hidden}"

    # ── ResultNode repr ────────────────────────────────────────────────────────

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

    # ── ResultNode HTML section helpers ────────────────────────────────────────

    @staticmethod
    def _node_value_section(node: ResultNode) -> str:
        """Return the Value fold-out section HTML, or empty string."""
        value_rows: list[tuple[str, Any]] = []
        if node.value_summary is not None:
            value_rows.append(("type", node.value_summary.python_type))
            if node.value_summary.shape is not None:
                value_rows.append(("shape", node.value_summary.shape))
            if node.value_summary.dtype is not None:
                value_rows.append(("dtype", node.value_summary.dtype))
            if node.value_summary.units is not None:
                value_rows.append(("units", node.value_summary.units))
            if node.value_summary.preview is not None:
                value_rows.append(("preview", node.value_summary.preview))
        if node.stored_value:
            value_rows.append(("public value", compact_repr(node.value, max_length=220)))
        if node.stored_raw:
            value_rows.append(("raw value", compact_repr(node.raw_value, max_length=220)))
        if not value_rows:
            return ""
        return html_details(
            "Value",
            html_scroll_x(
                html_table(
                    value_rows,
                    class_name="pynbodyext-calc-table pynbodyext-calc-table-nowrap pynbodyext-calc-monospace",
                ),
                min_width="56rem",
            ),
        )

    @staticmethod
    def _node_phases_section(node: ResultNode) -> str:
        """Return the Phases fold-out section HTML, or empty string."""
        if not node.phases:
            return ""
        phase_rows = [
            [
                phase.phase,
                format_time(phase.elapsed_s),
                format_mem(phase.memory_used),
                format_mem(phase.memory_peak),
                format_mem(phase.rss_used),
                html_badge(phase.status, tone=ResultRepr._tone_for_status(phase.status)),
            ]
            for phase in node.phases
        ]
        return html_details(
            "Phases",
            html_scroll_x(
                html_data_table(
                    ["phase", "time", "mem", "peak", "rss", "status"],
                    phase_rows,
                    escape_values=False,
                    class_name="pynbodyext-calc-data-table pynbodyext-calc-data-table-nowrap pynbodyext-calc-monospace",
                ),
                min_width="48rem",
            ),
        )

    @staticmethod
    def _node_observer_section(node: ResultNode) -> str:
        """Return Observer count table + fields fold-out HTML, or empty string."""
        if node.observation is None:
            return ""
        observation_rows = [
            ("events", node.observation.event_count),
            ("reads", len(node.observation.reads)),
            ("dirty", len(node.observation.dirty_fields)),
            ("deletes", len(node.observation.deletes)),
        ]
        counts_html = html_details(
            "Observer",
            html_scroll_x(
                html_table(
                    observation_rows,
                    class_name="pynbodyext-calc-table pynbodyext-calc-table-nowrap pynbodyext-calc-monospace",
                ),
                min_width="42rem",
            ),
        )

        observer_lines: list[str] = []
        if node.observation.reads:
            observer_lines.append(f"reads: {ResultRepr._format_field_set(node.observation.reads)}")
        if node.observation.dirty_fields:
            observer_lines.append(f"dirty: {ResultRepr._format_field_set(node.observation.dirty_fields)}")
        if node.observation.deletes:
            observer_lines.append(f"deletes: {ResultRepr._format_field_set(node.observation.deletes)}")
        fields_html = (
            html_details("Observer fields", html_pre("\n".join(observer_lines)), open=False) if observer_lines else ""
        )
        return counts_html + fields_html

    @staticmethod
    def _node_error_section(node: ResultNode) -> str:
        """Return Error section HTML, or empty string."""
        if node.error is None:
            return ""
        error_rows = [("type", node.error.error_type), ("message", node.error.message)]
        if node.error.phase is not None:
            error_rows.append(("phase", node.error.phase))
        error_html = html_section(
            "Error",
            html_scroll_x(html_table(error_rows, class_name="pynbodyext-calc-table pynbodyext-calc-table-nowrap")),
        )
        if node.error.traceback_text:
            error_html += html_details("Traceback", html_pre(node.error.traceback_text), open=True)
        return error_html

    @staticmethod
    def result_node_html(node: ResultNode) -> str:
        rows: list[tuple[str, Any]] = [
            ("label", node.label),
            ("ref", node.ref),
            ("kind", html_badge(display_value(node.kind), tone="info")),
            ("status", html_badge(display_value(node.status), tone=ResultRepr._tone_for_status(node.status))),
            (
                "stored",
                " ".join(
                    [
                        html_badge(
                            "value" if node.stored_value else "value: no", tone="ok" if node.stored_value else "neutral"
                        ),
                        html_badge(
                            "raw" if node.stored_raw else "raw: no", tone="ok" if node.stored_raw else "neutral"
                        ),
                    ]
                ),
            ),
        ]

        if node.calculator_type is not None:
            rows.append(("calculator", node.calculator_type))

        semantic_class = node.semantic_calculator_class_path or node.calculator_class_path
        if semantic_class is not None:
            rows.append(("class", ResultRepr._short_class_name(semantic_class)))

        if node.record_policy is not None:
            rows.append(("record", html_badge(display_value(node.record_policy), tone="neutral")))

        metrics_section = html_metric_grid(
            [
                ("parents", len(node.parent_ids)),
                ("children", len(node.children)),
                ("phases", len(node.phases)),
                ("events", node.observation.event_count if node.observation is not None else 0),
            ]
        )

        body = (
            metrics_section
            + ResultRepr._node_value_section(node)
            + ResultRepr._node_phases_section(node)
            + ResultRepr._node_observer_section(node)
            + ResultRepr._node_error_section(node)
        )

        return html_card("ResultNode", rows, body=body, escape_values=False)

    # ── Result repr ────────────────────────────────────────────────────────────

    @staticmethod
    def result_repr(result: Result[Any]) -> str:
        parts = [f"value={type(result.value).__name__}", f"ok={result.ok}", f"nodes={len(result.nodes)}"]
        if result.named:
            parts.append(f"named={tuple(result.named.keys())!r}")
        if result.warnings:
            parts.append(f"warnings={len(result.warnings)}")
        if result.errors:
            parts.append(f"errors={len(result.errors)}")
        if result.observations:
            parts.append(f"observations={len(result.observations)}")
        return f"Result({', '.join(parts)})"

    # ── Result HTML section helpers ────────────────────────────────────────────

    @staticmethod
    def _result_metrics_section(result: Result[Any]) -> str:
        """Return the top-level metric strip HTML."""
        hits = result.perf_summary.cache_hit_count
        misses = result.perf_summary.cache_miss_count
        stores = result.perf_summary.cache_store_count
        cache_total = hits + misses
        hit_rate = f"{hits / cache_total:.0%}" if cache_total else "-"
        return html_scroll_x(
            html_metric_strip(
                [
                    ("nodes", len(result.nodes)),
                    ("phases", result.perf_summary.phase_count),
                    ("warnings", len(result.warnings)),
                    ("errors", len(result.errors)),
                    ("time", format_time(result.perf_summary.total_time_s)),
                    ("cache", f"{hits} hit / {misses} miss"),
                    ("hit rate", hit_rate),
                    ("stores", stores),
                ]
            )
        )

    @staticmethod
    def _result_named_section(result: Result[Any]) -> str:
        """Return the Named values section HTML, or empty string."""
        named_values = result.named_values
        if not named_values:
            return ""
        return html_section(
            "Named values",
            html_scroll_x(
                html_data_table(
                    ["name", "value"],
                    [[key, compact_repr(value, max_length=180)] for key, value in named_values.items()],
                    class_name="pynbodyext-calc-data-table pynbodyext-calc-data-table-nowrap",
                )
            ),
        )

    @staticmethod
    def _result_provenance_section(result: Result[Any]) -> str:
        """Return the Provenance fold-out HTML, or empty string."""
        if result.provenance is None:
            return ""
        provenance_rows: list[tuple[str, Any]] = []
        if result.provenance.calculator_signature_hash is not None:
            provenance_rows.append(("calculator hash", result.provenance.calculator_signature_hash))
        elif result.provenance.calculator_signature_text is not None:
            provenance_rows.append(
                ("calculator", compact_repr(result.provenance.calculator_signature_text, max_length=180))
            )
        provenance_rows.append(("sim signature", compact_repr(result.provenance.sim_signature, max_length=180)))
        if result.provenance.finished_at is not None:
            provenance_rows.append(
                ("wall time", format_time(result.provenance.finished_at - result.provenance.started_at))
            )
        return html_details(
            "Provenance",
            html_scroll_x(html_table(provenance_rows, class_name="pynbodyext-calc-table pynbodyext-calc-table-nowrap")),
        )

    @staticmethod
    def _result_errors_section(result: Result[Any]) -> str:
        """Return the Errors section HTML, or empty string."""
        error_rows: list[list[Any]] = []
        for node in ResultQuery.find_error_nodes(result)[:8]:
            if node.error is None:
                continue
            error_rows.append(
                [
                    ResultQuery.node_label(node, show_ref=True, show_kind=True),
                    node.error.phase or "-",
                    f"{node.error.error_type}: {node.error.message}",
                ]
            )
        remaining_slots = max(0, 8 - len(error_rows))
        for error in result.errors[:remaining_slots]:
            error_rows.append(["<run>", error.phase or "-", f"{error.error_type}: {error.message}"])
        if not error_rows:
            return ""
        return html_section(
            "Errors",
            html_scroll_x(
                html_data_table(
                    ["node", "phase", "message"],
                    error_rows,
                    class_name="pynbodyext-calc-data-table pynbodyext-calc-data-table-nowrap",
                )
            ),
        )

    @staticmethod
    def result_html(result: Result[Any]) -> str:
        rows: list[tuple[str, Any]] = [
            ("root", result.root.label),
            ("value", type(result.value).__name__),
            ("status", html_badge("ok" if result.ok else "error", tone="ok" if result.ok else "error")),
        ]
        if result.named:
            rows.append(("named", compact_repr(tuple(result.named.keys()), max_length=96)))
        if result.provenance is not None and result.provenance.calculator_signature_hash is not None:
            rows.append(("signature", result.provenance.calculator_signature_hash[:12]))

        if _style() != "rich":
            hint = "Use .named, .provenance, .execution_tree, .performance and .cache for details"
            return html_card("Result", rows, body=html_pre(hint), escape_values=False)

        execution_tree_html = html_details(
            "Execution tree", html_pre(result.report_execution_tree()), open=not result.ok
        )

        perf_text = result.report_perf().strip()
        perf_html = html_details("Performance", html_pre(perf_text), open=False) if perf_text else ""

        cache_text = ResultRepr.cache_section(result).strip()
        cache_html = html_details("Cache", html_pre(cache_text), open=False) if cache_text else ""

        body = (
            ResultRepr._result_metrics_section(result)
            + ResultRepr._result_named_section(result)
            + ResultRepr._result_provenance_section(result)
            + ResultRepr._result_errors_section(result)
            + execution_tree_html
            + perf_html
            + cache_html
        )

        return html_card("Result", rows, body=body, escape_values=False)

    # ── Perf table ─────────────────────────────────────────────────────────────

    @staticmethod
    def perf_table(
        result: Result[Any], *, show_ids: bool = False, max_depth: int | None = None, max_children: int | None = None
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
                result, result.root, max_depth=max_depth, max_children=max_children
            )
            seen_node_ids: set[str] = set()
            nodes = []
            for node in visible_nodes:
                if node.node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node.node_id)
                nodes.append(node)

        for node in nodes:
            node_label = ResultQuery.node_label(node, show_ids=show_ids, show_ref=True, show_kind=True, max_width=30)
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
            lines.append(f"{hidden_label[:30]:<30} | {'-':<15} | {'-':>12} | {'-':>14} | {'-':>14} | {'-':>10}")

        lines.append("-" * len(header))
        lines.append(
            f"{'Total':<30} | {'-':<15} | "
            f"{format_time(result.perf_summary.total_time_s):>12} | "
            f"{'-':>14} | {'-':>14} | {'-':>10}"
        )
        lines.append("-" * len(header))
        return "\n".join(lines)

    # ── Cache section ──────────────────────────────────────────────────────────

    @staticmethod
    def cache_section(result: Result[Any], *, max_events: int = 12) -> str:
        lines = [
            "Runtime Cache",
            f"entries: {result.perf_summary.cache_store_count}",
            f"hits: {result.perf_summary.cache_hit_count}",
            f"misses: {result.perf_summary.cache_miss_count}",
            f"stores: {result.perf_summary.cache_store_count}",
        ]

        events = result.diagnostics.cache()
        if not events:
            return "\n".join(lines)

        lines.append("")
        lines.append("Recent events")
        for event in events[-max_events:]:
            if event.node_id and event.node_id in result.nodes:
                node = result.nodes[event.node_id]
                label = ResultQuery.node_label(node, show_ref=True, show_kind=True, max_width=48)
            else:
                label = "-"
            lines.append(f"- {event.event}: {label}")

        return "\n".join(lines)

    # ── Summary ────────────────────────────────────────────────────────────────

    @staticmethod
    def summary(result: Result[Any], *, include_cache_counts: bool = True) -> str:
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

    # ── Pipeline report ────────────────────────────────────────────────────────

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
            ResultQuery.node_tree(result, show_ids=show_ids, max_depth=max_depth, max_children=max_children),
        ]

        if include_execution_tree:
            execution_text = ResultQuery.execution_tree(
                result, show_ids=show_ids, max_depth=max_depth, max_children=max_children
            ).strip()
            if execution_text:
                sections.extend(["Execution", execution_text])

        if include_perf:
            perf_text = ResultRepr.perf_table(
                result, show_ids=show_ids, max_depth=max_depth, max_children=max_children
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
                    label = ResultQuery.node_label(node, show_ids=show_ids, show_ref=True, show_kind=True)
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
