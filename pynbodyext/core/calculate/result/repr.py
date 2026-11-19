"""Cards and text summaries for :class:`Result` and :class:`ResultNode`.

:class:`ResultRepr` is a stateless helper (static methods only).  The HTML of a
result object is described once, as a :class:`~pynbodyext.core.calculate.display.Card`
(``result_card`` / ``result_node_card``); the active repr style turns the card into
markup, so the ``rich`` and ``simple`` views always carry the same fields.

Card sections:

- node: Value, Phases, Observation, Observation fields, Error, Traceback
- result: Named values, Warnings, Errors, Provenance, Execution tree, Performance,
  Cache — each a drill-down named after the attribute that shows it
  (``result.execution_tree``), which the ``simple`` card lists instead of inlining.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.display import (
    Badge,
    Block,
    Card,
    Code,
    Fields,
    Metrics,
    Pre,
    Section,
    Table,
    compact_repr,
    display_value,
    format_mem,
    format_time,
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
    def _status_badge(status: Any) -> Badge:
        return Badge(display_value(status), ResultRepr._tone_for_status(status))

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

    # ── ResultNode ─────────────────────────────────────────────────────────────

    @staticmethod
    def result_node_repr(node: ResultNode) -> str:
        parts = [
            f"label={node.label!r}",
            f"kind={display_value(node.kind)!r}",
            f"status={display_value(node.status)!r}",
        ]
        if node.record.stored_value:
            parts.append(f"value={compact_repr(node.record.value, max_length=60)}")
        elif node.record.value_summary is not None:
            parts.append(f"summary={compact_repr(node.record.value_summary, max_length=80)}")
        if node.children:
            parts.append(f"children={len(node.children)}")
        if node.run.error is not None:
            parts.append(f"error={node.run.error.error_type!r}")
        if node.run.observation is not None and node.run.observation.event_count:
            parts.append(f"observer_events={node.run.observation.event_count}")
        return f"ResultNode({', '.join(parts)})"

    @staticmethod
    def _node_value_blocks(node: ResultNode) -> list[Block]:
        value_rows: list[tuple[str, Any]] = []
        if node.record.value_summary is not None:
            # One source of truth: the summary renders its own rows, so the node
            # card and a standalone ``node.record.value_summary`` always agree.
            value_rows.extend(node.record.value_summary._display_rows())
        if node.record.stored_value:
            value_rows.append(("public value", compact_repr(node.record.value, max_length=220)))
        if node.record.stored_raw:
            value_rows.append(("raw value", compact_repr(node.record.raw_value, max_length=220)))
        if not value_rows:
            return []
        return [Section("Value", [Fields(value_rows, nowrap=True, monospace=True, min_width="56rem")])]

    @staticmethod
    def _node_phase_blocks(node: ResultNode) -> list[Block]:
        if not node.run.phases:
            return []
        rows = [
            [
                phase.phase,
                format_time(phase.elapsed_s),
                format_mem(phase.memory_used),
                format_mem(phase.memory_peak),
                format_mem(phase.rss_used),
                ResultRepr._status_badge(phase.status),
            ]
            for phase in node.run.phases
        ]
        table = Table(
            ["phase", "time", "mem", "peak", "rss", "status"], rows, nowrap=True, monospace=True, min_width="48rem"
        )
        return [Section("Phases", [table])]

    @staticmethod
    def _node_observer_blocks(node: ResultNode) -> list[Block]:
        observation = node.run.observation
        if observation is None:
            return []
        blocks: list[Block] = [
            Section(
                "Observation",
                [Fields(observation._display_rows(), nowrap=True, monospace=True, min_width="42rem")],
            )
        ]
        lines: list[str] = []
        if observation.reads:
            lines.append(f"reads: {ResultRepr._format_field_set(observation.reads)}")
        if observation.dirty_fields:
            lines.append(f"dirty: {ResultRepr._format_field_set(observation.dirty_fields)}")
        if observation.deletes:
            lines.append(f"deletes: {ResultRepr._format_field_set(observation.deletes)}")
        if lines:
            blocks.append(Section("Observation fields", [Pre("\n".join(lines))]))
        return blocks

    @staticmethod
    def _node_error_blocks(node: ResultNode) -> list[Block]:
        error = node.run.error
        if error is None:
            return []
        blocks: list[Block] = [Section("Error", [Fields(error._display_rows(), nowrap=True)], collapsible=False)]
        if error.traceback_text:
            blocks.append(Section("Traceback", [Pre(error.traceback_text)], open=True))
        return blocks

    @staticmethod
    def result_node_card(node: ResultNode) -> Card:
        record = node.record
        rows: list[tuple[str, Any]] = [
            ("label", node.label),
            ("ref", node.ref),
            ("kind", Badge(display_value(node.kind), "info")),
            ("status", ResultRepr._status_badge(node.status)),
            (
                "stored",
                (
                    Badge("value" if record.stored_value else "value: no", "ok" if record.stored_value else "neutral"),
                    Badge("raw" if record.stored_raw else "raw: no", "ok" if record.stored_raw else "neutral"),
                ),
            ),
        ]
        if node.class_info.calculator_type is not None:
            rows.append(("calculator", node.class_info.calculator_type))
        semantic_class = node.class_info.semantic_calculator_class_path or node.class_info.calculator_class_path
        if semantic_class is not None:
            rows.append(("class", ResultRepr._short_class_name(semantic_class)))
        if record.policy is not None:
            rows.append(("record", Badge(display_value(record.policy))))

        metrics = Metrics(
            [
                ("parents", len(node.parent_ids)),
                ("children", len(node.children)),
                ("phases", len(node.run.phases)),
                ("events", node.run.observation.event_count if node.run.observation is not None else 0),
            ]
        )
        blocks: list[Block] = [
            metrics,
            *ResultRepr._node_value_blocks(node),
            *ResultRepr._node_phase_blocks(node),
            *ResultRepr._node_observer_blocks(node),
            *ResultRepr._node_error_blocks(node),
        ]
        return Card("ResultNode", rows, blocks)

    @staticmethod
    def result_node_html(node: ResultNode) -> str:
        return ResultRepr.result_node_card(node).html()

    # ── Result ─────────────────────────────────────────────────────────────────

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

    @staticmethod
    def _result_metrics(result: Result[Any]) -> Metrics:
        perf = result.perf_summary
        hits, misses = perf.cache_hit_count, perf.cache_miss_count
        cache_total = hits + misses
        return Metrics(
            [
                ("nodes", len(result.nodes)),
                ("phases", perf.phase_count),
                ("warnings", len(result.warnings)),
                ("errors", len(result.errors)),
                ("time", format_time(perf.total_time_s)),
                ("cache", f"{hits} hit / {misses} miss"),
                ("hit rate", f"{hits / cache_total:.0%}" if cache_total else "-"),
                ("stores", perf.cache_store_count),
            ],
            strip=True,
        )

    @staticmethod
    def _result_error_rows(result: Result[Any]) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for node in ResultQuery.find_error_nodes(result)[:8]:
            if node.run.error is None:
                continue
            rows.append(
                [
                    ResultQuery.node_label(node, show_ref=True, show_kind=True),
                    node.run.error.phase or "-",
                    f"{node.run.error.error_type}: {node.run.error.message}",
                ]
            )
        for error in result.errors[: max(0, 8 - len(rows))]:
            rows.append(["<run>", error.phase or "-", f"{error.error_type}: {error.message}"])
        return rows

    @staticmethod
    def _result_sections(result: Result[Any]) -> list[Block]:
        """The drill-down sections, each named after the attribute that shows it."""
        sections: list[Block] = []
        if result.diagnostics.get("named_values"):
            sections.append(
                Section.drilldown(
                    "Named values",
                    "named_values",
                    lambda: Fields(result.named_values._display_rows(), nowrap=True),
                    collapsible=False,
                )
            )
        if result.warnings:
            sections.append(
                Section.drilldown(
                    "Warnings",
                    "warnings",
                    lambda: Table(["#", "message"], list(enumerate(map(str, result.warnings), start=1)), nowrap=True),
                    collapsible=False,
                )
            )
        if result.errors or any(node.run.error is not None for node in result.nodes.values()):
            sections.append(
                Section.drilldown(
                    "Errors",
                    "errors",
                    lambda: Table(["node", "phase", "message"], ResultRepr._result_error_rows(result), nowrap=True),
                    collapsible=False,
                )
            )
        provenance = result.provenance
        if provenance is not None:
            sections.append(
                Section.drilldown("Provenance", "provenance", lambda: Fields(provenance._display_rows(), nowrap=True))
            )
        sections.extend(
            [
                Section.drilldown(
                    "Execution tree", "execution_tree", lambda: Pre(result.execution_tree), open=not result.ok
                ),
                Section.drilldown("Performance", "performance", lambda: Pre(result.performance.strip())),
                Section.drilldown("Cache", "cache", lambda: Pre(result.cache.strip())),
            ]
        )
        return sections

    @staticmethod
    def result_card(result: Result[Any]) -> Card:
        rows: list[tuple[str, Any]] = [
            ("root", result.root.label),
            ("value", type(result.value).__name__),
            ("status", Badge("ok", "ok") if result.ok else Badge("error", "error")),
        ]
        if result.named:
            rows.append(("named", compact_repr(tuple(result.named.keys()), max_length=96)))
        if result.provenance is not None and result.provenance.calculator_signature_hash is not None:
            rows.append(("signature", Code(result.provenance.calculator_signature_hash[:12])))
        return Card("Result", rows, [ResultRepr._result_metrics(result), *ResultRepr._result_sections(result)])

    @staticmethod
    def result_html(result: Result[Any]) -> str:
        return ResultRepr.result_card(result).html()

    # ── Reports ────────────────────────────────────────────────────────────────
    # The text builders live in ``reports.py``; these keep ``ResultRepr`` the one
    # presentation entry point ``Result.report_*()`` and callers already use.

    @staticmethod
    def perf_table(
        result: Result[Any], *, show_ids: bool = False, max_depth: int | None = None, max_children: int | None = None
    ) -> str:
        from .reports import perf_table

        return perf_table(result, show_ids=show_ids, max_depth=max_depth, max_children=max_children)

    @staticmethod
    def cache_section(result: Result[Any], *, max_events: int = 12) -> str:
        from .reports import cache_section

        return cache_section(result, max_events=max_events)

    @staticmethod
    def summary(result: Result[Any], *, include_cache_counts: bool = True) -> str:
        from .reports import summary

        return summary(result, include_cache_counts=include_cache_counts)

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
        from .reports import pipeline_report

        return pipeline_report(
            result,
            include_perf=include_perf,
            include_trace=include_trace,
            include_cache=include_cache,
            include_errors=include_errors,
            include_execution_tree=include_execution_tree,
            show_ids=show_ids,
            max_depth=max_depth,
            max_children=max_children,
        )
