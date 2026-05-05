"""Performance collection utilities for calculator phases.

This module contains the timing and memory helpers used to measure calculator
execution phases.  The engine records per-phase timing and optional memory
statistics, then aggregates them into :class:`PerfSummary` objects stored in the
final :class:`Result`.

Most users interact with performance data indirectly through run options and
result summaries rather than by instantiating the collector classes directly.

What This Module Provides
-------------------------
The key public pieces here are:

- :class:`PerfCollector`, which records phase timing and optional memory usage
- :class:`PerfFormatter`, which formats timing and memory values for display

How Performance Recording Is Used
---------------------------------
When performance collection is enabled, each node phase may record:

- wall-clock elapsed time
- memory change measured through ``tracemalloc``
- optional RSS change when ``psutil`` is available

These measurements are then folded into node-level phase records and the final
aggregate performance summary.

Typical User Workflow
---------------------
In normal analysis code, performance reporting is enabled through run options::

    result = calc.run(sim, progress="phase", perf_time=True, perf_memory=True)
    print(result.perf_summary)
    print(result.reports["perf"])

This is the recommended public-facing workflow.

Direct Collector Example
------------------------
Direct use of :class:`PerfCollector` is mainly useful in tests or internal
experiments::

    collector = PerfCollector()

    with collector.phase("example", measure_time=True, measure_memory=False) as record:
        total = sum(range(1000))

    print(record.elapsed_s)

Formatting Example
------------------
The formatter helpers are useful when building custom textual reports::

    print(PerfFormatter.format_time(0.0025))
    print(PerfFormatter.format_mem(1024 * 1024))

When This Module Matters
------------------------
This module is most relevant when you are:

- profiling a large calculator graph
- checking whether a shared dependency is a bottleneck
- writing custom performance reports
- testing framework-level timing and memory instrumentation

Notes
-----
Timing data is usually more portable than memory data.  Memory reporting depends
on the runtime environment and may differ depending on whether ``psutil`` is
installed and which Python allocator behavior is active.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

from .result import PerfSummary, PhaseRecord, ResultNode

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterator


def _get_rss() -> int | None:
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


class PerfFormatter:
    """Human-readable formatting helpers for timing and memory values."""

    @staticmethod
    def format_time(value: float | None) -> str:
        """Format seconds as a compact human-readable duration."""
        if value is None:
            return "-"
        if value < 1e-3:
            return f"{value * 1e6:.1f} us"
        if value < 1:
            return f"{value * 1e3:.2f} ms"
        if value < 60:
            return f"{value:.3f} s"
        return f"{value / 60:.2f} min"

    @staticmethod
    def format_mem(value: int | None) -> str:
        """Format bytes as a compact human-readable memory value."""
        if value is None:
            return "-"
        value_abs = abs(value)
        if value_abs < 1024:
            return f"{value:.1f} B"
        if value_abs < 1024**2:
            return f"{value / 1024:.1f} KiB"
        if value_abs < 1024**3:
            return f"{value / 1024**2:.2f} MiB"
        return f"{value / 1024**3:.2f} GiB"


@dataclass(slots=True)
class PerfCollector:
    """Collect timing and optional memory information for node phases."""

    tracemalloc_nframe: int = 1
    _tracing_started_here: bool = False

    @contextmanager
    def phase(
        self,
        phase_name: str,
        *,
        measure_time: bool = True,
        measure_memory: bool = False,
    ) -> Iterator[PhaseRecord]:
        """Context manager that records one phase.

        Parameters
        ----------
        phase_name : str
            Name stored in the resulting :class:`PhaseRecord`.
        measure_time, measure_memory : bool
            Enable wall-time and memory measurements.
        """
        record = PhaseRecord(phase=phase_name, started_at=time.perf_counter())
        mem_start = None
        rss_start = None

        try:
            if measure_memory:
                if not tracemalloc.is_tracing():
                    tracemalloc.start(self.tracemalloc_nframe)
                    self._tracing_started_here = True
                mem_start, _ = tracemalloc.get_traced_memory()
                tracemalloc.reset_peak()
                rss_start = _get_rss()

            yield record
            record.status = "ok"
        except Exception:
            record.status = "error"
            raise
        finally:
            record.finished_at = time.perf_counter()
            if measure_time:
                record.elapsed_s = record.finished_at - record.started_at

            if measure_memory:
                mem_end, mem_peak = tracemalloc.get_traced_memory()
                record.memory_used = None if mem_start is None else mem_end - mem_start
                record.memory_peak = None if mem_start is None else mem_peak - mem_start
                rss_end = _get_rss()
                record.rss_used = None if rss_start is None or rss_end is None else rss_end - rss_start

                if self._tracing_started_here:
                    tracemalloc.stop()
                    self._tracing_started_here = False

    def summary(self, nodes: dict[str, ResultNode], cache_hit_count: int = 0) -> PerfSummary:
        """Return aggregate performance counters for evaluated nodes."""
        started: list[float] = []
        finished: list[float] = []

        for node in nodes.values():
            for phase in node.phases:
                started.append(phase.started_at)
                if phase.finished_at is not None:
                    finished.append(phase.finished_at)

        total_time = None
        if started and finished:
            total_time = max(finished) - min(started)

        return PerfSummary(
            total_time_s=total_time,
            node_count=len(nodes),
            cache_hit_count=cache_hit_count,
        )

    def report_text(self, nodes: dict[str, ResultNode], title: str = "") -> str:
        """Render a text performance table."""
        lines: list[str] = []
        if title:
            lines.append(title)

        header = "Node                           | Phase           | Time         | Mem Used       | Peak Mem       | RSS Delta"
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))

        for node in nodes.values():
            node_name = node.name or node.kind
            node_label = f"{node_name} [{node.node_id}]"
            for phase in node.phases:
                lines.append(
                    f"{node_label[:30]:<30} | "
                    f"{phase.phase[:15]:<15} | "
                    f"{PerfFormatter.format_time(phase.elapsed_s):>12} | "
                    f"{PerfFormatter.format_mem(phase.memory_used):>14} | "
                    f"{PerfFormatter.format_mem(phase.memory_peak):>14} | "
                    f"{PerfFormatter.format_mem(phase.rss_used):>10}"
                )

        lines.append("-" * len(header))
        summary = self.summary(nodes)
        lines.append(
            f"{'Total':<30} | {'-':<15} | "
            f"{PerfFormatter.format_time(summary.total_time_s):>12} | "
            f"{'-':>14} | {'-':>14} | {'-':>10}"
        )
        lines.append("-" * len(header))
        return "\n".join(lines)

    def report(self, nodes: dict[str, ResultNode], logger: logging.Logger | None = None, title: str = "") -> str:
        """Return the performance report and optionally log it."""
        text = self.report_text(nodes, title=title)
        if logger is not None:
            logger.info("\n%s", text)
        return text
