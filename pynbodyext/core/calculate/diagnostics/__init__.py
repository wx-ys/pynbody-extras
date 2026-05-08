"""Diagnostics helpers for calculator runs."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AccessEvent",
    "AccessObservation",
    "PerfCollector",
    "TraceCollector",
    "TraceEvent",
]


def __getattr__(name: str) -> Any:
    if name in {"AccessEvent", "AccessObservation"}:
        from .observer import AccessEvent, AccessObservation

        return {"AccessEvent": AccessEvent, "AccessObservation": AccessObservation}[name]
    if name == "PerfCollector":
        from .perf import PerfCollector

        return PerfCollector
    if name in {"TraceCollector", "TraceEvent"}:
        from .trace import TraceCollector, TraceEvent

        return {"TraceCollector": TraceCollector, "TraceEvent": TraceEvent}[name]
    raise AttributeError(name)
