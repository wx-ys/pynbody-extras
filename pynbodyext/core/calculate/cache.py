"""Per-run cache objects used by the calculator engine.

The runtime cache is scoped to a single calculator run.  Its purpose is to
avoid duplicate work when several parts of the graph depend on the same
calculator node under the same effective input state.

This module defines the lightweight cache objects used internally by the engine
and exposed indirectly through result diagnostics and cache reports.

What This Module Provides
-------------------------
The main public types here are:

- :class:`ExecutionValue`, which stores raw and public values for one node
- :class:`CacheEvent`, which records hits, misses, and stores
- :class:`RuntimeCache`, the in-memory cache used during one run

Why The Cache Is Per-Run
------------------------
The cache is intentionally short-lived.  It is not a persistent memoization
layer across sessions or across unrelated runs.  Instead, it exists to make one
graph execution efficient and inspectable.

This design keeps cache semantics simple:

- no stale cross-run state
- no hidden global memoization
- cache events are easy to attribute to one result

Basic Mental Model
------------------
If two nodes in the same run request the same dependency with the same scope and
signature, the engine can reuse the cached runtime value instead of evaluating
that dependency again.

This is especially important for pipelines and reusable intermediate calculators.

Indirect Usage Example
----------------------
Most users see the cache through result reports rather than by instantiating
:class:`RuntimeCache` directly::

    result = calc.run(sim)
    print(result.reports["cache"])
    print(result.perf_summary.cache_hit_count)

Direct Usage Example
--------------------
Framework tests may sometimes inspect the cache class directly::

    cache = RuntimeCache(enabled=True)
    summary = cache.summary()
    print(summary["entries"])
    print(summary["hits"])

Reading Cache Diagnostics
-------------------------
Each cache access records a :class:`CacheEvent`.  These events are useful when
you want to understand whether a shared dependency was reused or recomputed.

The final :class:`Result` exposes those events through diagnostics, while the
human-readable cache report offers a quick summary.

When This Module Matters
------------------------
This module is most relevant when you are:

- debugging repeated evaluation of shared dependencies
- understanding cache-related performance behavior
- writing engine-level tests
- building custom debugging tools on top of cache diagnostics

Notes
-----
Most end users do not need to import this module directly.  It is part of the
public runtime model, but its main purpose is to support engine behavior,
reporting, and debugging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging


@dataclass(slots=True)
class ExecutionValue:
    """Raw and public values stored for one evaluated node."""

    node_id: str
    raw_value: Any
    public_value: Any


@dataclass(slots=True)
class CacheEvent:
    """Diagnostic event emitted for cache hit, miss, and store actions."""

    timestamp: float
    event: str
    key: tuple[Any, ...]
    node_id: str | None = None


@dataclass(slots=True)
class RuntimeCache:
    """Simple in-memory cache used during one evaluation run."""

    enabled: bool = True
    store: dict[tuple[Any, ...], ExecutionValue] = field(default_factory=dict)
    events: list[CacheEvent] = field(default_factory=list)
    hit_count: int = 0

    def get(self, key: tuple[Any, ...]) -> ExecutionValue | None:
        """Return a cached value and record a hit or miss event."""
        if not self.enabled:
            return None
        hit = self.store.get(key)
        self.events.append(
            CacheEvent(
                timestamp=time.perf_counter(),
                event="hit" if hit is not None else "miss",
                key=key,
                node_id=hit.node_id if hit is not None else None,
            )
        )
        if hit is not None:
            self.hit_count += 1
        return hit

    def set(self, key: tuple[Any, ...], value: ExecutionValue) -> None:
        """Store a value and record a store event."""
        if not self.enabled:
            return
        self.store[key] = value
        self.events.append(
            CacheEvent(
                timestamp=time.perf_counter(),
                event="store",
                key=key,
                node_id=value.node_id,
            )
        )

    def clear(self) -> None:
        """Remove all cached values and diagnostics."""
        self.store.clear()
        self.events.clear()
        self.hit_count = 0

    def summary(self) -> dict[str, Any]:
        """Return cache counters and entry count."""
        miss_count = sum(1 for event in self.events if event.event == "miss")
        store_count = sum(1 for event in self.events if event.event == "store")
        return {
            "enabled": self.enabled,
            "entries": len(self.store),
            "hits": self.hit_count,
            "misses": miss_count,
            "stores": store_count,
        }

    def report_text(self) -> str:
        """Render cache counters as text."""
        info = self.summary()
        lines = [
            "Runtime Cache",
            f"enabled: {info['enabled']}",
            f"entries: {info['entries']}",
            f"hits: {info['hits']}",
            f"misses: {info['misses']}",
            f"stores: {info['stores']}",
        ]
        return "\n".join(lines)

    def report(self, logger: logging.Logger | None = None) -> str:
        """Return the cache report and optionally log it."""
        text = self.report_text()
        if logger is not None:
            logger.info("\n%s", text)
        return text
