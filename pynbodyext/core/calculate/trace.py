"""Trace event collection for calculator evaluation.

The trace system records a lightweight event stream describing what the engine
did during one run.  This includes phase entry and exit events as well as cache
events, all annotated with timestamps, node ids, node names, and nesting depth.

Tracing is primarily a debugging and introspection tool.  It helps answer
questions such as:

- which node ran first
- which phases were entered for a given node
- where cache hits occurred
- how deeply nested a particular execution path became

What This Module Provides
-------------------------
The main public types here are:

- :class:`TraceEvent`, one structured trace record
- :class:`TraceCollector`, the per-run event collector used by the engine

How Trace Differs From ResultNode
---------------------------------
A :class:`ResultNode` describes the final recorded outcome of one node.
A :class:`TraceEvent` describes something that happened during execution.

In other words:

- result nodes are structural summaries
- trace events are runtime timeline records

This makes trace especially useful when debugging execution order and phase
boundaries.

Indirect Usage Example
----------------------
Most users access trace data through the final :class:`Result`::

    result = calc.run(sim, progress="phase")
    print(result.reports["trace_timeline"])
    print(result.reports["trace_tree"])

The timeline is useful for sequence debugging, while the tree is useful for
understanding graph structure.

Direct Usage Example
--------------------
Direct use of :class:`TraceCollector` is mostly relevant in tests or internal
tooling::

    collector = TraceCollector()
    print(collector.render_timeline())

Typical Questions Trace Helps Answer
------------------------------------
Trace output is especially useful when you want to know:

- why a dependency appeared to run more than once
- whether a phase was entered before an exception occurred
- how filter and transform scope changed execution depth
- whether cache reuse actually happened where expected

When This Module Matters
------------------------
This module is most relevant when you are:

- debugging a new calculator subclass
- inspecting graph execution order
- comparing two implementations of the same analysis
- building notebook or logging tools on top of trace output

Notes
-----
Trace is intentionally lightweight and text-friendly.  For most debugging
sessions, the rendered reports are easier to read than the raw event list, but
both are available through the final :class:`Result`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterator

    from .result import ResultNode

@dataclass(slots=True)
class TraceEvent:
    """One trace event emitted by the evaluation engine."""

    timestamp: float
    node_id: str
    node_name: str
    phase: str
    event: str
    depth: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceCollector:
    """Collect phase and cache trace events during one run."""

    events: list[TraceEvent] = field(default_factory=list)
    _stack: list[str] = field(default_factory=list)

    @contextmanager
    def phase(self, node_id: str, node_name: str, phase: str) -> Iterator[None]:
        """Record enter/leave events around a node phase."""
        depth = len(self._stack)
        self.events.append(
            TraceEvent(
                timestamp=time.perf_counter(),
                node_id=node_id,
                node_name=node_name,
                phase=phase,
                event="enter",
                depth=depth,
            )
        )
        self._stack.append(node_id)
        try:
            yield
        finally:
            self._stack.pop()
            self.events.append(
                TraceEvent(
                    timestamp=time.perf_counter(),
                    node_id=node_id,
                    node_name=node_name,
                    phase=phase,
                    event="leave",
                    depth=depth,
                )
            )

    def cache(self, node_id: str, node_name: str, event: str, **payload: Any) -> None:
        """Record a cache event in the trace stream."""
        self.events.append(
            TraceEvent(
                timestamp=time.perf_counter(),
                node_id=node_id,
                node_name=node_name,
                phase="cache",
                event=event,
                depth=len(self._stack),
                payload=payload,
            )
        )

    def events_for(self, node_id: str) -> list[TraceEvent]:
        """Return trace events for a single node id."""
        return [event for event in self.events if event.node_id == node_id]

    def render_timeline(self) -> str:
        """Return trace events sorted by timestamp as text."""
        lines: list[str] = []
        for event in sorted(self.events, key=lambda item: item.timestamp):
            indent = "  " * event.depth
            payload = f" {event.payload}" if event.payload else ""
            lines.append(
                f"{indent}{event.node_name} [{event.node_id}] "
                f"{event.phase}:{event.event}{payload}"
            )
        return "\n".join(lines)

    def render_tree(self, nodes: dict[str, ResultNode], root_id: str) -> str:
        """Render a result-node tree with node ids."""
        def render(node_id: str, prefix: str, is_last: bool) -> list[str]:
            node = nodes[node_id]
            branch = "└─" if is_last else "├─"
            label = node.label
            lines = [f"{prefix}{branch} {label} [{node.node_id}]"]
            child_prefix = prefix + ("   " if is_last else "│  ")
            for index, child_id in enumerate(node.children):
                lines.extend(render(child_id, child_prefix, index == len(node.children) - 1))
            return lines

        return "\n".join(render(root_id, "", True))

    def report_text(self, nodes: dict[str, ResultNode] | None = None, root_id: str | None = None) -> str:
        """Render trace timeline and optionally a tree section."""
        parts = ["Trace Timeline", self.render_timeline()]
        if nodes is not None and root_id is not None and root_id in nodes:
            parts.extend(["", "Trace Tree", self.render_tree(nodes, root_id)])
        return "\n".join(parts)

    def report(
        self,
        logger: logging.Logger | None = None,
        nodes: dict[str, ResultNode] | None = None,
        root_id: str | None = None,
    ) -> str:
        """Return the trace report and optionally log it."""
        text = self.report_text(nodes=nodes, root_id=root_id)
        if logger is not None:
            logger.info("\n%s", text)
        return text
