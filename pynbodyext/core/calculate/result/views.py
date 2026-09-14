"""Duck-typed view objects for :class:`Result` data members.

``Result.named`` / ``.errors`` / ``.warnings`` historically behave exactly like
a ``dict`` and a ``list``, but the presentation layer also wants to render them
as sectioned views across the three display styles.  Rather than change those
fields to a different type, this module wraps the existing dict/list in a
``ViewObject`` subclass that *keeps* the dict/list protocol (``__getitem__``,
``keys``, ``items``, ``__iter__``, ``__len__``, ``__bool__``, ``__eq__``) while
also being renderable by :class:`~pynbodyext.core.calculate.display.ViewObject`.

This module is intentionally a *leaf* (it imports only from ``display``) so both
``result.py`` and ``result/repr.py`` can import it without creating an import
cycle: ``repr.py`` -> ``query.py`` -> ``result.py``.
"""

from __future__ import annotations

from typing import Any

from pynbodyext.core.calculate.display import InfoView, ViewObject, compact_repr


def _line_count(text: str) -> str:
    lines = len(text.splitlines())
    return f"{lines} line" if lines == 1 else f"{lines} lines"


def as_view(value: Any, view_cls: Any, **kwargs: Any) -> Any:
    """Return *value* wrapped in *view_cls* unless it already is one.

    Typed loosely so ``Result.__post_init__`` can defensively wrap raw
    dicts/lists without tripping unreachable-branch checks.
    """
    return value if isinstance(value, view_cls) else view_cls(value, **kwargs)


class NamedView(ViewObject, dict[str, Any]):
    """A ``Result.named`` view that still behaves like the previous dict."""

    __hash__ = None

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        dict.__init__(self, data or {})

    def _title(self) -> str:
        return "Named"

    def _summary(self) -> str:
        return f"named={tuple(self.keys())!r}"

    def _sections(self) -> list[tuple[str | None, str]]:
        return [(None, ", ".join(compact_repr(value, max_length=80) for value in self.values()))]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NamedView):
            return dict(self) == dict(other)
        if isinstance(other, dict):
            return dict(self) == other
        return NotImplemented


class ErrorListView(ViewObject, list[Any]):
    """A ``Result.errors`` view that still behaves like the previous list."""

    __hash__ = None

    def __init__(self, data: list[Any] | None = None) -> None:
        list.__init__(self, data or [])

    def _title(self) -> str:
        return "Errors"

    def _summary(self) -> str:
        return f"errors={len(self)}"

    def _sections(self) -> list[tuple[str | None, str]]:
        return [(None, "\n".join(str(e) for e in self))]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ErrorListView):
            return list(self) == list(other)
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented


class WarningListView(ViewObject, list[Any]):
    """A ``Result.warnings`` view that still behaves like the previous list."""

    __hash__ = None

    def __init__(self, data: list[Any] | None = None) -> None:
        list.__init__(self, data or [])

    def _title(self) -> str:
        return "Warnings"

    def _summary(self) -> str:
        return f"warnings={len(self)}"

    def _sections(self) -> list[tuple[str | None, str]]:
        return [(None, "\n".join(str(w) for w in self))]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WarningListView):
            return list(self) == list(other)
        if isinstance(other, list):
            return list(self) == other
        return NotImplemented


class ReportsView(InfoView, dict[str, str]):
    """``Result.reports``: ``{name: report_text}`` with a :meth:`names` helper.

    Behaves like the previous plain ``dict`` — the report texts are still the
    values, reachable as ``result.reports[name]`` — but renders as a compact
    table of report names instead of dumping every report body into a repr.
    """

    __hash__ = None

    def __init__(self, data: dict[str, str] | None = None) -> None:
        dict.__init__(self, data or {})

    def names(self) -> tuple[str, ...]:
        """Names of the available reports."""
        return tuple(self.keys())

    def _display_title(self) -> str:
        return "Reports"

    def _display_rows(self) -> list[tuple[str, Any]]:
        if not self:
            return [("reports", "-")]
        return [(name, _line_count(text)) for name, text in self.items()]


class DiagnosticsView(InfoView, dict[str, Any]):
    """``Result.diagnostics``: raw diagnostic payloads plus typed accessors.

    Renders as a compact table of stream name -> payload size instead of dumping
    every raw event; the payloads stay reachable through the typed accessors.
    """

    __hash__ = None

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        dict.__init__(self, data or {})

    def names(self) -> tuple[str, ...]:
        """Names of the available diagnostic payloads."""
        return tuple(self.keys())

    def _display_title(self) -> str:
        return "Diagnostics"

    def _display_rows(self) -> list[tuple[str, Any]]:
        if not self:
            return [("streams", "-")]
        rows: list[tuple[str, Any]] = []
        for name, payload in self.items():
            size: Any = len(payload) if isinstance(payload, (list, dict, tuple, set)) else "-"
            rows.append((name, size))
        return rows

    def trace(self) -> list[Any]:
        """Raw trace events (``"trace_events"`` payload)."""
        return list(self.get("trace_events", []))

    def cache(self) -> list[Any]:
        """Raw cache events (``"cache_events"`` payload)."""
        return list(self.get("cache_events", []))

    def logs(self) -> list[Any]:
        """Runtime log events (``"log_events"`` payload)."""
        return list(self.get("log_events", []))

    def observer(self) -> list[Any]:
        """pynbody field-observer events (``"observer_events"`` payload)."""
        return list(self.get("observer_events", []))


class ObservationsView(InfoView, dict[str, Any]):
    """``Result.observations``: per-node field-access observations, ``.of(node)``.

    Keeps a back-reference to the owning :class:`Result` so a node *name* can be
    resolved to its observation.
    """

    __hash__ = None

    def __init__(self, data: dict[str, Any] | None = None, owner: Any = None) -> None:
        dict.__init__(self, data or {})
        self._owner = owner

    def _display_title(self) -> str:
        return "Observations"

    def _display_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = []
        for node_id, observation in self.items():
            label = getattr(observation, "node_label", None) or node_id
            events = len(getattr(observation, "events", ()) or ())
            rows.append((label, f"{events} event" if events == 1 else f"{events} events"))
        return rows or [("observations", "-")]

    def of(self, node: Any) -> Any:
        """Return the access observation for *node* (node id, name, or node object)."""
        from .result import ResultNode

        if isinstance(node, ResultNode):
            return self.get(node.node_id)
        if isinstance(node, str) and node not in self and self._owner is not None:
            try:
                return self.get(self._owner.node(node).node_id)
            except KeyError:
                return None
        return self.get(node)

    def all(self) -> dict[str, Any]:
        """Effective per-node observations, falling back to each node's own record."""
        if self:
            return dict(self)
        if self._owner is not None:
            return {
                node_id: node.observation
                for node_id, node in self._owner.nodes.items()
                if node.observation is not None
            }
        return {}
