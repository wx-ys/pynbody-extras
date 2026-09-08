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

from pynbodyext.core.calculate.display import ViewObject, compact_repr


class NamedView(ViewObject, dict[str, Any]):
    """A ``Result.named`` view that still behaves like the previous dict."""

    __hash__ = None

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        dict.__init__(self, data or {})
        self._data = dict(self)

    def _title(self) -> str:
        return "Named"

    def _summary(self) -> str:
        return f"named={tuple(self._data.keys())!r}"

    def _sections(self) -> list[tuple[str | None, str]]:
        return [(None, ", ".join(compact_repr(value, max_length=80) for value in self._data.values()))]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NamedView):
            return dict(self._data) == dict(other._data)
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented


class ErrorListView(ViewObject, list[Any]):
    """A ``Result.errors`` view that still behaves like the previous list."""

    __hash__ = None

    def __init__(self, data: list[Any] | None = None) -> None:
        list.__init__(self, data or [])
        self._data = list(self)

    def _title(self) -> str:
        return "Errors"

    def _summary(self) -> str:
        return f"errors={len(self._data)}"

    def _sections(self) -> list[tuple[str | None, str]]:
        return [(None, "\n".join(str(e) for e in self._data))]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ErrorListView):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return NotImplemented


class WarningListView(ViewObject, list[Any]):
    """A ``Result.warnings`` view that still behaves like the previous list."""

    __hash__ = None

    def __init__(self, data: list[Any] | None = None) -> None:
        list.__init__(self, data or [])
        self._data = list(self)

    def _title(self) -> str:
        return "Warnings"

    def _summary(self) -> str:
        return f"warnings={len(self._data)}"

    def _sections(self) -> list[tuple[str | None, str]]:
        return [(None, "\n".join(str(w) for w in self._data))]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WarningListView):
            return self._data == other._data
        if isinstance(other, list):
            return self._data == other
        return NotImplemented
