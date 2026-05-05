"""Shared display helpers for calculator objects."""

from __future__ import annotations

from html import escape
from typing import Any

HTML_STYLE = """
<style>
.pynbodyext-calc-card {
    display: inline-block;
    box-sizing: border-box;
    min-width: min(320px, 100%);
    max-width: min(720px, 100%);
    margin: 0.25rem 0;
    padding: 0.72rem 0.82rem;
    border: 1px solid rgba(127, 127, 127, 0.28);
    border-radius: 7px;
    background: rgba(127, 127, 127, 0.045);
    color: inherit;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.35;
}
.pynbodyext-calc-title {
    margin: 0 0 0.45rem;
    font-weight: 650;
    color: inherit;
}
.pynbodyext-calc-table {
    border-collapse: collapse;
    width: auto;
    max-width: 100%;
    margin: 0;
}
.pynbodyext-calc-table th {
    min-width: 6.4rem;
    padding: 2px 1.25rem 2px 0;
    text-align: left;
    vertical-align: top;
    white-space: nowrap;
    color: inherit;
    opacity: 0.68;
    font-weight: 500;
}
.pynbodyext-calc-table td {
    max-width: 34rem;
    padding: 2px 0;
    vertical-align: top;
    color: inherit;
    overflow-wrap: anywhere;
}
.pynbodyext-calc-pre {
    box-sizing: border-box;
    max-width: 100%;
    max-height: 16rem;
    margin: 0.62rem 0 0;
    padding: 0.55rem 0.65rem;
    border: 1px solid rgba(127, 127, 127, 0.22);
    border-radius: 5px;
    background: rgba(127, 127, 127, 0.08);
    color: inherit;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    line-height: 1.35;
    white-space: pre;
}
.pynbodyext-calc-section-title {
    margin: 0.68rem 0 0.2rem;
    font-weight: 650;
    color: inherit;
}
</style>
"""


def compact_repr(value: Any, *, max_length: int = 80) -> str:
    """Return a compact single-line string representation of a value, truncating if necessary."""
    try:
        text = repr(value)
    except Exception:
        text = f"<{value.__class__.__name__}>"
    text = " ".join(text.splitlines())
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def display_value(value: Any) -> Any:
    """Return value.value if it exists, otherwise value itself."""
    return getattr(value, "value", value)


def html_escape(value: Any) -> str:
    """Return an HTML-escaped string representation of value."""
    return escape(str(value), quote=True)


def html_table(rows: list[tuple[str, Any]]) -> str:
    """Return an HTML table for the given rows."""
    body = "".join(
        "<tr>"
        f"<th>{html_escape(key)}</th>"
        f"<td>{html_escape(value)}</td>"
        "</tr>"
        for key, value in rows
    )
    return f"<table class='pynbodyext-calc-table'>{body}</table>"


def html_pre(text: str) -> str:
    """Return an HTML <pre> block with the given text."""
    return f"<pre class='pynbodyext-calc-pre'>{html_escape(text)}</pre>"


def html_card(title: str, rows: list[tuple[str, Any]], *, body: str = "", html_style: str = HTML_STYLE) -> str:
    """Return an HTML card with a title, table of rows, and optional body."""
    return (
        f"{html_style}"
        "<div class='pynbodyext-calc-card'>"
        f"<div class='pynbodyext-calc-title'>{html_escape(title)}</div>"
        f"{html_table(rows)}"
        f"{body}"
        "</div>"
    )


def mimebundle(text: str, html: str) -> dict[str, str]:
    """Return a MIME bundle with plain text and HTML representations."""
    return {"text/plain": text, "text/html": html}


def format_time(value: float | None) -> str:
    """Format a time value in a human-friendly unit."""
    if value is None:
        return "-"
    if value < 1e-3:
        return f"{value * 1e6:.1f} us"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    if value < 60:
        return f"{value:.3f} s"
    return f"{value / 60:.2f} min"


def format_mem(value: int | None) -> str:
    """Format a memory size in a human-friendly unit."""
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


__all__ = [
    "HTML_STYLE",
    "compact_repr",
    "display_value",
    "html_escape",
    "html_table",
    "html_pre",
    "html_card",
    "mimebundle",
    "format_time",
    "format_mem",
]
