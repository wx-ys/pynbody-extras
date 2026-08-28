"""Shared display helpers for calculator objects."""

from __future__ import annotations

from html import escape
from typing import Any

HTML_STYLE = """
<style>
.pynbodyext-calc-card {
    --pyn-calc-bg:
        var(--vscode-editor-background,
        var(--jp-layout-color0,
        #f7f8fa));

    --pyn-calc-panel:
        var(--vscode-editorWidget-background,
        var(--jp-layout-color1,
        rgba(255, 255, 255, 0.72)));

    --pyn-calc-panel-strong:
        var(--vscode-sideBar-background,
        var(--jp-layout-color1,
        rgba(255, 255, 255, 0.9)));

    --pyn-calc-panel-muted:
        color-mix(in srgb, var(--pyn-calc-text) 4%, transparent);

    --pyn-calc-border:
        var(--vscode-widget-border,
        var(--vscode-panel-border,
        var(--jp-border-color2,
        rgba(15, 23, 42, 0.1))));

    --pyn-calc-border-soft:
        color-mix(in srgb, var(--pyn-calc-text) 8%, transparent);

    --pyn-calc-text:
        var(--vscode-foreground,
        var(--jp-ui-font-color1,
        #1f2937));

    --pyn-calc-text-muted:
        var(--vscode-descriptionForeground,
        var(--jp-ui-font-color2,
        rgba(31, 41, 55, 0.64)));

    --pyn-calc-title:
        var(--vscode-editor-foreground,
        var(--jp-ui-font-color0,
        #0f172a));

    --pyn-calc-accent:
        var(--vscode-textLink-foreground,
        var(--jp-brand-color1,
        #2563eb));

    --pyn-calc-accent-soft:
        color-mix(in srgb, var(--pyn-calc-accent) 14%, transparent);

    --pyn-calc-ok:
        var(--jp-success-color1,
        #15803d);

    --pyn-calc-warn:
        var(--jp-warn-color1,
        #b45309);

    --pyn-calc-error:
        var(--jp-error-color1,
        #b91c1c);

    --pyn-calc-ok-soft:
        color-mix(in srgb, var(--pyn-calc-ok) 14%, transparent);

    --pyn-calc-warn-soft:
        color-mix(in srgb, var(--pyn-calc-warn) 14%, transparent);

    --pyn-calc-error-soft:
        color-mix(in srgb, var(--pyn-calc-error) 14%, transparent);

    --pyn-calc-shadow:
        0 1px 2px color-mix(in srgb, var(--pyn-calc-text) 6%, transparent),
        0 8px 24px color-mix(in srgb, var(--pyn-calc-text) 5%, transparent);

    display: block;
    box-sizing: border-box;
    max-width: min(640px, 100%);
    margin: 0.18rem 0;
    padding: 0.62rem 0.74rem;
    border: 1px solid var(--pyn-calc-border);
    border-radius: 10px;
    background:
        linear-gradient(
            180deg,
            color-mix(in srgb, var(--pyn-calc-panel-strong) 92%, var(--pyn-calc-bg)) 0%,
            color-mix(in srgb, var(--pyn-calc-panel) 96%, var(--pyn-calc-bg)) 100%
        );
    box-shadow: var(--pyn-calc-shadow);
    color: var(--pyn-calc-text);
    overflow: hidden;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.32;
}

@supports not (color: color-mix(in srgb, white 50%, black)) {
    .pynbodyext-calc-card {
        --pyn-calc-panel-muted: rgba(127, 127, 127, 0.06);
        --pyn-calc-border-soft: rgba(127, 127, 127, 0.08);
        --pyn-calc-accent-soft: rgba(37, 99, 235, 0.12);
        --pyn-calc-ok-soft: rgba(22, 163, 74, 0.12);
        --pyn-calc-warn-soft: rgba(217, 119, 6, 0.12);
        --pyn-calc-error-soft: rgba(220, 38, 38, 0.12);
        --pyn-calc-shadow:
            0 1px 2px rgba(15, 23, 42, 0.04),
            0 8px 24px rgba(15, 23, 42, 0.04);
    }
}

@media (prefers-color-scheme: dark) {
    .pynbodyext-calc-card {
        --pyn-calc-ok: #4ade80;
        --pyn-calc-warn: #fbbf24;
        --pyn-calc-error: #f87171;
    }
}

.pynbodyext-calc-title {
    margin: 0 0 0.32rem;
    color: var(--pyn-calc-title);
    font-weight: 750;
    letter-spacing: 0.01em;
}

.pynbodyext-calc-table,
.pynbodyext-calc-data-table {
    display: block;
    width: 100%;
    max-width: 100%;
}

.pynbodyext-calc-kv-row {
    display: grid;
    grid-template-columns: minmax(4.1rem, max-content) minmax(0, 1fr);
    column-gap: 0.62rem;
    align-items: start;
    padding: 2px 0;
}
.pynbodyext-calc-kv-row + .pynbodyext-calc-kv-row {
    border-top: 1px solid var(--pyn-calc-border-soft);
}
.pynbodyext-calc-kv-key {
    white-space: nowrap;
    color: var(--pyn-calc-text-muted);
    font-weight: 600;
    line-height: 1.28;
}
.pynbodyext-calc-kv-value {
    min-width: 0;
    color: var(--pyn-calc-text);
    overflow-wrap: anywhere;
    text-align: left;
    line-height: 1.28;
}

.pynbodyext-calc-data-head,
.pynbodyext-calc-data-row {
    display: grid;
    grid-template-columns: var(--pynbodyext-calc-columns, max-content);
    column-gap: 0.82rem;
    align-items: start;
    min-width: 100%;
}
.pynbodyext-calc-data-head {
    padding-bottom: 0.18rem;
    border-bottom: 1px solid var(--pyn-calc-border);
}
.pynbodyext-calc-data-row {
    padding: 2px 0;
}
.pynbodyext-calc-data-row + .pynbodyext-calc-data-row {
    border-top: 1px solid var(--pyn-calc-border-soft);
}
.pynbodyext-calc-data-head-cell {
    white-space: nowrap;
    color: var(--pyn-calc-text-muted);
    font-weight: 650;
}
.pynbodyext-calc-data-cell {
    color: var(--pyn-calc-text);
    overflow-wrap: anywhere;
    text-align: left;
}

.pynbodyext-calc-table-nowrap .pynbodyext-calc-kv-value,
.pynbodyext-calc-data-table-nowrap .pynbodyext-calc-data-head-cell,
.pynbodyext-calc-data-table-nowrap .pynbodyext-calc-data-cell {
    white-space: nowrap;
    overflow-wrap: normal;
    word-break: normal;
}

.pynbodyext-calc-monospace .pynbodyext-calc-kv-value,
.pynbodyext-calc-monospace .pynbodyext-calc-data-cell {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    line-height: 1.33;
}

.pynbodyext-calc-scroll-x {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    overflow-y: hidden;
    scrollbar-gutter: stable;
    padding-bottom: 0.08rem;
}
.pynbodyext-calc-scroll-x-inner {
    display: inline-block;
    width: max-content;
    min-width: 100%;
    vertical-align: top;
}
.pynbodyext-calc-scroll-x-inner > .pynbodyext-calc-table,
.pynbodyext-calc-scroll-x-inner > .pynbodyext-calc-data-table {
    width: max-content;
    min-width: 100%;
    max-width: none;
}
.pynbodyext-calc-scroll-x-inner > .pynbodyext-calc-metric-strip {
    display: inline-flex;
    min-width: 100%;
}

.pynbodyext-calc-pre {
    box-sizing: border-box;
    max-width: 100%;
    max-height: 18rem;
    margin: 0.58rem 0 0;
    padding: 0.52rem 0.62rem;
    border: 1px solid var(--pyn-calc-border);
    border-radius: 7px;
    background: var(--pyn-calc-panel-muted);
    color: var(--pyn-calc-text);
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 12px;
    line-height: 1.33;
    white-space: pre;
}

.pynbodyext-calc-section {
    margin-top: 0.56rem;
}
.pynbodyext-calc-section-title {
    margin: 0 0 0.16rem;
    color: var(--pyn-calc-title);
    font-weight: 720;
}

.pynbodyext-calc-details {
    margin-top: 0.56rem;
}
.pynbodyext-calc-details summary {
    cursor: pointer;
    color: var(--pyn-calc-text);
    font-weight: 650;
    user-select: none;
    line-height: 1.25;
}
.pynbodyext-calc-details summary:hover {
    color: var(--pyn-calc-accent);
}
.pynbodyext-calc-details[open] summary {
    color: var(--pyn-calc-title);
}

.pynbodyext-calc-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(82px, 1fr));
    gap: 0.35rem;
    margin-top: 0.68rem;
}
.pynbodyext-calc-metric-strip {
    display: inline-flex;
    gap: 0.35rem;
    margin-top: 0.68rem;
}
.pynbodyext-calc-metric {
    min-width: 82px;
    padding: 0.38rem 0.5rem;
    border: 1px solid var(--pyn-calc-border);
    border-radius: 7px;
    background: var(--pyn-calc-panel-muted);
}
.pynbodyext-calc-metric-label {
    font-size: 10px;
    color: var(--pyn-calc-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.pynbodyext-calc-metric-value {
    margin-top: 0.08rem;
    color: var(--pyn-calc-title);
    font-size: 13px;
    font-weight: 680;
    overflow-wrap: anywhere;
}

.pynbodyext-calc-badge {
    display: inline-block;
    margin: 0 0.22rem 0.22rem 0;
    padding: 0.1rem 0.42rem;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    line-height: 1.2;
    border: 1px solid transparent;
    vertical-align: middle;
}
.pynbodyext-calc-badge-neutral {
    background: color-mix(in srgb, var(--pyn-calc-text) 8%, transparent);
    border-color: var(--pyn-calc-border);
    color: var(--pyn-calc-text);
}
.pynbodyext-calc-badge-info {
    background: var(--pyn-calc-accent-soft);
    border-color: color-mix(in srgb, var(--pyn-calc-accent) 18%, transparent);
    color: var(--pyn-calc-accent);
}
.pynbodyext-calc-badge-ok {
    background: var(--pyn-calc-ok-soft);
    border-color: color-mix(in srgb, var(--pyn-calc-ok) 18%, transparent);
    color: var(--pyn-calc-ok);
}
.pynbodyext-calc-badge-warn {
    background: var(--pyn-calc-warn-soft);
    border-color: color-mix(in srgb, var(--pyn-calc-warn) 18%, transparent);
    color: var(--pyn-calc-warn);
}
.pynbodyext-calc-badge-error {
    background: var(--pyn-calc-error-soft);
    border-color: color-mix(in srgb, var(--pyn-calc-error) 18%, transparent);
    color: var(--pyn-calc-error);
}
</style>
"""

REPR_STYLES: tuple[str, ...] = ("rich", "github", "plain")
_REPR_STYLE: dict[str, str] = {"style": "rich"}


def get_repr_style() -> str:
    """Return the current display style (``"rich"``, ``"github"``, or ``"plain"``)."""
    return _REPR_STYLE["style"]


def set_repr_style(style: str) -> None:
    """Set the global display style used by ``_repr_mimebundle_`` / ``_repr_html_``.

    Parameters
    ----------
    style : {"rich", "github", "plain"}
        - ``"rich"`` (default): full HTML for Jupyter/VSCode (cards, collapsible
          details, badges, metric grids, embedded CSS).
        - ``"github"``: GitHub-sanitizer-safe HTML (``table``/``pre``/``code``/
          ``strong``/``p``), no ``<style>``/``<details>``/grid divs, so notebooks
          render cleanly on GitHub.
        - ``"plain"``: plain-text only (the HTML mime type is omitted).

    Examples
    --------
    >>> set_repr_style("github")
    >>> set_repr_style("rich")
    """
    if style not in REPR_STYLES:
        raise ValueError(f"unknown repr style {style!r}; expected one of {REPR_STYLES}")
    _REPR_STYLE["style"] = style


def reset_repr_style() -> None:
    """Reset the display style back to the default ``"rich"``."""
    _REPR_STYLE["style"] = "rich"


def _style() -> str:
    return _REPR_STYLE["style"]


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


def _html_value(value: Any, *, escape_values: bool) -> str:
    return html_escape(value) if escape_values else str(value)


def html_badge(text: Any, tone: str = "neutral") -> str:
    """Return a small badge pill."""
    safe_tone = tone if tone in {"neutral", "info", "ok", "warn", "error"} else "neutral"
    if _style() != "rich":
        return f"<code>{html_escape(text)}</code>"
    return f"<span class='pynbodyext-calc-badge pynbodyext-calc-badge-{safe_tone}'>{html_escape(text)}</span>"


def html_table(
    rows: list[tuple[str, Any]], *, escape_values: bool = True, class_name: str = "pynbodyext-calc-table"
) -> str:
    """Return a two-column div grid for the given rows."""
    if _style() != "rich":
        body = "".join(
            f"<tr><th>{html_escape(key)}</th><td>{_html_value(value, escape_values=escape_values)}</td></tr>"
            for key, value in rows
        )
        return f"<table class='{class_name}'><tbody>{body}</tbody></table>"

    body = "".join(
        "<div class='pynbodyext-calc-kv-row'>"
        f"<div class='pynbodyext-calc-kv-key'>{html_escape(key)}</div>"
        f"<div class='pynbodyext-calc-kv-value'>{_html_value(value, escape_values=escape_values)}</div>"
        "</div>"
        for key, value in rows
    )
    return f"<div class='{class_name}'>{body}</div>"


def html_data_table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    escape_values: bool = True,
    class_name: str = "pynbodyext-calc-data-table",
) -> str:
    """Return a multi-column div grid."""
    if _style() != "rich":
        head = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_html_value(value, escape_values=escape_values)}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return f"<table class='{class_name}'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    if headers:
        column_template = " ".join("max-content" for _ in headers)
    else:
        column_template = "max-content"

    header_html = "".join(
        f"<div class='pynbodyext-calc-data-head-cell'>{html_escape(header)}</div>" for header in headers
    )
    body_html = "".join(
        "<div class='pynbodyext-calc-data-row'>"
        + "".join(
            f"<div class='pynbodyext-calc-data-cell'>{_html_value(value, escape_values=escape_values)}</div>"
            for value in row
        )
        + "</div>"
        for row in rows
    )
    return (
        f"<div class='{class_name}' style='--pynbodyext-calc-columns:{html_escape(column_template)};'>"
        f"<div class='pynbodyext-calc-data-head'>{header_html}</div>"
        f"{body_html}"
        "</div>"
    )


def html_scroll_x(body: str, *, min_width: str | None = None) -> str:
    """Return a horizontal scroll container."""
    if _style() != "rich":
        return body

    inner_style = ""
    if min_width is not None:
        inner_style = f" style='min-width:{html_escape(min_width)};'"

    return (
        "<div class='pynbodyext-calc-scroll-x'>"
        f"<div class='pynbodyext-calc-scroll-x-inner'{inner_style}>{body}</div>"
        "</div>"
    )


def html_metric_grid(metrics: list[tuple[str, Any]], *, escape_values: bool = True) -> str:
    """Return a compact responsive metric grid."""
    if _style() != "rich":
        rows = "".join(
            f"<tr><th>{html_escape(label)}</th><td>{_html_value(value, escape_values=escape_values)}</td></tr>"
            for label, value in metrics
        )
        return f"<table class='pynbodyext-calc-metric-grid'><tbody>{rows}</tbody></table>"

    body = "".join(
        "<div class='pynbodyext-calc-metric'>"
        f"<div class='pynbodyext-calc-metric-label'>{html_escape(label)}</div>"
        f"<div class='pynbodyext-calc-metric-value'>{_html_value(value, escape_values=escape_values)}</div>"
        "</div>"
        for label, value in metrics
    )
    return f"<div class='pynbodyext-calc-metric-grid'>{body}</div>"


def html_metric_strip(metrics: list[tuple[str, Any]], *, escape_values: bool = True) -> str:
    """Return a single-row metric strip intended for horizontal scrolling."""
    if _style() != "rich":
        rows = "".join(
            f"<tr><th>{html_escape(label)}</th><td>{_html_value(value, escape_values=escape_values)}</td></tr>"
            for label, value in metrics
        )
        return f"<table class='pynbodyext-calc-metric-strip'><tbody>{rows}</tbody></table>"

    body = "".join(
        "<div class='pynbodyext-calc-metric'>"
        f"<div class='pynbodyext-calc-metric-label'>{html_escape(label)}</div>"
        f"<div class='pynbodyext-calc-metric-value'>{_html_value(value, escape_values=escape_values)}</div>"
        "</div>"
        for label, value in metrics
    )
    return f"<div class='pynbodyext-calc-metric-strip'>{body}</div>"


def html_section(title: str, body: str) -> str:
    """Return a titled content section."""
    if _style() != "rich":
        return f"<h5>{html_escape(title)}</h5>{body}"

    return (
        "<div class='pynbodyext-calc-section'>"
        f"<div class='pynbodyext-calc-section-title'>{html_escape(title)}</div>"
        f"{body}"
        "</div>"
    )


def html_details(summary: str, body: str, *, open: bool = False) -> str:
    """Return a collapsible section."""
    if _style() != "rich":
        return f"<p><strong>{html_escape(summary)}</strong></p>{body}"

    open_attr = " open" if open else ""
    return (
        f"<details class='pynbodyext-calc-details'{open_attr}><summary>{html_escape(summary)}</summary>{body}</details>"
    )


def html_pre(text: str) -> str:
    """Return an HTML <pre> block with the given text."""
    if _style() != "rich":
        return f"<pre>{html_escape(text)}</pre>"

    return f"<pre class='pynbodyext-calc-pre'>{html_escape(text)}</pre>"


def html_card(
    title: str, rows: list[tuple[str, Any]], *, body: str = "", html_style: str = HTML_STYLE, escape_values: bool = True
) -> str:
    """Return an HTML card with a title, table of rows, and optional body."""
    if _style() != "rich":
        return f"<p><strong>{html_escape(title)}</strong></p>{html_table(rows, escape_values=escape_values)}{body}"

    return (
        f"{html_style}"
        "<div class='pynbodyext-calc-card'>"
        f"<div class='pynbodyext-calc-title'>{html_escape(title)}</div>"
        f"{html_table(rows, escape_values=escape_values)}"
        f"{body}"
        "</div>"
    )


def mimebundle(text: str, html: str) -> dict[str, str]:
    """Return a MIME bundle with plain text and HTML representations."""
    if _style() == "plain":
        return {"text/plain": text}
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
    "REPR_STYLES",
    "get_repr_style",
    "set_repr_style",
    "reset_repr_style",
    "compact_repr",
    "display_value",
    "html_escape",
    "html_badge",
    "html_table",
    "html_data_table",
    "html_scroll_x",
    "html_metric_grid",
    "html_metric_strip",
    "html_section",
    "html_details",
    "html_pre",
    "html_card",
    "mimebundle",
    "format_time",
    "format_mem",
]
