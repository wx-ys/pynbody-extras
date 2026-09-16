"""Shared display helpers for calculator objects."""

from __future__ import annotations

from functools import lru_cache
from html import escape
from importlib import resources
from typing import Any


@lru_cache(maxsize=1)
def _card_css() -> str:
    """Return the card stylesheet text, read from ``display.css`` beside this module.

    The CSS lives in a real ``.css`` file so it gets editor support, linters and
    readable diffs; it is shipped as package data (``[tool.setuptools.package-data]``
    in ``pyproject.toml``) because wheels otherwise drop non-``.py`` files.
    """
    return resources.files(__package__).joinpath("display.css").read_text(encoding="utf-8")


def card_style() -> str:
    """Return the ``<style>`` block injected into every rich card.

    Notebook output has to be self-contained (there is no URL a ``<link>`` could
    point at, and the GitHub sanitizer strips external stylesheets), so the
    stylesheet is inlined once per rendered card.
    """
    return f"<style>\n{_card_css()}</style>\n"


HTML_STYLE = card_style()

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


def _github_value(value: Any, *, escape_values: bool) -> str:
    """Render a table cell value for the GitHub-safe style.

    Short scalar/token values are wrapped in ``<code>`` (monospace) so they read
    cleanly on GitHub; longer text is left plain; pre-rendered HTML (``escape_values
    = False``) is inserted unchanged.
    """
    if not escape_values:
        return str(value)
    text = str(value)
    if len(text) <= 40 and isinstance(value, (str, int, float, bool)):
        return f"<code>{html_escape(text)}</code>"
    return html_escape(text)


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
            f"<tr><th>{html_escape(key)}</th><td>{_github_value(value, escape_values=escape_values)}</td></tr>"
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
            "<tr>" + "".join(f"<td>{_github_value(value, escape_values=escape_values)}</td>" for value in row) + "</tr>"
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
        body = "".join(
            f"<tr><td>{html_escape(label)}</td><td>{_github_value(value, escape_values=escape_values)}</td></tr>"
            for label, value in metrics
        )
        return (
            "<table class='pynbodyext-calc-metric-grid'>"
            "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

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
        body = "".join(
            f"<tr><td>{html_escape(label)}</td><td>{_github_value(value, escape_values=escape_values)}</td></tr>"
            for label, value in metrics
        )
        return (
            "<table class='pynbodyext-calc-metric-strip'>"
            "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

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
        return f"<h5>{html_escape(summary)}</h5>{body}"

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
    table = html_table(rows, escape_values=escape_values) if rows else ""
    if _style() != "rich":
        return f"<h4>{html_escape(title)}</h4>{table}{body}"

    return (
        f"{html_style}"
        "<div class='pynbodyext-calc-card'>"
        f"<div class='pynbodyext-calc-title'>{html_escape(title)}</div>"
        f"{table}"
        f"{body}"
        "</div>"
    )


def mimebundle(text: str, html: str) -> dict[str, str]:
    """Return a MIME bundle with plain text and HTML representations."""
    if _style() == "plain":
        return {"text/plain": text}
    return {"text/plain": text, "text/html": html}


class ViewObject:
    """A sectioned repr view rendered across all display styles.

    Subclasses provide ``_summary()`` (compact one-line text) and ``_sections()``
    (a list of ``(label, body)`` pairs).  The base class turns those into a
    style-appropriate representation:

    - ``rich``: a full HTML card with collapsible ``<details>`` sections.
    - ``github`` / ``plain``: the same sections rendered inline (a view exposes
      its detail directly, so no ''.attr for details'' hint is needed).
    """

    def _summary(self) -> str:
        raise NotImplementedError

    def _sections(self) -> list[tuple[str | None, str]]:
        raise NotImplementedError

    def _title(self) -> str:
        """Return a friendly display title (defaults to the class name)."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self._summary()

    def __str__(self) -> str:
        return self._summary()

    def _repr_html_(self) -> str:
        summary = self._summary()
        sections = self._sections()
        if _style() == "rich":
            body_parts = []
            for label, body in sections:
                heading = label if label else summary
                body_parts.append(html_details(heading, html_pre(body), open=False))
            return html_card(self._title(), [("value", summary)], body="".join(body_parts), escape_values=False)
        # github / plain: a content view renders its sections directly (no value row,
        # no per-section heading), so the body is the detail itself.
        body_parts = []
        for label, body in sections:
            if not body:
                continue
            if label:
                body_parts.append(html_section(label, html_pre(body)))
            else:
                body_parts.append(html_pre(body))
        return f"<h4>{html_escape(self._title())}</h4>" + "".join(body_parts)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        text = self._summary()
        if _style() == "plain":
            return {"text/plain": text}
        return mimebundle(text, self._repr_html_())


class InfoView:
    """Mixin that renders a value object from a single definition of its fields.

    A subclass describes itself once, as ``_display_rows()`` — a list of
    ``(label, value)`` pairs.  The mixin turns those rows into a compact text
    summary (``__repr__`` / ``__str__``) and into HTML for every style, because
    :func:`html_card` already picks the dialect (``rich`` cards with a CSS block
    vs. plain ``<table>`` for GitHub).  Rich and GitHub therefore always show
    the same fields; only the markup differs.

    Subclasses must define ``_display_rows``; ``_display_title`` and
    ``_display_body`` have defaults.  Dataclasses must pass ``repr=False`` so
    the generated ``__repr__`` does not shadow this mixin's summary.

    >>> class Point(InfoView):
    ...     def __init__(self, x, y):
    ...         self.x, self.y = x, y
    ...     def _display_rows(self):
    ...         return [("x", self.x), ("y", self.y)]
    >>> repr(Point(1, 2))
    'Point(x=1, y=2)'
    """

    __slots__ = ()

    def _display_title(self) -> str:
        """Title shown above the rows (defaults to the class name)."""
        return type(self).__name__

    def _display_rows(self) -> list[tuple[str, Any]]:
        """Rows of ``(label, value)`` shown in every display style."""
        raise NotImplementedError

    def _display_body(self) -> str:
        """Optional pre-rendered HTML appended under the rows."""
        return ""

    def _display_summary(self) -> str:
        parts = ", ".join(f"{key}={compact_repr(value, max_length=48)}" for key, value in self._display_rows())
        return f"{self._display_title()}({parts})"

    def __repr__(self) -> str:
        return self._display_summary()

    def __str__(self) -> str:
        # Mirror ``object.__str__``: a subclass with its own ``__repr__`` (a
        # hand-written one-line summary) keeps that text here too.
        return repr(self)

    def _repr_html_(self) -> str:
        return html_card(self._display_title(), self._display_rows(), body=self._display_body())

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())


class TextReport(str):
    """A text report that also renders itself in notebooks.

    Behaves exactly like :class:`str` — ``print``, slicing, ``in``, equality and
    every existing ``report_*`` caller are unchanged — but remembers a title and
    renders as an HTML card (``rich``) or ``<h4>`` + ``<pre>`` (``github``).
    This lets a report method be the single entry point for both the text and
    the notebook view, instead of adding a parallel ``.view`` attribute.
    """

    #: Title of the report (``str.title`` is a method, hence the distinct name).
    report_title: str

    def __new__(cls, title: str, text: str = "") -> TextReport:
        obj = super().__new__(cls, text)
        obj.report_title = title
        return obj

    def __repr__(self) -> str:
        return str.__repr__(self)

    def _repr_html_(self) -> str:
        return html_card(self.report_title, [], body=html_pre(str(self)), escape_values=False)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(str(self), self._repr_html_())

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self))


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
    if value < 3600:
        return f"{value // 60:.0f}m {value % 60:.0f}s"
    return f"{value // 3600:.0f}h {(value % 3600) // 60:.0f}m {value % 60:.0f}s"


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
    "ViewObject",
    "InfoView",
    "TextReport",
    "format_time",
    "format_mem",
]
