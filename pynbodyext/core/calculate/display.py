"""Shared display layer for calculator objects.

Every notebook-facing object of the calculator framework — calculators, results,
result nodes and the small value objects — describes *what* it shows once, as a
:class:`Card`, and a renderer turns that card into markup for the active
**repr style**:

``rich`` (default)
    The full card for Jupyter / VS Code: an inlined stylesheet, collapsible
    ``<details>`` sections, badges and metric grids.
``simple``
    Markup that survives GitHub's notebook sanitizer, so a notebook pushed to
    GitHub shows the same information: only ``<h4>``/``<h5>``/``<p>``/``<table>``/
    ``<pre>``/``<code>``/``<strong>``/``<br>`` tags, and no attributes at all (no
    ``class``, no ``style``).  A section that the simple card does not inline is
    named as the attribute that shows it (``.execution_tree``), so nothing is
    hidden without a way to reach it.  The ``text/plain`` repr is unchanged.

The card model keeps the two styles from drifting apart: there is one description
and two renderers, instead of every helper branching on the style.  A renderer for
another dialect can be added with :func:`register_repr_style`.

Values inside a card are *fragments*: plain values (escaped by the renderer),
:class:`Code` (monospace) or :class:`Badge` (a status pill in ``rich``, ``<code>``
in ``simple``), or a tuple of fragments.  No fragment carries pre-rendered HTML,
which is what lets the simple style guarantee sanitizer-safe output.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from html import escape
from importlib import resources
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Stylesheet (rich style only)
# ---------------------------------------------------------------------------


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
    point at), so the stylesheet is inlined once per rendered card.
    """
    return f"<style>\n{_card_css()}</style>\n"


HTML_STYLE = card_style()


# ---------------------------------------------------------------------------
# Repr style
# ---------------------------------------------------------------------------

DEFAULT_REPR_STYLE = "rich"
#: Retired style names, mapped to the style that replaced them.
_STYLE_ALIASES: dict[str, str] = {"github": "simple", "plain": "simple"}

#: The session-wide style, set by :func:`set_repr_style`.  A module global rather
#: than a context variable: IPython may run each cell in a copied context, so a
#: context variable set in one cell would not reach the next.
_GLOBAL_STYLE: dict[str, str] = {"style": DEFAULT_REPR_STYLE}
#: A temporary override, set by the :func:`repr_style` context manager.
_STYLE_OVERRIDE: ContextVar[str | None] = ContextVar("pynbodyext_repr_style", default=None)


def _canonical_style(style: str) -> str:
    if style in _STYLE_ALIASES:
        replacement = _STYLE_ALIASES[style]
        warnings.warn(
            f"repr style {style!r} was merged into {replacement!r}; use {replacement!r} instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return replacement
    if style not in _RENDERERS:
        raise ValueError(f"unknown repr style {style!r}; expected one of {tuple(_RENDERERS)}")
    return style


def get_repr_style() -> str:
    """Return the active display style (``"rich"`` or ``"simple"``)."""
    override = _STYLE_OVERRIDE.get()
    return override if override is not None else _GLOBAL_STYLE["style"]


def set_repr_style(style: str) -> None:
    """Set the display style used by ``_repr_html_`` / ``_repr_mimebundle_``.

    Parameters
    ----------
    style : {"rich", "simple"}
        - ``"rich"`` (default): full HTML for Jupyter / VS Code (cards, collapsible
          details, badges, metric grids, embedded CSS).
        - ``"simple"``: GitHub-sanitizer-safe HTML (tables, ``<pre>``, ``<code>``,
          headings; no attributes), so notebooks pushed to GitHub render cleanly.

        The retired names ``"github"`` and ``"plain"`` still work for now and mean
        ``"simple"``, with a :class:`DeprecationWarning`.

    Examples
    --------
    >>> set_repr_style("simple")
    >>> set_repr_style("rich")
    """
    _GLOBAL_STYLE["style"] = _canonical_style(style)


def reset_repr_style() -> None:
    """Reset the display style back to the default ``"rich"``."""
    _GLOBAL_STYLE["style"] = DEFAULT_REPR_STYLE


@contextmanager
def repr_style(style: str) -> Iterator[str]:
    """Render with *style* inside a ``with`` block, then restore the previous style.

    An asyncio task inherits the style active when it is created. The task keeps
    that override after its parent exits the block. Use ``repr_style`` inside the
    task to select another style there.

    >>> with repr_style("simple"):
    ...     get_repr_style()
    'simple'
    """
    token = _STYLE_OVERRIDE.set(_canonical_style(style))
    try:
        yield get_repr_style()
    finally:
        _STYLE_OVERRIDE.reset(token)


def _style() -> str:
    return get_repr_style()


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Card model
# ---------------------------------------------------------------------------

BADGE_TONES = frozenset({"neutral", "info", "ok", "warn", "error"})


@dataclass(frozen=True, slots=True)
class Badge:
    """A short status token: a coloured pill in ``rich``, ``<code>`` in ``simple``."""

    text: Any
    tone: str = "neutral"


@dataclass(frozen=True, slots=True)
class Code:
    """A value shown in monospace (hashes, expressions, identifiers)."""

    text: Any


#: A cell value: a plain value, :class:`Code`, :class:`Badge`, or a tuple of them
#: (rendered side by side).
Fragment: TypeAlias = Any


@dataclass(frozen=True, slots=True)
class Fields:
    """A two-column ``label -> value`` table."""

    rows: Sequence[tuple[str, Fragment]]
    nowrap: bool = False
    monospace: bool = False
    #: Minimum width before the block scrolls horizontally (rich only).
    min_width: str | None = None


@dataclass(frozen=True, slots=True)
class Table:
    """A multi-column table with a header row."""

    headers: Sequence[str]
    rows: Sequence[Sequence[Fragment]]
    nowrap: bool = False
    monospace: bool = False
    min_width: str | None = None


@dataclass(frozen=True, slots=True)
class Metrics:
    """A row of ``label: value`` counters."""

    items: Sequence[tuple[str, Fragment]]
    #: One scrolling line (``True``) instead of a wrapping grid (rich only).
    strip: bool = False


@dataclass(frozen=True, slots=True)
class Pre:
    """Preformatted text (trees, reports, tracebacks)."""

    text: str


_ContentBlock: TypeAlias = Fields | Table | Metrics | Pre


class _DrilldownBody(Sequence[_ContentBlock]):
    """A single content block, built when a renderer first reads it."""

    def __init__(self, build: Callable[[], _ContentBlock]) -> None:
        self._build = build

    @cached_property
    def _blocks(self) -> tuple[_ContentBlock]:
        return (self._build(),)

    def __len__(self) -> int:
        return 1

    @overload
    def __getitem__(self, index: int) -> _ContentBlock: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[_ContentBlock, ...]: ...
    def __getitem__(self, index: int | slice) -> _ContentBlock | tuple[_ContentBlock, ...]:
        return self._blocks[index]


@dataclass(frozen=True, slots=True)
class Section:
    """A titled group of blocks.

    ``attr`` names the attribute of the rendered object that shows this section
    on its own (``"execution_tree"`` for the *Execution tree* section).  A section
    with an ``attr`` is a *drill-down*: the simple card lists the attribute instead
    of inlining the content.  A section without one is always inlined.

    ``collapsible`` sections become ``<details>`` in the rich card (``open``
    controls their initial state); the others are plain titled groups.
    """

    title: str
    blocks: Sequence[Block]
    attr: str | None = None
    collapsible: bool = True
    open: bool = False

    @classmethod
    def drilldown(
        cls,
        title: str,
        attr: str,
        build: Callable[[], _ContentBlock],
        *,
        collapsible: bool = True,
        open: bool = False,
    ) -> Section:
        """Describe an attribute whose table or report is built only when displayed."""
        return cls(title, _DrilldownBody(build), attr=attr, collapsible=collapsible, open=open)


Block: TypeAlias = Fields | Table | Metrics | Pre | Section


@dataclass(frozen=True, slots=True)
class Card:
    """What an object shows in a notebook, independent of the repr style."""

    title: str
    rows: Sequence[tuple[str, Fragment]] = ()
    blocks: Sequence[Block] = field(default_factory=tuple)

    def sections(self) -> list[Section]:
        """Every section of the card, nested ones included, in document order."""
        out: list[Section] = []

        def walk(blocks: Sequence[Block]) -> None:
            for block in blocks:
                if isinstance(block, Section):
                    out.append(block)
                    if not isinstance(block.blocks, _DrilldownBody):
                        walk(block.blocks)

        walk(self.blocks)
        return out

    def drilldown_attrs(self) -> tuple[str, ...]:
        """Attribute names of the drill-down sections, in document order."""
        return tuple(section.attr for section in self.sections() if section.attr)

    def html(self, style: str | None = None) -> str:
        """Render the card as HTML in *style* (the active style by default)."""
        renderer = _RENDERERS[_canonical_style(style) if style is not None else get_repr_style()]
        return renderer.card(self)

    def mimebundle(self, text: str) -> dict[str, str]:
        """Return the notebook MIME bundle: *text* plus the card's HTML."""
        return {"text/plain": text, "text/html": self.html()}


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class CardRenderer(Protocol):
    """Turns a :class:`Card` into one HTML dialect."""

    def card(self, card: Card) -> str: ...


class RichRenderer:
    """Full notebook card: inlined CSS, ``<details>``, badges and grids."""

    css_prefix = "pynbodyext-calc"

    def _cls(self, *names: str) -> str:
        return " ".join(f"{self.css_prefix}-{name}" for name in names)

    def fragment(self, value: Fragment) -> str:
        if isinstance(value, Badge):
            tone = value.tone if value.tone in BADGE_TONES else "neutral"
            return f"<span class='{self._cls('badge', f'badge-{tone}')}'>{html_escape(value.text)}</span>"
        if isinstance(value, Code):
            return f"<code>{html_escape(value.text)}</code>"
        if isinstance(value, tuple):
            return " ".join(self.fragment(item) for item in value)
        return html_escape(value)

    def _scroll(self, body: str, min_width: str | None) -> str:
        inner_style = f" style='min-width:{html_escape(min_width)};'" if min_width is not None else ""
        return (
            f"<div class='{self._cls('scroll-x')}'>"
            f"<div class='{self._cls('scroll-x-inner')}'{inner_style}>{body}</div>"
            "</div>"
        )

    def fields(self, block: Fields, *, scroll: bool = True) -> str:
        names = ["table"] + (["table-nowrap"] if block.nowrap else []) + (["monospace"] if block.monospace else [])
        body = "".join(
            f"<div class='{self._cls('kv-row')}'>"
            f"<div class='{self._cls('kv-key')}'>{html_escape(key)}</div>"
            f"<div class='{self._cls('kv-value')}'>{self.fragment(value)}</div>"
            "</div>"
            for key, value in block.rows
        )
        html = f"<div class='{self._cls(*names)}'>{body}</div>"
        return self._scroll(html, block.min_width) if scroll and (block.nowrap or block.min_width) else html

    def table(self, block: Table) -> str:
        names = ["data-table"]
        names += ["data-table-nowrap"] if block.nowrap else []
        names += ["monospace"] if block.monospace else []
        columns = " ".join("max-content" for _ in block.headers) or "max-content"
        head = "".join(f"<div class='{self._cls('data-head-cell')}'>{html_escape(h)}</div>" for h in block.headers)
        body = "".join(
            f"<div class='{self._cls('data-row')}'>"
            + "".join(f"<div class='{self._cls('data-cell')}'>{self.fragment(value)}</div>" for value in row)
            + "</div>"
            for row in block.rows
        )
        html = (
            f"<div class='{self._cls(*names)}' style='--pynbodyext-calc-columns:{html_escape(columns)};'>"
            f"<div class='{self._cls('data-head')}'>{head}</div>{body}</div>"
        )
        return self._scroll(html, block.min_width)

    def metrics(self, block: Metrics) -> str:
        body = "".join(
            f"<div class='{self._cls('metric')}'>"
            f"<div class='{self._cls('metric-label')}'>{html_escape(label)}</div>"
            f"<div class='{self._cls('metric-value')}'>{self.fragment(value)}</div>"
            "</div>"
            for label, value in block.items
        )
        if block.strip:
            return self._scroll(f"<div class='{self._cls('metric-strip')}'>{body}</div>", None)
        return f"<div class='{self._cls('metric-grid')}'>{body}</div>"

    def pre(self, block: Pre) -> str:
        return f"<pre class='{self._cls('pre')}'>{html_escape(block.text)}</pre>"

    def section(self, block: Section) -> str:
        body = self.blocks(block.blocks)
        if block.collapsible:
            open_attr = " open" if block.open else ""
            return (
                f"<details class='{self._cls('details')}'{open_attr}>"
                f"<summary>{html_escape(block.title)}</summary>{body}</details>"
            )
        return (
            f"<div class='{self._cls('section')}'>"
            f"<div class='{self._cls('section-title')}'>{html_escape(block.title)}</div>{body}</div>"
        )

    def block(self, block: Block) -> str:
        if isinstance(block, Fields):
            return self.fields(block)
        if isinstance(block, Table):
            return self.table(block)
        if isinstance(block, Metrics):
            return self.metrics(block)
        if isinstance(block, Pre):
            return self.pre(block)
        if isinstance(block, Section):
            return self.section(block)
        raise TypeError(f"unsupported card block: {type(block)!r}")

    def blocks(self, blocks: Sequence[Block]) -> str:
        return "".join(self.block(block) for block in blocks)

    def card(self, card: Card) -> str:
        rows = self.fields(Fields(card.rows), scroll=False) if card.rows else ""
        return (
            f"{HTML_STYLE}"
            f"<div class='{self._cls('card')}'>"
            f"<div class='{self._cls('title')}'>{html_escape(card.title)}</div>"
            f"{rows}{self.blocks(card.blocks)}"
            "</div>"
        )


#: Tags the simple style may emit.  GitHub's notebook renderer sanitizes HTML
#: output; these tags (with no attributes) pass it unchanged.  Pinned by
#: ``tests/test_calculate_render.py``.
SIMPLE_HTML_TAGS = frozenset({"h4", "h5", "p", "br", "strong", "code", "pre", "table", "thead", "tbody", "tr", "th", "td"})


class SimpleRenderer:
    """GitHub-sanitizer-safe card: headings, tables, ``<pre>`` and ``<code>`` only.

    Emits no attributes and none of ``<style>``, ``<div>``, ``<span>`` or
    ``<details>``.  Drill-down sections (those with an ``attr``) are listed as
    attributes under the card instead of being inlined.
    """

    #: Plain values up to this length are shown as ``<code>`` tokens.
    code_max_length = 40

    def fragment(self, value: Fragment) -> str:
        if isinstance(value, (Badge, Code)):
            return f"<code>{html_escape(value.text)}</code>"
        if isinstance(value, tuple):
            return " ".join(self.fragment(item) for item in value)
        text = str(value)
        if isinstance(value, (str, int, float, bool)) and len(text) <= self.code_max_length:
            return f"<code>{html_escape(text)}</code>"
        return html_escape(text)

    def fields(self, block: Fields) -> str:
        body = "".join(f"<tr><th>{html_escape(key)}</th><td>{self.fragment(value)}</td></tr>" for key, value in block.rows)
        return f"<table><tbody>{body}</tbody></table>"

    def table(self, block: Table) -> str:
        head = "".join(f"<th>{html_escape(header)}</th>" for header in block.headers)
        body = "".join("<tr>" + "".join(f"<td>{self.fragment(v)}</td>" for v in row) + "</tr>" for row in block.rows)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def metrics(self, block: Metrics) -> str:
        head = "".join(f"<th>{html_escape(label)}</th>" for label, _ in block.items)
        body = "".join(f"<td>{self.fragment(value)}</td>" for _, value in block.items)
        return f"<table><thead><tr>{head}</tr></thead><tbody><tr>{body}</tr></tbody></table>"

    def pre(self, block: Pre) -> str:
        return f"<pre>{html_escape(block.text)}</pre>"

    def section(self, block: Section) -> str:
        if block.attr:
            return ""  # listed under the card as a drill-down attribute
        return f"<h5>{html_escape(block.title)}</h5>{self.blocks(block.blocks)}"

    def block(self, block: Block) -> str:
        if isinstance(block, Fields):
            return self.fields(block)
        if isinstance(block, Table):
            return self.table(block)
        if isinstance(block, Metrics):
            return self.metrics(block)
        if isinstance(block, Pre):
            return self.pre(block)
        if isinstance(block, Section):
            return self.section(block)
        raise TypeError(f"unsupported card block: {type(block)!r}")

    def blocks(self, blocks: Sequence[Block]) -> str:
        return "".join(self.block(block) for block in blocks)

    def drilldowns(self, card: Card) -> str:
        attrs = card.drilldown_attrs()
        if not attrs:
            return ""
        names = ", ".join(f"<code>.{html_escape(attr)}</code>" for attr in attrs)
        return f"<p><strong>Details:</strong> {names}</p>"

    def card(self, card: Card) -> str:
        rows = self.fields(Fields(card.rows)) if card.rows else ""
        return f"<h4>{html_escape(card.title)}</h4>{rows}{self.blocks(card.blocks)}{self.drilldowns(card)}"


_RENDERERS: dict[str, CardRenderer] = {"rich": RichRenderer(), "simple": SimpleRenderer()}


def register_repr_style(name: str, renderer: CardRenderer) -> None:
    """Register (or replace) the renderer behind repr style *name*."""
    if name in _STYLE_ALIASES:
        raise ValueError(f"{name!r} is a retired style name and cannot be registered")
    _RENDERERS[name] = renderer


def repr_styles() -> tuple[str, ...]:
    """Return the registered repr style names."""
    return tuple(_RENDERERS)


#: Built-in style names (see :func:`repr_styles` for the registered ones).
REPR_STYLES: tuple[str, ...] = ("rich", "simple")


def mimebundle(text: str, html: str) -> dict[str, str]:
    """Return a MIME bundle with plain text and HTML representations."""
    return {"text/plain": text, "text/html": html}


# ---------------------------------------------------------------------------
# Renderable bases
# ---------------------------------------------------------------------------


class ViewObject:
    """A sectioned view of a container (named values, warnings, errors).

    Subclasses provide ``_summary()`` (compact one-line text) and ``_sections()``
    (a list of ``(label, body)`` pairs of text).  The card shows the summary as
    its value and every section as preformatted text — collapsible in ``rich``,
    inlined in ``simple`` (a view exposes its detail directly, so there is no
    drill-down attribute to name).
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

    def _repr_card(self) -> Card:
        summary = self._summary()
        blocks = [Section(label or summary, [Pre(body)]) for label, body in self._sections() if body]
        return Card(self._title(), [("value", Code(summary))], blocks)

    def _repr_html_(self) -> str:
        return self._repr_card().html()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return self._repr_card().mimebundle(self._summary())


class InfoView:
    """Mixin that renders a value object from a single definition of its fields.

    A subclass describes itself once, as ``_display_rows()`` — a list of
    ``(label, value)`` pairs.  The mixin turns those rows into a compact text
    summary (``__repr__`` / ``__str__``) and into a :class:`Card`, so every
    style shows the same fields; only the markup differs.

    Subclasses must define ``_display_rows``; ``_display_title`` and
    ``_display_blocks`` have defaults.  Dataclasses must pass ``repr=False`` so
    the generated ``__repr__`` does not shadow this mixin's summary.

    The 0.0.1 card-model change (PR #17, revision ``458e808``) replaced the
    ``_display_body()`` HTML hook with ``_display_blocks()``. Subclasses that
    supplied a preformatted message can return ``[Pre(self.message)]`` from the
    new hook. Use ``Fields`` or ``Table`` for tabular data. Each renderer formats
    the blocks for its own style and escapes their values.

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

    def _display_blocks(self) -> list[Block]:
        """Optional blocks shown under the rows."""
        return []

    def _display_summary(self) -> str:
        parts = ", ".join(f"{key}={compact_repr(value, max_length=48)}" for key, value in self._display_rows())
        return f"{self._display_title()}({parts})"

    def __repr__(self) -> str:
        return self._display_summary()

    def __str__(self) -> str:
        # Mirror ``object.__str__``: a subclass with its own ``__repr__`` (a
        # hand-written one-line summary) keeps that text here too.
        return repr(self)

    def _repr_card(self) -> Card:
        return Card(self._display_title(), self._display_rows(), self._display_blocks())

    def _repr_html_(self) -> str:
        return self._repr_card().html()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return self._repr_card().mimebundle(repr(self))


class TextReport(str):
    """A text report that also renders itself in notebooks.

    Behaves exactly like :class:`str` — ``print``, slicing, ``in``, equality and
    every existing ``report_*`` caller are unchanged — but remembers a title and
    renders as a card holding the text.  This lets a report attribute be the
    single entry point for both the text and the notebook view.
    """

    #: Title of the report (``str.title`` is a method, hence the distinct name).
    report_title: str

    def __new__(cls, title: str, text: str = "") -> TextReport:
        obj = super().__new__(cls, text)
        obj.report_title = title
        return obj

    def __repr__(self) -> str:
        return str.__repr__(self)

    def _repr_card(self) -> Card:
        return Card(self.report_title, blocks=[Pre(str(self))])

    def _repr_html_(self) -> str:
        return self._repr_card().html()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return self._repr_card().mimebundle(str(self))

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(str(self))


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tree text
# ---------------------------------------------------------------------------


def validate_tree_limits(max_depth: int | None, max_children: int | None) -> None:
    """Refuse a negative truncation limit, for every tree renderer."""
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be non-negative or None")
    if max_children is not None and max_children < 0:
        raise ValueError("max_children must be non-negative or None")


def subtree_nodes(
    roots: Sequence[T],
    children: Callable[[T], Sequence[T]],
    *,
    key: Callable[[T], Any],
) -> list[T]:
    """Return the nodes of the subtrees rooted at *roots*, each node once.

    *key* identifies a node, so a graph that shares one node between two parents
    counts it once — which is what the hidden-count summaries report.
    """
    out: list[T] = []
    seen: set[Any] = set()
    stack = list(reversed(list(roots)))
    while stack:
        node = stack.pop()
        node_key = key(node)
        if node_key in seen:
            continue
        seen.add(node_key)
        out.append(node)
        stack.extend(reversed(list(children(node))))
    return out


def hidden_summary(
    nodes: list[T],
    *,
    key: Callable[[T], Any],
    children: Callable[[T], Sequence[T]],
) -> str:
    """Describe the *nodes* a truncated tree leaves out, by subtree size."""
    count = len(subtree_nodes(nodes, children, key=key))
    suffix = "node" if count == 1 else "nodes"
    return f"... {count} {suffix} hidden"


def tree_lines(
    root: T,
    *,
    children: Callable[[T], Sequence[T]],
    label: Callable[[T], str],
    hidden: Callable[[list[T]], str],
    max_depth: int | None = None,
    max_children: int | None = None,
) -> list[str]:
    """Render *root* and its descendants as box-drawing tree lines.

    One implementation of the tree convention — the ``├─`` / ``└─`` branches, the
    ``│  `` guide under a non-last child, and the truncation policy — so the
    calculator dependency tree and the two result trees cannot drift apart.

    *label* renders one node's line and *hidden* describes the nodes a
    ``max_depth`` / ``max_children`` cut leaves out. The root carries no branch.

    Call order is part of the contract, so a caller may keep walk state in its
    callbacks (the execution tree does, to expand a shared node only once):
    nodes are visited depth-first in pre-order, ``label(node)`` is called before
    ``children(node)``, and ``children`` is called **at most once** per rendered
    line.
    """
    validate_tree_limits(max_depth, max_children)
    lines: list[str] = [label(root)]

    def siblings(node_children: list[T], prefix: str, depth: int) -> Iterator[tuple[list[T], str, int]]:
        if not node_children:
            return
        if max_depth is not None and depth >= max_depth:
            lines.append(f"{prefix}└─ {hidden(node_children)}")
            return

        visible = node_children if max_children is None else node_children[:max_children]
        omitted = [] if max_children is None else node_children[max_children:]
        for index, child in enumerate(visible):
            is_last = index == len(visible) - 1 and not omitted
            branch = "└─" if is_last else "├─"
            lines.append(f"{prefix}{branch} {label(child)}")
            yield list(children(child)), prefix + ("   " if is_last else "│  "), depth + 1
        if omitted:
            lines.append(f"{prefix}└─ {hidden(omitted)}")

    # Each iterator resumes the remaining siblings after a child's subtree.
    stack = [siblings(list(children(root)), "", 1)]
    while stack:
        branch = next(stack[-1], None)
        if branch is None:
            stack.pop()
        else:
            stack.append(siblings(*branch))
    return lines


__all__ = [
    "HTML_STYLE",
    "REPR_STYLES",
    "SIMPLE_HTML_TAGS",
    "DEFAULT_REPR_STYLE",
    "get_repr_style",
    "set_repr_style",
    "reset_repr_style",
    "repr_style",
    "repr_styles",
    "register_repr_style",
    "card_style",
    "compact_repr",
    "display_value",
    "html_escape",
    "Badge",
    "Code",
    "Fields",
    "Table",
    "Metrics",
    "Pre",
    "Section",
    "Card",
    "CardRenderer",
    "RichRenderer",
    "SimpleRenderer",
    "mimebundle",
    "ViewObject",
    "InfoView",
    "TextReport",
    "format_time",
    "format_mem",
    "validate_tree_limits",
    "subtree_nodes",
    "hidden_summary",
    "tree_lines",
]
