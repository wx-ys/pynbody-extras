"""The two repr styles, and the GitHub-safety of ``simple``.

``rich`` is the notebook card; ``simple`` is what a notebook shows once it is pushed
to GitHub, whose renderer sanitizes HTML output.  Every object here is described
once as a :class:`~pynbodyext.core.calculate.display.Card`, so these tests check
the two renderers rather than each object's markup.
"""

from __future__ import annotations

import asyncio
import warnings
from html.parser import HTMLParser

import pytest
from calculate_helpers import make_pipeline, make_sim

from pynbodyext.core.calculate import display
from pynbodyext.core.calculate.display import (
    SIMPLE_HTML_TAGS,
    Badge,
    Card,
    Code,
    Fields,
    Metrics,
    Pre,
    Section,
    Table,
    get_repr_style,
    register_repr_style,
    repr_style,
    reset_repr_style,
    set_repr_style,
)


class _TagCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.attributes: list[tuple[str, list[tuple[str, str | None]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        if attrs:
            self.attributes.append((tag, attrs))

    handle_startendtag = handle_starttag


def _renderables() -> list[object]:
    """One of every notebook-facing object of the calculator framework."""
    from pynbodyext.core.calculate import RunOptions
    from pynbodyext.core.calculate.display import TextReport
    from pynbodyext.core.calculate.result.result import ErrorInfo
    from pynbodyext.filters import FamilyFilter, Sphere
    from pynbodyext.properties import ParamContain

    pipeline = make_pipeline()
    result = pipeline.run(make_sim())
    result.warnings.append("synthetic <warning> & co")
    result.errors.append(ErrorInfo(error_type="RuntimeError", message="boom <b>", traceback_text="Traceback ..."))
    return [
        pipeline,
        ParamContain().filter(Sphere("30 kpc") & FamilyFilter("star")),
        result,
        result.get_named("m"),
        result.provenance,
        result.perf_summary,
        result.named,
        result.named_values,
        result.warnings,
        result.errors,
        result.errors[0],
        result.reports,
        result.diagnostics,
        result.observations,
        result.execution_tree,
        pipeline.dependency_tree,
        RunOptions(),
        TextReport("Report", "line <1>\nline 2"),
    ]


def test_simple_html_passes_the_github_sanitizer() -> None:
    """``simple`` emits only allow-listed tags, and no attributes at all.

    GitHub strips ``<style>``, ``class``/``style`` attributes and most layout tags
    from notebook output, which turns the rich card into an unstyled jumble.  The
    simple renderer must stay inside the subset that survives untouched.
    """
    for obj in _renderables():
        with repr_style("simple"):
            html = obj._repr_html_()
        parser = _TagCollector()
        parser.feed(html)
        name = type(obj).__name__
        assert parser.tags, name
        assert set(parser.tags) <= SIMPLE_HTML_TAGS, (name, sorted(set(parser.tags) - SIMPLE_HTML_TAGS))
        assert parser.attributes == [], (name, parser.attributes[:3])


def test_simple_html_escapes_values() -> None:
    """Values are text: markup in a message must not reach the page as markup."""
    result = _renderables()[2]
    with repr_style("simple"):
        html = result._repr_html_()
    assert "<b>" not in html
    assert "<warning>" not in html


def test_simple_html_preserves_literal_text_in_visible_names() -> None:
    pipeline = make_pipeline("r < 3 & blue")
    result = pipeline.run(make_sim())
    with repr_style("simple"):
        for obj in (result, result.root):
            assert "r &lt; 3 &amp; blue" in obj._repr_html_()


def test_both_styles_render_every_object() -> None:
    for obj in _renderables():
        for style in ("rich", "simple"):
            with repr_style(style):
                bundle = obj._repr_mimebundle_()
            assert set(bundle) == {"text/plain", "text/html"}, type(obj).__name__
            assert bundle["text/html"]


def test_retired_style_names_mean_simple_with_a_warning() -> None:
    for retired in ("github", "plain"):
        with pytest.warns(DeprecationWarning, match="simple"):
            set_repr_style(retired)
        try:
            assert get_repr_style() == "simple"
        finally:
            reset_repr_style()
    with pytest.warns(DeprecationWarning), repr_style("github"):
        assert get_repr_style() == "simple"


def test_unknown_style_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown repr style"):
        set_repr_style("fancy")
    assert get_repr_style() == "rich"


def test_repr_style_context_manager_restores_the_previous_style() -> None:
    assert get_repr_style() == "rich"
    with repr_style("simple"):
        assert get_repr_style() == "simple"
        with repr_style("rich"):
            assert get_repr_style() == "rich"
        assert get_repr_style() == "simple"
    assert get_repr_style() == "rich"


def test_task_created_inside_repr_style_keeps_its_own_override() -> None:
    async def read_style():
        return get_repr_style()

    async def inherited_style():
        with repr_style("simple"):
            task = asyncio.create_task(read_style())
        return get_repr_style(), await task

    assert asyncio.run(inherited_style()) == ("rich", "simple")


def test_set_repr_style_is_session_wide() -> None:
    set_repr_style("simple")
    try:
        assert get_repr_style() == "simple"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert "<style>" not in make_pipeline()._repr_html_()
    finally:
        reset_repr_style()
    assert "<style>" in make_pipeline()._repr_html_()


def _sample_card() -> Card:
    return Card(
        "Sample",
        [("status", Badge("ok", "ok")), ("hash", Code("abc123")), ("pair", (Code("1"), Badge("default")))],
        [
            Metrics([("nodes", 3)], strip=True),
            Section("Inline", [Fields([("a", 1)]), Table(["x", "y"], [[1, 2]])], collapsible=False),
            Section("Folded", [Pre("text")]),
            Section("Drill", [Pre("hidden in simple")], attr="drill"),
        ],
    )


def test_the_simple_card_inlines_sections_without_an_attribute() -> None:
    html = _sample_card().html("simple")
    assert "<h5>Inline</h5>" in html
    assert "<h5>Folded</h5>" in html
    assert "hidden in simple" not in html
    assert html.endswith("<p><strong>Details:</strong> <code>.drill</code></p>")


def test_the_rich_card_folds_collapsible_sections() -> None:
    html = _sample_card().html("rich")
    assert html.startswith(display.HTML_STYLE)
    assert "<summary>Folded</summary>" in html
    assert "<summary>Drill</summary>" in html
    assert "hidden in simple" in html
    assert "pynbodyext-calc-badge-ok" in html


def test_card_lists_nested_drilldowns() -> None:
    card = Card("Nested", blocks=[Section("Outer", [Section("Inner", [Pre("x")], attr="inner")], attr="outer")])
    assert card.drilldown_attrs() == ("outer", "inner")


def test_a_style_can_be_registered() -> None:
    class Upper:
        def card(self, card: Card) -> str:
            return card.title.upper()

    register_repr_style("upper", Upper())
    try:
        with repr_style("upper"):
            assert make_pipeline()._repr_html_() == "PIPELINE"
    finally:
        display._RENDERERS.pop("upper")
    with pytest.raises(ValueError):
        register_repr_style("github", Upper())
