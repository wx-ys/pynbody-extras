"""Characterization tests for the calculator result presentation layer.

These pin the *stable, deterministic* text output of the result and node
repr/report layer (``core/calculate/result/repr.py`` and the report helpers that
back it).  They exist so the presentation layer can be refactored safely: any
change to a rendered string that users see will be caught here, while timing- and
address-dependent fields are asserted only structurally.
"""

from __future__ import annotations

import re

from calculate_helpers import make_pipeline, make_sim

from pynbodyext.core.calculate.result.repr import ResultRepr


def _result():
    return make_pipeline().run(make_sim())


def test_result_repr_is_stable() -> None:
    """repr(Result) must not change when the presentation layer is refactored."""
    result = _result()
    assert repr(result) == "Result(value=dict, ok=True, nodes=4, named=('p', 'm', 't'), observations=4)"


def test_result_node_repr_is_stable() -> None:
    """The first node's repr must pin the label/kind/status/value surface."""
    result = _result()
    node = result.get_named("p")
    text = ResultRepr.result_node_repr(node)
    assert text == "ResultNode(label='p', kind='combined', status='ok', value={'m': 10.0, 't': 20.0}, children=2)"


def test_node_tree_is_stable() -> None:
    """The tree report must pin the exact relationship/dependency rendering."""
    result = _result()
    assert result.report_node_tree() == (
        "p<combined>\n"
        "├─ MassSum<property>\n"
        "│  └─ RBelow<filter>\n"
        "└─ TempMean<property>\n"
        "   └─ RBelow<filter>"
    )


def test_summary_contains_expected_sections() -> None:
    """Summary must keep its measured structure (stable lines) for callers."""
    result = _result()
    text = result.report_summary()
    lines = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    assert lines["root"] == " p"
    assert lines["value_type"] == " dict"
    assert lines["nodes"] == " 4"
    assert lines["warnings"] == " 0"
    assert lines["errors"] == " 0"
    # timing counters are asserted structurally, not by exact value
    assert re.fullmatch(r"\d+\.\d+", lines["total_time_s"].strip())


def test_observer_report_lists_all_nodes_and_phases() -> None:
    """Observer report must enumerate every evaluated node and its key phases."""
    result = _result()
    text = result.report_observer()
    # all three nodes + the combined root appear as row heads
    for name in ("RBelow", "MassSum", "TempMean"):
        assert name in text
    assert "calculate" in text
    assert "resolve_params" in text
    assert "Reads" in text and "Dirty" in text


def test_perf_summary_is_reachable() -> None:
    """Perf table helper must render a non-empty header/rows for a run."""
    result = _result()
    table = ResultRepr.perf_table(result)
    assert table  # non-empty
    assert "Total" in table or "phase" in table.lower()


def test_view_object_repr_is_compact_in_plain() -> None:
    from pynbodyext.core.calculate.display import ViewObject

    class Dummy(ViewObject):
        def _summary(self):
            return "Dummy(a=1)"

        def _sections(self):
            return [("config", "a=1"), ("tree", "root{__}child")]

    obj = Dummy()
    assert repr(obj) == "Dummy(a=1)"


def test_view_object_html_renders_sections_in_github_and_plain() -> None:
    from pynbodyext.core.calculate.display import ViewObject, reset_repr_style, set_repr_style

    class Dummy(ViewObject):
        def _summary(self):
            return "Dummy(a=1)"

        def _sections(self):
            return [("config", "a=1"), ("tree", "root{__}child")]

    obj = Dummy()
    for style in ("github", "plain"):
        set_repr_style(style)
        try:
            html = obj._repr_html_()
            # a view exposes its detail directly: sections are rendered, no hint
            assert "a=1" in html
            assert "root{__}child" in html
            assert "for details" not in html
        finally:
            reset_repr_style()


def test_result_named_is_named_view() -> None:
    from pynbodyext.core.calculate.result.views import NamedView

    result = _result()
    assert isinstance(result.named, NamedView)


def test_result_named_view_protocol_matches_dict() -> None:
    result = _result()
    named = result.named
    assert "m" in named
    assert named["m"].label == "MassSum"
    assert set(named.keys()) == {"p", "m", "t"}
    assert len(named) == 3
    assert named == {"p": result.named["p"], "m": result.named["m"], "t": result.named["t"]}
    assert isinstance(result.named.get("m"), type(result.named["m"]))


def test_result_errors_warnings_protocol_matches_list() -> None:
    import numpy as np
    import pynbody

    from pynbodyext.core.calculate import ErrorPolicy, Pipeline, PropertyBase

    @PropertyBase.dataclass
    class AlwaysFails(PropertyBase[float]):
        def calculate(self, sim, params=None):
            raise RuntimeError("boom")

    @PropertyBase.dataclass
    class Good(PropertyBase[float]):
        def calculate(self, sim, params=None):
            return 1.0

    sim = pynbody.new(3)
    sim["x"] = np.arange(3.0)
    pipe = Pipeline({"ok": Good(), "bad": AlwaysFails()}, name="p")
    result = pipe.run(sim, errors=ErrorPolicy.COLLECT)
    assert bool(result.errors) is True
    assert len(result.errors) >= 1
    assert result.errors[0].message == "boom"
    assert isinstance(list(result.errors)[0].message, str)
    assert bool(result.warnings) is False


def test_named_view_summary_tracks_mutation() -> None:
    """NamedView must not cache a private copy that drifts from the dict.

    Regression guard: the view class used to keep a ``self._data`` copy while the
    dict/list protocol mutated in place, so repr/summary diverged from the real
    contents. Reading the live dict keeps them consistent.
    """
    result = _result()
    named = result.named
    before_keys = set(named.keys())
    named["zz"] = named[next(iter(named))]
    assert "zz" in named
    assert set(named.keys()) == before_keys | {"zz"}
    assert "zz" in named._summary()


def test_result_removed_aliases_gone() -> None:
    """Pure aliases that duplicated a canonical method must not be reintroduced."""
    from pynbodyext.core.calculate.result.result import Result

    for alias in ("value_of", "trace_timeline", "trace_tree", "cache_report"):
        assert not hasattr(Result, alias), f"removed alias {alias!r} was reintroduced"


def test_result_find_relations_and_constants() -> None:
    """``Result.find`` supports relation queries and string constants."""
    result = _result()
    root = result.root
    children = result.find("children", relative_to=root)
    assert {n.label for n in children} == {"MassSum", "TempMean"}

    prop = children[0]
    assert {n.label for n in result.find("parents", relative_to=prop)} == {"p"}
    assert {n.label for n in result.find("ancestors", relative_to=prop)} == {"p"}
    assert {n.label for n in result.find("descendants", relative_to=root)} == {"MassSum", "RBelow", "TempMean"}
    assert [n.node_id for n in result.find("root", relative_to=prop)] == [root.node_id]

    assert len(result.find("all")) == len(result.nodes)
    assert result.find("errors") == []


def test_calculator_base_options_overrides_run_options() -> None:
    """``options(**kwargs)`` overrides the calculator's default run options."""
    calc = make_pipeline()
    overridden = calc.options(cache=False, progress=False)
    assert overridden.default_options.cache is False
    assert overridden.default_options.progress is False
    # original calculator is unchanged (options returns a clone)
    assert calc.default_options.cache is True


def test_param_repr_is_readable() -> None:
    """``Param(...)`` must not leak dataclasses.Field internals in its repr."""
    from pynbodyext.core.calculate import Param

    text = repr(Param(default=0.0, field_name="pos"))
    assert text.startswith("Param(")
    assert "field_name='pos'" in text
    assert "mappingproxy" not in text
    assert "object at 0x" not in text


def test_calculator_base_signature_cache_and_clone_isolation() -> None:
    """``to_signature()`` is cached per instance, but clones get fresh signatures."""
    from pynbodyext.core.calculate import FilterBase, Param

    @FilterBase.dataclass
    class _RB(FilterBase):
        radius: Param[float] = Param(field_name="r")

        def calculate(self, sim, params=None):
            return sim["r"] < self.radius

    base = make_pipeline()
    s1 = base.to_signature()
    s2 = base.to_signature()
    assert s1 is s2, "to_signature() must reuse the memoised result"

    cloned = base.filter(_RB(5.0))
    assert cloned.to_signature() is not s1, "cloned calculator must not reuse the base signature"
    assert cloned.signature_hash() != base.signature_hash()
def test_calculator_base_config_and_dependency_tree_attributes() -> None:
    """The dependency tree is an attribute, named after the card section.

    ``.config`` was a pure view of the summary rows (name/record/scope), so it
    was folded into the repr card.  The tree is one thing, reachable as
    ``.dependency_tree`` — no ``format_tree()`` call, and no second name.
    """
    from pynbodyext.core.calculate.display import TextReport

    calc = make_pipeline()
    assert not hasattr(calc, "config")
    assert not hasattr(calc, "format_tree")
    tree = calc.dependency_tree
    assert isinstance(tree, TextReport)
    # still a plain str for every existing caller
    assert tree.strip().startswith("p<")
    assert isinstance(tree._repr_html_(), str)
    # the old ``.config`` rows (name/record/scope) now live in the summary card
    from calculate_helpers import MassSum, RBelow

    from pynbodyext.core.calculate import RecordPolicy

    scoped = MassSum().filter(RBelow(5.0)).record(RecordPolicy.SUMMARY)
    assert {row[0] for row in scoped._repr_summary_rows()} >= {"label", "record", "scope"}


def test_calculator_base_repr_html_tail_hint_in_github_plain() -> None:
    from pynbodyext.core.calculate.display import reset_repr_style, set_repr_style

    calc = make_pipeline()
    for style in ("github", "plain"):
        set_repr_style(style)
        try:
            html = calc._repr_html_()
            assert ".dependency_tree" in html
        finally:
            reset_repr_style()


def test_result_section_attributes_render_without_a_call() -> None:
    """Each card section is reachable as an attribute of the same name.

    ``.execution_tree`` / ``.performance`` / ``.cache`` used to be both a
    ``report_*()`` method and a view-only property; they are now one attribute
    that renders a :class:`TextReport`.
    """
    from pynbodyext.core.calculate.display import TextReport

    result = _result()
    for method in ("report_execution_tree", "report_perf", "report_cache"):
        assert not hasattr(result, method), f"superseded method {method!r} was reintroduced"
    for name in ("execution_tree", "performance", "cache"):
        report = getattr(result, name)
        assert isinstance(report, TextReport), name
        assert repr(report)
        assert isinstance(report._repr_html_(), str)


def test_result_repr_html_tail_hint_in_github_plain() -> None:
    from pynbodyext.core.calculate.display import reset_repr_style, set_repr_style

    result = _result()
    hint = (
        "Use .named_values, .warnings, .errors, .provenance, "
        ".execution_tree, .performance and .cache for details"
    )
    for style in ("github", "plain"):
        set_repr_style(style)
        try:
            html = result._repr_html_()
            assert hint in html
        finally:
            reset_repr_style()


def test_provenance_shows_the_same_fields_in_every_style() -> None:
    """Regression: ``result.provenance`` used to dump its raw dataclass repr.

    The rich ``Result`` card folded provenance open with a clean table, while
    ``result.provenance`` itself printed every encoded signature payload.  Both
    must now show the same fields; only the markup differs.
    """
    from pynbodyext.core.calculate.display import html_escape, reset_repr_style, set_repr_style

    result = _result()
    provenance = result.provenance
    assert provenance is not None
    rows = provenance._display_rows()
    assert [label for label, _ in rows] == ["calculator hash", "sim signature", "wall time"]

    summary = repr(provenance)
    assert summary.startswith("Provenance(")
    assert "calculator hash" in summary
    assert "('dict'," not in summary  # no raw encoded payload

    set_repr_style("rich")
    try:
        card = result._repr_html_()
    finally:
        reset_repr_style()

    for style in ("rich", "github"):
        set_repr_style(style)
        try:
            standalone = provenance._repr_html_()
        finally:
            reset_repr_style()
        for label, value in rows:
            assert html_escape(label) in standalone
            assert html_escape(str(value)) in standalone
            assert html_escape(label) in card


def test_value_objects_self_render_instead_of_leaking_a_dataclass_repr() -> None:
    """Every user-facing value object renders; none prints a bare dataclass repr."""
    result = _result()
    node = result.get_named("m")
    summary = node.record.value_summary
    assert summary is not None

    for obj in (result.provenance, result.perf_summary, result.reports, result.diagnostics, result.observations, summary):
        text = repr(obj)
        assert text
        assert "object at 0x" not in text
        assert "('dict'," not in text
        html = obj._repr_html_()
        assert isinstance(html, str) and html


def test_error_info_is_renderable() -> None:
    from pynbodyext.core.calculate.result.result import ErrorInfo

    error = ErrorInfo(error_type="RuntimeError", message="boom", phase="calculate", traceback_text="Traceback ...")
    assert "boom" in repr(error)
    assert "phase='calculate'" in repr(error)
    html = error._repr_html_()
    assert "boom" in html
    assert "Traceback ..." in html


def test_text_report_is_a_str_that_renders() -> None:
    from pynbodyext.core.calculate.display import TextReport, reset_repr_style, set_repr_style

    report = TextReport("Node tree", "root\n└─ leaf")
    # str behaviour is preserved for every existing caller
    assert report == "root\n└─ leaf"
    assert report.splitlines() == ["root", "└─ leaf"]
    assert repr(report) == repr("root\n└─ leaf")

    for style in ("rich", "github", "plain"):
        set_repr_style(style)
        try:
            html = report._repr_html_()
            assert "Node tree" in html
            assert "leaf" in html
        finally:
            reset_repr_style()


def test_run_options_text_and_html_list_the_same_fields() -> None:
    """A curated text repr must not hide rows the HTML table shows."""
    from pynbodyext.core.calculate import RunOptions
    from pynbodyext.core.calculate.display import html_escape

    options = RunOptions()
    text = repr(options)
    html = options._repr_html_()
    for name, label, _value in options._fields():
        assert f"{name}=" in text, f"RunOptions repr omits {name!r}"
        assert html_escape(label) in html


def test_result_card_reuses_the_section_attributes() -> None:
    """The rich card must render exactly what the section attributes return."""
    from pynbodyext.core.calculate.display import html_escape, reset_repr_style, set_repr_style

    result = _result()
    reports = (result.execution_tree, result.performance, result.cache)

    set_repr_style("rich")
    try:
        html = result._repr_html_()
    finally:
        reset_repr_style()

    for report in reports:
        assert html_escape(str(report)) in html, f"{report.report_title!r} body differs from the card"


_CARD_TITLE_RE = re.compile(
    r"<summary>([^<]+)</summary>" r"|<div class='pynbodyext-calc-section-title'>([^<]+)</div>"
)


def _card_titles(html: str) -> list[str]:
    """Section titles of a rendered card, in document order."""
    return [summary or section for summary, section in _CARD_TITLE_RE.findall(html)]


def _snake(title: str) -> str:
    return title.lower().replace(" ", "_")


def test_result_card_sections_are_nameable_attributes() -> None:
    """Every card section is reachable as an attribute of the same name.

    This is the contract behind the github/plain hint: a reader sees a section
    title in the rich card, and the same name (snake_cased) as an attribute
    whose repr shows that section — no method call.
    """
    from pynbodyext.core.calculate.display import reset_repr_style, set_repr_style
    from pynbodyext.core.calculate.result.result import ErrorInfo

    result = _result()
    result.warnings.append("synthetic warning")
    result.errors.append(ErrorInfo(error_type="RuntimeError", message="synthetic error"))

    set_repr_style("rich")
    try:
        titles = _card_titles(result._repr_html_())
    finally:
        reset_repr_style()

    assert titles == ["Named values", "Warnings", "Errors", "Provenance", "Execution tree", "Performance", "Cache"]
    for title in titles:
        assert hasattr(result, _snake(title)), f"section {title!r} has no .{_snake(title)} attribute"

    set_repr_style("github")
    try:
        hint_html = result._repr_html_()
    finally:
        reset_repr_style()
    for title in titles:
        assert f".{_snake(title)}" in hint_html
    assert "()" not in hint_html.split("<pre>")[1]


def test_calculator_card_section_is_a_nameable_attribute() -> None:
    from pynbodyext.core.calculate.display import reset_repr_style, set_repr_style

    calc = make_pipeline()
    set_repr_style("rich")
    try:
        titles = _card_titles(calc._repr_html_())
    finally:
        reset_repr_style()

    assert titles == ["Dependency tree"]
    for title in titles:
        assert hasattr(calc, _snake(title))

    set_repr_style("github")
    try:
        hint_html = calc._repr_html_()
    finally:
        reset_repr_style()
    assert ".dependency_tree" in hint_html
    assert "()" not in hint_html.split("<pre>")[1]


def _emitted_card_classes(html: str) -> set[str]:
    """``pynbodyext-calc-*`` classes referenced by a rendered card."""
    classes = set()
    for attribute in re.findall(r"class='([^']*)'", html):
        classes.update(name for name in attribute.split() if name.startswith("pynbodyext-calc-"))
    return classes


def test_card_css_is_a_packaged_file_defining_every_emitted_class() -> None:
    """The stylesheet lives in ``display.css`` and covers the class contract.

    The rich helpers emit ``pynbodyext-calc-*`` class names and the stylesheet is
    the only place they are defined, so a renamed or missing class would silently
    render an unstyled card.  The CSS is package data (``display.css`` next to
    ``display.py``), hence the ``importlib.resources`` check: an unshipped file
    must fail here rather than only in someone's installed wheel.
    """
    import importlib.resources

    from pynbodyext.core.calculate import display

    css_file = importlib.resources.files("pynbodyext.core.calculate").joinpath("display.css")
    assert css_file.is_file(), "display.css is missing from the installed package"
    css = css_file.read_text(encoding="utf-8")
    assert ".pynbodyext-calc-card" in css
    assert display.HTML_STYLE == display.card_style()
    assert ".pynbodyext-calc-card" in display.HTML_STYLE

    result = _result()
    result.warnings.append("synthetic warning")
    node = result.get_named("m")
    rich_cards = [
        make_pipeline()._repr_html_(),
        result._repr_html_(),
        node._repr_html_(),
        result.provenance._repr_html_(),
        result.reports._repr_html_(),
    ]
    emitted = set().union(*(_emitted_card_classes(html) for html in rich_cards))
    assert emitted, "expected the rich cards to reference card classes"
    missing = sorted(name for name in emitted if f".{name}" not in css)
    assert not missing, f"class names emitted but not defined in display.css: {missing}"


def _scoped_contain():
    """``ParamContain().filter(...)`` — a property carrying a filter scope.

    A non-transform calculator takes its scope through ``ScopeSpec``, which the
    signature encodes as a ``{"node": "scoped", ...}`` wrapper.
    """
    from pynbodyext.filters import FamilyFilter, Sphere
    from pynbodyext.properties import ParamContain

    return ParamContain().filter(Sphere("30 kpc") & FamilyFilter("stars"))


def test_scoped_node_tree_label_is_not_a_raw_payload() -> None:
    """A scoped node labels with its base name; the scope shows as a child."""
    tree = _scoped_contain().dependency_tree
    assert "'node'" not in tree
    assert tree.strip().startswith('ParamContain(0.5, "r", "mass")<prop>')
    assert "└─ AndFilter<filt>" in tree


def test_scoped_node_pretty_is_not_a_raw_payload() -> None:
    """``pretty()`` is the human/store-facing key and must render, not dump a dict."""
    pretty = _scoped_contain().to_signature().pretty()
    assert "'node'" not in pretty
    assert pretty.startswith('ParamContain(0.5, "r", "mass").filter(')


def test_nested_scoped_argument_renders_compactly() -> None:
    """The reported case: a scoped property nested in another node's init args."""
    from pynbodyext.filters import FamilyFilter, Sphere
    from pynbodyext.transforms import ShiftVelTo

    node = ShiftVelTo().filter(Sphere(0.5 * _scoped_contain()) & FamilyFilter("stars"))
    tree = node.dependency_tree
    assert "'node'" not in tree
    assert 'Sphere(0.5 * ParamContain(0.5, "r", "mass"))<filt>' in tree


def test_all_default_node_spells_out_its_defaults() -> None:
    """A node whose parameters are all defaults must describe itself.

    Otherwise it renders as a bare class name (``ParamContain``), which tells
    the reader nothing about what the node computes.
    """
    from pynbodyext.properties import ParamContain

    assert repr(ParamContain()) == 'ParamContain(0.5, "r", "mass")'
    assert ParamContain().to_signature().pretty() == 'ParamContain(0.5, "r", "mass")'


def test_node_with_explicit_argument_keeps_compact_label() -> None:
    """Defaults stay hidden when the node already shows a meaningful argument."""
    from pynbodyext.filters import Sphere

    assert repr(Sphere("30 kpc")) == 'Sphere("30 kpc")'
    assert Sphere("30 kpc").dependency_tree.strip() == 'Sphere("30 kpc")<filt>'


def _scoped_chain():
    """The reported chain: transforms holding scope-filtered properties as args."""
    from pynbodyext.filters import FamilyFilter, Sphere
    from pynbodyext.properties import AngMomVec, ParamContain
    from pynbodyext.transforms import AlignVec, ShiftPosTo, ShiftVelTo, WrapBox

    re = ParamContain().filter(Sphere("30 kpc") & FamilyFilter("stars"))
    return (
        WrapBox()
        .then(ShiftPosTo("ssc"))
        .then(ShiftVelTo().filter(Sphere(0.5 * re) & FamilyFilter("stars")))
        .then(AlignVec(AngMomVec().filter(Sphere(2 * re) & FamilyFilter("stars"))))
        .revert(False)
    )


def _walk(node, seen: set[int] | None = None):
    seen = set() if seen is None else seen
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for child in node.children():
        yield from _walk(child, seen)


def test_scoped_chain_renders_no_raw_payloads_anywhere() -> None:
    """No node in the graph may leak its encoded payload into a display string.

    This guards the whole class of bug reported for scoped properties: every
    signature payload node type needs a renderer in both printers, otherwise
    nested arguments and ``pretty()`` fall back to ``repr(dict)``.
    """
    chain = _scoped_chain()
    assert "'node'" not in chain.dependency_tree
    for node in _walk(chain):
        assert "'node'" not in node.dependency_tree, node.tree_label
        assert "'node'" not in node.to_signature().pretty(), node.tree_label


def test_both_printers_cover_every_payload_node_type() -> None:
    """Both printers must dispatch on every ``payload["node"]`` type.

    The scoped-property regression happened because a payload node type existed
    in the encoder with no renderer, so both printers silently fell back to
    ``repr(payload)``.  Keep the dispatch tables in sync with the node types the
    signature can carry.
    """
    from pynbodyext.core.calculate.result.render import SignaturePrinter, TreePrinter
    from pynbodyext.core.calculate.result.signature import _Decoder

    # Decoder-known node types, plus the two non-constructible display-only ones.
    node_types = set(_Decoder._SPECIAL_DECODERS) | {"dataclass", "lambda_property", "generic"}
    assert node_types <= set(SignaturePrinter._CALCULATOR_HANDLERS)
    assert node_types <= set(TreePrinter._CALCULATOR_HEADERS)
