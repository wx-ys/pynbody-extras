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
    calc = make_pipeline()
    assert hasattr(calc, "config")
    assert hasattr(calc, "dependency_tree")
    assert "Configuration" in calc.config._repr_html_() or "config" in calc.config._summary()
    # dependency_tree summary is the root/head line, not the whole multiline tree
    assert calc.dependency_tree._summary().startswith("p<")


def test_calculator_base_repr_html_tail_hint_in_github_plain() -> None:
    from pynbodyext.core.calculate.display import reset_repr_style, set_repr_style

    calc = make_pipeline()
    for style in ("github", "plain"):
        set_repr_style(style)
        try:
            html = calc._repr_html_()
            assert ".config" in html or ".dependency_tree" in html
        finally:
            reset_repr_style()


def test_result_has_detail_view_attributes() -> None:
    """Result must expose .execution_tree/.performance/.cache as view objects."""
    from pynbodyext.core.calculate.display import ViewObject

    result = _result()
    assert isinstance(result.execution_tree, ViewObject)
    assert isinstance(result.performance, ViewObject)
    assert isinstance(result.cache, ViewObject)
    # each view must be renderable and non-empty
    for view in (result.execution_tree, result.performance, result.cache):
        assert repr(view)
        assert isinstance(view._repr_html_(), str)


def test_result_repr_html_tail_hint_in_github_plain() -> None:
    from pynbodyext.core.calculate.display import reset_repr_style, set_repr_style

    result = _result()
    hint = "Use .named, .provenance, .execution_tree, .performance and .cache for details"
    for style in ("github", "plain"):
        set_repr_style(style)
        try:
            html = result._repr_html_()
            assert hint in html
        finally:
            reset_repr_style()
