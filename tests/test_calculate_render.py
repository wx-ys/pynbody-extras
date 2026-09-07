"""Characterization tests for the calculator result presentation layer.

These pin the *stable, deterministic* text output of the result and node
repr/report layer (``core/calculate/result/repr.py`` and the report helpers that
back it).  They exist so the presentation layer can be refactored safely: any
change to a rendered string that users see will be caught here, while timing- and
address-dependent fields are asserted only structurally.
"""

from __future__ import annotations

import re

from pynbodyext.core.calculate.result.repr import ResultRepr

from test_calculate_core import make_pipeline, make_sim


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
