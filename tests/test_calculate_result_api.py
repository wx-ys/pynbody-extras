"""Tests for the consolidated Result reports / diagnostics / observations views."""

from __future__ import annotations

from calculate_helpers import make_pipeline, make_sim

from pynbodyext.core.calculate.result.result import Result, ResultNode
from pynbodyext.core.calculate.result.views import DiagnosticsView, ObservationsView, ReportsView


def _result():
    return make_pipeline().run(make_sim())


def test_reports_view_is_dict_like() -> None:
    reports = _result().reports
    assert isinstance(reports, ReportsView)
    assert "execution_tree" in reports
    assert set(reports.names()) == set(reports.keys())
    assert reports["cache"] == reports.get("cache")


def test_diagnostics_view_typed_accessors() -> None:
    diagnostics = _result().diagnostics
    assert isinstance(diagnostics, DiagnosticsView)
    assert isinstance(diagnostics.cache(), list)
    assert isinstance(diagnostics.trace(), list)
    assert isinstance(diagnostics.logs(), list)
    assert isinstance(diagnostics.observer(), list)
    assert diagnostics.cache() == diagnostics["cache_events"]


def test_observations_view_lookup() -> None:
    result = _result()
    observations = result.observations
    assert isinstance(observations, ObservationsView)
    root = result.root
    assert observations.of(root) is observations.get(root.node_id)
    # name resolution goes through the owning Result
    assert observations.of("p") is observations.get(root.node_id)
    assert observations.all()


def test_consolidated_result_members_removed() -> None:
    for name in (
        "report",
        "available_reports",
        "diagnostic",
        "available_diagnostics",
        "trace_events",
        "cache_events",
        "log_events",
        "observer_events",
        "observation_of",
        "access_observations",
    ):
        assert not hasattr(Result, name), f"Result.{name} should be folded into a view"


def test_result_node_fields_are_grouped() -> None:
    """ResultNode exposes grouped sub-objects instead of 21 flat fields."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(ResultNode)}
    assert {"class_info", "record", "run"} <= names
    flat = {
        "calculator_type",
        "calculator_class_path",
        "semantic_calculator_class_path",
        "record_policy",
        "raw_value",
        "value",
        "value_summary",
        "stored_raw",
        "stored_value",
        "phases",
        "artifacts",
        "observation",
        "error",
    }
    assert names & flat == set(), f"ungrouped fields remain: {names & flat}"
