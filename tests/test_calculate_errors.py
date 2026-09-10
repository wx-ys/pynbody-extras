"""Tests for calculator error diagnostics (traceback + context)."""

from __future__ import annotations

import sys

import pytest
from calculate_helpers import make_sim

from pynbodyext.core.calculate import ErrorPolicy, Pipeline, PropertyBase


@PropertyBase.dataclass
class _Boom(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        raise RuntimeError("boom")


def _failed_result():
    return Pipeline({"bad": _Boom()}, name="p").run(make_sim(), errors=ErrorPolicy.COLLECT_PARTIAL)


def test_collected_error_captures_traceback() -> None:
    """ErrorInfo must carry a traceback so collected errors are diagnosable."""
    result = _failed_result()
    assert result.errors, "expected a collected error"
    error = result.errors[0]
    assert error.error_type == "RuntimeError"
    assert "boom" in error.message
    assert error.phase == "calculate"
    assert error.traceback_text, "traceback_text must be populated"
    assert "boom" in error.traceback_text


@pytest.mark.skipif(sys.version_info < (3, 11), reason="add_note requires Python 3.11+")
def test_raised_error_carries_location_note() -> None:
    """RAISE policy should annotate the propagated exception with node/phase."""
    with pytest.raises(RuntimeError) as excinfo:
        Pipeline({"bad": _Boom()}, name="p").run(make_sim(), errors=ErrorPolicy.RAISE)
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("pynbodyext.calculate" in note for note in notes), notes
