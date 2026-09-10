"""Characterization tests for the lightweight batch execution path.

``CalculatorBase.batch`` and the underlying ``EvalEngine.run_light`` /
``_evaluate_light`` path are not covered elsewhere.  These pin the observable
behavior so that splitting the engine's execution pipeline (evaluate ->
materialize -> assemble) cannot silently change batch semantics.
"""

from __future__ import annotations

from calculate_helpers import MassSum, make_pipeline, make_sim

from pynbodyext.core.calculate import ErrorPolicy, PropertyBase


def test_batch_returns_public_value_per_sim() -> None:
    """``batch`` yields a callable that returns the public value for each sim."""
    pipeline = make_pipeline()
    with pipeline.batch(cache=False) as run_one:
        first = run_one(make_sim())
        second = run_one(make_sim())
    assert first == {"m": 10.0, "t": 20.0}
    assert second == {"m": 10.0, "t": 20.0}


def test_batch_minimal_node_uses_run_minimal_path() -> None:
    """A node with no dependencies takes the ultra-minimal engine path."""
    calc = MassSum()
    with calc.batch(cache=False) as run_one:
        value = run_one(make_sim())
    assert value == 15.0


def test_run_light_matches_run_value() -> None:
    """``EvalEngine.run_light`` and ``run`` agree on the public value."""
    from pynbodyext.core.calculate.runtime.engine import EvalEngine

    calc = MassSum()
    sim = make_sim()
    engine = EvalEngine()
    from pynbodyext.core.calculate.runtime.options import RunOptions

    light = engine.run_light(calc, sim, RunOptions(), _precomputed_node_sig=calc.signature())
    full = engine.run(calc, sim).value
    assert light == full == 15.0


def test_batch_propagates_raise_error_policy() -> None:
    """A failing node under RAISE policy must raise through the batch caller."""
    sim = make_sim()

    @PropertyBase.dataclass
    class AlwaysFails(PropertyBase[float]):
        def calculate(self, sim, params=None) -> float:
            raise RuntimeError("boom")

    with AlwaysFails().batch(cache=False, errors=ErrorPolicy.RAISE) as run_one:
        try:
            run_one(sim)
            raise AssertionError("expected RuntimeError to propagate")
        except RuntimeError as exc:
            assert "boom" in str(exc)


def test_batch_collect_policy_returns_none_on_error() -> None:
    """Under a non-RAISE policy the batch caller swallows errors (returns None)."""
    sim = make_sim()

    @PropertyBase.dataclass
    class AlwaysFails(PropertyBase[float]):
        def calculate(self, sim, params=None) -> float:
            raise RuntimeError("boom")

    with AlwaysFails().batch(cache=False, errors=ErrorPolicy.COLLECT) as run_one:
        assert run_one(sim) is None
