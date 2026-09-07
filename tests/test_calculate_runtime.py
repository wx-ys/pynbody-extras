"""Characterization tests for the runtime engine and transform lifecycle.

These pin the currently-observable runtime behaviour so the engine and the
calculator mixins can be split safely.  They intentionally use a small, custom
transform and a failing node so the apply lifecycle and the engine error
policies are exercised even though the existing suite did not cover them.

Behavioural note: as of this snapshot a standalone ``.run()`` applies a bound
transform but does *not* auto-revert it after a successful run (cleanup is
only wired on some chain/error paths).  These tests pin that observed contract
so a refactor cannot silently change it.
"""

from __future__ import annotations

import numpy as np
import pynbody
from pynbody.transformation import GenericTranslation

from pynbodyext.core.calculate import ErrorPolicy, Pipeline, PropertyBase, RevertPolicy, TransformBase


def make_sim() -> pynbody.SimSnap:
    sim = pynbody.new(5)
    sim["x"] = np.arange(5.0)
    sim["mass"] = np.arange(5.0)
    return sim


@TransformBase.dataclass
class ShiftX(TransformBase[object]):
    dx: float = 1.0

    def build_handle(self, sim, target, params=None) -> object:
        return GenericTranslation(target, "x", -self.dx, description="ShiftX")


@PropertyBase.dataclass
class RangeMean(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(np.asarray(sim["x"]).mean())


@PropertyBase.dataclass
class MassMax(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(np.asarray(sim["mass"]).max())


@PropertyBase.dataclass
class AlwaysFails(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        raise RuntimeError("boom")


def test_transform_is_applied_while_computing() -> None:
    """A bound transform is applied to the data seen during compute."""
    sim = make_sim()
    before = np.asarray(sim["x"]).copy()
    result = RangeMean().transform(ShiftX(1.0)).run(sim)
    # The property read the shifted array at compute time.
    assert result.value == np.asarray(before - 1.0).mean()


def test_transform_revert_never_keeps_mutation() -> None:
    """RevertPolicy.NEVER leaves the transform applied after the run."""
    sim = make_sim()
    before = np.asarray(sim["x"]).copy()
    _ = RangeMean().transform(ShiftX(1.0).revert(RevertPolicy.NEVER)).run(sim)
    assert np.allclose(np.asarray(sim["x"]), before - 1.0)


def test_engine_raise_policy_propagates() -> None:
    """Default RAISE policy re-raises the first node error."""
    sim = make_sim()
    pipe = Pipeline({"ok": MassMax(), "bad": AlwaysFails()}, name="p")
    try:
        pipe.run(sim, errors=ErrorPolicy.RAISE)
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")


def test_engine_collect_policy_records_errors() -> None:
    """COLLECT policy records the failing node and the root as errors."""
    sim = make_sim()
    pipe = Pipeline({"ok": MassMax(), "bad": AlwaysFails()}, name="p")
    result = pipe.run(sim, errors=ErrorPolicy.COLLECT)
    assert not result.ok
    assert result.value is None
    error_labels = {node.label for node in result.find_error_nodes()}
    assert {"p", "AlwaysFails"}.issubset(error_labels)
    # The root is recorded as an error because it could not be assembled.
    assert result.root.status.value == "error"


def test_engine_collect_partial_keeps_successful_values() -> None:
    """COLLECT_PARTIAL keeps successful outputs and marks only failures."""
    sim = make_sim()
    pipe = Pipeline({"ok": MassMax(), "bad": AlwaysFails()}, name="p")
    result = pipe.run(sim, errors=ErrorPolicy.COLLECT_PARTIAL)
    assert result.value is not None
    assert result.value["ok"] == 4.0
    assert result.value["bad"] is None
    error_nodes = result.find_error_nodes()
    badges = [node for node in error_nodes if node.label == "AlwaysFails"]
    assert len(badges) == 1
    assert "boom" in badges[0].error.message


def test_transform_chain_composes_steps() -> None:
    """Chained transforms compose into a TransformChain in apply order."""
    from pynbodyext.core.calculate.nodes.transforms import TransformChain

    chain = ShiftX(1.0).then(ShiftX(2.0))
    assert isinstance(chain, TransformChain)
    assert len(chain.transforms) == 2
