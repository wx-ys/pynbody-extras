"""Tests for the calculator core engine and signature behavior.

These exercise the runtime engine and the structured-signature path directly,
independent of the ``pynbodyext.calculate`` facade (which is covered by other
modules).
"""

from __future__ import annotations

import numpy as np
import pynbody

from pynbodyext.core.calculate import FilterBase, Param, Pipeline, PropertyBase


def make_sim() -> pynbody.SimSnap:
    sim = pynbody.new(6)
    sim["mass"] = np.arange(6.0)
    sim["temp"] = np.arange(6.0) * 10.0
    sim["r"] = np.arange(6.0)
    return sim


@PropertyBase.dataclass
class MassSum(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(sim["mass"].sum())


@FilterBase.dataclass
class RBelow(FilterBase):
    radius: Param[float] = Param(field_name="r")

    def calculate(self, sim, params=None) -> bool:
        return sim["r"] < self.radius


@PropertyBase.dataclass
class TempMean(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(sim["temp"].mean())


def make_pipeline(root_name: str = "p") -> Pipeline:
    shared = RBelow(5.0)
    return Pipeline(
        {"m": MassSum().filter(shared), "t": TempMean().filter(shared)},
        name=root_name,
    )


def test_engine_computes_structured_signature_once_per_node(monkeypatch) -> None:
    """Each evaluated node's structured signature is computed once per run.

    Regression guard: the engine used to call ``CalculatorBase.to_signature``
    twice per node (once for the cache-key plan, once when creating the
    :class:`ResultNode`) plus a third time for the root provenance.  With a
    cache-disabled run, every node is evaluated exactly once, so the number of
    engine-level signature computations must equal the number of registered
    result nodes.
    """
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

    original = CalculatorBase.to_signature
    calls: list[str] = []

    def counting(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CalculatorBase, "to_signature", counting)

    sim = make_sim()
    result = make_pipeline().run(sim, cache=False)

    evaluated = len(result.nodes)
    assert calls, "engine should have requested at least one structured signature"
    assert len(calls) == evaluated, (
        f"expected {evaluated} structured-signature computations (one per node), "
        f"got {len(calls)}; a node signature was computed more than once"
    )


def test_provenance_and_result_signatures_stay_correct() -> None:
    """Refactoring must not change the provenance or result signatures."""
    calc = make_pipeline()
    sim = make_sim()
    result = calc.run(sim)

    expected_hash = calc.signature_hash()
    assert result.provenance is not None
    assert result.provenance.calculator_signature_hash == expected_hash
    assert result.provenance.calculator_signature_text == calc.signature_text()
    assert result.root.signature == calc.signature()
    mask = sim["r"] < 5.0
    assert result.value["m"] == float(sim["mass"][mask].sum())
    assert result.value["t"] == float(sim["temp"][mask].mean())


def test_sim_identity_is_pluggable() -> None:
    """The simulation identity used for provenance is supplied by a pluggable provider.

    The default provider is id-based (unique per object in-process), but a
    persistence-aware caller can inject a stable, address-based identity so the
    stored provenance survives restarts.  This is the seam the future database
    store plugs into.
    """
    from pynbodyext.core.calculate.runtime.engine import EvalEngine

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    engine = EvalEngine(sim_identity=path_identity)
    sim = make_sim()
    assert engine.make_sim_signature(sim) == ("sim", "snapshot_103", "halo_0")


def test_sim_identity_flows_to_provenance() -> None:
    """An injected identity provider is used in the run's stored provenance."""
    from pynbodyext.core.calculate.runtime.engine import EvalEngine

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    engine = EvalEngine(sim_identity=path_identity)
    sim = make_sim()
    result = engine.run(MassSum(), sim)
    assert result.provenance is not None
    assert result.provenance.sim_signature == ("sim", "snapshot_103", "halo_0")


def test_default_sim_identity_is_id_based() -> None:
    """The default provider stays id-based so cache semantics are unchanged."""
    from pynbodyext.core.calculate.runtime.engine import EvalEngine

    sim = make_sim()
    assert EvalEngine().make_sim_signature(sim) == ("sim", id(sim))


def test_signature_cache_key_is_memoized(monkeypatch) -> None:
    """Deriving the cache key freezes the payload once, not once per call.

    The engine reuses a single :class:`CalculatorSignature` across the plan,
    the result node, and provenance.  Freezing the (immutable) payload must
    therefore be memoized so the repeated ``cache_key()`` requests do not
    re-walk the whole payload each time.
    """
    from pynbodyext.core.calculate import calculator_to_signature
    from pynbodyext.core.calculate.result import signature as sigmod

    calc = make_pipeline()
    sig = calculator_to_signature(calc)

    original = sigmod._freeze_signature_value
    calls: list[int] = []

    def counting(value):
        calls.append(1)
        return original(value)

    monkeypatch.setattr(sigmod, "_freeze_signature_value", counting)

    key1 = sig.cache_key()
    first_freezes = len(calls)
    assert first_freezes > 0, "first cache_key() should freeze the payload"

    key2 = sig.cache_key()
    assert key1 == key2
    assert len(calls) == first_freezes, (
        "second cache_key() must not re-freeze the payload; "
        "freezing should be memoized on the CalculatorSignature"
    )


def test_calculator_base_mixin_surface_is_intact() -> None:
    """The god-object split must not remove anything from the public surface.

    :class:`CalculatorBase` was decomposed into focused mixins (signature,
    graph/params, logging, display, run, compose).  This guards the refactor:
    every public entry point and the ``super()``-based composition chain must
    still resolve on a concrete dataclass calculator and on the wrappers.
    """
    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.nodes.filters import FilterBase
    from pynbodyext.core.calculate.nodes.runtime_base import RuntimeCalculatorBase
    from pynbodyext.core.calculate.nodes.transforms import TransformBase

    calc = make_pipeline()

    # Mixin-provided surface still present on a concrete instance.
    for meth in (
        "run", "__call__", "value", "batch", "signature", "signature_text",
        "signature_hash", "to_signature", "from_signature", "format_tree",
        "named", "record", "filter", "transform", "keep", "options",
        "config", "dependency_tree", "dependencies", "children",
        "resolve_params_for_sim", "resolve_dynamic_params", "resolve_dynamic_param",
        "has_dynamic_param", "dynamic_param_names", "dynamic_param_spec",
        "is_dynamic_value", "debug", "info", "warning", "error",
    ):
        assert hasattr(calc, meth), f"CalculatorBase lost public method {meth!r}"

    # ``RuntimeCalculatorBase`` overrides with_filter/transform and calls
    # ``super()``; the mixin must be reachable through the composed MRO.
    assert issubclass(FilterBase, RuntimeCalculatorBase)
    bound = MassSum().filter(RBelow(5.0))
    assert isinstance(bound, MassSum)  # filter preserves the concrete type
    assert bound.scope.filter is not None
    assert issubclass(TransformBase, CalculatorBase)

    # Wrapper classes that override relocated methods still resolve correctly.
    from pynbodyext.core.calculate.nodes.base import CombinedCalculator

    combined = MassSum() & MassSum()
    assert isinstance(combined, CombinedCalculator)
    assert combined.children()  # declared_dependencies -> items

    # Signature helpers still wired through the signature mixin.
    assert calc.signature() == calc.to_signature().cache_key()
