"""Value-consuming property nodes keep the existing DAG/runtime/store contracts."""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray
from pynbody.transformation import GenericTranslation

import pynbodyext.calculate as public
import pynbodyext.core.calculate as core
import pynbodyext.core.calculate.nodes as nodes
from pynbodyext.calculate import PostprocessorBase, PropertyBase
from pynbodyext.core.calculate import (
    InMemoryResultStore,
    LambdaProperty,
    Param,
    Pipeline,
    RecordRef,
    RevertPolicy,
    TransformBase,
    calculator_from_signature,
)
from pynbodyext.core.calculate import interfaces


@PropertyBase.dataclass
class ArrayValues(PropertyBase[SimArray]):
    offset: float = 0.0
    calls: ClassVar[int] = 0

    def calculate(self, sim, params=None):
        type(self).calls += 1
        return sim["mass"].copy() + SimArray(self.offset, sim["mass"].units)


@PropertyBase.dataclass
class PositionValues(PropertyBase[SimArray]):
    calls: ClassVar[int] = 0

    def calculate(self, sim, params=None):
        type(self).calls += 1
        return sim["pos"].copy()


@PostprocessorBase.dataclass
class ScaleValues(PostprocessorBase[SimArray]):
    factor: float = 2.0

    def process(self, value):
        return value * self.factor


@PostprocessorBase.dataclass
class OffsetValues(ScaleValues):
    offset: float = 1.0

    def process(self, value):
        return super().process(value) + SimArray(self.offset, value.units)


@PostprocessorBase.dataclass
class WeightedValues(PostprocessorBase[Any]):
    weights: np.ndarray

    def process(self, value):
        return value


@PostprocessorBase.dataclass
class IdentityValues(PostprocessorBase[Any]):
    def process(self, value):
        return value


@PostprocessorBase.dataclass
class FailValues(PostprocessorBase[Any]):
    def process(self, value):
        raise RuntimeError("postprocessor failed")


@TransformBase.dataclass
class ShiftPositions(TransformBase[object]):
    dx: float = 1.0
    cleanup_order: ClassVar[list[float]] = []

    def build_handle(self, sim, target, params=None):
        return GenericTranslation(target, "pos", np.array([self.dx, 0.0, 0.0]), description="test-shift")

    def cleanup(self, ctx, handle):
        ShiftPositions.cleanup_order.append(self.dx)
        super().cleanup(ctx, handle)


@TransformBase.dataclass
class NonrevertibleShift(ShiftPositions):
    def is_revertible(self, handle):
        return False


@TransformBase.dataclass
class CleanupFails(ShiftPositions):
    def cleanup(self, ctx, handle):
        raise ValueError("cleanup failed deliberately")


@pytest.fixture(autouse=True)
def reset_calls():
    ArrayValues.calls = 0
    PositionValues.calls = 0
    ShiftPositions.cleanup_order.clear()


def mass_sim():
    sim = pynbody.new(3)
    sim["mass"] = SimArray([1.0, 2.0, 4.0], "Msol")
    sim["pos"] = SimArray(np.zeros((3, 3)), "kpc")
    return sim


def test_exports_are_the_same_role():
    for module in (public, core, nodes):
        assert module.PostprocessorBase is PostprocessorBase
        assert "PostprocessorBase" in module.__all__
    assert issubclass(PostprocessorBase, PropertyBase)


def test_source_is_required_keyword_only_and_inherited_with_static_defaults():
    assert inspect.signature(ScaleValues).parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="source"):
        ScaleValues()
    with pytest.raises(TypeError):
        ScaleValues(3.0, ArrayValues())
    assert ScaleValues(3.0, source=ArrayValues()).factor == 3.0
    original = OffsetValues(3.0, 2.0, source=ArrayValues())
    rebuilt = calculator_from_signature(original.to_signature().to_json())
    assert type(rebuilt) is OffsetValues
    assert rebuilt.factor == 3.0 and rebuilt.offset == 2.0
    assert rebuilt.to_signature().hash == original.to_signature().hash
    np.testing.assert_array_equal(rebuilt.run(mass_sim()).value, [5.0, 8.0, 14.0])


def test_units_and_signature_roundtrip():
    calc = ScaleValues(3.0, source=ArrayValues())
    original = calc.run(mass_sim()).value
    rebuilt = calculator_from_signature(calc.to_signature().to_json())
    repeated = rebuilt.run(mass_sim()).value
    assert isinstance(original, SimArray)
    assert original.units == repeated.units == SimArray([], "Msol").units
    np.testing.assert_array_equal(original, [3.0, 6.0, 12.0])
    np.testing.assert_array_equal(repeated, original)
    assert type(rebuilt) is ScaleValues
    assert rebuilt.to_signature().payload == calc.to_signature().payload
    assert rebuilt.to_signature().hash == calc.to_signature().hash


@pytest.mark.parametrize("value", [SimArray([4.0], "Msol"), SimArray([1.0, 2.0], "kpc"), None])
def test_resolved_value_is_delivered_unchanged(value):
    source = LambdaProperty(lambda sim: value)
    assert IdentityValues(source=source).run(mass_sim()).value is value


def test_two_pipeline_outputs_share_the_upstream_once():
    source = ArrayValues()
    result = Pipeline({"two": ScaleValues(source=source), "three": ScaleValues(3.0, source=source)}).run(mass_sim())
    assert ArrayValues.calls == 1
    np.testing.assert_array_equal(result.value["two"], [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(result.value["three"], [3.0, 6.0, 12.0])


@pytest.mark.parametrize("source", [None, 2.0, lambda sim: 1, [ArrayValues()], {"source": ArrayValues()}])
def test_source_must_be_one_calculator(source):
    with pytest.raises(TypeError, match="source.*Calculator"):
        ScaleValues(source=source)


def test_result_and_record_reference_sources_are_refused():
    result = ArrayValues().run(mass_sim(), store=InMemoryResultStore())
    assert isinstance(result.store_ref, RecordRef)
    for source in (result, result.store_ref):
        with pytest.raises(TypeError, match="source.*Calculator"):
            ScaleValues(source=source)


def test_additional_dynamic_param_is_refused():
    @PostprocessorBase.dataclass
    class ExtraDynamic(IdentityValues):
        extra: Param[float] = 1.0

    with pytest.raises(TypeError, match="only.*dynamic"):
        ExtraDynamic(source=ArrayValues())


def test_manually_declared_dynamic_spec_is_refused():
    @PostprocessorBase.dataclass
    class ExtraSpec(IdentityValues):
        dynamic_param_specs = {"extra": None}
        extra: float = 1.0

    with pytest.raises(TypeError, match="only.*dynamic"):
        ExtraSpec(source=ArrayValues())


@pytest.mark.parametrize("declaration", [Param.static(kw_only=True), Param(signature=False, kw_only=True)])
def test_source_cannot_lose_dynamic_signature_identity(declaration):
    @PostprocessorBase.dataclass
    class InvalidSource(IdentityValues):
        source: Any = declaration

    with pytest.raises(TypeError, match="source.*signature"):
        InvalidSource(source=ArrayValues())


@pytest.mark.parametrize(
    "declaration",
    [
        Param(field_name="mass", kw_only=True),
        Param(target_units="Msol", kw_only=True),
        Param(optional_units=False, kw_only=True),
    ],
)
def test_source_unit_conversion_is_refused(declaration):
    @PostprocessorBase.dataclass
    class ConvertingSource(IdentityValues):
        source: Any = declaration

    with pytest.raises(TypeError, match="source.*unit"):
        ConvertingSource(source=ArrayValues())


@pytest.mark.parametrize("base", [PostprocessorBase, ScaleValues])
def test_calculate_override_is_refused_before_signature_registration(base):
    registered = dict(interfaces._SIGNATURE_TYPES)
    with pytest.raises(TypeError, match="process"):

        class BypassAdapter(base):
            def calculate(self, sim, params=None):
                return 123

            def process(self, value):
                return value

    assert interfaces._SIGNATURE_TYPES == registered


def test_source_parameters_and_algorithm_version_change_signature(monkeypatch):
    calc = ScaleValues(source=ArrayValues())
    original = calc.to_signature()
    assert original.hash == ScaleValues(source=ArrayValues()).to_signature().hash
    assert original.hash != ScaleValues(3.0, source=ArrayValues()).to_signature().hash
    assert original.hash != ScaleValues(source=ArrayValues(1.0)).to_signature().hash
    calc.name = "label only"
    assert calc.to_signature().hash == original.hash
    monkeypatch.setattr(ScaleValues, "signature_version", 1)
    # Existing calculator signatures are memoized; a changed algorithm creates
    # a fresh recipe rather than mutating an already signed live instance.
    current = ScaleValues(source=ArrayValues()).to_signature()
    assert current.hash != original.hash
    assert current.payload["version"] == 1


def test_stability_and_constructibility_remain_independent():
    stable = ScaleValues(source=ArrayValues()).to_signature()
    assert stable.stable and stable.constructible
    unstable = ScaleValues(source=LambdaProperty(lambda sim: sim["mass"])).to_signature()
    assert not unstable.stable and not unstable.constructible
    assert any("source" in path for path in unstable.unstable_paths)
    with pytest.raises(ValueError):
        calculator_from_signature(unstable.to_json())
    hashed = WeightedValues(np.arange(1000.0), source=ArrayValues()).to_signature()
    assert hashed.stable and not hashed.constructible
    assert hashed.hash != WeightedValues(np.arange(1000.0) + 1, source=ArrayValues()).to_signature().hash


@pytest.mark.parametrize("errors", ["collect", "raise"])
def test_process_failure_reverts_transform_and_never_persists(errors):
    sim, store = mass_sim(), InMemoryResultStore()
    before = sim["pos"].copy()
    calc = FailValues(source=ArrayValues()).transform(ShiftPositions())
    identity = ("sim", "failed-postprocessor")
    kwargs = {"errors": errors, "store": store, "sim_identity": lambda sim: identity}
    if errors == "raise":
        with pytest.raises(RuntimeError, match="postprocessor failed"):
            calc.run(sim, **kwargs)
    else:
        result = calc.run(sim, **kwargs)
        assert not result.ok and result.value is None and result.store_ref is None
        assert result.root.status.value == "error"
        assert "postprocessor failed" in result.root.run.error.message
    assert ArrayValues.calls == 1
    np.testing.assert_array_equal(sim["pos"], before)
    assert store.list_refs() == []
    assert store.list_versions(RecordRef(identity, calc.to_signature().hash)) == []


def test_store_hit_preserves_exact_pinned_receipt_and_refresh_history():
    sim, store = mass_sim(), InMemoryResultStore()
    calc = ScaleValues(source=ArrayValues())
    kwargs = {"store": store, "sim_identity": lambda sim: ("sim", "postprocessor-test")}
    first = calc.run(sim, **kwargs)
    second = calc.run(sim, **kwargs)
    assert first.store_ref is not None and first.store_ref.version_id
    assert second.from_store and second.store_ref == first.store_ref
    assert store.list_versions(first.store_ref) == [first.store_ref]
    assert ArrayValues.calls == 1
    assert second.value.units == first.value.units
    np.testing.assert_array_equal(second.value, first.value)
    sim["mass"] *= 2
    refreshed = calc.run(sim, reuse=False, **kwargs)
    assert refreshed.store_ref != first.store_ref and not refreshed.from_store
    assert ArrayValues.calls == 2
    historical = store.fetch(first.store_ref)
    assert historical.store_ref == first.store_ref
    np.testing.assert_array_equal(historical.value, [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(refreshed.value, [4.0, 8.0, 16.0])


@pytest.mark.parametrize("errors", ["collect", "raise"])
def test_chained_transforms_are_cleaned_in_reverse_after_process_failure(errors):
    sim = mass_sim()
    before = sim["pos"].copy()
    calc = FailValues(source=ArrayValues()).transform(ShiftPositions(2.0)).transform(ShiftPositions(3.0))
    if errors == "raise":
        with pytest.raises(RuntimeError, match="postprocessor failed"):
            calc.run(sim, errors=errors)
    else:
        assert not calc.run(sim, errors=errors).ok
    np.testing.assert_array_equal(sim["pos"], before)
    # pynbody's last native handle owns the compound transform. Its revert
    # unwinds both; TransformChain skips the already-reverted earlier handle.
    assert ShiftPositions.cleanup_order == [3.0]


@pytest.mark.parametrize("policy", ["scope-never", "transform-never", "chain-never", "nonrevertible"])
def test_cleanup_respects_explicit_nonreverting_transforms(policy):
    sim = mass_sim()
    transform = ShiftPositions()
    kwargs = {}
    expected_dx = 1.0
    if policy == "scope-never":
        kwargs["revert"] = RevertPolicy.NEVER
    elif policy == "transform-never":
        transform = transform.revert(RevertPolicy.NEVER)
    elif policy == "chain-never":
        transform = transform.then(ShiftPositions(2.0)).revert(RevertPolicy.NEVER)
        expected_dx = 3.0
    else:
        transform = NonrevertibleShift()
    result = FailValues(source=ArrayValues()).transform(transform, **kwargs).run(sim, errors="collect")
    assert not result.ok
    np.testing.assert_array_equal(sim["pos"][:, 0], [expected_dx] * 3)


@pytest.mark.parametrize("errors", ["collect", "raise"])
def test_cleanup_failure_preserves_original_process_error(errors, caplog):
    calc = FailValues(source=ArrayValues()).transform(CleanupFails())
    if errors == "raise":
        with pytest.raises(RuntimeError, match="postprocessor failed"):
            calc.run(mass_sim(), errors=errors)
    else:
        result = calc.run(mass_sim(), errors=errors)
        assert result.root.run.error.error_type == "RuntimeError"
        assert result.root.run.error.message == "postprocessor failed"
    assert "cleanup failed deliberately" in caplog.text


def test_inherited_ancestor_transform_is_not_cleaned_by_child_processor():
    sim = mass_sim()
    child = FailValues(source=ArrayValues())
    # The Pipeline owns this scope; the child receives it through NodeInput.
    parent = Pipeline({"bad": child}).transform(ShiftPositions())
    assert not parent.run(sim, errors="collect").ok
    np.testing.assert_array_equal(sim["pos"][:, 0], [1.0] * 3)
    assert ShiftPositions.cleanup_order == []


def test_native_cleanup_callback_retains_its_existing_cascading_semantics():
    sim = mass_sim()
    child = FailValues(source=ArrayValues()).transform(ShiftPositions(2.0))
    parent = Pipeline({"bad": child}).transform(ShiftPositions(1.0))
    assert not parent.run(sim, errors="collect").ok
    # The adapter invokes only the child's callback, but that native handle
    # represents a compound transformation and its revert also unwinds parent.
    np.testing.assert_array_equal(sim["pos"][:, 0], [0.0] * 3)
    assert ShiftPositions.cleanup_order == [2.0]


def test_native_never_step_can_prevent_earlier_cleanup_without_masking_error(caplog):
    sim = mass_sim()
    transform = ShiftPositions(2.0).then(ShiftPositions(3.0).revert(RevertPolicy.NEVER))
    result = FailValues(source=ArrayValues()).transform(transform).run(sim, errors="collect")
    assert not result.ok
    assert result.root.run.error.message == "postprocessor failed"
    # Native deregistration requires the latest transform first. The retained
    # NEVER step prevents reverting the older handle, so cleanup logs failure.
    np.testing.assert_array_equal(sim["pos"][:, 0], [5.0] * 3)
    assert ShiftPositions.cleanup_order == [2.0]
    assert "transform cleanup failed after error" in caplog.text


def test_collect_partial_cleanup_leaves_successful_sibling_science_correct():
    sim, store = mass_sim(), InMemoryResultStore()
    source = PositionValues()
    pipeline = Pipeline(
        {
            "bad": FailValues(source=source).transform(ShiftPositions(2.0)),
            "good": ScaleValues(3.0, source=source).transform(ShiftPositions(5.0)),
        }
    )
    result = pipeline.run(sim, errors="collect_partial", store=store)
    assert not result.ok and result.value["bad"] is None
    np.testing.assert_array_equal(result.value["good"][:, 0], [15.0] * 3)
    assert PositionValues.calls == 2
    # Successful property scopes retain their established applied-state behavior.
    np.testing.assert_array_equal(sim["pos"][:, 0], [5.0] * 3)
    assert result.store_ref is None and store.list_refs() == []
