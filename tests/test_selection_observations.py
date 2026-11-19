"""Selection origins freeze the particles and coordinates actually used by a computation."""

import gc
import json
import weakref

import numpy as np
import pynbody
import pytest
from pynbody import units
from pynbody.analysis.angmom import calc_faceon_matrix
from pynbody.array import SimArray

from pynbodyext.calculate import FilterBase, Param, Pipeline, PropertyBase, TransformBase
from pynbodyext.filters import Disc, FamilyFilter, Sphere
from pynbodyext.properties import AngMomVec, ParamContain, ParamSum
from pynbodyext.transforms import AlignVec, ShiftPosTo, ShiftVelTo, WrapBox


@PropertyBase.dataclass
class CountMembers(PropertyBase):
    def calculate(self, sim, params=None):
        return len(sim)


@FilterBase.dataclass
class CountedAperture(FilterBase):
    radius: Param[float] = Param(field_name="pos")
    calls = 0

    def calculate(self, sim, params=None):
        type(self).calls += 1
        return np.asarray(sim["r"]) < params.radius


def snapshot():
    sim = pynbody.new(dm=3, star=3)
    xyz = np.zeros((6, 3))
    xyz[:, 0] = [10, 20, 30, 1, 2, 6]
    sim["pos"] = SimArray(xyz, "kpc")
    sim["vel"] = SimArray(np.arange(18.0).reshape(6, 3), "km s^-1")
    sim["mass"] = SimArray([10.0, 10.0, 10.0, 1.0, 1.0, 1.0], "Msol")
    return sim


def observations(result):
    assert getattr(result, "selection_observations", None) is not None, "missing actual selection observations"
    return result.selection_observations


def test_dynamic_aperture_uses_actual_input():
    sim = snapshot()
    rhalf = ParamContain(frac=0.5, cal_key="rxy", parameter="mass")
    star, aperture = FamilyFilter("star"), Sphere(2 * rhalf)
    mass = ParamSum("mass")
    assert float(rhalf.run(sim).value) == 14.0
    assert float(rhalf.run(sim.star).value) == 2.0
    joint = mass.filter(star & aperture)
    assert mass.filter(star).filter(aperture).to_signature().identity() == joint.to_signature().identity()
    all_result = joint.run(sim)
    nested = Pipeline({"mass": mass.filter(aperture)}).filter(star).run(sim)
    assert float(all_result.value) == 3.0
    assert float(nested.value["mass"]) == 2.0
    combined, branches = observations(all_result), observations(nested)
    assert len(combined.origins) == 1
    outer, inner = branches.origins
    assert outer.indices == inner.parent_indices == (3, 4, 5)
    assert inner.parent_origin_id == outer.origin_id
    assert inner.indices == (3, 4)
    assert outer.parent_indices is combined.origins[0].parent_indices is None
    assert json.loads(combined.origins[0].filter_signature) == (star & aperture).to_signature().as_dict(full=True)
    for origin, expected in ((combined.origins[0], 28), (inner, 4)):
        (radius,) = (p for p in origin.parameters if p.path[-1] == "radius")
        assert float(units.Unit(radius.units).ratio("kpc")) * radius.value == expected
    assert [p.path for p in inner.parameters if p.path[-1] == "radius"] == [("radius",)]
    assert len(next(p.path for p in combined.origins[0].parameters if p.path[-1] == "radius")) == 2
    assert {u.output_path for u in branches.uses} == {("mass",)}
    assert len({u.occurrence_path for u in branches.uses}) == 2


@pytest.mark.parametrize("observe", [True, False])
def test_shared_filter_and_cached_property_retain_each_output_use(observe):
    sim = snapshot()
    CountedAperture.calls = 0
    aperture = CountedAperture(5.0)
    # The second field reuses the entire leaf, including its inner filter.
    leaf = CountMembers().filter(aperture)
    result = (
        Pipeline({"pair": Pipeline({"a": leaf, "b": leaf}), "all_stars": CountMembers()})
        .filter(FamilyFilter("star"))
        .run(sim, observe=observe)
    )
    assert result.value == {"pair": {"a": 2, "b": 2}, "all_stars": 3}
    assert CountedAperture.calls == 1
    facts = observations(result)
    assert len(facts.origins) == 2
    parent, child = facts.origins
    assert parent.indices == (3, 4, 5) and child.indices == (3, 4)
    assert child.parent_origin_id == parent.origin_id
    assert {u.output_path for u in facts.uses if u.origin_id == parent.origin_id} == {
        ("pair", "a"),
        ("pair", "b"),
        ("all_stars",),
    }
    children = [u for u in facts.uses if u.origin_id == child.origin_id]
    assert {u.output_path for u in children} == {("pair", "a"), ("pair", "b")}
    assert len({u.occurrence_path for u in children}) == 2
    assert len(facts.uses) == 5
    assert {u.role for u in facts.uses} == {"output_filter"}


@pytest.mark.parametrize("negate", [False, True])
def test_only_a_consumed_view_creates_a_filter_origin(negate):
    a, b = CountedAperture(5.0), FamilyFilter("star")
    filt = ~(a & b) if negate else a | b
    CountedAperture.calls = 0
    sim = snapshot()
    mask = filt.run(sim)
    assert CountedAperture.calls == 1
    recorded = observations(mask)
    assert recorded.origins == recorded.uses == ()
    result = CountMembers().filter(filt).run(sim)
    assert CountedAperture.calls == 2
    (origin,) = observations(result).origins
    assert origin.indices == tuple(np.flatnonzero(mask.value))
    assert json.loads(origin.filter_signature) == filt.to_signature().as_dict(full=True)
    (small,) = (p for p in origin.parameters if p.path[-1] == "radius")
    assert small.value == 5.0
    assert small.path == (("child", "left", "radius") if negate else ("left", "radius"))


def test_measurement_filter_precedes_rotation_and_each_leaf_has_its_frame():
    sim = snapshot()
    sim["pos"][3:] = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
    sim["vel"][3:] = [[0, 0, 1], [0, -1, 0], [0, 0, 1]]
    stars = FamilyFilter("star")
    face_on = AlignVec(AngMomVec()).measure_with(stars)
    thin = Disc(4.0, 0.1)
    result = Pipeline({"thin": CountMembers().transform(face_on).filter(thin), "native": CountMembers()}).run(sim)
    facts = observations(result)
    before, after = facts.origins
    assert before.indices == after.indices == (3, 4, 5)
    assert before.context.steps == ()
    assert [(use.output_path, use.role) for use in facts.uses] == [
        (("thin",), "transform_measure"),
        (("thin",), "output_filter"),
    ]
    leaf = {o.output_path: o.context for o in facts.outputs}
    assert leaf[("thin",)] == after.context
    assert leaf[("native",)] == before.context
    assert after.context.complete
    assert after.context.family_layout == (("dm", 3), ("star", 3))
    (step,) = after.context.steps
    assert step.kind == "rotation" and step.target_indices is None
    assert json.loads(step.signature) == face_on.to_signature().as_dict(full=True)
    np.testing.assert_allclose(step.values, calc_faceon_matrix([1.0, 0.0, 0.0], up=[0.0, 1.0, 0.0]))


def test_repeated_filter_in_two_roles_preserves_both_actual_uses():
    sim = snapshot()
    filt = FamilyFilter("star")
    transform = AlignVec(np.array([1.0, 0.0, 0.0])).measure_with(filt)
    facts = observations(CountMembers().transform(transform).filter(filt).run(sim))
    assert {u.role for u in facts.uses} == {"transform_measure", "output_filter"}
    assert len({u.occurrence_path for u in facts.uses}) == 2
    assert len(facts.origins) == 2
    assert {len(o.context.steps) for o in facts.origins} == {0, 1}
    assert {o.indices for o in facts.origins} == {(3, 4, 5)}


def test_ordered_targets_shifts_and_indices_detach_from_live_particles():
    sim = snapshot()
    parent = sim[[5, 3, 5, 4]]
    center = np.array([1.0, 0.0, 0.0])
    velocity = np.array([0.0, 2.0, 0.0])
    prop = (
        CountMembers()
        .filter(CountedAperture(8.0))
        .transform(ShiftPosTo(center, move_all=False))
        .transform(ShiftVelTo(velocity))
    )
    ref = weakref.ref(sim)
    result = Pipeline({"measured": prop, "restored": CountMembers()}).run(parent)
    assert result.value == {"measured": 4, "restored": 4}
    facts = observations(result)
    (origin,) = facts.origins
    assert origin.parent_indices == origin.indices == (5, 3, 5, 4)
    assert origin.context.root_signature == result._save_input.root_signature
    assert origin.context.root_count == 6
    p, v = origin.context.steps
    assert (p.kind, p.target_indices) == ("shift_pos", (5, 3, 5, 4))
    assert (v.kind, v.target_indices) == ("shift_vel", None)
    assert units.Unit(p.units) == units.Unit("kpc")
    assert units.Unit(v.units) == units.Unit("km s^-1")
    assert origin.context.complete
    sim["pos"] += 200
    center[:] = velocity[:] = 30
    del sim, parent
    gc.collect()
    assert ref() is None
    assert p.values == (-1.0, 0.0, 0.0) and v.values == (0.0, -2.0, 0.0)
    assert origin.indices == (5, 3, 5, 4)
    assert type(facts).from_dict(json.loads(json.dumps(facts.to_dict()))) == facts
    with pytest.raises((TypeError, AttributeError)):
        origin.indices = ()


def test_wrap_external_and_opaque_transform_report_incomplete_steps():
    @TransformBase.dataclass
    class Opaque(TransformBase):
        def build_handle(self, sim, target, params=None):
            return target.translate([1.0, 0.0, 0.0])

    sim = snapshot()
    ext = sim.translate([1.0, 0.0, 0.0])
    try:
        facts = observations(
            Pipeline({"a": CountMembers().transform(Opaque()).transform(WrapBox(90.0)), "b": CountMembers()}).run(sim)
        )
        a, b = facts.outputs
        assert not a.context.complete and not b.context.complete
        assert [s.kind for s in a.context.steps] == ["external", "unsupported", "unsupported"]
        assert [s.kind for s in b.context.steps] == ["external"]
        assert all(s.reason for s in a.context.steps)
    finally:
        ext.revert()


def test_pure_and_unobserved_slices_stay_usable_and_batch_creates_no_evidence(monkeypatch):
    @PropertyBase.dataclass
    class CustomSlice(PropertyBase):
        def calculate(self, sim, params=None):
            return len(sim[:2])

    assert CustomSlice().run([9, 8, 7]).value == 2
    facts = observations(CustomSlice().run(snapshot()))
    assert facts.origins == () and facts.uses == ()
    assert [o.output_path for o in facts.outputs] == [()]
    from pynbodyext.core.calculate.runtime.selection_observations import SelectionRecorder

    def forbidden(*args, **kwargs):
        pytest.fail("lightweight execution must not construct a selection recorder")

    monkeypatch.setattr(SelectionRecorder, "__init__", forbidden)
    with CountMembers().filter(FamilyFilter("star")).batch() as count:
        assert count(snapshot()) == 3


def test_resolved_dependency_occurrences_and_actual_raw_filter_consumption():
    @PropertyBase.dataclass
    class UsesRaw(PropertyBase):
        def declared_dependencies(self):
            return [self.aperture]

        def execute(self, ctx, input):
            real_filter = ctx.raw_value(self.aperture, input)
            first = real_filter.filtered_sim
            assert real_filter.filtered_sim is first
            return len(first)

        aperture = FamilyFilter("star")

    @PropertyBase.dataclass
    class SumInputs(PropertyBase):
        x: Param[float]
        y: Param[float]

        def calculate(self, sim, params=None):
            return params.x + params.y

    source = UsesRaw()
    result = SumInputs(source, source).run(snapshot())
    assert result.value == 6
    facts = observations(result)
    assert len(facts.origins) == 1
    assert facts.origins[0].indices == (3, 4, 5)
    assert len(facts.uses) == 2
    assert {u.occurrence_path[:2] for u in facts.uses} == {("init", "x"), ("init", "y")}


def test_physical_context_equality_preserves_root_identity_types():
    sim = snapshot()
    measure = CountMembers()
    as_bool = observations(measure.run(sim, sim_identity=lambda s: ("sim", True))).outputs[0].context
    as_int = observations(measure.run(sim, sim_identity=lambda s: ("sim", 1))).outputs[0].context
    assert as_bool != as_int


def test_nonpersistable_particle_view_still_runs(monkeypatch):
    sim = snapshot()
    view = sim.star

    def not_addressable(*args):
        raise TypeError("custom view has no root address")

    monkeypatch.setattr(view, "get_index_list", not_addressable)
    result = CountMembers().run(view)
    assert result.value == 3
    assert result.selection_observations is None


def test_external_native_handle_created_during_run_marks_later_measurement_incomplete():
    @PropertyBase.dataclass
    class ExternalMutation(PropertyBase):
        def calculate(self, sim, params=None):
            sim.translate([1.0, 0.0, 0.0])
            return len(sim)

    sim = snapshot()
    try:
        result = Pipeline({"mutation": ExternalMutation(), "science": CountMembers()}).run(sim)
        ctx = {o.output_path: o.context for o in observations(result).outputs}[("science",)]
        assert not ctx.complete
        assert [s.kind for s in ctx.steps] == ["external"]
    finally:
        sim.current_transformation().revert()


def test_shared_cached_filter_copies_its_root_indices_once(monkeypatch):
    from pynbody.snapshot.subsnap import SubSnapBase

    sim = snapshot()
    counts = []
    original = SubSnapBase.get_index_list

    def count(sim, root, *args, **kwargs):
        counts.append(sim)
        return original(sim, root, *args, **kwargs)

    monkeypatch.setattr(SubSnapBase, "get_index_list", count)
    filt = FamilyFilter("star")
    result = Pipeline({"n": CountMembers().filter(filt), "mass": ParamSum("mass").filter(filt)}).run(sim)
    assert result.value["n"] == 3 and float(result.value["mass"]) == 3.0
    assert len(observations(result).uses) == 2
    assert len(counts) == 1


def test_unloaded_coordinates_have_no_particle_io_and_late_units_are_recorded(monkeypatch):
    from pynbody.snapshot import SimSnap

    sim = snapshot()
    del sim["pos"]
    del sim["vel"]
    io = []
    original = SimSnap._get_array_with_lazy_actions

    def check_field(self, name):
        if name in {"pos", "vel"}:
            io.append(name)
        return original(self, name)

    monkeypatch.setattr(SimSnap, "_get_array_with_lazy_actions", check_field)
    result = CountMembers().filter(FamilyFilter("star")).run(sim)
    assert result.value == 3 and io == []
    assert observations(result).outputs[0].context.position_units is None

    @PropertyBase.dataclass
    class LoadsPositions(PropertyBase):
        def calculate(self, sim, params=None):
            sim["pos"] = SimArray(np.ones((len(sim), 3)), "pc")
            return np.sum(sim["pos"])

    # The first actual position read during the science hook supplies its units.
    measured = LoadsPositions().run(sim)
    assert units.Unit(observations(measured).outputs[0].context.position_units) == units.Unit("pc")


def test_boolean_evidence_paths_follow_the_complete_stored_filter_signature():
    sim = snapshot()
    star, sphere = FamilyFilter("star"), Sphere(7.0)
    forward = observations(CountMembers().filter(star & sphere).run(sim)).origins[0]
    reverse = observations(CountMembers().filter(sphere & star).run(sim)).origins[0]
    assert forward.filter_signature == reverse.filter_signature
    assert forward.parameters == reverse.parameters
    encoded = json.loads(forward.filter_signature)["payload"]
    (aperture,) = (p for p in forward.parameters if p.path[-1] == "radius")
    assert encoded[aperture.path[0]]["class"] == "pynbodyext.filters.Sphere"


def test_scientific_bounds_and_mapping_parameters_round_trip_as_strict_json():
    @FilterBase.dataclass
    class Limits(FilterBase):
        bounds: Param[object]

        def calculate(self, sim, params=None):
            return np.asarray(sim["r"]) <= params.bounds["max"]

    bounds = {"max": float("inf"), "min": float("-inf"), "weights": np.array([2.0, 3.0])}
    result = CountMembers().filter(Limits(bounds)).run(snapshot())
    assert result.value == 6
    facts = observations(result)
    (param,) = facts.origins[0].parameters
    assert param.reason is None
    assert param.value["max"] == float("inf")
    assert param.value["weights"] == (2.0, 3.0)
    bounds["weights"][:] = 0.0
    assert param.value["weights"] == (2.0, 3.0)
    loaded = type(facts).from_dict(json.loads(json.dumps(facts.to_dict(), allow_nan=False)))
    assert loaded.origins[0].parameters == (param,)


def test_filter_local_rotation_is_frozen_before_its_consumer_restores_native_frame():
    sim = snapshot()
    sim["pos"][3:] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    rotation = AlignVec(np.array([1.0, 0.0, 0.0]))
    tilted_disc = Disc(3.0, 0.1).transform(rotation)
    result = Pipeline({"native_measurement": CountMembers().filter(tilted_disc)}).run(sim)
    assert result.value["native_measurement"] == 2
    facts = observations(result)
    (origin,) = facts.origins
    (output,) = facts.outputs
    assert origin.indices == (4, 5)
    assert [step.kind for step in origin.context.steps] == ["rotation"]
    assert output.context.steps == ()


def test_boolean_operands_in_different_coordinate_frames_report_the_mixed_selection():
    sim = snapshot()
    sim["pos"][3:] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    native = Disc(3.0, 0.1)
    tilted = native.transform(AlignVec(np.array([1.0, 0.0, 0.0])))
    result = Pipeline({"mixed": CountMembers().filter(native & tilted)}).run(sim)
    (origin,) = observations(result).origins
    assert origin.indices == (4,)
    assert not origin.context.complete
    assert any("operand" in step.reason for step in origin.context.steps if step.reason)


def test_capture_uses_prepared_filter_parameters_and_marks_opaque_objects():
    from types import SimpleNamespace

    @FilterBase.dataclass
    class PreparedRadius(FilterBase):
        radius: Param[float] = Param(field_name="pos")
        opaque: bool = False

        def prepare_params(self, sim, values):
            return object() if self.opaque else SimpleNamespace(radius=2 * values["radius"])

        def calculate(self, sim, params=None):
            if self.opaque:
                return np.full(len(sim), True)
            return np.asarray(sim["r"]) < params.radius

    known = observations(CountMembers().filter(PreparedRadius(3.0)).run(snapshot())).origins[0]
    (radius,) = (p for p in known.parameters if p.path == ("radius",))
    assert radius.value == 6 and units.Unit(radius.units) == units.Unit("kpc")
    opaque = observations(CountMembers().filter(PreparedRadius(3.0, opaque=True)).run(snapshot())).origins[0]
    assert opaque.parameters and all(p.reason for p in opaque.parameters)


@pytest.mark.parametrize("observe", [False, True])
def test_transient_external_frames_inside_a_science_hook_are_incomplete(observe):
    @PropertyBase.dataclass
    class TemporaryFrame(PropertyBase):
        def calculate(self, sim, params=None):
            with sim.translate([10.0, 0.0, 0.0]):
                return float(np.sum(sim["x"]))

    sim = snapshot()
    native = float(sim["x"].sum())
    result = Pipeline({"temporary": TemporaryFrame(), "next": CountMembers()}).run(sim, observe=observe)
    assert result.value["temporary"] == native + 60
    temporary, following = observations(result).outputs
    assert not temporary.context.complete
    assert [s.kind for s in temporary.context.steps] == ["external"]
    assert following.context.complete and not following.context.steps
    assert sim.current_transformation() is None


def test_failed_leaf_has_no_measured_output_or_successful_use():
    @PropertyBase.dataclass
    class FailedLeaf(PropertyBase):
        def calculate(self, sim, params=None):
            raise ValueError("science failed")

    result = FailedLeaf().filter(FamilyFilter("dm")).run(snapshot(), errors="collect")
    assert result.errors
    facts = observations(result)
    assert len(facts.origins) == 1
    assert not facts.outputs and not facts.uses


def test_many_transforms_on_one_filtered_target_share_one_index_sequence(monkeypatch):
    from pynbody.snapshot.subsnap import SubSnapBase

    count = 0
    original = SubSnapBase.get_index_list

    def indices(view, root, *args):
        nonlocal count
        count += 1
        return original(view, root, *args)

    monkeypatch.setattr(SubSnapBase, "get_index_list", indices)
    pipe = Pipeline(
        {str(i): CountMembers().transform(ShiftPosTo(np.array([i + 1.0, 0.0, 0.0]), move_all=False)) for i in range(10)}
    ).filter(FamilyFilter("star"))
    facts = observations(pipe.run(snapshot()))
    assert count == 1
    assert all(o.context.steps[0].target_indices is facts.origins[0].indices for o in facts.outputs)


def test_concurrent_and_failed_runs_restore_the_native_registration_hook():
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from threading import Barrier

    from pynbody.transformation import Transformable

    original = Transformable._register_transformation
    ready = Barrier(2)

    @PropertyBase.dataclass
    class Overlapping(PropertyBase):
        offset: float

        def calculate(self, sim, params=None):
            with sim.translate([self.offset, 0.0, 0.0]) if self.offset else nullcontext():
                ready.wait(timeout=5)
                return len(sim)

    def run(offset):
        return observations(Overlapping(offset).run(snapshot(), observe=False)).outputs[0].context

    with ThreadPoolExecutor(max_workers=2) as pool:
        shifted, native = list(pool.map(run, [3.0, 0.0]))
    assert not shifted.complete and native.complete
    assert Transformable._register_transformation is original

    @PropertyBase.dataclass
    class Failing(PropertyBase):
        def calculate(self, sim, params=None):
            with sim.translate([1.0, 0.0, 0.0]):
                raise RuntimeError("original failure")

    with pytest.raises(RuntimeError, match="original failure"):
        Failing().run(snapshot())
    assert Transformable._register_transformation is original


def test_preexisting_external_step_keeps_its_subset_target():
    sim = snapshot()
    target = sim.star
    handle = target.translate(np.array([3.0, 0.0, 0.0]))
    try:
        frame = observations(CountMembers().run(sim)).outputs[0].context
        (step,) = frame.steps
        assert not frame.complete and step.kind == "external"
        assert step.target_indices == tuple(target.get_index_list(sim))
    finally:
        handle.revert()
