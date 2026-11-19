"""Each named Pipeline output measures particles in its own coordinate frame."""

import numpy as np
import pynbody
import pytest

from pynbodyext.calculate import Pipeline, PropertyBase
from pynbodyext.filters import Disc
from pynbodyext.transforms import AlignVec


@PropertyBase.dataclass
class RootIndices(PropertyBase):
    def calculate(self, sim, params=None):
        return np.asarray(sim.get_index_list(sim.ancestor)).copy()


def particle_snapshot():
    sim = pynbody.new(dm=2, star=4)
    sim["pos"][:] = [[10, 10, 10], [11, 11, 11], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    sim["vel"][:] = np.arange(18.0).reshape(6, 3)
    return sim


@pytest.mark.parametrize("order", [("rotated", "native"), ("native", "rotated")])
def test_named_output_order_restores_entry_frame(order):
    sim = particle_snapshot()
    before = {key: np.asarray(sim[key]).copy() for key in ("pos", "vel")}
    entry = sim.current_transformation()
    disc = Disc(3.0, 0.1)
    face_on = AlignVec(np.array([1.0, 0.0, 0.0]))
    outputs = {"rotated": RootIndices().transform(face_on).filter(disc), "native": RootIndices().filter(disc)}
    expected = {"native": [2, 3, 5], "rotated": [3, 4, 5]}
    for key in order:
        fresh = particle_snapshot()
        np.testing.assert_array_equal(outputs[key].run(fresh.star).value, expected[key])
    result = Pipeline({key: outputs[key] for key in order}).run(sim.star)
    assert result.ok
    for key in order:
        np.testing.assert_array_equal(result.value[key], expected[key])
    for key, original in before.items():
        np.testing.assert_allclose(sim[key], original, atol=1e-12, rtol=0)
    assert sim.current_transformation() is entry


@PropertyBase.dataclass
class Coordinates(PropertyBase):
    def calculate(self, sim, params=None):
        return {"pos": sim["pos"], "vel": sim["vel"]}


@PropertyBase.dataclass
class Fails(PropertyBase):
    def calculate(self, sim, params=None):
        raise RuntimeError("measurement failed")


def test_nested_frames_and_borrowed_arrays():
    from pynbody.array import SimArray

    from pynbodyext.transforms import ShiftPosTo

    sim = particle_snapshot()
    sim["pos"].units = "kpc"
    sim["vel"].units = "km s^-1"
    native_pos, native_vel = sim["pos"].copy(), sim["vel"].copy()
    outer = sim.translate(SimArray([4.0, 2.0, 1.0], "kpc"))
    at_entry = sim["pos"].copy()
    shift = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    rotation = AlignVec(np.array([1.0, 0.0, 0.0]))
    inner = Pipeline({"disk": Coordinates().transform(rotation), "parent": Coordinates()}).transform(shift)
    value = Pipeline({"inner": inner, "outer": Coordinates()}).run(sim).value
    rotated = particle_snapshot()
    rotated["pos"][:] = np.asarray(at_entry) - [1.0, 0.0, 0.0]
    rotation.run(rotated)
    np.testing.assert_allclose(value["inner"]["disk"]["pos"], rotated["pos"])
    np.testing.assert_allclose(value["inner"]["disk"]["vel"], rotated["vel"])
    assert str(value["inner"]["disk"]["pos"].units) == "kpc"
    assert value["inner"]["disk"]["vel"].units == sim["vel"].units
    np.testing.assert_allclose(value["inner"]["parent"]["pos"], np.asarray(at_entry) - [1.0, 0.0, 0.0])
    np.testing.assert_allclose(value["outer"]["pos"], at_entry)
    assert sim.current_transformation() is outer
    outer.revert()
    np.testing.assert_allclose(sim["pos"], native_pos)
    np.testing.assert_allclose(sim["vel"], native_vel)
    np.testing.assert_allclose(value["outer"]["pos"], at_entry)


@pytest.mark.parametrize("errors", ["success", "raise", "collect", "collect_partial"])
def test_nested_output_exit_keeps_external_frame(errors):
    from pynbodyext.core.calculate import InMemoryResultStore
    from pynbodyext.transforms import ShiftPosTo

    cleanup = []

    class ChildRotation(AlignVec):
        def cleanup(self, ctx, handle):
            cleanup.append("child")
            super().cleanup(ctx, handle)

    class ParentShift(ShiftPosTo):
        def cleanup(self, ctx, handle):
            cleanup.append("parent")
            super().cleanup(ctx, handle)

    sim = particle_snapshot()
    native = sim["pos"].copy()
    outside = sim.translate(np.array([2.0, 0.0, 0.0]))
    start = sim["pos"].copy()
    velocity = sim["vel"].copy()
    rotation = ChildRotation(np.array([1.0, 0.0, 0.0]))
    parent = ParentShift(np.array([1.0, 0.0, 0.0]))
    store = InMemoryResultStore()
    first = Coordinates() if errors == "success" else Fails()
    calc = Pipeline({"first": first.transform(rotation), "sibling": Coordinates()}).transform(parent)
    if errors == "raise":
        with pytest.raises(RuntimeError, match="measurement failed"):
            calc.run(sim, errors=errors, store=store)
    else:
        result = calc.run(sim, errors="raise" if errors == "success" else errors, store=store)
        assert result.ok == (errors == "success")
        if errors in ("success", "collect_partial"):
            np.testing.assert_allclose(result.value["sibling"]["pos"], np.asarray(start) - [1.0, 0.0, 0.0])
            np.testing.assert_allclose(result.value["sibling"]["vel"], velocity)
            if errors == "collect_partial":
                assert result.value["first"] is None
        if errors != "success":
            assert result.store_ref is None
    if errors != "success":
        assert store.list_refs() == []
    assert cleanup == ["child", "parent"]
    np.testing.assert_allclose(sim["pos"], start)
    np.testing.assert_allclose(sim["vel"], velocity)
    assert sim.current_transformation() is outside
    outside.revert()
    np.testing.assert_allclose(sim["pos"], native)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_shared_transform_restores_after_each_measurement(cache, observe, batch):
    sim = particle_snapshot()
    before = sim["pos"].copy()
    rotate = AlignVec(np.array([1.0, 0.0, 0.0]))
    measure = Coordinates()
    rotated = measure.transform(rotate)
    calc = Pipeline({"first": rotated, "native": measure, "again": rotated})
    if batch:
        with calc.batch(cache=cache, observe=observe) as run_one:
            values = run_one(sim)
    else:
        values = calc.run(sim, cache=cache, observe=observe).value
    expected = particle_snapshot()
    rotate.run(expected)
    for key in ("first", "again"):
        np.testing.assert_allclose(values[key]["pos"], expected["pos"])
    np.testing.assert_allclose(values["native"]["pos"], before)
    np.testing.assert_allclose(sim["pos"], before)
    assert sim.current_transformation() is None


def test_named_transform_scope_and_standalone_handle_lifetimes():
    sim = particle_snapshot()
    before = sim["pos"].copy()
    rotate = AlignVec(np.array([1.0, 0.0, 0.0]))
    value = Pipeline({"rotation": rotate, "native": Coordinates()}).run(sim).value
    np.testing.assert_allclose(value["native"]["pos"], before)
    assert sim.current_transformation() is None
    direct = rotate.run(sim).value
    assert sim.current_transformation() is direct
    direct.revert()
    np.testing.assert_allclose(sim["pos"], before)


@pytest.mark.parametrize("level", ["scope", "transform", "chain"])
def test_explicit_retention_is_visible_to_sibling(level):
    from pynbodyext.transforms import ShiftPosTo

    sim = particle_snapshot()
    before = sim["pos"].copy()
    step = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    if level == "scope":
        prop = Coordinates().transform(step, revert=False)
    elif level == "transform":
        prop = Coordinates().transform(step.revert(False))
    else:
        prop = Coordinates().transform(step.then(step).revert(False))
    result = Pipeline({"shifted": prop, "after": Coordinates()}).run(sim)
    expected = np.asarray(before) - [2.0 if level == "chain" else 1.0, 0.0, 0.0]
    np.testing.assert_allclose(result.value["shifted"]["pos"], expected)
    np.testing.assert_allclose(result.value["after"]["pos"], expected)
    np.testing.assert_allclose(sim["pos"], expected)
    while (handle := sim.current_transformation()) is not None:
        handle.revert()
    np.testing.assert_allclose(sim["pos"], before)


def test_subset_target_is_independent_of_an_inherited_root_transform():
    from pynbodyext.transforms import ShiftPosTo

    @PropertyBase.dataclass
    class RootPositions(PropertyBase):
        def calculate(self, sim, params=None):
            return sim.ancestor["pos"]

    sim = particle_snapshot()
    before = sim["pos"].copy()
    ids = [2, 4]
    shift_root = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    shift_subset = ShiftPosTo(np.array([0.0, 1.0, 0.0]), move_all=False)
    calc = Pipeline({"subset": RootPositions().transform(shift_subset), "parent": RootPositions()}).transform(
        shift_root
    )
    result = calc.run(sim[ids])
    parent = np.asarray(before) - [1.0, 0.0, 0.0]
    subset = parent.copy()
    subset[ids, 1] -= 1
    np.testing.assert_allclose(result.value["subset"], subset)
    np.testing.assert_allclose(result.value["parent"], parent)
    np.testing.assert_allclose(sim["pos"], before)


@pytest.mark.parametrize("phase", ["filter", "check_input", "materialize", "materialize_public"])
def test_failures_after_transform_restore_before_next_output(phase):
    from pynbodyext.calculate import FilterBase

    @FilterBase.dataclass
    class BrokenFilter(FilterBase):
        def calculate(self, sim, params=None):
            raise RuntimeError("filter failed")

    @PropertyBase.dataclass
    class BrokenProperty(Coordinates):
        def check_input(self, sim):
            if phase == "check_input":
                raise RuntimeError("input failed")

        def materialize(self, ctx, value):
            if phase == "materialize":
                raise RuntimeError("materialize failed")
            return value

        def materialize_public(self, ctx, value):
            if phase == "materialize_public":
                raise RuntimeError("public failed")
            return value

    sim = particle_snapshot()
    before = sim["pos"].copy()
    node = BrokenProperty().transform(AlignVec(np.array([1.0, 0.0, 0.0])))
    if phase == "filter":
        node = node.filter(BrokenFilter())
    res = Pipeline({"bad": node, "good": Coordinates()}).run(sim, errors="collect_partial")
    assert not res.ok and res.value["bad"] is None
    np.testing.assert_allclose(res.value["good"]["pos"], before)
    assert sim.current_transformation() is None


@pytest.mark.parametrize("cleanup_failure", ["refuses", "raises"])
@pytest.mark.parametrize("science_fails", [False, True])
def test_unrestored_frame_stops_partial_run(cleanup_failure, science_fails, caplog):
    from pynbody.transformation import GenericTranslation

    from pynbodyext.calculate import TransformBase
    from pynbodyext.core.calculate import InMemoryResultStore

    @TransformBase.dataclass
    class Unrestorable(TransformBase):
        def build_handle(self, sim, target, params=None):
            return GenericTranslation(target, "pos", [1.0, 0.0, 0.0])

        def is_revertible(self, handle):
            return cleanup_failure != "refuses"

        def cleanup(self, ctx, handle):
            raise ValueError("cannot undo translation")

    visited = []

    @PropertyBase.dataclass
    class Later(PropertyBase):
        def calculate(self, sim, params=None):
            visited.append(True)
            return 1

    sim = particle_snapshot()
    store = InMemoryResultStore()
    body = Fails() if science_fails else Coordinates()
    res = Pipeline({"bad": body.transform(Unrestorable()), "later": Later()}).run(
        sim, errors="collect_partial", store=store
    )
    assert not res.ok and res.value is None and not visited
    expected = (
        "measurement failed"
        if science_fails
        else ("no reversible handle" if cleanup_failure == "refuses" else "cannot undo translation")
    )
    assert expected in res.root.run.error.message
    assert "cleanup failed" in caplog.text and store.list_refs() == []


def test_partial_transform_chain_unwinds_known_steps():
    from pynbodyext.calculate import TransformBase
    from pynbodyext.transforms import ShiftPosTo

    @TransformBase.dataclass
    class CannotApply(TransformBase):
        def build_handle(self, sim, target, params=None):
            raise RuntimeError("cannot apply next step")

    sim = particle_snapshot()
    before = sim["pos"].copy()
    steps = ShiftPosTo(np.array([1.0, 0.0, 0.0])).then(CannotApply())
    with pytest.raises(RuntimeError, match="cannot apply next step"):
        Pipeline({"bad": Coordinates().transform(steps)}).run(sim)
    assert sim.current_transformation() is None
    np.testing.assert_allclose(sim["pos"], before)


def test_bins_eager_queries_use_transformed_frame_and_lazy_queries_use_restored_particles():
    from pynbodyext.core.calculate.bins.nodes import Bin1D
    from pynbodyext.transforms import ShiftPosTo

    sim = pynbody.new(dm=3)
    sim["x"][:] = [1.0, 2.0, 4.0]
    calc = Bin1D("x", vmin=-0.5, vmax=3.5, nbins=2).with_active(["x.sum"])
    shifted = calc.transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])))
    bins = Pipeline({"profile": shifted}).run(sim).value["profile"]
    np.testing.assert_allclose(bins["x.sum"], [1.0, 3.0])
    np.testing.assert_allclose(bins["x.mean"], [1.5, 4.0])
    np.testing.assert_allclose(sim["x"], [1.0, 2.0, 4.0])


@pytest.mark.parametrize("outer_shift", [0.0, 1.0])
def test_eager_per_bin_calculator_restores_each_bin_before_the_next(outer_shift):
    from pynbodyext.core.calculate.bins.nodes import Bin1D
    from pynbodyext.properties import ParamSum
    from pynbodyext.transforms import ShiftPosTo

    sim = pynbody.new(dm=3)
    sim["x"][:] = [1.0, 2.0, 4.0]
    per_bin = ParamSum("x").transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])))
    profile = Bin1D("x", vmin=-outer_shift, vmax=5.0 - outer_shift, nbins=2).with_active([per_bin])
    if outer_shift:
        profile = profile.transform(ShiftPosTo(np.array([outer_shift, 0.0, 0.0])))
    result = Pipeline({"profile": profile, "native": ParamSum("x")}).run(sim)

    np.testing.assert_allclose(result.value["profile"][per_bin], [1.0 - 2 * outer_shift, 3.0 - outer_shift])
    assert result.value["native"] == 7.0
    np.testing.assert_array_equal(sim["x"], [1.0, 2.0, 4.0])
    assert sim.current_transformation() is None


def test_eager_bin_filter_restores_frame_before_sibling():
    from pynbodyext.core.calculate.bins.nodes import Bin1D
    from pynbodyext.filters import HighPass
    from pynbodyext.properties import ParamSum
    from pynbodyext.transforms import ShiftPosTo

    sim = pynbody.new(dm=3)
    sim["x"][:] = [1.0, 2.0, 4.0]
    selector = HighPass("x", 0.0).transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])))
    farther = HighPass("x", 0.0).transform(ShiftPosTo(np.array([3.0, 0.0, 0.0])))
    profile = Bin1D("x", vmin=0, vmax=5, nbins=2).with_active([selector, farther])
    result = Pipeline({"profile": profile, "native": ParamSum("x")}).run(sim)

    bins = result.value["profile"]
    selected = bins[selector]
    assert bins[selector] is selected
    np.testing.assert_array_equal(selected.sim.get_index_list(sim), [1, 2])
    np.testing.assert_array_equal(bins[farther].sim.get_index_list(sim), [2])
    assert result.value["native"] == 7.0
    np.testing.assert_array_equal(sim["x"], [1.0, 2.0, 4.0])
    assert sim.current_transformation() is None


def test_eager_bin_subresult_remeasures_scoped_axis_in_its_own_frame():
    from pynbodyext.core.calculate.bins.nodes import Bin1D
    from pynbodyext.filters import HighPass
    from pynbodyext.transforms import ShiftPosTo

    @PropertyBase.dataclass
    class AxisX(PropertyBase):
        def calculate(self, sim, params=None):
            return sim["x"]

    sim = pynbody.new(dm=3)
    sim["x"][:] = [1.0, 2.0, 4.0]
    axis = AxisX().transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])))
    selector = HighPass("x", 1.0)
    profile = Bin1D(axis, vmin=0, vmax=4, nbins=2).with_active([selector])
    result = Pipeline({"profile": profile, "native": AxisX()}).run(sim)
    selected = result.value["profile"][selector]
    np.testing.assert_array_equal(selected["count"], [1, 1])
    np.testing.assert_array_equal(result.value["native"], [1.0, 2.0, 4.0])
    assert sim.current_transformation() is None


def test_eager_bin_cleanup_failure_stops_the_enclosing_partial_pipeline():
    from pynbodyext.core.calculate.bins.nodes import Bin1D
    from pynbodyext.properties import ParamSum
    from pynbodyext.transforms import ShiftPosTo

    class FailingCleanup(ShiftPosTo):
        def cleanup(self, ctx, handle):
            raise ValueError("bin transform cleanup failed")

    sim = pynbody.new(dm=2)
    sim["x"][:] = [1.0, 3.0]
    query = ParamSum("x").transform(FailingCleanup(np.array([1.0, 0.0, 0.0])))
    profile = Bin1D("x", vmin=0, vmax=4, nbins=2).with_active([query])
    res = Pipeline({"profile": profile, "native": Coordinates()}).run(sim, errors="collect_partial")
    assert not res.ok and res.value is None and "native" not in res.named
    assert "bin transform cleanup failed" in res.root.run.error.message
    sim.current_transformation().revert()
    np.testing.assert_array_equal(sim["x"], [1.0, 3.0])


@pytest.mark.parametrize(("vector", "up", "error"), [
    (np.zeros(3), None, "zero or NaN"),
    (np.array([0., np.inf, 0.]), None, "finite"),
    (np.array([0., 0., 1.]), np.array([0., 0., 1.]), "parallel"),
])
def test_invalid_alignment_before_rotation_preserves_partial_sibling(vector, up, error):
    sim = particle_snapshot()
    before = sim["pos"].copy()
    invalid = Coordinates().transform(AlignVec(vector, up=up))
    result = Pipeline({"invalid": invalid, "native": Coordinates()}).run(sim, errors="collect_partial")
    assert not result.ok and error in result.errors[0].message
    assert result.value["invalid"] is None
    np.testing.assert_array_equal(result.value["native"]["pos"], before)
    assert sim.current_transformation() is None


@pytest.mark.parametrize("borrowed_container", [False, True])
def test_object_arrays_preserve_shape_and_frame_of_ragged_cells(borrowed_container):
    from pynbodyext.transforms import ShiftPosTo

    @PropertyBase.dataclass
    class Groups(PropertyBase):
        def calculate(self, sim, params=None):
            groups = np.empty((2, 2 if borrowed_container else 1), dtype=object)
            groups[0, 0] = sim["x"][:1]
            groups[1, 0] = sim["x"][1:]
            return groups[:, :1] if borrowed_container else groups

    sim = pynbody.new(dm=3)
    sim["x"][:] = [10.0, 20.0, 30.0]
    sim["pos"].units = "kpc"
    groups = Pipeline({"groups": Groups().transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])))}).run(sim).value["groups"]

    assert groups.dtype == object and groups.shape == (2, 1)
    np.testing.assert_array_equal(groups[0, 0], [9.0])
    np.testing.assert_array_equal(groups[1, 0], [19.0, 29.0])
    assert groups[1, 0].units == sim["pos"].units
    np.testing.assert_array_equal(sim["x"], [10.0, 20.0, 30.0])


@pytest.mark.parametrize("observe", [True, False])
def test_shared_outputs_reapply_retained_effects_inside_a_composite(observe):
    from pynbodyext.calculate import PostprocessorBase
    from pynbodyext.transforms import ShiftPosTo

    @PostprocessorBase.dataclass
    class Forward(PostprocessorBase):
        def process(self, value):
            return value

    sim = particle_snapshot()
    before = sim["pos"].copy()
    prop = Forward(source=Coordinates().transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])), revert=False))
    values = Pipeline({"one": prop, "two": prop}).run(sim, observe=observe, cache=True).value
    np.testing.assert_allclose(values["one"]["pos"], np.asarray(before) - [1.0, 0.0, 0.0])
    np.testing.assert_allclose(values["two"]["pos"], np.asarray(before) - [2.0, 0.0, 0.0])
    np.testing.assert_allclose(sim["pos"], np.asarray(before) - [2.0, 0.0, 0.0])


@pytest.mark.parametrize("hook", ["compute", "wrap_raw"])
def test_custom_transform_failure_only_continues_with_a_recoverable_frame(hook):
    from pynbodyext.calculate import TransformBase

    @TransformBase.dataclass
    class BrokenHook(TransformBase):
        def compute(self, runtime, params):
            handle = runtime.target.translate([2.0, 0.0, 0.0])
            if hook == "compute":
                raise RuntimeError("compute failed after applying")
            return handle

        def wrap_raw(self, runtime, computed):
            raise RuntimeError("wrapping returned handle failed")

    sim = particle_snapshot()
    before = sim["pos"].copy()
    calc = Pipeline({"bad": Coordinates().transform(BrokenHook()), "later": Coordinates()})
    res = calc.run(sim, errors="collect_partial")
    assert not res.ok
    if hook == "compute":
        assert res.value is None and "later" not in res.named
        assert "compute failed after applying" in res.root.run.error.message
        sim.current_transformation().revert()
    else:
        assert res.value["bad"] is None
        np.testing.assert_allclose(res.value["later"]["pos"], before)
    np.testing.assert_allclose(sim["pos"], before)
    assert sim.current_transformation() is None


def test_low_level_transform_result_owns_returned_handle():
    from pynbodyext.calculate import TransformBase
    from pynbodyext.core.calculate.runtime.input import TransformResult

    @TransformBase.dataclass
    class DirectShift(TransformBase):
        def execute(self, ctx, input):
            handle = input.active_sim.translate([1.0, 0.0, 0.0])
            return TransformResult(handle=handle, target=input.active_sim, sim_after=input.active_sim)

    sim = particle_snapshot()
    before = sim["pos"].copy()
    res = Pipeline({"shift": Coordinates().transform(DirectShift()), "later": Coordinates()}).run(sim)
    assert res.ok
    np.testing.assert_allclose(res.value["shift"]["pos"], np.asarray(before) + [1.0, 0.0, 0.0])
    np.testing.assert_allclose(res.value["later"]["pos"], before)
    assert sim.current_transformation() is None


@pytest.mark.parametrize("observe", [True, False])
def test_cached_component_and_filter_follow_frame_mutations(observe):
    from pynbodyext.filters import HighPass
    from pynbodyext.transforms import ShiftPosTo

    @PropertyBase.dataclass
    class FirstX(PropertyBase):
        def calculate(self, sim, params=None):
            return float(sim["x"][0])

    sim = pynbody.new(dm=3)
    sim["x"][:] = [0.2, 0.8, 2.2]
    x, selected = FirstX(), RootIndices().filter(HighPass("x", 0.5))
    shift = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    pipeline = Pipeline(
        {
            "entry_x": x,
            "entry_ids": selected,
            "shifted": Pipeline({"moved_x": x, "moved_ids": selected}).transform(shift),
            "final_x": x,
            "final_ids": selected,
        }
    )
    values = pipeline.run(sim, observe=observe, cache=True).value
    assert values["entry_x"] == pytest.approx(0.2)
    assert values["final_x"] == pytest.approx(0.2)
    assert values["shifted"]["moved_x"] == pytest.approx(-0.8)
    np.testing.assert_array_equal(values["shifted"]["moved_ids"], [2])
    np.testing.assert_array_equal(values["entry_ids"], [1, 2])
    np.testing.assert_array_equal(values["final_ids"], [1, 2])


@pytest.mark.parametrize("role", ["transform", "chain", "combined"])
def test_bound_scope_releases_body_transforms_before_own_handle(role):
    from pynbodyext.transforms import ShiftPosTo

    a = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    b = ShiftPosTo(np.array([2.0, 0.0, 0.0]))
    body = {"transform": b, "chain": b.then(b), "combined": b & Coordinates()}[role]
    sim = particle_snapshot()
    before = sim["pos"].copy()
    result = Pipeline({"body": body.transform(a), "after": Coordinates()}).run(sim)
    assert result.ok and sim.current_transformation() is None
    np.testing.assert_allclose(result.value["after"]["pos"], before)


@pytest.mark.parametrize("observe", [False, True])
def test_pipeline_does_not_reuse_live_view_from_outside(observe):
    from pynbodyext.transforms import ShiftPosTo

    sim = particle_snapshot()
    before = sim["pos"].copy()
    read = Coordinates()
    shift = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    inside = Pipeline({"entry": read, "shifted": read.transform(shift, revert=False)})
    _, outputs = (read & inside).run(sim, observe=observe).value
    np.testing.assert_allclose(outputs["entry"]["pos"], before)
    np.testing.assert_allclose(outputs["shifted"]["pos"], np.asarray(before) - [1.0, 0.0, 0.0])


def test_named_containers_preserve_measured_fields_and_types():
    from collections import OrderedDict, namedtuple

    from pynbodyext.transforms import ShiftPosTo

    class Point(namedtuple("Coordinates", "position")):
        def __new__(cls, position):
            point = super().__new__(cls, position)
            point.self_ref = point
            return point

    @PropertyBase.dataclass
    class PointMeasurement(PropertyBase):
        def calculate(self, sim, params=None):
            return OrderedDict([("point", Point(sim["pos"]))])

    sim = particle_snapshot()
    before = sim["pos"].copy()
    node = PointMeasurement().transform(ShiftPosTo(np.array([1.0, 0.0, 0.0])))
    value = Pipeline({"point": node}).run(sim).value["point"]
    assert isinstance(value, OrderedDict) and isinstance(value["point"], Point)
    assert value["point"].self_ref is value["point"]
    np.testing.assert_allclose(value["point"].position, np.asarray(before) - [1.0, 0.0, 0.0])
    np.testing.assert_allclose(sim["pos"], before)


def test_retained_native_handle_can_block_outer_cleanup(caplog):
    from pynbodyext.transforms import ShiftPosTo

    sim = particle_snapshot()
    start = sim["pos"].copy()
    a = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    b = ShiftPosTo(np.array([2.0, 0.0, 0.0]))
    retained = Coordinates().transform(b, revert=False)
    nested = Pipeline({"retained": retained}).transform(a)
    result = Pipeline({"nested": nested, "later": Coordinates()}).run(sim, errors="collect_partial")
    assert not result.ok and result.value is None and "later" not in result.named
    assert "cleanup failed" in caplog.text
    np.testing.assert_allclose(sim["pos"], np.asarray(start) - [3.0, 0.0, 0.0])
    sim.current_transformation().revert()
    sim.current_transformation().revert()
    np.testing.assert_allclose(sim["pos"], start)


def test_standalone_property_keeps_its_default_transform():
    from pynbodyext.transforms import ShiftPosTo

    sim = particle_snapshot()
    start = sim["pos"].copy()
    Coordinates().transform(ShiftPosTo(np.array([1.0, 0.0, 0.0]))).run(sim)
    np.testing.assert_allclose(sim["pos"], np.asarray(start) - [1.0, 0.0, 0.0])
    sim.current_transformation().revert()
    np.testing.assert_allclose(sim["pos"], start)


@pytest.mark.parametrize("hook", ["wrap_raw", "execute"])
def test_advanced_receipt_can_declare_handle_nonrevertible(hook):
    from pynbodyext.transforms import ShiftPosTo

    class ReceiptInWrap(ShiftPosTo):
        def wrap_raw(self, runtime, computed):
            receipt = super().wrap_raw(runtime, computed)
            receipt.revertible = False
            return receipt

    class ReceiptInExecute(ShiftPosTo):
        def execute(self, ctx, input):
            receipt = super().execute(ctx, input)
            receipt.revertible = False
            return receipt

    sim = particle_snapshot()
    before = sim["pos"].copy()
    typ = ReceiptInWrap if hook == "wrap_raw" else ReceiptInExecute
    result = Pipeline({"a": Coordinates().transform(typ(np.array([1.0, 0.0, 0.0]))), "b": Coordinates()}).run(
        sim, errors="collect_partial"
    )
    assert not result.ok and result.value is None and "b" not in result.named
    np.testing.assert_allclose(sim["pos"], np.asarray(before) - [1.0, 0.0, 0.0])
    sim.current_transformation().revert()


@pytest.mark.parametrize("delegate", [False, True], ids=["own-receipt", "tracked-steps"])
def test_custom_transform_chain_releases_its_receipt_once(delegate):
    from pynbodyext.calculate import TransformBase
    from pynbodyext.core.calculate.nodes.transforms import TransformChain, TransformStep
    from pynbodyext.core.calculate.runtime.input import TransformResult

    cleanup_calls = []

    @TransformBase.dataclass
    class Shift(TransformBase):
        def build_handle(self, sim, target, params=None):
            return target.translate([2., 0., 0.])

        def cleanup(self, ctx, handle):
            cleanup_calls.append("shift")
            super().cleanup(ctx, handle)

    class CustomChain(TransformChain):
        def execute(self, ctx, input):
            if delegate:
                return super().execute(ctx, input)
            target = input.active_sim
            step = self.transforms[0]
            handle = step.build_handle(target, target)
            applied = TransformResult(handle=handle, target=target, sim_after=target)
            return TransformResult(handle=(TransformStep(step, applied),), target=target, sim_after=target)

    sim = particle_snapshot()
    original = sim["pos"].copy()
    scope = CustomChain(Shift())
    values = Pipeline({"moved": Coordinates().transform(scope), "sibling": Coordinates()}).run(sim).value
    np.testing.assert_array_equal(values["moved"]["pos"], original + [2., 0., 0.])
    np.testing.assert_array_equal(values["sibling"]["pos"], original)
    np.testing.assert_array_equal(sim["pos"], original)
    assert cleanup_calls == ["shift"]
    assert sim.current_transformation() is None


@pytest.mark.parametrize("direct_first", [True, False])
def test_custom_chain_restores_direct_and_delegated_steps_in_receipt_order(direct_first):
    from dataclasses import replace

    from pynbodyext.calculate import TransformBase
    from pynbodyext.core.calculate.nodes.transforms import TransformChain, TransformStep
    from pynbodyext.core.calculate.runtime.input import TransformResult

    cleanup = []

    @TransformBase.dataclass
    class Shift(TransformBase):
        dx: float

        def build_handle(self, sim, target, params=None):
            return target.translate([self.dx, 0., 0.])

        def cleanup(self, ctx, handle):
            cleanup.append(self.dx)
            super().cleanup(ctx, handle)

    class MixedChain(TransformChain):
        def execute(self, ctx, input):
            def direct_step():
                target = input.active_sim
                step = Shift(2.)
                handle = step.build_handle(target, target)
                return TransformStep(step, TransformResult(handle=handle, target=target, sim_after=target))

            before = (direct_step(),) if direct_first else ()
            tracked = super().execute(ctx, input)
            after = () if direct_first else (direct_step(),)
            return replace(tracked, handle=before + tracked.handle + after)

    sim = particle_snapshot()
    original = sim["pos"].copy()
    node = Coordinates().transform(MixedChain(Shift(1.)))
    result = Pipeline({"moved": node, "sibling": Coordinates()}).run(sim, errors="collect_partial")
    assert result.ok
    np.testing.assert_array_equal(result.value["moved"]["pos"], original + [3., 0., 0.])
    np.testing.assert_array_equal(result.value["sibling"]["pos"], original)
    assert cleanup == ([1., 2.] if direct_first else [2., 1.])
    assert sim.current_transformation() is None


@pytest.mark.parametrize("container", ["dataclass", "mapping", "proxy", "tuple", "tuple_attribute"])
def test_custom_measurement_containers_keep_their_values_and_type(container):
    from collections import UserDict
    from dataclasses import dataclass, field
    from types import MappingProxyType

    from pynbodyext.transforms import ShiftPosTo

    @dataclass(frozen=True, slots=True)
    class Measurement:
        position: object
        first: object = field(init=False)

        def __post_init__(self):
            object.__setattr__(self, "first", self.position[0])

    class TaggedTuple(tuple):
        def __new__(cls, position, label):
            obj = super().__new__(cls, (position,) if container == "tuple" else ())
            obj.position = position
            obj.label = label
            return obj

    @PropertyBase.dataclass
    class Read(PropertyBase):
        def calculate(self, sim, params=None):
            position = sim["pos"]
            if container == "dataclass":
                return Measurement(position)
            if container == "mapping":
                return UserDict({"position": position})
            if container == "proxy":
                return MappingProxyType({"position": position})
            return TaggedTuple(position, "stars")

    sim = particle_snapshot()
    sim["pos"].units = "kpc"
    original = sim["pos"].copy()
    node = Read().transform(ShiftPosTo(np.array([1., 0., 0.])))
    result = Pipeline({"reading": node}).run(sim).value["reading"]
    expected = original - [1., 0., 0.]
    if container == "dataclass":
        assert type(result) is Measurement
        np.testing.assert_array_equal(result.first, expected[0])
        measured = result.position
    elif container in ("mapping", "proxy"):
        assert type(result) is (UserDict if container == "mapping" else MappingProxyType)
        measured = result["position"]
    else:
        assert type(result) is TaggedTuple and result.label == "stars"
        if container == "tuple":
            assert result.position is result[0]
        measured = result.position
    np.testing.assert_array_equal(measured, expected)
    assert measured.units == original.units
    np.testing.assert_array_equal(sim["pos"], original)


@pytest.mark.parametrize("container", ["dict", "list"])
@pytest.mark.parametrize("attribute_only", [False, True])
@pytest.mark.parametrize("slotted", [False, True], ids=["instance-dict", "slots"])
def test_mutable_measurement_container_attributes_follow_the_measured_frame(container, attribute_only, slotted):
    from pynbodyext.transforms import ShiftPosTo

    class Readings(dict if container == "dict" else list):
        if slotted:
            __slots__ = ("position", "self_ref")

    @PropertyBase.dataclass
    class Read(PropertyBase):
        def calculate(self, sim, params=None):
            readings = Readings()
            readings.position = sim["pos"]
            readings.self_ref = readings
            if not attribute_only:
                if container == "dict":
                    readings["position"] = readings.position
                else:
                    readings.append(readings.position)
            return readings

    sim = particle_snapshot()
    original = sim["pos"].copy()
    moved = Read().transform(ShiftPosTo(np.array([1., 0., 0.])))
    readings = Pipeline({"moved": moved}).run(sim).value["moved"]
    assert type(readings) is Readings and readings.self_ref is readings
    if not attribute_only:
        assert readings.position is readings["position" if container == "dict" else 0]
    np.testing.assert_array_equal(readings.position, original - [1., 0., 0.])
    np.testing.assert_array_equal(sim["pos"], original)


@pytest.mark.parametrize("container", ["dict", "list"])
def test_cyclic_measurement_reports_error_and_restores_frame(container):
    from pynbodyext.transforms import ShiftPosTo

    @PropertyBase.dataclass
    class CyclicMeasurement(PropertyBase):
        def calculate(self, sim, params=None):
            if container == "dict":
                reading = {"position": sim["pos"]}
                reading["self"] = reading
            else:
                reading = [sim["pos"]]
                reading.append(reading)
            return reading

    sim = particle_snapshot()
    before = sim["pos"].copy()
    moved = CyclicMeasurement().transform(ShiftPosTo(np.array([1., 0., 0.])))
    result = Pipeline({"moved": moved, "after": Coordinates()}).run(sim, errors="collect_partial")
    assert result.errors and "cyclic" in result.errors[0].message.lower()
    assert result.value["moved"] is None
    np.testing.assert_array_equal(result.value["after"]["pos"], before)
    np.testing.assert_array_equal(sim["pos"], before)
