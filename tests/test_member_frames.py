"""Replay a witnessed frame on independent arrays and compare it to the measurement."""

from __future__ import annotations

import numpy as np
import pytest
from pynbody.array import SimArray

import pynbodyext as pe
from pynbodyext.calculate import Pipeline, PropertyBase
from pynbodyext.filters import FamilyFilter, Sphere
from pynbodyext.transforms import AlignVec, ShiftPosTo, ShiftVelTo, WrapBox
from test_database_member_catalogue_workflows import make_catalogue_input


@PropertyBase.dataclass
class FrameProbe(PropertyBase):
    def calculate(self, sim, params=None):
        return {
            "pos": sim["pos"].copy(),
            "vel": sim["vel"].copy(),
            "indices": np.array(sim.get_index_list(sim.ancestor), copy=True),
        }


def assert_frame(view, measured):
    np.testing.assert_array_equal(view.get_index_list(view.ancestor), measured["indices"])
    for key in ("pos", "vel"):
        expected = measured[key]
        assert view[key].units == expected.units
        np.testing.assert_allclose(view[key], expected, rtol=3e-6, atol=2e-6)


def operations():
    shift = ShiftPosTo(SimArray([1400.0, 0.0, 0.0], "pc"))
    spin = AlignVec(np.array([1.0, 0.0, 0.0]))
    boost = ShiftVelTo(SimArray([0.0, 1000.0, 0.0], "m s^-1"))
    return {
        "rotation": spin,
        "translation": shift,
        "velocity": boost,
        "first_shift": shift.then(spin),
        "first_spin": spin.then(shift),
    }


def test_saved_frames_match_live_positions_velocities_units_and_order(tmp_path, monkeypatch):
    sim = make_catalogue_input(tmp_path / "snap")
    native = FrameProbe().run(sim.dm).value
    methods = operations()
    family = FamilyFilter("dm")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        measured = Pipeline({name: FrameProbe().transform(op).filter(family) for name, op in methods.items()}).run(snap)
        assert measured.ok
        assert not np.allclose(measured.value["first_shift"]["pos"], measured.value["first_spin"]["pos"])
    with pe.Database(tmp_path / "db", read_only=True) as db:
        snap = next(iter(db.snapshots))
        # Deliberately reuse one live root on every load.
        db.bind_loader("default", lambda path: sim)

        def forbidden(*args, **kwargs):
            pytest.fail("frame navigation re-executed a calculator")

        monkeypatch.setattr(FamilyFilter, "calculate", forbidden)
        monkeypatch.setattr(AlignVec, "build_handle", forbidden)
        members = snap.members[family]
        assert len(members.origins) == len(methods)
        assert_frame(members.load(), native)
        kept = {}
        for name, method in methods.items():
            chosen = snap.members.where(filter=FamilyFilter, transform=method).one()
            assert chosen.origin.context.complete
            assert snap.members.where(context=chosen.origin.context).one().origin.id == chosen.origin.id
            kept[name] = chosen.load(frame="selection")
            assert_frame(kept[name], measured.value[name])
        for name, view in kept.items():
            assert_frame(view, measured.value[name])
        assert_frame(sim.dm, native)


def test_partial_target_replay_applies_shift_once_to_selected_particle_identities(tmp_path):
    from pynbody.halo.number_array import HaloNumberCatalogue

    sim = make_catalogue_input(tmp_path / "snap")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        cat = db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp"), name="groups")
        sub = cat.halos[42].subhalos.register(7, particles=sim[[0, 0, 1, 3, 5]])
        small = Sphere("9 kpc")
        moved = ShiftPosTo(SimArray([0.5, 0.0, 0.0], "kpc"), move_all=False)
        prepare = Pipeline({"move": FrameProbe().transform(moved, revert=False)}).filter(small)
        read = FrameProbe().filter(FamilyFilter("dm"))
        raw = FrameProbe().run(sub.load().dm).value
        measured = Pipeline({"prepare": prepare, "read": read}).run(sub)
        assert measured.ok
        assert np.any(measured.value["read"]["pos"] != raw["pos"])
        np.testing.assert_array_equal(measured.value["read"]["indices"], [0, 0, 1, 3])
        chosen = sub.members.where(filter=FamilyFilter("dm")).one()
        assert_frame(chosen.load(frame="selection"), measured.value["read"])
        assert_frame(chosen.load(), raw)


def test_unsupported_wrap_and_absent_file_leave_metadata_browsable(tmp_path):
    path = tmp_path / "snap"
    sim = make_catalogue_input(path)
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        result = Pipeline({"probe": FrameProbe().transform(WrapBox(10.0)).filter(FamilyFilter("dm"))}).run(snap)
        assert result.ok
        chosen = snap.members.where(filter="dm").one()
        context = chosen.origin.context
        assert not context.complete
        assert len(snap.members.where(context=context)) == 0
        assert len(chosen.load()) == len(sim.dm)
        with pytest.raises(ValueError, match="(?i)wrap|unsupported"):
            chosen.load(frame="selection")
        path.rename(tmp_path / "moved")
        assert chosen.origin.context == context
        with pytest.raises((OSError, RuntimeError, ValueError)):
            chosen.load()


def test_changed_family_layout_is_rejected_even_with_same_root_count(tmp_path):
    import pynbody

    sim = make_catalogue_input(tmp_path / "snap")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099", loader="custom")
        db.bind_loader("custom", lambda path: sim)
        result = FrameProbe().filter(FamilyFilter("dm")).run(snap)
        assert result.ok
        chosen = snap.members["dm"]
        replacement = pynbody.new(dm=len(sim))
        replacement._filename = sim.filename
        db.bind_loader("custom", lambda path: replacement)
        with pytest.raises(ValueError, match="family|layout"):
            chosen.load(frame="selection")


def test_missing_root_prevents_both_load_modes_and_preserves_metadata(tmp_path):
    path = tmp_path / "snap"
    sim = make_catalogue_input(path)
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        probe = FrameProbe().transform(operations()["rotation"]).filter(FamilyFilter("dm"))
        ref = probe.run(snap).store_ref
    path.rename(tmp_path / "moved")
    with pe.Database(tmp_path / "db", read_only=True) as db:
        hit = db.results.where(ref=ref).one()
        use = hit.members.where(role="output_filter").one()
        assert use.origin.context.complete
        for frame in ("snapshot", "selection"):
            with pytest.raises((OSError, RuntimeError, ValueError), match="snap|location|available"):
                use.load(frame=frame)
        assert db.results.where(ref=ref).one().members.one().origin.id == use.origin.id


def test_replay_converts_only_its_private_arrays(tmp_path):
    sim = make_catalogue_input(tmp_path / "snap")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        method = ShiftVelTo(SimArray([1.0, 0.0, 2.0], "km s^-1"))
        result = Pipeline({"probe": FrameProbe().transform(method).filter(FamilyFilter("dm"))}).run(snap)
        sim.physical_units(distance="pc", velocity="m s^-1", mass="Msol")
        originals = {key: sim[key].copy() for key in ("pos", "vel")}
        before_basis = tuple(sim._autoconvert)
        db.bind_loader("default", lambda path: sim)
        assert_frame(snap.members["dm"].load(frame="selection"), result.value["probe"])
        for key, original in originals.items():
            assert sim[key].units == original.units
            np.testing.assert_array_equal(sim[key], original)
        assert tuple(sim._autoconvert) == before_basis


@pytest.mark.data
def test_gadget_snapshot_replays_saved_frame_after_reopen(tmp_path, _test_data):
    import pynbody

    path = _test_data / "gadget3/data/snapshot_103/snap_103.hdf5"
    sim = pynbody.load(str(path))
    sim.physical_units()
    sub = sim[[3, 1, 3, 0]]
    family = FamilyFilter(str(sub.families()[0]))
    calc = FrameProbe().transform(AlignVec(np.array([1.0, 0.0, 0.0]))).filter(family)
    with pe.Database(tmp_path / "gadget.sqlite") as db:
        result = Pipeline({"frame": calc}).run(sub, database=db)
        assert result.ok
    with pe.Database(tmp_path / "gadget.sqlite", read_only=True) as db:
        hit = db.results.where(ref=result.store_ref, output=("frame",)).one()
        origin = hit.members.where(role="output_filter", filter=family).one()
        assert_frame(origin.load(frame="selection"), result.value["frame"])
        np.testing.assert_array_equal(result.value["frame"]["indices"], [3, 1, 3, 0])
