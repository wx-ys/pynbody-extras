"""Membership is an observed particle sequence, distinct from a result or a catalogue label."""

import json
import warnings

import numpy as np
import pytest
from pynbody.halo.number_array import HaloNumberCatalogue
from sqlalchemy import event, select

import pynbodyext as pe
from pynbodyext.calculate import FilterBase, Pipeline
from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamSum

from test_database import write_snapshot


@FilterBase.dataclass
class MassBelow(FilterBase):
    limit: float = 2.5

    def calculate(self, sim, params=None):
        return np.asarray(sim["mass"].in_units("Msol")) < self.limit


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected particle load or filter evaluation")


def assert_indices(view, expected):
    np.testing.assert_array_equal(view.get_index_list(view.ancestor), expected)


def test_both_save_paths_keep_observed_members_after_mutation_and_reopen(tmp_path, monkeypatch):
    sim = write_snapshot(tmp_path / "particles")
    calc = ParamSum("mass").filter(MassBelow())
    deferred = calc.run(sim)
    with pe.Database(tmp_path / "direct") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        calc.run(snap)
    sim["mass"][:] = 100
    monkeypatch.setattr(MassBelow, "calculate", forbidden)
    with pe.Database(tmp_path / "deferred") as db:
        deferred.save(db)
    for filename in ("direct", "deferred"):
        with pe.Database(tmp_path / filename, read_only=True) as db:
            snap = next(iter(db.snapshots))
            db.bind_loader("default", forbidden)
            selected = snap.members[MassBelow(2.5)]
            assert selected.origin.filter_signature.hash == MassBelow().to_signature().hash
            assert selected.origin.source_locator == snap.object_locator
            assert selected.origin.context.root_count == 4
            db.bind_loader("default", __import__("pynbody").load)
            view = selected.load()
            assert str(view["mass"].units) == str(sim["mass"].units)
            assert_indices(view, [0, 1])


def test_explicit_registration_unknown_empty_no_results_and_metadata_only_lookup(tmp_path, monkeypatch):
    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        assert hasattr(snap, "members"), "registered inputs must expose their member collection"
        dm = snap.members.register("dm")
        with warnings.catch_warnings(record=True) as emitted:
            empty = snap.members.register(FamilyFilter("star"))
        assert not any("ConstantProperty" in str(w.message) for w in emitted)
        assert_indices(dm.load(), [0, 1, 2, 3])
        assert_indices(empty.load(), [])
        assert empty.origin.id == snap.members["star"].origin.id
        assert dm.origin.id == snap.members[FamilyFilter("dm")].origin.id
        assert len(snap.results) == len(db.runs) == 0
        sql = []
        event.listen(db._engine, "before_cursor_execute", lambda c, cur, statement, *args: sql.append(statement))
        db.bind_loader("default", forbidden)
        monkeypatch.setattr(FamilyFilter, "calculate", forbidden)
        assert snap.members.register("dm").origin.id == dm.origin.id
        with pytest.raises(KeyError, match="registered|recorded"):
            snap.members[MassBelow()]
        assert len(snap.members.where(filter=MassBelow)) == 0
        assert all(s.lstrip().split(None, 1)[0].upper() in {"SELECT", "PRAGMA", "BEGIN"} for s in sql)
        assert list(empty.used_by) == []


def test_same_exact_set_keeps_origins_grouped_and_nested_navigation_pinned(tmp_path):
    from pynbodyext.transforms import AlignVec

    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        direct = snap.members.register("dm")
        face = AlignVec([1., 0., 0.])
        ParamSum("mass").transform(face).filter(FamilyFilter("dm")).run(snap)
        group = snap.members["dm"]
        assert_indices(group.load(), [0, 1, 2, 3])
        assert group.input_id == direct.input_id
        assert len(group.origins) == 2
        with pytest.raises(ValueError, match="ambiguous|2 origins"):
            _ = group.origin
        rotated = snap.members.where(filter=FamilyFilter, transform=face).one()
        assert rotated.origin.id != direct.origin.id
        assert snap.members.where(context=rotated.origin.context).one().origin.id == rotated.origin.id
        child = rotated.members.register(MassBelow())
        assert child.origin.parent_origin_id == rotated.origin.id
        assert rotated.members[MassBelow()].input_id == group.members[MassBelow()].input_id
        with pytest.raises(KeyError):
            _ = direct.members[MassBelow()]
        # A direct calculation on the chosen member must keep the same origin path.
        ParamSum("mass").filter(MassBelow(3.5)).run(rotated)
        assert_indices(rotated.members[MassBelow(3.5)].load(), [0, 1, 2])
        with pytest.raises(KeyError):
            _ = direct.members[MassBelow(3.5)]


def test_same_selector_different_particles_lists_candidates(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        calc = ParamSum("mass").filter(MassBelow())
        calc.run(sim, database=db)
        sim["mass"][:] = [4., 3., 2., 1.]
        calc.run(sim, database=db, mode="refresh")
        snap = next(iter(db.snapshots))
        candidates = list(snap.members.where(filter=MassBelow()))
        assert len(candidates) == 2
        observed = [v.load() for v in candidates]
        assert {tuple(v.get_index_list(v.ancestor)) for v in observed} == {(0, 1), (2, 3)}
        with pytest.raises(ValueError, match="ambiguous") as error:
            _ = snap.members[MassBelow()]
        assert all(member.origin.id in str(error.value) for member in candidates)
        with pytest.raises(ValueError, match="2"):
            snap.members.where(filter=MassBelow).one()


def test_order_and_multiplicity_survive_nested_runtime_capture(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        selections = []
        for indices in ([2, 0, 2], [0, 2, 2], [2, 0, 0]):
            pipeline = Pipeline({"mass": ParamSum("mass").filter(MassBelow(5.0))}).filter(FamilyFilter("dm"))
            result = pipeline.run(sim[indices], database=db)
            obj = db._registered_input(result.store_ref.sim_signature)
            assert hasattr(obj, "members"), "archive the exact input as the observation's direct source"
            selected = obj.members["dm"]
            child = selected.members[MassBelow(5.0)]
            assert child.origin.parent_origin_id == selected.origin.id
            assert_indices(child.load(), indices)
            selections.append(child.input_id)
        assert len(set(selections)) == 3


def test_shared_catalogue_cache_hit_retains_only_the_producer_origin(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    sim["grp"] = np.array([5, 5, 8, -1])
    catalogue = HaloNumberCatalogue(sim, array="grp", ignore=-1)
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        cat_a = db.register_catalogue(snap, catalogue, name="a")
        cat_b = db.register_catalogue(snap, catalogue, name="b")
        a, b = cat_a.halos[5], cat_b.halos[5]
        calc = ParamSum("mass").filter(FamilyFilter("dm"))
        first = calc.run(a)
        second = calc.run(b)
        assert second.from_store and first.store_ref == second.store_ref
        assert a.input_id == b.input_id
        assert a.members["dm"].origin.source_locator == a.object_locator
        with pytest.raises(KeyError):
            _ = b.members["dm"]
        assert b.members.register("dm").origin.id != a.members["dm"].origin.id
        assert len(db.runs) == 2
        assert len(list(a.members["dm"].used_by)) == 1
        assert not list(b.members["dm"].used_by)


def test_large_index_sequence_is_stored_once_for_ten_outputs(tmp_path):
    import pynbody
    from pynbodyext.database.models import InputInfo
    from pynbodyext.orm.models import Selection

    root = pynbody.new(dm=12000)
    root._filename = str(tmp_path / "virtual")
    root["mass"] = np.ones(len(root))
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(root, simulation="virtual", snapshot="s", loader="virtual")
        db.bind_loader("virtual", lambda _: root)
        calc = Pipeline({f"out_{i}": ParamSum("mass") for i in range(10)}).filter(FamilyFilter("dm"))
        ref = calc.run(snap).store_ref
        assert hasattr(snap, "members"), "share the index block for all consumers"
        from pynbodyext.database.member_models import SelectionUse
        selected = snap.members["dm"]
        with db._repo.session() as session:
            definitions = session.scalars(select(Selection.definition_json).where(Selection.view_input_id == selected.input_id)).all()
            info = session.get(InputInfo, selected.input_id)
            assert len(definitions) == 1
            assert json.loads(definitions[0])["indices"] == list(range(12000))
            assert len(info.description) < 1000
            uses = session.scalars(select(SelectionUse).where(SelectionUse.version_id == ref.version_id)).all()
            assert {tuple(json.loads(use.output_path_json)) for use in uses} == {(f"out_{i}",) for i in range(10)}
            assert {use.origin_id for use in uses} == {selected.origin.id}
        # The embedded metadata must not contain further copies of the ordered indices.
        with db._engine.connect() as connection:
            dump = "\n".join(connection.connection.driver_connection.iterdump())
        assert dump.count("11995,11996,11997,11998,11999") == 1


def test_deferred_raw_halo_uses_only_its_identified_live_catalogue(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    sim["grp"] = np.array([5, 5, 8, -1])
    catalogue = HaloNumberCatalogue(sim, array="grp", ignore=-1)
    result = ParamSum("mass").filter(FamilyFilter("dm")).run(catalogue[5])
    with pe.Database(tmp_path / "db") as db:
        result.save(db)
        halo = next(iter(db.snapshots)).halos[5]
        assert halo.members["dm"].origin.source_locator == halo.object_locator
        # An identical exact halo registered under another scientific catalogue cannot claim this observation.
        other = db.register_catalogue(next(iter(db.snapshots)), catalogue, name="other").halos[5]
        assert not list(other.members.where())


def test_grouped_member_with_multiple_parent_origins_registers_only_an_exact_input_child(tmp_path):
    from pynbodyext.transforms import AlignVec

    sim = write_snapshot(tmp_path / "particles")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        original = snap.members.register("dm")
        face = AlignVec([1., 0., 0.])
        ParamSum("mass").transform(face).filter(FamilyFilter("dm")).run(snap)
        grouped = snap.members["dm"]
        rotated = snap.members.where(transform=face).one()
        pinned = rotated.members.register(MassBelow())
        unpinned = grouped.members.register(MassBelow())
        assert unpinned.origin.id != pinned.origin.id
        assert unpinned.origin.parent_origin_id is None
        assert {o.id for o in grouped.members[MassBelow()].origins} == {unpinned.origin.id, pinned.origin.id}
        assert not list(original.members.where(filter=MassBelow))
        assert rotated.members[MassBelow()].origin.id == pinned.origin.id


def test_unfiltered_output_keeps_its_scientific_producer_when_a_second_halo_hits(tmp_path):
    from pynbodyext.database.member_models import OutputObservation
    sim = write_snapshot(tmp_path / "particles")
    sim["grp"] = np.array([5, 5, 8, -1])
    catalogue = HaloNumberCatalogue(sim, array="grp", ignore=-1)
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        a = db.register_catalogue(snap, catalogue, name="a").halos[5]
        b = db.register_catalogue(snap, catalogue, name="b").halos[5]
        calc = ParamSum("mass")
        ref = calc.run(a).store_ref
        assert calc.run(b).from_store
        with db._repo.session() as session:
            observation = session.get(OutputObservation, (ref.version_id, "[]"))
            assert tuple(json.loads(observation.source_locator_json)) == a.object_locator


def test_cross_family_source_rejected_before_deferred_registration(tmp_path):
    import pynbody
    from pynbodyext.calculate import PropertyBase
    sim = pynbody.new(dm=3, star=3)
    sim._filename = str(tmp_path / "file")
    with pe.Database(tmp_path / "db") as db:
        # Pynbody rejects this view before an invalid particle order can enter the registry.
        with pytest.raises(ValueError, match="Families.*ordering|family-block"):
            PropertyBase.as_property(0).filter(FamilyFilter("dm")).run(sim[[4, 0, 1]]).save(db)
        assert len(db.snapshots) == 0


def test_nonfinite_physical_aperture_round_trips_without_restricting_the_science(tmp_path):
    from pynbodyext.filters import Sphere
    sim = write_snapshot(tmp_path / "particles")
    aperture = Sphere(float("inf"))
    result = ParamSum("mass").filter(aperture).run(sim)
    with pe.Database(tmp_path / "db") as db:
        result.save(db)
    with pe.Database(tmp_path / "db", read_only=True) as db:
        member = next(iter(db.snapshots)).members[aperture]
        radius = next(p for p in member.origin.parameters if p.path == ("radius",))
        assert np.isposinf(radius.value) and radius.units == "kpc"
        assert_indices(member.load(), [0, 1, 2, 3])


def test_explicit_dynamic_aperture_chooses_the_direct_filter_and_keeps_real_auxiliary_origins(tmp_path):
    from pynbodyext.filters import Sphere
    from pynbodyext.properties import ParamContain

    sim = write_snapshot(tmp_path / "particles")
    radius = ParamContain(frac=0.5, cal_key="rxy", parameter="mass").filter(FamilyFilter("dm"))
    aperture = Sphere(2 * radius)
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="s")
        member = snap.members.register(aperture)
        assert member.origin.filter_signature.hash == aperture.to_signature().hash
        assert_indices(member.load(), np.flatnonzero(aperture.run(sim).value))
        assert snap.members["dm"].origin.filter_signature.hash == FamilyFilter("dm").to_signature().hash
        assert not list(member.used_by) and not list(snap.members["dm"].used_by)
        assert not db.runs and not len(snap.results)
