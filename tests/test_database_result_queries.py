"""Discovery works from stored recipes and versioned observations alone."""

import sqlite3

import numpy as np
import pynbody
import pytest
from sqlalchemy import event
from test_database import write_snapshot

import pynbodyext as pe
from pynbodyext.calculate import Pipeline, PropertyBase
from pynbodyext.core.calculate import Scope
from pynbodyext.core.calculate.study.table import SavedResultRef
from pynbodyext.filters import Disc, FamilyFilter, Sphere
from pynbodyext.properties import KappaRot, ParamSum
from pynbodyext.transforms import AlignVec, ShiftPosTo


def paths(hits):
    return [hit.output_path for hit in hits]


def forbidden(*args, **kwargs):
    pytest.fail("metadata discovery attempted to read particles, decode a payload, or reconstruct a calculator")


@PropertyBase.dataclass
class ProjectedHalfLight(PropertyBase):
    def calculate(self, sim, params=None):
        radius = np.hypot(sim["x"], sim["y"])
        order = np.argsort(radius)
        cumulative = np.cumsum(sim["r_band_luminosity"][order])
        return float(radius[order[np.searchsorted(cumulative, cumulative[-1] / 2)]])


def test_same_output_predicates_and_pinned_root_with_projected_stellar_measurements(tmp_path, monkeypatch):
    from pynbodyext.core.calculate.store.codec import ValueCodec

    face = AlignVec((0.0, 0.0, 1.0))
    # Keep the edge-on sightline between stars so every projected circular velocity is defined.
    edge = AlignVec((1.0, 1.0, 0.0))
    disc = Disc(10.0, 1.0)
    recipe = Pipeline(
        {
            "kinematics": Pipeline({"krot": KappaRot().filter(disc)}).transform(face),
            "Re": ProjectedHalfLight().transform(face),
            "edge": ParamSum("mass").transform(edge).filter(Sphere(30.0)),
        }
    )
    path = tmp_path / "stars"
    (tmp_path / "units.param").write_text("dKpcUnit = 1\ndMsolUnit = 1\nbComove = 0\n")
    sim = pynbody.new(star=6)
    sim["pos"][:] = [[1, 0, 0], [0, 2, 0], [-3, 0, 0], [0, -4, 0], [5, 0, 0], [0, 6, 0]]
    sim["vel"][:] = [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]
    sim["mass"][:] = 1
    sim.properties["time"] = 0.0
    sim.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=str(path))
    stars = pynbody.load(str(path))
    stars["r_band_luminosity"] = [1.0, 1.0, 8.0, 2.0, 1.0, 1.0]
    with pe.Database(tmp_path / "db") as db:
        assert hasattr(db, "results"), "saved outputs must be discoverable without reading their values"
        measured = recipe.run(stars)
        assert measured.value["Re"] == pytest.approx(3)
        assert measured.value["kinematics"]["krot"] == pytest.approx(1)
        assert float(measured.value["edge"]) == pytest.approx(6)
        ref = measured.save(db)
        source = next(iter(db.snapshots))
        monkeypatch.setattr(ValueCodec, "decode", forbidden)
        db.bind_loader("default", forbidden)
        assert paths(source.results.where(transform=face)) == [("kinematics", "krot"), ("Re",)]
        assert paths(db.results.where(objects=[source], transform=face, filter=Disc)) == [("kinematics", "krot")]
        assert paths(source.results.where(transform=face, filter=Sphere)) == []
        assert paths(source.results.where(filter=Disc(1.0, 1.0))) == []
        assert paths(source.results.where(filter=disc)) == [("kinematics", "krot")]
        hits = source.results.where(calculator=recipe.to_signature())
        assert paths(hits) == paths(source.results.where(calculator=recipe))
        assert {hit.ref.record_ref for hit in hits} == {ref}
        assert source.results.where(output=()).one().ref.record_ref == ref
        assert len(db.versions(ref)) == 1
        assert {hit.object_locator for hit in hits} == {source.object_locator}


def test_filter_type_finds_both_apertures_and_instance_pins_its_parameters(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    mass = ParamSum("mass")
    recipe = Pipeline({"small": mass.filter(Disc(3.0, 3.0)), "large": mass.filter(Disc(20.0, 20.0))})
    with pe.Database(tmp_path / "db") as db:
        source = db.register(sim, snapshot="099", simulation="s")
        measured = recipe.run(source)
        assert float(measured.value["small"]) == pytest.approx(1)
        assert float(measured.value["large"]) == pytest.approx(10)
        assert paths(source.results.where(filter=Disc)) == [("small",), ("large",)]
        for name, size in (("small", 3.0), ("large", 20.0)):
            hit = source.results.where(filter=Disc(size, size)).one()
            assert hit.output_path == (name,)
            assert hit.ref.record_ref == measured.store_ref


def test_transform_order_scope_layers_and_declared_use_roles(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    center = ShiftPosTo(np.array([1.0, 0.0, 0.0]))
    face = AlignVec((0.0, 0.0, 1.0))
    sphere, disc = Sphere(50.0), Disc(50.0, 50.0)
    mass = ParamSum("mass")
    recipe = Pipeline(
        {
            "direct": mass.filter(sphere),
            "and": mass.filter(sphere & disc),
            "not": mass.filter(~sphere),
            "measure": mass.transform(face.measure_with(sphere)),
            "two": Pipeline({"nested": mass.transform(face)}).transform(center),
            "flat": mass.transform(center).transform(face),
            "reverse": mass.transform(face).transform(center),
        }
    )
    with pe.Database(tmp_path / "db") as db:
        assert hasattr(db, "results"), "query declared output scopes independently of execution observations"
        recipe.run(sim).save(db)
        q = db.results
        assert paths(q.where(filter=Sphere)) == [("direct",)]
        use = {hit.output_path: hit for hit in q.where(uses=sphere)}
        assert set(use) == {("direct",), ("and",), ("not",), ("measure",)}
        assert use[("measure",)].role == "transform_measure"
        assert use[("direct",)].role == "output_filter"
        assert use[("and",)].occurrence_path[-1] in {"left", "right"}
        assert use[("not",)].occurrence_path[-1] == "child"
        assert paths(q.where(transform=(center, face))) == [("two", "nested"), ("flat",)]
        assert paths(q.where(transform=(face, center))) == [("reverse",)]
        assert paths(q.where(scope=(Scope(transform=center), Scope(transform=face)))) == [("two", "nested")]
        assert paths(q.where(scope=Scope(transforms=(center, face)))) == [("flat",)]
        assert paths(q.where(scope=Scope(transforms=(center, face), revert_policy=False))) == []


def test_readonly_legacy_recipes_without_new_tables_or_science_reads(tmp_path, monkeypatch):
    from pynbodyext.core.calculate.store.codec import ValueCodec
    from pynbodyext.database.models import Base

    sim = write_snapshot(tmp_path / "particles")
    recipe = Pipeline({"mass": ParamSum("mass").filter(Sphere(20.0)), "plain": ParamSum("x")})
    path = tmp_path / "db"
    with pe.Database(path) as db:
        assert hasattr(db, "results"), "legacy signatures must be queryable without a schema upgrade"
        ref = recipe.run(sim).save(db)
        from pynbodyext.database.member_models import OutputObservation, SelectionUse
        from pynbodyext.database.models import RecipeOutput

        tables = {cls.__table__ for cls in (OutputObservation, SelectionUse, RecipeOutput)}
        with db._engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                if table in tables:
                    table.drop(conn)
    with pe.Database(path, read_only=True) as db:
        statements = []
        event.listen(db._engine, "before_cursor_execute", lambda c, cur, stmt, *args: statements.append(stmt))
        db.bind_loader("default", forbidden)
        monkeypatch.setattr(ValueCodec, "decode", forbidden)
        monkeypatch.setattr(Pipeline, "__init__", forbidden)
        hit = db.results.where(filter=Sphere, ref=SavedResultRef(db.database_id, ref)).one()
        assert hit.output_path == ("mass",)
        assert hit.context is None and not hit.members.recorded
        assert not list(hit.members)
        assert all(s.lstrip().split(None, 1)[0].upper() in {"BEGIN", "SELECT", "PRAGMA"} for s in statements)
        assert all("record_json" not in s for s in statements)


def test_frozen_study_keeps_actual_members_after_refresh_and_checks_ownership(tmp_path, monkeypatch):
    @PropertyBase.dataclass
    class Direction(PropertyBase):
        direction = (0.0, 0.0, 1.0)

        def calculate(self, sim, params=None):
            return np.array(type(self).direction)

    sim = write_snapshot(tmp_path / "particles")
    face = AlignVec(Direction())
    selected = Disc(30.0, 4.5)
    recipe = Pipeline({"mass": ParamSum("mass").filter(selected), "plain": ParamSum("mass")}).transform(face)
    with pe.Database(tmp_path / "db") as db, pe.Database(tmp_path / "other") as other:
        assert hasattr(db, "results"), "query actual uses by pinned version and scientific output"
        snap = db.register(sim, simulation="galaxies", snapshot="0")
        first = recipe.run(snap)
        study = db.study(objects=[snap], calculations={"profiles": recipe})
        exported = tmp_path / "table.json"
        study.collect().save(exported)
        frozen = pe.StudyTable.load(exported)
        old_ref = frozen.cell(frozen.objects[0], "profiles").ref
        monkeypatch.setattr(Direction, "direction", (1.0, 0.0, 0.0))
        second = recipe.run(snap, mode="refresh")
        assert first.store_ref != second.store_ref
        current = db.results.where(output=("mass",)).one()
        assert current.ref.record_ref == second.store_ref
        hit = db.results.where(ref=old_ref, output=("mass",)).one()
        use = hit.members.where(role="output_filter", filter=selected).one()
        assert use.ref == old_ref and use.output_path == ("mass",)
        assert use.role == "output_filter"
        assert hit.context == use.origin.context
        assert hit.context != current.context
        assert paths(db.results.where(context=hit.context)) == []
        assert {row.output_path for row in db.results.where(ref=old_ref, context=hit.context)} == {
            ("mass",),
            ("plain",),
        }
        particles = use.load()
        np.testing.assert_array_equal(particles.get_index_list(particles.ancestor), [0])
        current_particles = current.members.one().load()
        np.testing.assert_array_equal(current_particles.get_index_list(current_particles.ancestor), [0, 1])
        prior_uses = [x for x in use.member.used_by if x.ref == old_ref]
        assert [(x.output_path, x.occurrence_path) for x in prior_uses] == [(hit.output_path, use.occurrence_path)]
        plain = db.results.where(ref=old_ref, output=("plain",)).one()
        assert plain.members.recorded and not list(plain.members)
        assert db.fetch(hit.ref).value["mass"] == first.value["mass"]
        with pytest.raises(ValueError, match="another database"):
            other.results.where(ref=old_ref)


def test_same_filter_occurrences_and_exported_receipts(tmp_path):
    pytest.importorskip("pandas")
    sim = write_snapshot(tmp_path / "particles")
    small, large = Disc(30.0, 3.0), Disc(30.0, 10.0)
    mass = ParamSum("mass")
    recipe = Pipeline(
        {"small": mass.filter(small), "large": mass.filter(large), "repeated": mass.filter(large) & mass.filter(large)}
    )
    with pe.Database(tmp_path / "db") as db:
        ref = recipe.run(sim).save(db)
        snap = next(iter(db.snapshots))
        assert paths(snap.results.where(filter=Disc)) == [("small",), ("large",)]
        assert paths(snap.results.where(filter=small)) == [("small",)]
        hits = snap.results.where(uses=large, output=("repeated",))
        assert len(hits) == 2
        assert len({item.occurrence_path for item in hits}) == 2
        uses = snap.results.where(output=("repeated",)).one().members
        assert len(uses) == 2
        assert len({u.origin.id for u in uses}) == 1
        assert len({u.occurrence_path for u in uses}) == 2
        frame = hits.to_dataframe()
        assert set(frame["object_locator"]) == {snap.object_locator}
        assert set(frame["ref"]) == {SavedResultRef(db.database_id, ref)}
        assert set(frame["occurrence_path"]) == {hit.occurrence_path for hit in hits}


def test_read_snapshot_pins_receipt_while_another_connection_refreshes(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    recipe = ParamSum("mass").filter(FamilyFilter("dm"))
    path = tmp_path / "db"
    with pe.Database(path) as writer:
        with writer._engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        snap = writer.register(sim, simulation="run", snapshot="0")
        first = recipe.run(snap).store_ref
        refreshed = []
        with pe.Database(path, read_only=True) as reader:

            def refresh_after_receipt_selection(conn, cursor, statement, *args):
                if not refreshed and f"FROM {writer._store._versions.name} " in statement:
                    refreshed.append(recipe.run(snap, mode="refresh").store_ref)

            event.listen(reader._engine, "after_cursor_execute", refresh_after_receipt_selection)
            hit = reader.results.where(calculator=recipe).one()
            assert len(refreshed) == 1 and first != refreshed[0]
            assert hit.ref.record_ref == first
            assert hit.members.one().ref == hit.ref
            assert reader.results.where(calculator=recipe).one().ref.record_ref == refreshed[0]


def test_large_object_recipe_and_version_sets_share_one_read_snapshot(tmp_path):
    from test_database import CountMass

    sim = write_snapshot(tmp_path / "particles")
    path = tmp_path / "db"
    with pe.Database(path) as writer:
        with writer._engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        objects = [writer.register(sim, simulation="run", snapshot=str(i)) for i in range(13)]
        recipes = [CountMass(float(i)) for i in range(len(objects))]
        first = [recipe.run(obj).store_ref for obj, recipe in zip(objects, recipes, strict=True)]
        with pe.Database(path, read_only=True) as reader:
            # A low SQLite limit exercises the same boundary as a catalogue with thousands of entries.
            event.listen(
                reader._engine, "checkout", lambda conn, *_: conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
            )
            refreshed = []

            def refresh_after_first_batch(conn, cursor, statement, *args):
                if not refreshed and f"FROM {writer._store._versions.name} " in statement:
                    refreshed.append(recipes[0].run(objects[0], mode="refresh").store_ref)

            event.listen(reader._engine, "after_cursor_execute", refresh_after_first_batch)
            repeated = list(reversed(list(reader.snapshots))) * 3
            hits = reader.results.where(objects=repeated)
            assert [hit.ref.record_ref for hit in hits] == first
            assert all(hit.context_recorded for hit in hits)
            assert len(refreshed) == 1 and refreshed[0] != first[0]
            assert [hit.ref.record_ref for hit in reader.results.where()] == first[1:] + refreshed
            old = reader.results.where(objects=repeated, calculator=recipes[0], ref=first[0]).one()
            assert old.ref.record_ref == first[0]


def test_large_reverse_membership_keeps_every_pinned_use_in_version_order(tmp_path):
    sim = write_snapshot(tmp_path / "particles")
    path = tmp_path / "db"
    with pe.Database(path) as db:
        snap = db.register(sim, simulation="run", snapshot="0")
        receipts = []
        for index in range(12):
            shift = ShiftPosTo(np.array([float(index), 0., 0.]))
            recipe = ParamSum("mass").transform(shift).filter(FamilyFilter("dm"))
            receipts.append(recipe.run(snap).store_ref)
    with pe.Database(path, read_only=True) as db:
        event.listen(db._engine, "checkout",
                     lambda conn, *_: conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8))
        member = next(iter(db.snapshots)).members["dm"]
        assert len(member.origins) == len(receipts)
        hits = member.used_by
        assert [hit.ref.record_ref for hit in hits] == receipts
        assert {hit.origin.id for hit in hits} == {origin.id for origin in member.origins}


def test_repeating_catalogue_handles_keeps_one_row_per_scientific_locator(tmp_path):
    from pynbody.halo.number_array import HaloNumberCatalogue

    sim = write_snapshot(tmp_path / "particles")
    sim["grp"] = np.array([5, 5, 8, -1])
    native = HaloNumberCatalogue(sim, array="grp", ignore=-1)
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="run", snapshot="0")
        a, b = [db.register_catalogue(snap, native, name=name).halos[5] for name in ("a", "b")]
        receipt = ParamSum("mass").run(a).store_ref
        locators = [a.object_locator, b.object_locator]
        statements = []
        counts = []
        event.listen(db._engine, "before_cursor_execute", lambda c, cur, sql, *args: statements.append(sql))
        for size in (1, 1000):
            statements.clear()
            hits = db.results.where(objects=[a, b] * size)
            assert [hit.object_locator for hit in hits] == locators
            assert [hit.ref.record_ref for hit in hits] == [receipt, receipt]
            counts.append(len(statements))
        assert counts[0] == counts[1]


def test_global_discovery_batches_composite_subhalo_host_keys(tmp_path, monkeypatch):
    from pynbody.halo.number_array import HaloNumberCatalogue

    sim = write_snapshot(tmp_path / "particles")
    sim["grp"] = np.array([5, 5, 8, 8])
    path = tmp_path / "db"
    expected = []
    with pe.Database(path) as db:
        snap = db.register(sim, simulation="run", snapshot="0")
        catalogue = db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp"), name="groups")
        for number, index in ((5, 0), (8, 2)):
            host = catalogue.halos[number]
            for order, indices in enumerate(([index], [index + 1], [index, index + 1])):
                child = host.subhalos.register(f"{number}-{order}", particles=sim[indices])
                receipt = ParamSum("mass").run(child).store_ref
                expected.append((child.object_locator, host.object_locator, receipt))
    with pe.Database(path, read_only=True) as db:
        event.listen(
            db._engine, "checkout", lambda conn, *_: conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        )
        db.bind_loader("default", forbidden)
        monkeypatch.setattr(db._store.codec, "decode", forbidden)
        hits = db.results.where(calculator=ParamSum("mass"))
        assert [(hit.object_locator, hit.host_locator, hit.ref.record_ref) for hit in hits] == expected
