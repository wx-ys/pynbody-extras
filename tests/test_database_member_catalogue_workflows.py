"""Catalogue inventories preserve the scientific population and typed join keys."""

from __future__ import annotations

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray
from pynbody.halo.number_array import HaloNumberCatalogue
from sqlalchemy import event, insert

import pynbodyext as pe
from pynbodyext.calculate import PropertyBase
from pynbodyext.filters import FamilyFilter
from pynbodyext.orm.catalogues import typed_object_key
from pynbodyext.orm.models import CatalogueObject
from pynbodyext.properties import ParamSum


@PropertyBase.dataclass
class LowestMass(PropertyBase):
    def calculate(self, sim, params=None):
        return sim["mass"].min()


def make_catalogue_input(path):
    sim = pynbody.new(dm=5, star=3)
    (path.parent / "units.param").write_text("dKpcUnit = 1\ndMsolUnit = 1\nbComove = 0\n")
    sim["pos"] = SimArray(np.arange(24.0).reshape(8, 3), "kpc")
    sim["vel"] = SimArray(np.arange(24.0, 48.0).reshape(8, 3), "km s^-1")
    sim["mass"] = SimArray(np.arange(1.0, 9.0), "Msol")
    sim.properties["time"] = 0.0
    sim.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=str(path))
    sim = pynbody.load(str(path))
    sim.physical_units()
    sim["grp"] = np.array([42, 42, 42, 43, 43, 42, 42, 43])
    return sim


def test_complete_catalogue_is_the_denominator_and_study_can_be_reopened(tmp_path):
    pd = pytest.importorskip("pandas")
    sim = make_catalogue_input(tmp_path / "snap_099")
    mass = ParamSum("mass")
    stellar_mass = mass.filter(FamilyFilter("star"))
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        native = HaloNumberCatalogue(sim, array="grp")
        a = db.register_catalogue(snap, native, name="A")
        sim["grp_b"] = np.array([42, 42, 42, 42, 43, 42, 42, 43])
        b = db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp_b"), name="B")
        a7 = a.halos[42].subhalos.register(7, particles=sim[[0, 5, 6]])
        a8 = a.halos[42].subhalos.register(8, particles=sim[[1]])
        a9 = a.halos[43].subhalos.register(9, particles=sim[[3]])
        a.halos[42].subhalos.register(10)
        b7 = b.halos[42].subhalos.register(7, particles=sim[[4, 7]])
        assert a7.host.number == b7.host.number == 42
        assert a7.host.object_locator != b7.host.object_locator
        children = [a7, a8, a9]
        with pytest.warns(UserWarning, match="snapshot with no particles"):
            produced = [stellar_mass.run(obj) for obj in children]
        assert [float(res.value.in_units("Msol")) for res in produced] == [13.0, 0.0, 0.0]
        for halo in (a.halos[42], b.halos[42]):
            mass.run(halo)
        assert len(a8.members["star"].load()) == 0
        assert len(a9.members["star"].load()) == 0
        assert len(a7.members.register("dm").load()) == 1
        with pytest.raises((KeyError, NotImplementedError)):
            a.subhalos[10].members["star"]
        study = db.study(objects=children, calculations={"Mstar": stellar_mass})
        table = study.collect(missing="keep")
        for obj, saved in zip(table.objects, produced, strict=True):
            assert table.cell(obj, "Mstar").ref.record_ref == saved.store_ref
        table.save(tmp_path / "frozen.json")
        with pytest.raises(ValueError, match="duplicate"):
            db.study(objects=[a8.members["star"], a9.members["star"]], calculations={"M": mass})
        inventory = pd.concat([a.subhalos.to_dataframe(), b.subhalos.to_dataframe()], ignore_index=True)
        hits = db.results.where(objects=children + [a.subhalos[10], b7], calculator=stellar_mass).to_dataframe()
        hosts = db.results.where(objects=[a.halos[42], a.halos[43], b.halos[42]], calculator=mass).to_dataframe()
        compare = inventory.merge(
            hits[["object_locator", "ref"]], how="left", on="object_locator", validate="one_to_one"
        )
        right = hosts[["object_locator", "ref"]].rename(columns={"object_locator": "host_locator", "ref": "host_ref"})
        joined = compare.merge(right, how="left", on="host_locator", validate="many_to_one")
        assert len(compare) == len(joined) == 5
        assert joined["ref"].notna().tolist() == [True, True, True, False, False]
        assert joined["host_ref"].notna().tolist() == [True, True, False, True, True]
        assert joined["membership_known"].tolist() == [True, True, True, False, True]
        assert joined["host_ref"][0] == joined["host_ref"][1] == joined["host_ref"][3]
        assert joined["host_ref"][0] != joined["host_ref"][4]
        combined = db.study(
            objects=[*children, a.halos[42], a.halos[43], b.halos[42]],
            calculations={"Mstar": stellar_mass, "Mhost": mass},
        ).collect(missing="keep")
        assert [combined.cell(obj, "Mstar").ref for obj in children] == list(joined["ref"][:3])
        assert combined.cell(a.halos[42], "Mhost").ref == joined["host_ref"][0]
        assert combined.cell(a.halos[43], "Mhost").ref is None
        assert combined.cell(b.halos[42], "Mhost").ref == joined["host_ref"][4]
        with pytest.raises(pd.errors.MergeError):
            compare.merge(pd.concat([right, right.iloc[:1]]), how="left", on="host_locator", validate="many_to_one")
        minima = db.study(objects=[a8, a9], calculations={"min": LowestMass().filter(FamilyFilter("star"))})
        with pytest.warns(UserWarning, match="snapshot with no particles"):
            failures = minima.run(errors="collect")
        assert [failures.cell(obj, "min").status for obj in failures.objects] == ["failed", "failed"]

        def unavailable(path):
            raise OSError("saved particles unavailable")

        db.bind_loader("default", unavailable)
        failed_refresh = study.run(mode="refresh", errors="collect")
        assert all(not failed_refresh.cell(obj, "Mstar").has_value for obj in failed_refresh.objects)
        assert study.collect().cell(a7, "Mstar").ref == table.cell(a7, "Mstar").ref
        global_hits = db.results.where(calculator=stellar_mass)
        assert {(h.object_locator, h.host_locator) for h in global_hits} == {
            (child.object_locator, child.host_locator) for child in children
        }
    loaded = pe.StudyTable.load(tmp_path / "frozen.json")
    assert [float(loaded.cell(obj, "Mstar").value) for obj in loaded.objects] == [13.0, 0.0, 0.0]


def test_catalogue_identity_covers_root_revision_namespace_and_typed_id(tmp_path):
    sim = make_catalogue_input(tmp_path / "snap_099")
    catalogue = HaloNumberCatalogue(sim, array="grp")
    with pe.Database(tmp_path / "db") as db:
        locators = set()
        for step in ("098", "099"):
            snap = db.register(sim, simulation="s", snapshot=step)
            for revision in ("v1", "v2"):
                entries = db.register_catalogue(snap, catalogue, name="groups", revision=revision)
                for number in (7, "7"):
                    obj = entries.subhalos.register(number)
                    assert obj.host is None
                    assert entries.subhalos[number].object_locator == obj.object_locator
                    assert obj.object_locator not in locators
                    locators.add(obj.object_locator)
        assert len(locators) == 8


def test_inventory_page_and_point_lookup_have_bounded_sql_and_no_particle_reads(tmp_path):
    pytest.importorskip("pandas")
    sim = make_catalogue_input(tmp_path / "snap_099")

    def forbidden(*args, **kwargs):
        pytest.fail("metadata navigation touched particles or payloads")

    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        db.bind_loader("default", forbidden)
        db._store.codec.decode = forbidden
        counts = []
        for size in (100, 1000):
            cat = db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp"), name=f"cat-{size}")
            host = cat.halos[42]
            with db._repo.session() as session, session.begin():
                session.execute(
                    insert(CatalogueObject),
                    [
                        dict(
                            catalogue_fk=cat.catalogue_id,
                            kind="subhalo",
                            host_id=host.object_id,
                            typed_object_key=typed_object_key(i),
                        )
                        for i in range(size)
                    ],
                )
            sql = []

            def capture(conn, cursor, statement, params, *unused):
                if statement.lstrip().upper().startswith("SELECT"):
                    sql.append((statement, params))

            event.listen(db._engine, "before_cursor_execute", capture)
            try:
                assert list(host.subhalos.to_dataframe(limit=6)["number"]) == list(range(6))
                assert host.subhalos[17].number == 17
                assert str(size) in repr(host.subhalos)
            finally:
                event.remove(db._engine, "before_cursor_execute", capture)
            counts.append(len(sql))
            point, params = next((q, p) for q, p in sql if "typed_object_key = ?" in q)
            with db._engine.connect() as conn:
                details = " ".join(str(row[-1]) for row in conn.exec_driver_sql("EXPLAIN QUERY PLAN " + point, params))
            assert "INDEX" in details and "typed_object_key=?" in details
            assert any("host_id = ?" in q for q, p in sql)
            pages = [cat.subhalos.to_dataframe(limit=83, offset=offset) for offset in range(0, size, 83)]
            keys = [key for page in pages for key in page.object_locator]
            assert len(keys) == len(set(keys)) == size
        assert counts[0] == counts[1] < 20


def test_inventory_original_numbers_roundtrip_as_mixed_scalar_types(tmp_path):
    pytest.importorskip("pandas")
    sim = make_catalogue_input(tmp_path / "snap")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        cat = db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp"), name="groups")
        numbers = [2**60 + 1, 0.5]
        for key in numbers:
            cat.subhalos.register(key)
        exported = cat.subhalos.to_dataframe()
        for original, value, locator in zip(numbers, exported["number"], exported["object_locator"], strict=True):
            assert type(value) is type(original) and value == original
            assert cat.subhalos[value].object_locator == locator


def test_registering_a_known_halo_preserves_its_existing_calculation_address(tmp_path):
    sim = make_catalogue_input(tmp_path / "snap")
    native = HaloNumberCatalogue(sim, array="grp")
    mass = ParamSum("mass")
    with pe.Database(tmp_path / "db") as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        cat = db.register_catalogue(snap, native, name="groups")
        halo = cat.halos[42]
        saved = mass.run(halo)
        view = sim[np.flatnonzero(sim["grp"] == 42)]
        registered = cat.halos.register(42, particles=view)
        assert registered.input_id == halo.input_id
        assert mass.run(registered).store_ref == saved.store_ref
        known = cat.halos[43]
        cat.halos.register(43, particles=known)
        assert mass.run(known).store_ref == mass.run(native[43], database=db).store_ref


def test_two_writers_can_register_the_same_new_subhalo(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    sim = make_catalogue_input(tmp_path / "snap")
    db_path = tmp_path / "db"
    with pe.Database(db_path) as db:
        snap = db.register(sim, simulation="s", snapshot="099")
        db.register_catalogue(snap, HaloNumberCatalogue(sim, array="grp"), name="groups")
    observed_missing = Barrier(2)
    start = Barrier(2)

    def register():
        with pe.Database(db_path) as db:
            host = db.snapshots["099"].catalogues["groups"].halos[42]

            def after_read(conn, cursor, statement, params, *args):
                if (
                    statement.startswith("SELECT")
                    and "typed_object_key = ?" in statement
                    and typed_object_key(700) in params
                    and not conn.connection.driver_connection.in_transaction
                ):
                    observed_missing.wait(timeout=10)

            event.listen(db._engine, "after_cursor_execute", after_read)
            start.wait(timeout=10)
            return host.subhalos.register(700).object_id

    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(register) for _ in range(2)]
        assert calls[0].result() == calls[1].result()
