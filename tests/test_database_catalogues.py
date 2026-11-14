"""Catalogue browsing records names and exact selections without eager halo loads."""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pynbody
import pytest
from pynbody.halo import HaloCatalogue
from pynbody.halo.number_array import HaloNumberCatalogue

import pynbodyext as pe
from pynbodyext.properties import ParamSum
from test_database import write_snapshot


@pytest.fixture
def env(tmp_path):
    sim = write_snapshot(tmp_path / 'snapshot_099')
    sim['grp'] = np.array([5, 5, 8, -1])
    catalogue = HaloNumberCatalogue(sim, array='grp', ignore=-1)
    with pe.Database(tmp_path/'db') as db:
        ParamSum('mass').run(sim).save(db)
        yield db, db.snapshots['snapshot_099'], sim, catalogue


def test_register_and_browse_do_not_materialize_halos(env, monkeypatch):
    db, snap, sim, catalogue = env
    def forbidden(*args):
        raise AssertionError('halo membership loaded')
    monkeypatch.setattr(HaloNumberCatalogue, '__getitem__', forbidden)
    cat = db.register_catalogue(snap, catalogue, name='groups', revision='r1')
    assert len(snap.catalogues) == 1
    assert snap.catalogues['groups'].key == cat.key
    assert len(cat.halos) == 2
    assert [halo.number for halo in cat.halos] == [5, 8]
    assert snap.halos[5].number == 5
    assert 'groups' in repr(snap.catalogues)
    assert '5' in repr(cat.halos)
    assert len(cat.halos[5].results) == 0
    assert '0 results' in repr(cat.halos[5].results)
    with pytest.raises(KeyError):
        cat.halos[5].results[ParamSum('mass')]
    with pytest.raises(KeyError):
        cat.halos[0]
    with pytest.raises(KeyError):
        cat.halos['5']
    assert len(cat.subhalos) == 0


def test_materialization_freezes_indices_and_reopens_without_catalogue(env, tmp_path):
    db, snap, sim, catalogue = env
    cat = db.register_catalogue(snap, catalogue, name='groups')
    halo = cat.halos[5]
    np.testing.assert_allclose(halo.load()['mass'], [1., 2.])
    first = ParamSum('mass').run(halo)
    assert first.value == 3.
    assert halo.results[ParamSum('mass')].store_ref == first.store_ref
    with pe.Database(tmp_path/'db') as reopened:
        halo2 = reopened.snapshots['snapshot_099'].halos[5]
        np.testing.assert_allclose(halo2.load()['mass'], [1., 2.])
        def forbidden(path):
            raise AssertionError('particle loader invoked')
        reopened.bind_loader('default', forbidden)
        assert ParamSum('mass').run(halo2).store_ref == first.store_ref
        assert halo2.results[ParamSum('mass')].value == 3.


def test_unmaterialized_object_reopens_using_copied_membership(env, tmp_path):
    db, snap, sim, catalogue = env
    db.register_catalogue(snap, catalogue, name='groups')
    sim['grp'][:] = 99
    with pe.Database(tmp_path/'db') as reopened:
        halo = reopened.snapshots['snapshot_099'].halos[8]
        np.testing.assert_allclose(halo.load()['mass'], [3.])


def test_catalogue_namespaces_and_default_are_explicit(env):
    db, snap, sim, first_cat = env
    first = db.register_catalogue(snap, first_cat, name='groups', revision='r1')
    sim['other_grp'] = np.array([5, 8, 5, -1])
    second_cat = HaloNumberCatalogue(sim, array='other_grp', ignore=-1)
    second = db.register_catalogue(snap, second_cat, name='groups', revision='r2')
    assert first.key != second.key
    with pytest.raises(ValueError, match='ambiguous'):
        snap.catalogues['groups']
    with pytest.raises(ValueError, match='ambiguous'):
        snap.halos
    assert snap.catalogues[first.key].catalogue_id == first.catalogue_id
    a = ParamSum('mass').run(first.halos[5])
    b = ParamSum('mass').run(second.halos[5])
    assert a.value == 3. and b.value == 4.
    assert a.store_ref != b.store_ref
    db.set_default_catalogue(snap, second.key)
    assert snap.halos[5].key == second.halos[5].key


def test_same_membership_can_share_exact_input_but_not_catalogue_handles(env):
    db, snap, sim, catalogue = env
    a = db.register_catalogue(snap, catalogue, name='a')
    b = db.register_catalogue(snap, catalogue, name='b')
    one = ParamSum('mass').run(a.halos[5])
    two = ParamSum('mass').run(b.halos[5])
    assert a.halos[5].key != b.halos[5].key
    assert one.store_ref == two.store_ref


def test_registration_rejects_membership_drift_without_new_revision(env):
    db, snap, sim, catalogue = env
    db.register_catalogue(snap, catalogue, name='groups', revision='r1')
    sim['grp'][:] = [5, 8, 5, -1]
    changed = HaloNumberCatalogue(sim, array='grp', ignore=-1)
    with pytest.raises(ValueError, match='revision|membership'):
        db.register_catalogue(snap, changed, name='groups', revision='r1')
    np.testing.assert_allclose(snap.halos[5].load()['mass'], [1., 2.])


def test_registration_does_not_retain_snapshot_or_catalogue(env):
    db, snap, original_sim, original_cat = env
    # Separate weak-reference check independent of the fixture's strong references.
    other = write_snapshot(db.path.parent / 'other')
    other['grp'] = np.array([1, 1, 2, 2])
    cat = HaloNumberCatalogue(other)
    ParamSum('mass').run(other).save(db)
    db.register_catalogue(db.snapshots['other'], cat, name='groups')
    weak_sim, weak_cat = weakref.ref(other), weakref.ref(cat)
    del other, cat
    gc.collect()
    assert weak_sim() is None and weak_cat() is None
    assert len(db.snapshots['other'].halos) == 2


def test_unsupported_catalogue_and_wrong_root_fail_before_registration(env):
    db, snap, sim, catalogue = env
    with pytest.raises((TypeError, NotImplementedError), match='adapter|catalogue'):
        db.register_catalogue(snap, object(), name='bad')
    class UnknownCatalogue(HaloCatalogue):
        pass
    unknown = UnknownCatalogue(sim, catalogue.number_mapper)
    with pytest.raises(NotImplementedError, match='adapter'):
        db.register_catalogue(snap, unknown, name='unknown')
    other = write_snapshot(db.path.parent / 'other')
    other['grp'] = np.array([1, 1, 2, 2])
    with pytest.raises(ValueError, match='root|snapshot'):
        db.register_catalogue(snap, HaloNumberCatalogue(other), name='wrong')
    assert len(snap.catalogues) == 0


def test_associate_raw_halo_only_with_registered_catalogue(env):
    from pynbodyext.database.catalogues import associate_halo
    from pynbodyext.core.calculate.runtime.persistence import capture_input
    db, snap, sim, catalogue = env
    raw = catalogue[5]
    data = db._record_input(capture_input(raw))
    assert associate_halo(db, raw, data.id) is False
    db.register_catalogue(snap, catalogue, name='groups')
    assert associate_halo(db, raw, data.id) is True
    assert snap.halos[5]._materialized().input_id == data.id


def test_saved_live_halo_automatically_registers_supported_catalogue(env):
    import pynbodyext.database.catalogues as api
    from pynbodyext.core.calculate.runtime.persistence import capture_input
    db, snap, sim, catalogue = env
    raw = catalogue[8]
    data = db._record_input(capture_input(raw))
    assert api.associate_saved_catalogue(db, catalogue, 8, data.id)
    assert snap.halos[8].input_id == data.id
    assert len(snap.catalogues) == 1
    assert api.associate_saved_catalogue(db, catalogue, 8, data.id)
    assert len(snap.catalogues) == 1


def test_registered_subhalo_browses_and_replays_only_explicit_selection(env):
    from pynbodyext.database.models import ObjectSelection
    db, snap, sim, catalogue = env
    cat = db.register_catalogue(snap, catalogue, name='groups')
    host = cat.halos[5]
    metadata = db._repo.register_catalogue_object(cat.catalogue_id, 12, kind='subhalo', host_id=host.object_id)
    subhalo = cat.subhalos[12]
    assert len(cat.subhalos) == 1
    assert len(host.subhalos) == 1
    assert subhalo.number == 12
    assert len(subhalo.results) == 0
    with pytest.raises(NotImplementedError, match='subhalo'):
        subhalo.load()
    saved = ParamSum('mass').run(sim[[2]], database=db)
    selected = next(item for item in snap.selections if item._row.signature == saved.store_ref.sim_signature)
    with db._repo.session() as session, session.begin():
        session.add(ObjectSelection(object_id=metadata.id, input_id=selected.input_id))
    np.testing.assert_allclose(subhalo.load()['mass'], [3.])
    assert subhalo.results[ParamSum('mass')].store_ref == saved.store_ref
    assert ParamSum('mass').run(subhalo).from_store


def test_logical_catalogue_registration_does_not_bind_an_unrelated_root(env):
    db, snap, sim, catalogue = env
    logical = db.register(sim, simulation='simulation', snapshot='099')
    other = write_snapshot(db.path.parent / 'other')
    other['grp'] = np.array([5, 5, 8, -1])
    unrelated = HaloNumberCatalogue(other, ignore=-1)
    with pytest.raises(ValueError, match='root|snapshot|location'):
        db.register_catalogue(logical, unrelated, name='wrong')
    assert len(logical.catalogues) == 0


def test_uint64_halo_ids_keep_distinct_memberships(tmp_path):
    sim = write_snapshot(tmp_path/'snapshot_099')
    sim['grp'] = np.array([2**64-1, 2**64-2, 0, 0], dtype=np.uint64)
    cat = HaloNumberCatalogue(sim, ignore=0)
    with pe.Database(tmp_path/'db') as db:
        ParamSum('mass').run(sim).save(db)
        stored = db.register_catalogue(db.snapshots['snapshot_099'], cat, name='big-ids')
        np.testing.assert_allclose(stored.halos[2**64-1].load()['mass'], [1.])
        np.testing.assert_allclose(stored.halos[2**64-2].load()['mass'], [2.])


def test_first_halo_load_accepts_explicit_logical_location(tmp_path):
    first = write_snapshot(tmp_path/'a'/'snapshot_099')
    second = write_snapshot(tmp_path/'b'/'snapshot_099')
    first['grp'] = np.array([5, 5, 8, -1], dtype=np.int64)
    cat = HaloNumberCatalogue(first, ignore=-1)
    with pe.Database(tmp_path/'db') as db:
        snap = db.register(first, simulation='sim-A', snapshot='099')
        db.register(second, simulation='sim-A', snapshot='099')
        stored = db.register_catalogue(snap, cat, name='groups')
        location = db._repo.get_input(snap.input_id).locations[0].id
        np.testing.assert_allclose(stored.halos[5].load(location_id=location)['mass'], [1., 2.])


def test_first_halo_run_loads_particles_once_and_records_load_failure(env):
    db, snap, sim, cat = env
    stored = db.register_catalogue(snap, cat, name='groups')
    calls = []
    def loader(path):
        calls.append(path)
        return pynbody.load(path)
    db.bind_loader('default', loader)
    assert ParamSum('mass').run(stored.halos[5]).value == 3.
    assert len(calls) == 1
    db.bind_loader('default', lambda _: (_ for _ in ()).throw(OSError('offline')))
    with pytest.raises(OSError, match='offline'):
        ParamSum('mass').run(stored.halos[8])
    assert db.runs[-1].status == 'failed'
