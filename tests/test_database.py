"""Database onboarding contracts, exercised through real local pynbody snapshots."""
from __future__ import annotations

import gc
import subprocess
import sys
import weakref
from dataclasses import replace

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray

import pynbodyext as pe
from pynbodyext.calculate import PropertyBase
from pynbodyext.core.calculate.store import InMemoryResultStore
from pynbodyext.properties import ParamSum


@PropertyBase.dataclass
class CountMass(PropertyBase):
    scale: float = 1.
    calls = 0
    def calculate(self, sim, params=None):
        type(self).calls += 1
        return sim['mass'].sum() * self.scale


@pytest.fixture
def sim(tmp_path):
    return write_snapshot(tmp_path / 'snapshot_099')


def write_snapshot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / 'units.param').write_text('dKpcUnit = 1\ndMsolUnit = 1\nbComove = 0\n')
    s = pynbody.new(dm=4)
    s['pos'] = SimArray(np.arange(12.).reshape(4, 3), 'kpc')
    s['vel'] = SimArray(np.zeros((4, 3)), 'km s^-1')
    s['mass'] = SimArray([1., 2., 3., 4.], 'Msol')
    s.properties['time'] = 0.
    s.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=str(path))
    loaded = pynbody.load(str(path))
    loaded.physical_units()
    return loaded


def test_plain_run_save_reopen_and_load(sim, tmp_path):
    cc = ParamSum('mass')
    res = cc.run(sim)
    db = pe.Database(tmp_path / 'results.sqlite')
    ref = res.save(db, name='mass')
    assert res.store_ref == ref
    db.close()
    with pe.Database(tmp_path / 'results.sqlite') as db:
        snap = db.snapshots['snapshot_099']
        saved = snap.results['mass']
        assert snap.results[cc].store_ref == ref
        np.testing.assert_allclose(saved.value.in_units('Msol'), 10.)
        assert saved.calculator.to_signature().hash == cc.to_signature().hash
        np.testing.assert_allclose(snap.load()['mass'].in_units('Msol'), sim['mass'])
        assert len(db.simulations) == 0
        assert db.runs == []


def test_capture_precedes_mutation_and_does_not_retain_sim(tmp_path):
    sim = write_snapshot(tmp_path / "snapshot_099")
    cc = CountMass(scale=2.)
    original = cc.to_signature().hash
    res = cc.run(sim)
    cc.scale = 9.
    weak = weakref.ref(sim)
    del sim
    gc.collect()
    assert weak() is None
    with pe.Database(tmp_path/'db') as db:
        ref = res.save(db)
        assert ref.calculator_signature_hash == original
        assert db.snapshots['snapshot_099'].results[CountMass(scale=2.)].value == 20.


def test_modes_and_immutable_versions(sim, tmp_path):
    CountMass.calls = 0
    cc = CountMass()
    with pe.Database(tmp_path/'db') as db:
        with db.use():
            a = cc.run(sim)
            b = cc.run(sim)
            assert b.from_store and CountMass.calls == 1
            sim['mass'] *= 2
            c = cc.run(sim, mode='refresh')
            assert c.value == 20. and CountMass.calls == 2
            assert db.fetch(a.store_ref).value == 10.
            manual = cc.run(sim, mode='manual')
            assert manual.store_ref is None
            manual.save(name='mass')
        assert len(db.runs) == 3
        assert a.store_ref != c.store_ref
        assert b.store_ref == a.store_ref
        assert cc.run(sim).store_ref is None


def test_handle_hit_does_not_load_and_missing_data_refresh_fails(sim, tmp_path):
    with pe.Database(tmp_path/'db') as db:
        cc = CountMass()
        first = cc.run(sim, database=db)
        snap = db.snapshots['snapshot_099']
        def no_load(path):
            raise AssertionError('loader invoked')
        db.bind_loader('default', no_load)
        assert cc.run(snap).store_ref == first.store_ref
        with pytest.raises(Exception, match='loader invoked|execution failed'):
            cc.run(snap, mode='refresh')
        assert db.runs[-1].status == 'failed'
        assert snap.results[cc].store_ref == first.store_ref


def test_context_nesting_low_level_isolation_and_ownership(sim, tmp_path):
    cc = CountMass()
    with pe.Database(tmp_path/'a') as a, pe.Database(tmp_path/'b') as b:
        with a.use():
            cc.run(sim)
            with pytest.raises(RuntimeError):
                with b.use():
                    cc.run(sim)
                    raise RuntimeError('exit')
            assert cc.run(sim).from_store
            low = cc.run(sim, store=InMemoryResultStore())
            assert low.store_ref is not None
            with pytest.raises(ValueError):
                cc.run(sim, database=a, reuse=False)
            with pytest.raises(ValueError):
                cc.run(sim, mode='unknown')
        snap = a.snapshots['snapshot_099']
        with b.use(mode='refresh'):
            assert cc.run(snap).from_store
        with pytest.raises(ValueError):
            cc.run(snap, database=b)
        assert len(a.runs) == 3 and len(b.runs) == 1


def test_alias_conflict_and_duplicate_names(sim, tmp_path):
    with pe.Database(tmp_path/'db') as db:
        CountMass().run(sim).save(db, name='mass')
        with pytest.raises(ValueError, match='alias|name'):
            CountMass(scale=2.).run(sim).save(db, name='mass')
        other = write_snapshot(tmp_path/'other'/'snapshot_099')
        CountMass().run(other).save(db)
        with pytest.raises(ValueError, match='ambiguous'):
            db.snapshots['snapshot_099']
        assert len(db.snapshots) == 2
        for snap in db.snapshots:
            assert db.snapshots[snap.key].key == snap.key


def test_missing_destination_unstable_inputs_and_closed_database(sim, tmp_path):
    res = CountMass().run(sim)
    with pytest.raises(ValueError, match='database|Database'):
        res.save()
    with pe.Database(tmp_path/'db') as db:
        memory = pynbody.new(2)
        memory['mass'] = [1., 2.]
        with pytest.raises(ValueError):
            CountMass().run(memory, database=db)
        with sim.translate([1., 0., 0.]):
            with pytest.raises(ValueError):
                CountMass().run(sim).save(db)
        res.save(db)
    with pytest.raises(RuntimeError, match='closed'):
        res.save()


def test_selection_roundtrip_and_offline_hit(sim, tmp_path):
    view = sim[[2, 0, 2]]
    with pe.Database(tmp_path/'db') as db:
        first = CountMass().run(view, database=db)
        snap = db.snapshots['snapshot_099']
        assert len(snap.selections) == 1
        selected = next(iter(snap.selections))
        np.testing.assert_allclose(selected.load()['mass'], [3., 1., 3.])
    with pe.Database(tmp_path/'db') as db:
        selected = next(iter(db.snapshots['snapshot_099'].selections))
        db.bind_loader('default', lambda _: (_ for _ in ()).throw(AssertionError('IO')))
        assert CountMass().run(selected).store_ref == first.store_ref


def test_logical_registration_custom_loading_and_revision_ambiguity(sim, tmp_path):
    with pe.Database(tmp_path/'db') as db:
        snap = db.register(sim, simulation='sim-A', snapshot='099', revision='r1', loader='custom')
        db.bind_loader('custom', lambda _: sim)
        a = CountMass().run(snap)
        assert db.simulations['sim-A'].snapshots['099'].key == snap.key
        with db.use():
            assert CountMass().run(sim).store_ref == a.store_ref
        db.register(sim, simulation='sim-A', snapshot='099', revision='r2', loader='custom')
        with db.use(), pytest.raises(ValueError, match='ambiguous'):
            CountMass().run(sim)
    with pe.Database(tmp_path/'db') as db:
        snap = db.snapshots[snap.key]
        assert CountMass().run(snap).store_ref == a.store_ref
        with pytest.raises(KeyError, match='loader'):
            snap.load()


def test_save_recovery_does_not_append_versions(sim, tmp_path, monkeypatch):
    with pe.Database(tmp_path/'db') as db:
        res = CountMass().run(sim)
        finish = db._finish_save
        def fail(*args, **kwargs):
            raise RuntimeError('association failure')
        monkeypatch.setattr(db, '_finish_save', fail)
        with pytest.raises(Exception) as err:
            res.save(db, name='mass')
        ref = err.value.result_ref
        monkeypatch.setattr(db, '_finish_save', finish)
        db.recover(ref)
        db.recover(ref)
        assert db.snapshots['snapshot_099'].results['mass'].store_ref == ref
        assert len(db.versions(ref)) == 1


def test_plain_import_does_not_import_sqlalchemy():
    subprocess.run([sys.executable, '-c', "import pynbodyext; from pynbodyext.properties import ParamSum; assert 'sqlalchemy' not in __import__('sys').modules"], check=True)


def test_result_lookup_listing_and_historical_fetch_are_read_only(sim, tmp_path, monkeypatch):
    with pe.Database(tmp_path/'db') as db:
        cc = CountMass()
        first = cc.run(sim, database=db)
        snap = db.snapshots['snapshot_099']
        def forbidden(*args, **kwargs):
            raise AssertionError('unexpected registration or particle IO')
        monkeypatch.setattr(db._repo, 'register_input', forbidden)
        db.bind_loader('default', forbidden)
        assert db.fetch(first.store_ref).store_ref == first.store_ref
        assert snap.results[cc].store_ref == first.store_ref
        def decode_forbidden(*args):
            raise AssertionError('repr decoded scientific payload')
        monkeypatch.setattr(db._store.codec, 'decode', decode_forbidden)
        assert 'results' in repr(snap.results)
        assert 'registered inputs' in repr(db.snapshots)


def test_failed_encoding_and_failed_refresh_preserve_previous_result(sim, tmp_path):
    with pe.Database(tmp_path/'db') as db:
        cc = CountMass()
        first = cc.run(sim, database=db)
        changed = cc.run(sim)
        changed.value = np.array([(1, object())], dtype=[('n', 'i4'), ('value', object)])
        with pytest.raises(Exception):
            changed.save(db)
        assert db.snapshots['snapshot_099'].results[cc].store_ref == first.store_ref
        assert db.versions(first.store_ref) == [first.store_ref]


def test_context_save_destination_and_copy_reopen(sim, tmp_path):
    with pe.Database(tmp_path/'a') as a, pe.Database(tmp_path/'b') as b:
        cc = CountMass()
        r = cc.run(sim, database=a)
        original = r.store_ref
        with b.use(mode='manual'):
            copied = r.save(name='copy')
        assert copied.sim_signature == original.sim_signature
        assert a.fetch(original).value == b.fetch(copied).value
        assert len(b.runs) == 0


def test_auto_run_association_recovery_keeps_running_receipt(sim, tmp_path, monkeypatch):
    with pe.Database(tmp_path/'db') as db:
        finish = db._repo.finish_run
        def failed(*args, **kwargs):
            raise RuntimeError('finalize injected failure')
        monkeypatch.setattr(db._repo, 'finish_run', failed)
        with pytest.raises(Exception) as err:
            CountMass().run(sim, database=db)
        ref = err.value.result_ref
        assert db.runs[-1].status == 'running'
        monkeypatch.setattr(db._repo, 'finish_run', finish)
        db.recover(ref)
        assert db.runs[-1].status == 'succeeded'
        assert len(db.versions(ref)) == 1


def test_partial_root_cannot_be_cached_as_full_file(sim, tmp_path):
    partial = pynbody.load(sim.filename, take=[0, 1])
    partial.physical_units()
    with pe.Database(tmp_path/'db') as db:
        with pytest.raises(ValueError, match='partial|stable|loader'):
            CountMass().run(partial, database=db)
        with pytest.raises(ValueError, match='partial|stable|loader'):
            CountMass().run(partial).save(db)
        assert len(db.snapshots) == 0
        assert CountMass().run(sim, database=db).value == 10.


def test_recovery_rejects_corrupted_input_metadata(sim, tmp_path, monkeypatch):
    from pynbodyext.orm.models import DataObject
    with pe.Database(tmp_path/'db') as db:
        finish = db._finish_save
        monkeypatch.setattr(db, '_finish_save', lambda *args: (_ for _ in ()).throw(RuntimeError('injected')))
        with pytest.raises(Exception) as err:
            CountMass().run(sim).save(db)
        ref = err.value.result_ref
        monkeypatch.setattr(db, '_finish_save', finish)
        with db._repo.session() as session, session.begin():
            row = session.get(DataObject, next(iter(db.snapshots)).input_id)
            row.identity_key = 'CORRUPT'
        with pytest.raises(ValueError):
            db.recover(ref)


def test_result_listing_uses_bounded_metadata_without_reading_payloads(sim, tmp_path, monkeypatch):
    with pe.Database(tmp_path/'db') as db:
        CountMass().run(sim).save(db, name='mass')
        snap = db.snapshots['snapshot_099']
        def forbidden(*args, **kwargs):
            raise AssertionError('listing accessed result payload')
        monkeypatch.setattr(db._store, '_list', forbidden)
        monkeypatch.setattr(db._store, '_read_ref', forbidden)
        assert len(snap.results) == 1
        text = repr(snap.results)
        assert 'Msol' in text and 'shape' in text and 'mass' in text


def test_opted_in_path_repository_study_preserves_root_identity(sim, tmp_path):
    from pynbodyext.orm.study import run_study
    with pe.Database(tmp_path/'db') as db:
        CountMass().run(sim).save(db)
        snap = db.snapshots['snapshot_099']
        report = run_study(db._repo, [snap.input_id], {'mass': CountMass()}, store_name='default')
        assert len(report.cells) == 1


def test_low_level_store_and_database_share_exact_cache_identity(sim, tmp_path):
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore
    cc = CountMass()
    store = SQLAlchemyResultStore(f'sqlite:///{tmp_path / "db"}')
    first = cc.run(sim, store=store)
    store.engine.dispose()
    with pe.Database(tmp_path/'db') as db:
        hit = cc.run(sim, database=db)
        assert hit.from_store and hit.store_ref == first.store_ref
        assert len(db.snapshots['snapshot_099'].results) == 1


def test_recovering_old_receipt_does_not_promote_old_summary(sim, tmp_path):
    with pe.Database(tmp_path/'db') as db:
        cc = CountMass()
        first = cc.run(sim, database=db)
        sim['mass'] *= 2
        second = cc.run(sim, database=db, mode='refresh')
        db.recover(first.store_ref)
        snap = db.snapshots['snapshot_099']
        assert snap.results[cc].store_ref == second.store_ref
        assert second.store_ref.version_id in repr(snap.results)
        assert first.store_ref.version_id not in repr(snap.results)


def test_low_level_refresh_remains_visible_without_loading_payloads(sim, tmp_path, monkeypatch):
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore
    with pe.Database(tmp_path/'db') as db:
        cc = CountMass()
        cc.run(sim, database=db)
        store = SQLAlchemyResultStore(db._engine)
        refreshed = cc.run(sim, store=store, reuse=False)
        snap = db.snapshots['snapshot_099']
        monkeypatch.setattr(db._store, '_read_ref', lambda *args: (_ for _ in ()).throw(AssertionError('payload IO')))
        assert len(snap.results) == 1
        assert refreshed.store_ref.version_id in repr(snap.results)
        assert 'summary unavailable' in repr(snap.results)


def test_recovery_rejects_corrupted_particle_description(sim, tmp_path, monkeypatch):
    import json
    from pynbodyext.database.models import InputInfo
    with pe.Database(tmp_path/'db') as db:
        finish = db._finish_save
        monkeypatch.setattr(db, '_finish_save', lambda *args: (_ for _ in ()).throw(RuntimeError('injected')))
        with pytest.raises(Exception) as err:
            CountMass().run(sim[[0, 1]]).save(db)
        ref = err.value.result_ref
        monkeypatch.setattr(db, '_finish_save', finish)
        selected = next(iter(db.snapshots['snapshot_099'].selections))
        with db._repo.session() as session, session.begin():
            info = session.get(InputInfo, selected.input_id)
            desc = json.loads(info.description)
            desc['indices'] = [2, 3]
            info.description = json.dumps(desc)
        with pytest.raises(ValueError, match='selection|digest'):
            db.recover(ref)
