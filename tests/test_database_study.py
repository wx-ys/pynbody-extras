"""ADR-021 acceptance scenarios through the public Database API."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pynbody
import pytest
from pynbody.halo.number_array import HaloNumberCatalogue

import pynbodyext as pe
from pynbodyext.properties import ParamSum
from test_database import CountMass, write_snapshot


@pytest.fixture
def study_env(tmp_path):
    sim = write_snapshot(tmp_path / "snapshot")
    with pe.Database(tmp_path / "results.sqlite") as db:
        obj = db.register(sim, simulation="box", snapshot="99", loader="research")
        db.bind_loader("research", pynbody.load)
        yield db, obj, sim


def test_a19_a31_definition_reopens_without_construction(study_env, monkeypatch):
    db, obj, _ = study_env
    cc = ParamSum("mass")
    signature = cc.to_signature()
    study = db.study(objects=[obj], calculations={"mass": signature, "alias": cc})
    definition_id = study.save(name="published")
    assert study.save(name="published") == definition_id
    assert study.save(name="another-name") == definition_id
    signature.payload.clear()
    study.calculations["mass"].payload.clear()
    with pytest.raises(ValueError, match="already"):
        db.study(objects=[obj], calculations={"other": cc}).save(name="published")

    def forbidden(*args, **kwargs):
        pytest.fail("calculator constructed during inspection")

    monkeypatch.setattr(ParamSum, "__init__", forbidden)
    with pe.Database(db.path, read_only=True) as reopened:
        saved = reopened.studies["published"]
        assert saved.definition_id == definition_id
        assert saved.objects == study.objects
        assert tuple(saved.calculations) == ("mass", "alias")
        assert saved.calculations["mass"].payload
        with pytest.raises(TypeError):
            saved.calculations["another"] = signature


def test_a05_duplicate_exact_catalogue_inputs_are_rejected(study_env):
    db, obj, sim = study_env
    sim["grp"] = np.array([5, 5, 8, -1])
    catalogue = HaloNumberCatalogue(sim, array="grp", ignore=-1)
    a = db.register_catalogue(obj, catalogue, name="a").halos[5]
    b = db.register_catalogue(obj, catalogue, name="b").halos[5]
    with pytest.raises(ValueError, match="duplicate"):
        db.study(objects=[a, b], calculations={"mass": ParamSum("mass")})
    assert len(obj.selections) == 0
    assert db.runs == []


def test_a10_exact_live_bindings(study_env):
    db, obj, _ = study_env
    cc = CountMass()
    study = db.study(objects=[obj], calculations={"mass": cc})
    with pytest.raises(ValueError, match="signature"):
        study.bind_calculators({"mass": CountMass(scale=2)})
    bound = study.bind_calculators({"mass": CountMass()})
    assert bound is not study
    assert bound.definition_id == study.definition_id
    cc.scale = 2
    with pytest.raises(ValueError, match="signature"):
        study.plan()
    assert db.runs == []


def test_a21_foreign_reference_rejected_even_for_same_input(study_env):
    db, obj, sim = study_env
    saved = db.study(objects=[obj], calculations={})
    with pe.Database(db.path.with_name("foreign.sqlite")) as foreign:
        other = foreign.register(sim, simulation="box", snapshot="99", loader="research")
        assert other.input_id == obj.input_id and other.key == obj.key
        with pytest.raises(ValueError, match="database"):
            foreign.study(objects=saved.objects, calculations={})


def test_a27_read_only_does_not_create_or_upgrade(tmp_path, study_env):
    missing = tmp_path / "absent.sqlite"
    with pytest.raises(FileNotFoundError):
        pe.Database(missing, read_only=True)
    assert not missing.exists()

    db, obj, sim = study_env
    calc = ParamSum("mass")
    ref = calc.run(obj).store_ref
    path = db.path
    with pe.Database(path, read_only=True) as reader:
        obj = reader.snapshots[obj.key]
        assert obj.results.get(calc).store_ref == ref
        assert obj.results.get(CountMass()) is None
        study = reader.study(objects=[obj], calculations={"mass": calc})
        with pytest.raises(PermissionError, match="read.only"):
            study.run()
        with pytest.raises(PermissionError, match="read.only"):
            study.save(name="no")
        with pytest.raises(PermissionError, match="read.only"):
            calc.run(obj)
        with pytest.raises(PermissionError, match="read.only"):
            reader.register(sim, simulation="no", snapshot="no")

    object_key = obj.key
    db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE pynbodyext_database_namespace")
    old_bytes = path.read_bytes()
    with pe.Database(path, read_only=True) as older:
        assert older.snapshots[object_key].results.get(calc).store_ref == ref
        with pytest.raises(RuntimeError, match="schema"):
            _ = older.studies
    assert path.read_bytes() == old_bytes


def test_a01_a02_a03_a06_share_exact_versions(study_env):
    db, obj, sim = study_env
    cc = CountMass()
    first = cc.run(sim, database=db, mode="manual").save(db)
    study = db.study(objects=[obj], calculations={"renamed": cc, "alias": cc})
    study.save(name="first")
    study.save(name="second")
    loads = []

    def load(path):
        loads.append(path)
        return pynbody.load(path)

    db.bind_loader("research", load)
    results = study.run()
    assert results.cell(obj, "renamed").ref.record_ref == first
    assert results.cell(obj, "alias").ref.record_ref == first
    assert len(loads) == 0
    expanded = db.study(objects=study.objects, calculations={**study.calculations, "twice": CountMass(scale=2)})
    second = expanded.run()
    exact = second.cell(obj, "twice").ref.record_ref
    assert CountMass(scale=2).run(obj).store_ref == exact
    assert db.versions(exact) == [exact]
    assert len(loads) == 1
    fresh = study.run(mode="refresh")
    refs = [fresh.cell(obj, label).ref.record_ref for label in study.calculations]
    assert db.versions(first) == [first, *refs]
    assert len({fresh.cell(obj, label).run_id for label in study.calculations}) == 2
    assert len(loads) == 3


def test_study_compares_nan_bindings_and_keeps_strict_json_definitions(study_env):
    from pynbodyext.core.calculate import ConstantProperty

    db, obj, _ = study_env
    cc = ConstantProperty(float("nan"))
    study = db.study(objects=[obj], calculations={"signal": cc})
    rebound = study.bind_calculators({"signal": ConstantProperty(float("nan"))})
    assert rebound.calculations["signal"].hash == cc.to_signature().hash
    with pytest.raises(ValueError, match="JSON compliant"):
        rebound.save(name="nan")


def test_a04_scientific_input_and_calculator_changes_miss(study_env, monkeypatch):
    db, obj, sim = study_env
    cc = CountMass()
    old = cc.run(obj).store_ref
    assert CountMass(scale=2).run(obj).store_ref.cache_key != old.cache_key
    changed = db.register(sim, simulation="box", snapshot="99", loader="research", revision="new")
    assert cc.run(changed).store_ref.cache_key != old.cache_key
    sim["grp"] = np.array([5, 5, 8, -1])
    cat = db.register_catalogue(obj, HaloNumberCatalogue(sim, array="grp", ignore=-1), name="groups")
    report = db.study(objects=[cat.halos[5], cat.halos[8]], calculations={"mass": cc}).run()
    assert report.cell(cat.halos[5], "mass").value == 3
    assert report.cell(cat.halos[8], "mass").value == 3
    assert report.cell(cat.halos[5], "mass").ref.cache_key != report.cell(cat.halos[8], "mass").ref.cache_key
    monkeypatch.setattr(CountMass, "signature_version", 2, raising=False)
    assert CountMass().run(obj).store_ref.cache_key != old.cache_key


def test_a07_unmaterialized_catalogue_reads_are_metadata_only(study_env, monkeypatch):
    db, obj, sim = study_env
    sim["grp"] = np.array([5, 5, 8, -1])
    cat = db.register_catalogue(obj, HaloNumberCatalogue(sim, array="grp", ignore=-1), name="groups")
    halo = cat.halos[5]
    calc = ParamSum("mass")
    study = db.study(objects=[halo], calculations={"mass": calc})
    study.save(name="halo")

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected particle IO, payload decode or materialization")

    with pe.Database(db.path, read_only=True) as rd:
        halo = rd.snapshots[obj.key].catalogues[cat.key].halos[5]
        monkeypatch.setattr(type(halo), "_materialized", forbidden)
        monkeypatch.setattr(rd._store.codec, "decode", forbidden)
        rd.bind_loader("research", forbidden)
        bound = rd.studies["halo"]
        assert bound.plan()[0].action == "blocked"
        assert halo.results.get(calc) is None
        assert bound.collect(missing="keep").cell(halo, "mass").status == "missing"
        with pytest.raises(KeyError, match="missing"):
            bound.collect(missing="raise")
        assert len(rd.snapshots[obj.key].selections) == 0
        assert rd.runs == []


def test_a08_a09_reopen_cache_hits_without_calculator_or_loader(study_env, monkeypatch):
    db, obj, sim = study_env
    calc = CountMass()
    study = db.study(objects=[obj], calculations={"cached": calc})
    study.run()
    study.save(name="hit")
    db.study(objects=[obj], calculations={"cached": calc, "absent": CountMass(scale=2)}).save(name="miss")

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected calculator construction or particle load")

    monkeypatch.setattr(CountMass, "__init__", forbidden)
    Path(sim.filename).unlink()
    with pe.Database(db.path) as reopened:
        cached = reopened.studies["hit"]
        assert cached.plan()[0].action == "reuse"
        assert cached.collect().cell(obj, "cached").value == 10
        assert cached.run().cell(obj, "cached").from_store
        count = len(reopened.runs)
        missing = reopened.studies["miss"]
        plan = missing.plan()
        assert [cell.action for cell in plan] == ["reuse", "blocked"]
        assert "loader" in plan[1].reason
        reopened.bind_loader("research", forbidden)
        assert "calculator type" in missing.plan()[1].reason
        with pytest.raises(ValueError, match="blocked"):
            missing.run()
        assert len(reopened.runs) == count


def test_a11_a13_a14_failed_refresh_does_not_report_old_success(study_env, monkeypatch):
    db, obj, _ = study_env
    good, bad = CountMass(), CountMass(scale=2)
    study = db.study(objects=[obj], calculations={"bad": bad, "good": good})
    past = study.run()
    original = CountMass.calculate

    def fail(self, *args, **kwargs):
        if self.scale == 2:
            raise ArithmeticError("scientific failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CountMass, "calculate", fail)
    report = study.run(mode="refresh", errors="collect")
    cell = report.cell(obj, "bad")
    assert cell.status == "failed" and cell.ref is None
    assert "scientific failure" in cell.error.message
    assert report.cell(obj, "good").status == "succeeded"
    assert study.collect().cell(obj, "bad").ref == past.cell(obj, "bad").ref
    from pynbodyext.database import StudyExecutionError

    with pytest.raises(StudyExecutionError) as caught:
        study.run(mode="refresh")
    assert caught.value.report.cell(obj, "bad").status == "failed"
    assert caught.value.report.cell(obj, "good").status == "not_run"
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("stage", ["before", "after", "record"])
def test_a15_a16_persistence_stops_with_exact_receipt(study_env, tmp_path, monkeypatch, stage):
    db, obj, _ = study_env
    from pynbodyext.database import StudyPersistenceError

    study = db.study(objects=[obj], calculations={"first": CountMass(), "second": CountMass(scale=2)})

    def fail(*args, **kwargs):
        raise OSError("metadata unavailable")

    with monkeypatch.context() as patch:
        if stage == "before":
            patch.setattr(db, "_begin_invocation", fail)
        elif stage == "after":
            patch.setattr(db, "_finish_save", fail)
        else:
            patch.setattr(db._repo, "fail_run", fail)
            db.bind_loader("research", fail)
        with pytest.raises(StudyPersistenceError) as caught:
            study.run(errors="collect")
    error = caught.value
    cell = error.report.cell(obj, "first")
    assert cell.status == ("not_run" if stage == "before" else "incomplete")
    assert cell.ref is None and error.report.cell(obj, "second").status == "not_run"
    if stage == "after":
        path = tmp_path / "interrupted.json"
        error.report.save(path)
        assert error.result_ref is not None
        assert len(db.versions(error.result_ref)) == 1
        db.recover(error.result_ref)
        assert len(db.versions(error.result_ref)) == 1
        assert pe.StudyTable.load(path).cell(obj, "first").status == "incomplete"
    else:
        assert error.result_ref is None


def test_a17_a18_retries_are_exact_and_definition_bound(study_env, monkeypatch):
    db, obj, sim = study_env
    other = db.register(sim, simulation="box", snapshot="second", loader="research")
    cc, second = CountMass(), CountMass(scale=2)
    study = db.study(objects=[obj, other], calculations={"a": cc, "b": second})
    original = CountMass.calculate
    attempts = []

    def fail_diagonal(self, *args, **kwargs):
        attempts.append(self.scale)
        if len(attempts) in (1, 4):
            raise ArithmeticError("diagonal")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CountMass, "calculate", fail_diagonal)
    report = study.run(errors="collect")
    cells = report.cells(status="failed")
    retry = study.run(cells=cells)
    assert [(c.input_key, c.label) for c in retry.cells(status="succeeded")] == [(obj.key, "a"), (other.key, "b")]
    assert len(attempts) == 6
    count = len(db.runs)
    with pytest.raises(ValueError, match="selector"):
        db.study(objects=study.objects[::-1], calculations=study.calculations).run(cells=cells)
    with pe.Database(db.path.with_name("second.sqlite")) as foreign:
        foreign_obj = foreign.register(sim, simulation="box", snapshot="99", loader="research")
        with pytest.raises(ValueError, match="selector"):
            foreign.study(objects=[foreign_obj], calculations={"a": cc}).run(cells=cells)
    assert len(db.runs) == count


def test_a23_collect_pins_all_versions_before_payload_read(study_env, monkeypatch):
    db, obj, _ = study_env
    cc, second = CountMass(), CountMass(scale=2)
    study = db.study(objects=[obj], calculations={"first": cc, "second": second})
    initial = study.run()
    original = db._store._get_version
    refreshed = []

    def refresh_between_reads(version):
        if not refreshed:
            with pe.Database(db.path) as writer:
                writer.bind_loader("research", pynbody.load)
                target = writer.snapshots[obj.key]
                result = second.run(target, mode="refresh")
                refreshed.append(result.store_ref)
        return original(version)

    monkeypatch.setattr(db._store, "_get_version", refresh_between_reads)
    collected = study.collect()
    assert collected.cell(obj, "second").ref == initial.cell(obj, "second").ref
    assert refreshed[0] != collected.cell(obj, "second").ref.record_ref


def test_a28_manual_empty_axes_and_empty_selectors(study_env):
    db, obj, _ = study_env
    cc = CountMass()
    with pe.Database(db.path, read_only=True) as reader:
        reader.bind_loader("research", pynbody.load)
        study = reader.study(objects=[reader.snapshots[obj.key]], calculations={"mass": cc})
        with reader.use(mode="manual"):
            report = study.run()
        cell = report.cell(obj, "mass")
        assert cell.value == 10 and cell.run_id is None and cell.ref is None and cell.from_store is False
    study = db.study(objects=[obj], calculations={"mass": cc})
    assert study.run(cells=[]).cell(obj, "mass").status == "not_run"
    assert db.study(objects=[], calculations={"mass": cc}).run().objects == ()
    assert not db.study(objects=[obj], calculations={}).run().cells()
    assert not db.runs and obj.results.get(cc) is None


def test_a32_interrupt_propagates_with_partial_report(study_env, monkeypatch):
    db, obj, _ = study_env

    def interrupt(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(CountMass, "calculate", interrupt)
    study = db.study(objects=[obj], calculations={"first": CountMass(), "second": CountMass(scale=2)})
    with pytest.raises(KeyboardInterrupt) as caught:
        study.run(errors="collect")
    assert caught.value.report.cell(obj, "first").status == "incomplete"
    assert caught.value.report.cell(obj, "second").status == "not_run"
    assert len(db.runs) == 1 and db.runs[0].status == "running"


def test_a10_rejects_mutation_before_create_or_rebind(study_env):
    db, obj, _ = study_env
    stable = db.study(objects=[obj], calculations={"mass": CountMass()})
    cc = CountMass()
    cc.scale = 3
    with pytest.raises(ValueError, match="signature"):
        db.study(objects=[obj], calculations={"mass": cc})
    with pytest.raises(ValueError, match="signature"):
        stable.bind_calculators({"mass": cc})
    assert not db.runs


def test_a16_unconfirmed_current_result_requires_explicit_recovery(study_env, monkeypatch):
    db, obj, _ = study_env
    from pynbodyext.database import DatabasePersistenceError, StudyPersistenceError

    study = db.study(objects=[obj], calculations={"mass": CountMass()})

    def fail(*args, **kwargs):
        raise OSError("could not associate version")

    with monkeypatch.context() as patch:
        patch.setattr(db, "_finish_save", fail)
        with pytest.raises(StudyPersistenceError) as caught:
            study.run()
    error = caught.value
    assert error.result_ref is not None
    count = len(db.runs)
    assert study.plan()[0].action == "blocked"
    with pytest.raises(DatabasePersistenceError, match="recover") as preflight:
        study.run()
    assert preflight.value.result_ref == error.result_ref
    with pytest.raises(DatabasePersistenceError, match="recover"):
        study.collect()
    assert len(db.runs) == count
    db.recover(error.result_ref)
    assert study.run().cell(obj, "mass").ref.record_ref == error.result_ref
    assert len(db.versions(error.result_ref)) == 1


def test_a16_commit_acknowledgment_failure_preserves_tracked_receipt(study_env, monkeypatch):
    db, obj, _ = study_env
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore
    from pynbodyext.database import StudyPersistenceError

    original = SQLAlchemyResultStore._put

    def commit_then_raise(store, record):
        original(store, record)
        raise OSError("commit acknowledgment lost")

    study = db.study(objects=[obj], calculations={"mass": CountMass()})
    with monkeypatch.context() as patch:
        patch.setattr(SQLAlchemyResultStore, "_put", commit_then_raise)
        with pytest.raises(StudyPersistenceError) as caught:
            study.run(errors="collect")
    error = caught.value
    cell = error.report.cell(obj, "mass")
    assert cell.status == "incomplete"
    assert cell.recovery_ref.record_ref == error.result_ref
    db.recover(cell.recovery_ref)
    assert study.collect().cell(obj, "mass").ref == cell.recovery_ref


def test_a32_process_exit_leaves_retriable_running_invocation(study_env):
    db, obj, _ = study_env
    script = """
import os
import sys
import pynbody
from pynbodyext import Database
from pynbodyext.calculate import PropertyBase

@PropertyBase.dataclass
class ForcedExit(PropertyBase):
    code: int = 37
    def calculate(self, sim, params=None):
        os._exit(self.code)

with Database(sys.argv[1]) as db:
    db.bind_loader("research", pynbody.load)
    study = db.study(objects=[db.snapshots[sys.argv[2]]], calculations={"exit": ForcedExit()})
    study.run(errors="collect")
"""
    result = subprocess.run([sys.executable, "-c", script, str(db.path), obj.key], check=False, timeout=30)
    assert result.returncode == 37
    assert len(db.runs) == 1 and db.runs[0].status == "running"
    retry = db.study(objects=[obj], calculations={"mass": ParamSum("mass")}).run()
    assert retry.cell(obj, "mass").status == "succeeded"
    assert len(db.runs) == 2


def test_a10_legacy_calculator_may_have_a_cyclic_runtime_memo(study_env):
    from dataclasses import field

    from pynbodyext.calculate import PropertyBase

    @PropertyBase.dataclass
    class Memoized(PropertyBase):
        factor: int = 2
        memo: dict = field(init=False)

        def __post_init__(self):
            self.memo = {}
            self.memo["self"] = self.memo

        def calculate(self, sim, params=None):
            return self.factor * 5

    _, _, sim = study_env
    assert Memoized().run(sim).value == 10


def test_a10_unauditable_normalized_parameter_cannot_poison_a_study(study_env):
    from pynbodyext.calculate import PropertyBase

    class OpaqueScale:
        def __init__(self, value):
            self.value = value

    @PropertyBase.dataclass
    class Normalized(PropertyBase):
        scale: int = 1

        def __post_init__(self):
            self.scale = OpaqueScale(self.scale)

        def calculate(self, sim, params=None):
            return self.scale.value

    db, obj, sim = study_env
    calculator = Normalized()
    assert calculator.run(sim).value == 1
    calculator.scale.value = 3
    with pytest.raises(ValueError, match="signature"):
        db.study(objects=[obj], calculations={"scale": calculator})
    assert not db.runs


def test_a10_rechecks_binding_after_project_loader(study_env):
    db, obj, _ = study_env
    from pynbodyext.database import StudyExecutionError

    calculator = CountMass()
    study = db.study(objects=[obj], calculations={"mass": calculator, "later": ParamSum("mass")})

    def change_then_load(path):
        calculator.scale = 3
        return pynbody.load(path)

    db.bind_loader("research", change_then_load)
    with pytest.raises(StudyExecutionError, match="signature") as caught:
        study.run(errors="collect")
    assert caught.value.report.cell(obj, "mass").status == "failed"
    assert caught.value.report.cell(obj, "later").status == "not_run"
    assert len(db.runs) == 1 and db.runs[0].status == "failed"
    assert obj.results.get(CountMass()) is None
