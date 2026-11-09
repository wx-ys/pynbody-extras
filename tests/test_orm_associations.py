"""Metadata relationships track exact result versions across failures and restarts."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import replace

import pynbody
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext.calculate import ConfiguredSnapshotIdentity, LogicalSnapshotIdentity, PropertyBase
from pynbodyext.core.calculate import LambdaProperty, Pipeline, RunOptions
from pynbodyext.core.calculate.store import RecordRef, SQLAlchemyResultStore
from pynbodyext.properties import ParamSum


def api():
    return importlib.import_module("pynbodyext.orm")


def sim(path, value=6):
    result = pynbody.new(2)
    result["mass"] = [float(value), 0.0]
    result._filename = str(path)
    return result


@pytest.fixture
def environment(tmp_path):
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'payload.sqlite'}")
    path = tmp_path / "snap"
    identity = LogicalSnapshotIdentity("project", "simulation", "099")
    provider = ConfiguredSnapshotIdentity({path: identity})
    yield tmp_path, store, path, identity, provider
    store.engine.dispose()


def repository(env):
    return api().AnalysisRepository(f"sqlite:///{env[0] / 'metadata.sqlite'}", stores={"results": env[1]})


def prepare(repo, env):
    data = repo.register_input(env[3], location=str(env[2]))
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    return data, recipe, repo.begin_run(data.id, recipe.hash, store_name="results")


def test_full_relationship_chain_queries_and_reopen(environment):
    env, orm = environment, api()
    repo = repository(env)
    calc = Pipeline({"mass": ParamSum("mass")})
    recorded = repo.run(calc, sim(env[2]), sim_identity=env[4], store_name="results")
    run = recorded.run
    assert run.status == "succeeded" and run.reused is False
    assert run.input.timestep.logical_id == "099"
    assert run.input.timestep.simulation.logical_id == "simulation"
    assert run.input.timestep.simulation.project_id == "project"
    assert run.recipe.hash == calc.to_signature().hash
    assert RecordRef.from_json(run.result_link.ref_json) == recorded.result.store_ref
    assert dict(repo.fetch_result(run.id).value)["mass"] == 6
    with repo.session() as session:
        found = session.scalars(
            sa.select(orm.Run)
            .join(orm.Run.input)
            .join(orm.DataObject.timestep)
            .join(orm.Timestep.simulation)
            .where(orm.Simulation.project_id == "project")
        ).one()
        assert found.id == run.id
        assert len(found.input.timestep.simulation.timesteps) == 1
        assert found.input.locations[0].uri == str(env[2])
        assert found.recipe.runs[0].id == run.id
    repo.engine.dispose()
    reopened = repository(env)
    assert reopened.get_run(run.id).recipe.hash == calc.to_signature().hash
    assert reopened.fetch_result(run.id).store_ref == recorded.result.store_ref
    assert reopened.audit() == []
    assert reopened.runs(input_id=run.input_id, recipe_hash=run.recipe_hash, status="succeeded")[0].id == run.id
    reopened.engine.dispose()


def test_registration_keeps_revisions_views_and_locations_separate(environment):
    env, orm = environment, api()
    repo = repository(env)
    first = repo.register_input(env[3], location="/old/snap")
    again = repo.register_input(env[3], location="/new/snap")
    assert first.id == again.id
    for field in ("project_id", "simulation_id", "timestep_id", "revision", "loader_id"):
        other = repo.register_input(replace(env[3], **{field: "other"}))
        assert other.id != first.id
    root = sim(env[2])
    child = repo.register_input(env[4](root[[0]]))
    assert child.parent_id == first.id and child.kind == "subset"
    with repo.session() as session:
        original = session.get(orm.DataObject, first.id)
        assert {x.uri for x in original.locations} == {"/old/snap", "/new/snap"}
        assert original.children[0].id == child.id
    recipe = ParamSum("mass").to_signature()
    assert repo.register_recipe(recipe).hash == repo.register_recipe(recipe).hash


def test_cache_hit_has_own_run_and_refresh_preserves_old_result(environment):
    env, repo = environment, repository(environment)
    calc, snapshot = ParamSum("mass"), sim(env[2])
    opts = dict(sim_identity=env[4], store_name="results")
    first = repo.run(calc, snapshot, **opts)
    snapshot["mass"] *= 2
    hit = repo.run(calc, snapshot, **opts)
    fresh = repo.run(calc, snapshot, reuse=False, **opts)
    assert len({x.run.id for x in (first, hit, fresh)}) == 3
    assert hit.run.reused and not fresh.run.reused
    assert first.result.store_ref == hit.result.store_ref != fresh.result.store_ref
    env[1].delete(fresh.result.store_ref)  # invalidate current cache, keep history
    assert repo.fetch_result(first.run.id).value == 6
    assert repo.fetch_result(fresh.run.id).value == 12
    assert repo.audit() == []


@PropertyBase.dataclass
class Explodes(PropertyBase):
    def calculate(self, sim, params=None):
        raise RuntimeError("scientific failure")


def test_raised_and_collected_failures_are_recorded(environment):
    env, orm, repo = environment, api(), repository(environment)
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run(Explodes(), sim(env[2]), sim_identity=env[4], store_name="results")
    run = repo.get_run(caught.value.run_id)
    assert run.status == "failed" and run.result_link is None
    assert "scientific failure" in run.error_message
    assert caught.value.__cause__ is not None
    recorded = repo.run(
        Explodes(), sim(env[2]), sim_identity=env[4], store_name="results", options=RunOptions(errors="collect")
    )
    assert recorded.result.errors and recorded.run.status == "failed"
    assert recorded.run.result_link is None and env[1].list_refs() == []


def test_begin_failure_prevents_calculation(environment, monkeypatch):
    env, repo = environment, repository(environment)

    def fail(*args, **kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(repo, "begin_run", fail)
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    assert env[1].list_refs() == []


def test_final_commit_failure_rolls_back_and_explicit_recovery_uses_saved_receipt(environment):
    env, orm, repo = environment, api(), repository(environment)

    def fail_link(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_result_links"):
            raise RuntimeError("injected metadata failure")

    sa.event.listen(repo.engine, "before_cursor_execute", fail_link)
    try:
        with pytest.raises(orm.RunPersistenceError) as caught:
            repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", fail_link)
    error = caught.value
    assert error.result_ref is not None and error.result.value == 6
    assert repo.get_run(error.run_id).status == "running"
    assert repo.get_run(error.run_id).result_link is None
    assert [(issue.run_id, issue.code) for issue in repo.audit()] == [(error.run_id, "unfinished")]
    assert repo.unlinked_versions(error.run_id) == [error.result_ref]
    recovered = repo.finish_run(error.run_id, error.result_ref)
    assert recovered.status == "succeeded" and recovered.reused is None
    assert repo.finish_run(error.run_id, error.result_ref).id == recovered.id
    assert repo.unlinked_versions(error.run_id) == []
    assert repo.fetch_result(error.run_id).value == 6
    assert repo.audit() == []


def test_interrupted_run_candidates_are_never_attached_automatically(environment):
    env, repo = environment, repository(environment)
    data, recipe, running = prepare(repo, env)
    saved = ParamSum("mass").run(sim(env[2]), store=env[1], sim_identity=env[4])
    newer = ParamSum("mass").run(sim(env[2], 12), store=env[1], sim_identity=env[4], reuse=False)
    assert repo.unlinked_versions(running.id) == [saved.store_ref, newer.store_ref]
    assert repo.get_run(running.id).result_link is None
    repo.finish_run(running.id, saved.store_ref)
    assert repo.fetch_result(running.id).value == 6
    assert repo.unlinked_versions(running.id) == [newer.store_ref]
    with pytest.raises(ValueError):
        repo.finish_run(running.id, newer.store_ref)
    with pytest.raises(ValueError):
        repo.fail_run(running.id, RuntimeError("too late"))


@pytest.mark.parametrize("change", ["unpinned", "missing", "input", "recipe"])
def test_finish_rejects_invalid_receipts_without_changing_running(environment, change):
    env, repo = environment, repository(environment)
    _, _, running = prepare(repo, env)
    saved = ParamSum("mass").run(sim(env[2]), store=env[1], sim_identity=env[4]).store_ref
    if change == "unpinned":
        ref = replace(saved, version_id=None)
    elif change == "missing":
        ref = replace(saved, version_id="absent")
    elif change == "input":
        ref = replace(saved, sim_signature=replace(env[3], timestep_id="other").signature())
    else:
        ref = replace(saved, calculator_signature_hash="0" * 64)
    with pytest.raises((ValueError, KeyError)):
        repo.finish_run(running.id, ref)
    assert repo.get_run(running.id).status == "running"
    assert repo.get_run(running.id).result_link is None


def test_audit_checks_existence_without_decoding_and_reports_missing_payload(environment, monkeypatch):
    env, repo = environment, repository(environment)
    recorded = repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    monkeypatch.setattr(env[1].codec, "decode", lambda value: pytest.fail("audit must not decode payload"))
    assert repo.audit() == []
    with env[1].engine.begin() as conn:
        conn.execute(env[1]._table.delete())
        conn.execute(env[1]._versions.delete())
    assert repo.audit()[0].code == "missing_payload"
    with pytest.raises(KeyError):
        repo.fetch_result(recorded.run.id)


def test_missing_backend_and_corrupt_link_are_detectable(environment):
    env, orm, repo = environment, api(), repository(environment)
    recorded = repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    unbound = orm.AnalysisRepository(f"sqlite:///{env[0] / 'metadata.sqlite'}")
    assert unbound.audit()[0].code == "missing_store"
    with repo.session() as session, session.begin():
        session.get(orm.ResultLink, recorded.run.id).ref_json = "{}"
    assert repo.audit()[0].code == "invalid_link"


def test_failed_run_cannot_be_completed(environment):
    env, repo = environment, repository(environment)
    _, _, running = prepare(repo, env)
    repo.fail_run(running.id, RuntimeError("cancelled by caller"))
    saved = ParamSum("mass").run(sim(env[2]), store=env[1], sim_identity=env[4])
    with pytest.raises(ValueError):
        repo.finish_run(running.id, saved.store_ref)
    assert repo.get_run(running.id).status == "failed"


def test_fk_and_uniqueness_are_enforced(environment):
    env, orm, repo = environment, api(), repository(environment)
    data, recipe, run = prepare(repo, env)
    with pytest.raises(sa.exc.IntegrityError):
        with repo.session() as session, session.begin():
            session.execute(sa.delete(orm.DataObject).where(orm.DataObject.id == data.id))
    with pytest.raises(sa.exc.IntegrityError):
        with repo.session() as session, session.begin():
            session.add(orm.Simulation(project_id="project", logical_id="simulation"))
    assert repo.get_run(run.id).input_id == data.id


@pytest.mark.parametrize(
    "identity", [("session", 123), ("sim", "/some/path"), ("sim", "pynbodyext.logical-snapshot/v1", "p")]
)
def test_nonlogical_inputs_are_not_promoted_to_scientific_objects(environment, identity):
    repo = repository(environment)
    with pytest.raises((TypeError, ValueError)):
        repo.register_input(identity)


def test_unstable_recipe_and_unknown_store_fail_before_execution(environment):
    env, repo = environment, repository(environment)
    with pytest.raises(ValueError):
        repo.register_recipe(LambdaProperty(lambda sim: 1).to_signature())
    with pytest.raises((ValueError, KeyError)):
        repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="unknown")
    assert repo.runs() == [] and env[1].list_refs() == []


def test_metadata_tables_have_no_payload_columns(environment):
    repo = repository(environment)
    inspector = sa.inspect(repo.engine)
    tables = inspector.get_table_names()
    assert len(tables) == 10
    assert "pynbodyext_orm_selections" in tables
    assert not any(
        column["name"] in {"value", "named", "record_json", "payload"}
        for table in tables
        for column in inspector.get_columns(table)
    )


def test_pure_calculation_does_not_import_sqlalchemy():
    program = """
import sys
class NoSQL:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sqlalchemy" or fullname.startswith("sqlalchemy."):
            raise ImportError("SQLAlchemy intentionally unavailable")
sys.meta_path.insert(0, NoSQL())
import pynbody
from pynbodyext.properties import ParamSum
snapshot = pynbody.new(1)
snapshot["mass"] = [4.0]
assert ParamSum("mass").run(snapshot).value == 4
assert "sqlalchemy" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_audit_rejects_broken_reference_shape_as_invalid_link(environment):
    env, orm, repo = environment, api(), repository(environment)
    recorded = repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    with repo.session() as session, session.begin():
        session.get(orm.ResultLink, recorded.run.id).ref_json = json.dumps({"schema": "pynbodyext.result.ref/v1"})
    assert repo.audit()[0].code == "invalid_link"


def test_audit_checks_input_hierarchy_not_just_payload_identity(environment):
    env, orm, repo = environment, api(), repository(environment)
    recorded = repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    other = repo.register_input(replace(env[3], timestep_id="other"))
    with repo.session() as session, session.begin():
        session.get(orm.DataObject, recorded.run.input_id).timestep_id = other.timestep_id
    assert repo.audit()[0].code == "invalid_link"
    with pytest.raises(ValueError):
        repo.fetch_result(recorded.run.id)


def test_concurrent_completions_keep_one_terminal_reference(environment):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    env, repo = environment, repository(environment)
    _, _, running = prepare(repo, env)
    refs = [
        ParamSum("mass").run(sim(env[2], value), store=env[1], sim_identity=env[4], reuse=False).store_ref
        for value in (6, 12)
    ]
    barrier = Barrier(2)

    def complete(ref):
        barrier.wait(timeout=10)
        try:
            repo.finish_run(running.id, ref)
            return ref
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(complete, refs))
    winners = [ref for ref in outcomes if ref is not None]
    assert len(winners) == 1
    assert repo.fetch_result(running.id).store_ref == winners[0]
    assert len(repo.unlinked_versions(running.id)) == 1
    assert repo.audit() == []


def test_payload_write_failure_records_failed_run(environment, monkeypatch):
    env, orm, repo = environment, api(), repository(environment)

    def broken_store(*args, **kwargs):
        raise OSError("payload storage unavailable")

    monkeypatch.setattr(env[1], "store", broken_store)
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run(ParamSum("mass"), sim(env[2]), sim_identity=env[4], store_name="results")
    run = repo.get_run(caught.value.run_id)
    assert run.status == "failed" and run.result_link is None
    assert run.error_type == "OSError" and "payload storage unavailable" in run.error_message
    assert env[1].list_refs() == []


def test_failure_recording_interruption_leaves_inspectable_run(environment):
    env, orm, repo = environment, api(), repository(environment)

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE pynbodyext_orm_runs"):
            raise RuntimeError("metadata unavailable")

    sa.event.listen(repo.engine, "before_cursor_execute", fail_update)
    try:
        with pytest.raises(orm.RunPersistenceError) as caught:
            repo.run(Explodes(), sim(env[2]), sim_identity=env[4], store_name="results")
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", fail_update)
    run_id = caught.value.run_id
    assert repo.get_run(run_id).status == "running"
    assert caught.value.result_ref is None
    assert repo.audit()[0].code == "unfinished"
    assert repo.fail_run(run_id, RuntimeError("confirmed failed execution")).status == "failed"
