"""Registered input loading joins real snapshots, ORM runs and immutable results."""
from __future__ import annotations

from dataclasses import replace

import pynbody
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.calculate import LogicalSnapshotIdentity
from pynbodyext.core.calculate import LambdaProperty
from pynbodyext.core.calculate.store import SQLAlchemyResultStore
from pynbodyext.properties import ParamSum


def snapshot(path, value=6):
    sim = pynbody.new(2)
    sim["mass"] = [float(value), 0.0]
    sim["pos"] = [[0., 0., 0.], [1., 0., 0.]]
    sim._filename = str(path)
    return sim


@pytest.fixture
def env(tmp_path):
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'payload.sqlite'}")
    repos = []
    url = f"sqlite:///{tmp_path / 'metadata.sqlite'}"
    def make(loaders=None):
        repo = orm.AnalysisRepository(url, stores={"results": store}, loaders=loaders)
        repos.append(repo)
        return repo
    yield make, LogicalSnapshotIdentity("project", "sim", "099"), tmp_path / "snap", store
    for repo in repos:
        repo.engine.dispose()
    store.engine.dispose()


def test_load_reopen_detached_metadata_and_existing_run(env):
    make, identity, path, _ = env
    repo = make({"default": snapshot})
    data = repo.register_input(identity, location=str(path))
    repo.engine.dispose()
    repo = make({"default": snapshot})
    found = repo.get_input(data.id)
    assert found.timestep.simulation.project_id == "project"
    assert found.parent is None and found.locations[0].uri == str(path)
    loaded = repo.load_input(data.id)
    assert isinstance(loaded, orm.LoadedInput)
    assert loaded.input.id == data.id and loaded.location.id == found.locations[0].id
    assert loaded.sim_identity(loaded.sim) == identity.signature()
    assert loaded.sim_identity(loaded.sim[[1]])[:7] == identity.signature()
    assert repo.runs() == []
    run = repo.run(ParamSum("mass"), loaded.sim, sim_identity=loaded.sim_identity, store_name="results")
    assert run.result.value == 6 and run.run.input_location_id is None
    with loaded.sim.translate([1., 0., 0.]):
        assert loaded.sim_identity(loaded.sim)[0] == "session"


def test_bindings_are_copied_and_no_implicit_default(env):
    make, identity, path, _ = env
    bindings = {"default": snapshot}
    repo = make(bindings)
    bindings.clear()
    data = repo.register_input(identity, location=str(path))
    assert len(repo.load_input(data.id).sim) == 2
    with pytest.raises(KeyError, match="loader"):
        make().load_input(data.id)


@pytest.mark.parametrize("bindings, message", [({"": snapshot}, "loader_id"), ({"default": 42}, "callable")])
def test_invalid_loader_bindings(env, bindings, message):
    with pytest.raises((TypeError, ValueError), match=message):
        env[0](bindings)


def test_locations_require_choice_and_ownership_without_io(env):
    make, identity, path, _ = env
    calls = []
    def loader(uri):
        calls.append(uri)
        return snapshot(uri)
    repo = make({"default": loader})
    data = repo.register_input(identity)
    with pytest.raises(ValueError, match="location"):
        repo.load_input(data.id)
    repo.register_input(identity, location=str(path))
    repo.register_input(identity, location=str(path) + '.alias')
    foreign = repo.register_input(replace(identity, timestep_id="100"), location=str(path))
    foreign_location = repo.get_input(foreign.id).locations[0].id
    for location_id in (None, foreign_location, 99999):
        with pytest.raises((ValueError, KeyError)):
            repo.run_input(ParamSum("mass"), data.id, store_name="results", location_id=location_id)
    assert calls == [] and repo.runs() == []
    location = repo.get_input(data.id).locations[0]
    loaded = repo.load_input(data.id, location_id=location.id)
    assert loaded.location.id == location.id and calls == [location.uri]


@pytest.mark.parametrize("uri", ["relative/snap", "https://example.invalid/snap", "file:///data/snap"])
def test_unsupported_locations_fail_preflight(env, uri):
    make, identity, _, _ = env
    repo = make({"default": lambda uri: pytest.fail("must not load")})
    data = repo.register_input(identity, location=uri)
    with pytest.raises(ValueError, match="local|absolute"):
        repo.run_input(ParamSum("mass"), data.id, store_name="results")
    assert repo.runs() == []


def test_prefix_need_not_exist_and_filename_can_be_normalized(env):
    make, identity, path, _ = env
    assert not path.exists()
    normalized = str(path) + ".0.hdf5"
    repo = make({"default": lambda uri: snapshot(normalized)})
    data = repo.register_input(identity, location=str(path))
    loaded = repo.load_input(data.id)
    assert loaded.sim.filename == normalized
    assert loaded.sim_identity(loaded.sim) == identity.signature()
    assert loaded.location.uri == str(path)


@pytest.mark.parametrize("problem", ["missing", "view", "hierarchy", "kind", "identity_key", "parent"])
def test_invalid_inputs_fail_before_loading(env, problem):
    make, identity, path, _ = env
    repo = make({"default": lambda uri: pytest.fail("must not load")})
    data = repo.register_input(identity, location=str(path))
    input_id = data.id
    if problem == "missing":
        input_id = 99999
    elif problem == "view":
        input_id = repo.register_input(identity.signature() + ("subset", "a" * 16)).id
    else:
        with repo.session() as session, session.begin():
            row = session.get(orm.DataObject, data.id)
            if problem == "hierarchy":
                row.timestep.simulation.logical_id = "different"
            elif problem == "parent":
                row.parent_id = row.id
            elif problem == "kind":
                row.kind = "halo"
            else:
                row.identity_key = "corrupt"
    with pytest.raises((KeyError, ValueError)):
        repo.run_input(ParamSum("mass"), input_id, store_name="results")
    assert repo.runs() == []


@pytest.mark.parametrize("problem", ["store", "loader", "recipe"])
def test_missing_binding_and_unstable_recipe_do_not_start_run(env, problem):
    make, identity, path, _ = env
    repo = make({"default": lambda uri: pytest.fail("must not load")} if problem != "loader" else {})
    data = repo.register_input(identity, location=str(path))
    calculator = LambdaProperty(lambda sim: 1) if problem == "recipe" else ParamSum("mass")
    with pytest.raises((KeyError, ValueError)):
        repo.run_input(calculator, data.id, store_name="absent" if problem == "store" else "results")
    assert repo.runs() == []


@pytest.mark.parametrize("problem", ["type", "view", "filename", "relative", "remote", "transformed"])
def test_invalid_loaded_root_fails_the_run(env, problem):
    make, identity, path, _ = env
    root = snapshot(path)
    if problem == "type":
        output = object()
    elif problem == "view":
        output = root[[0]]
    else:
        output = root
        if problem == "filename":
            root._filename = "<created>"
        elif problem == "relative":
            root._filename = "relative/snap"
        elif problem == "remote":
            root._filename = "s3://bucket/snap"
        elif problem == "transformed":
            root.translate([1., 0., 0.])
    repo = make({"default": lambda uri: output})
    data = repo.register_input(identity, location=str(path))
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run_input(ParamSum("mass"), data.id, store_name="results")
    run = repo.get_run(caught.value.run_id)
    assert run.status == "failed" and run.result_link is None
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))
    assert run.input_location.uri == str(path)


def test_running_is_committed_before_io_without_held_connection(env):
    make, identity, path, _ = env
    def loader(uri):
        assert repo.engine.pool.checkedout() == 0
        observer = make()
        running = observer.runs(status="running")
        assert len(running) == 1 and running[0].input_location.uri == str(path)
        return snapshot(uri)
    repo = make({"default": loader})
    data = repo.register_input(identity, location=str(path))
    recorded = repo.run_input(ParamSum("mass"), data.id, store_name="results")
    assert recorded.run.status == "succeeded"
    assert repo.fetch_result(recorded.run.id).store_ref == recorded.result.store_ref
    assert repo.runs()[0].input_location.id == recorded.run.input_location_id


def test_alias_cache_refresh_and_reopen_preserve_exact_versions(env):
    make, identity, path, store = env
    calls = []
    def loader(uri):
        calls.append(uri)
        return snapshot(uri, 6 * len(calls))
    repo = make({"default": loader})
    data = repo.register_input(identity, location=str(path))
    first = repo.run_input(ParamSum("mass"), data.id, store_name="results")
    repo.register_input(identity, location=str(path) + ".moved")
    moved = next(x for x in repo.get_input(data.id).locations if x.uri.endswith(".moved"))
    hit = repo.run_input(ParamSum("mass"), data.id, location_id=moved.id, store_name="results")
    fresh = repo.run_input(ParamSum("mass"), data.id, location_id=moved.id, store_name="results", reuse=False)
    assert len(calls) == 3
    assert first.result.value == hit.result.value == 6
    assert fresh.result.value == 18 and hit.run.reused and not fresh.run.reused
    assert first.result.store_ref == hit.result.store_ref != fresh.result.store_ref
    assert first.run.input_location_id != hit.run.input_location_id
    reopened = make()
    store.delete(fresh.result.store_ref)
    assert reopened.fetch_result(first.run.id).value == 6
    assert reopened.fetch_result(fresh.run.id).value == 18
    assert reopened.audit() == []


@pytest.mark.parametrize("field", ["revision", "loader_id"])
def test_revision_and_loader_isolation(env, field):
    make, identity, path, _ = env
    repo = make({"default": snapshot, "other": lambda uri: snapshot(uri, 12)})
    a = repo.register_input(identity, location=str(path))
    b = repo.register_input(replace(identity, **{field: "other"}), location=str(path))
    first = repo.run_input(ParamSum("mass"), a.id, store_name="results")
    second = repo.run_input(ParamSum("mass"), b.id, store_name="results")
    assert first.result.store_ref.cache_key != second.result.store_ref.cache_key
    assert not second.run.reused
    assert second.result.value == (12 if field == "loader_id" else 6)


def test_loader_failure_and_failure_write_recovery(env):
    make, identity, path, _ = env
    def broken(uri):
        raise OSError("snapshot unavailable")
    repo = make({"default": broken})
    data = repo.register_input(identity, location=str(path))
    with pytest.raises(OSError):
        repo.load_input(data.id)
    assert repo.runs() == []
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run_input(ParamSum("mass"), data.id, store_name="results")
    assert isinstance(caught.value.__cause__, OSError)
    assert repo.get_run(caught.value.run_id).error_message == "snapshot unavailable"
    def block_status(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE pynbodyext_orm_runs"):
            raise RuntimeError("metadata unavailable")
    sa.event.listen(repo.engine, "before_cursor_execute", block_status)
    try:
        with pytest.raises(orm.RunPersistenceError) as caught:
            repo.run_input(ParamSum("mass"), data.id, store_name="results")
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", block_status)
    assert caught.value.result_ref is None
    pending = repo.get_run(caught.value.run_id)
    assert pending.status == "running" and pending.input_location.uri == str(path)
    assert repo.fail_run(pending.id, OSError("snapshot unavailable")).status == "failed"


def test_final_metadata_failure_preserves_location_and_explicit_receipt(env):
    make, identity, path, _ = env
    repo = make({"default": snapshot})
    data = repo.register_input(identity, location=str(path))
    def block_link(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_result_links"):
            raise RuntimeError("metadata unavailable")
    sa.event.listen(repo.engine, "before_cursor_execute", block_link)
    try:
        with pytest.raises(orm.RunPersistenceError) as caught:
            repo.run_input(ParamSum("mass"), data.id, store_name="results")
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", block_link)
    err = caught.value
    pending = repo.get_run(err.run_id)
    assert pending.status == "running" and pending.input_location.uri == str(path)
    assert err.result.value == 6
    assert repo.finish_run(err.run_id, err.result_ref).status == "succeeded"
    assert repo.fetch_result(err.run_id).value == 6 and repo.audit() == []


def test_begin_run_validates_location_ownership(env):
    make, identity, path, _ = env
    repo = make()
    a = repo.register_input(identity, location=str(path))
    b = repo.register_input(replace(identity, timestep_id="other"), location=str(path))
    location_id = repo.get_input(b.id).locations[0].id
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    with pytest.raises(ValueError, match="location"):
        repo.begin_run(a.id, recipe.hash, store_name="results", input_location_id=location_id)
    assert repo.runs() == []


@pytest.mark.data
def test_real_pynbody_loading_after_database_reopen(env, _test_data):
    make, identity, _, _ = env
    path = _test_data / "gadget3/data/snapshot_103/snap_103.hdf5"
    repo = make()
    data = repo.register_input(identity, location=str(path.resolve()))
    repo.engine.dispose()
    reopened = make({"default": pynbody.load})
    recorded = reopened.run_input(ParamSum("mass"), data.id, store_name="results")
    assert recorded.run.status == "succeeded"
    assert float(recorded.result.value) > 0
    assert make().fetch_result(recorded.run.id).store_ref == recorded.result.store_ref


def test_selected_location_failure_does_not_try_alias(env):
    make, identity, path, _ = env
    calls = []
    def loader(uri):
        calls.append(uri)
        raise OSError("selected copy unavailable")
    repo = make({"default": loader})
    data = repo.register_input(identity, location=str(path))
    repo.register_input(identity, location=str(path) + ".alias")
    selected = repo.get_input(data.id).locations[0]
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run_input(ParamSum("mass"), data.id, store_name="results", location_id=selected.id)
    assert calls == [selected.uri]
    assert repo.get_run(caught.value.run_id).input_location_id == selected.id


def test_begin_commit_failure_prevents_loader_io(env):
    make, identity, path, _ = env
    repo = make({"default": lambda uri: pytest.fail("must not load")})
    data = repo.register_input(identity, location=str(path))
    def block_begin(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_runs"):
            raise RuntimeError("metadata unavailable")
    sa.event.listen(repo.engine, "before_cursor_execute", block_begin)
    try:
        with pytest.raises(RuntimeError, match="metadata unavailable"):
            repo.run_input(ParamSum("mass"), data.id, store_name="results")
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", block_begin)
    assert repo.runs() == []


def test_payload_write_failure_marks_loaded_run_failed(env, monkeypatch):
    make, identity, path, store = env
    repo = make({"default": snapshot})
    data = repo.register_input(identity, location=str(path))
    def fail_put(record):
        raise OSError("payload unavailable")
    monkeypatch.setattr(store, "_put", fail_put)
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run_input(ParamSum("mass"), data.id, store_name="results")
    assert isinstance(caught.value.__cause__, OSError)
    failed = repo.get_run(caught.value.run_id)
    assert failed.status == "failed" and failed.input_location.uri == str(path)
    assert failed.result_link is None


def test_collected_calculation_errors_keep_loaded_run_failed(env):
    from pynbodyext.core.calculate import RunOptions
    make, identity, path, store = env
    repo = make({"default": snapshot})
    data = repo.register_input(identity, location=str(path))
    recorded = repo.run_input(
        ParamSum("absent_field"), data.id, store_name="results", options=RunOptions(errors="collect")
    )
    assert recorded.result.errors and recorded.run.status == "failed"
    assert recorded.run.input_location.uri == str(path)
    assert recorded.run.result_link is None and store.list_refs() == []


def test_corrupted_run_location_ownership_is_reported(env):
    make, identity, path, _ = env
    repo = make({"default": snapshot})
    data = repo.register_input(identity, location=str(path))
    other = repo.register_input(replace(identity, timestep_id="other"), location=str(path))
    foreign_location = repo.get_input(other.id).locations[0].id
    recorded = repo.run_input(ParamSum("mass"), data.id, store_name="results")
    with repo.session() as session, session.begin():
        session.get(orm.Run, recorded.run.id).input_location_id = foreign_location
    assert repo.audit()[0].code == "invalid_link"
    with pytest.raises(ValueError, match="location"):
        repo.fetch_result(recorded.run.id)


def test_context_snapshots_selected_prefix_before_loader_io(env):
    make, identity, path, _ = env
    identity = replace(identity, revision="r2", loader_id="custom")
    metadata = {"tool_versions": {"script": "v1"}, "random_seed": 7}
    observed = []

    def loader(uri):
        assert repo.engine.pool.checkedout() == 0
        pending = make().runs(status="running")[0]
        context = pending.execution_context
        assert context["input"] == {"uri": uri, "loader_id": "custom", "revision": "r2"}
        assert context["tool_versions"] == {"script": "v1"}
        assert context["random_seed"] == 7
        observed.append(context)
        metadata["tool_versions"]["script"] = "mutated during loading"
        return snapshot(uri + ".0.hdf5")

    repo = make({"custom": loader})
    data = repo.register_input(identity, location=str(path))
    repo.register_input(identity, location=str(path) + ".alias")
    location = next(item for item in repo.get_input(data.id).locations if item.uri.endswith(".alias"))
    recorded = repo.run_input(
        ParamSum("mass"), data.id, location_id=location.id, store_name="results", execution_metadata=metadata
    )
    assert recorded.run.execution_context == observed[0]
    assert recorded.run.execution_context["input"]["uri"] == location.uri
    with repo.session() as session, session.begin():
        session.get(orm.DataLocation, location.id).uri = str(path) + ".moved"
    repo.engine.dispose()
    reopened = make().get_run(recorded.run.id)
    assert reopened.input_location.uri.endswith(".moved")
    assert reopened.execution_context == observed[0]


@pytest.mark.parametrize("failure", ["loader", "returned_root", "failure_metadata", "result_metadata"])
def test_load_failures_and_recovery_retain_context(env, failure):
    make, identity, path, _ = env

    def loader(uri):
        if failure in ("loader", "failure_metadata"):
            raise OSError("snapshot unavailable")
        return object() if failure == "returned_root" else snapshot(uri)

    repo = make({"default": loader})
    data = repo.register_input(identity, location=str(path))

    def block_metadata(conn, cursor, statement, parameters, context, executemany):
        prefix = "UPDATE pynbodyext_orm_runs" if failure == "failure_metadata" else "INSERT INTO pynbodyext_orm_result_links"
        if statement.startswith(prefix):
            raise RuntimeError("metadata unavailable")

    if failure.endswith("metadata"):
        sa.event.listen(repo.engine, "before_cursor_execute", block_metadata)
    try:
        error = orm.RunPersistenceError if failure.endswith("metadata") else orm.RunExecutionError
        with pytest.raises(error) as caught:
            repo.run_input(
                ParamSum("mass"), data.id, store_name="results",
                execution_metadata={"tool_versions": {"script": "v1"}, "random_seed": 5},
            )
    finally:
        if failure.endswith("metadata"):
            sa.event.remove(repo.engine, "before_cursor_execute", block_metadata)
    run = repo.get_run(caught.value.run_id)
    original = run.execution_context
    assert original["input"] == {"uri": str(path), "loader_id": "default", "revision": None}
    assert original["tool_versions"] == {"script": "v1"} and original["random_seed"] == 5
    repo.engine.dispose()
    reopened = make()
    assert reopened.get_run(run.id).execution_context == original
    if failure == "result_metadata":
        assert run.status == "running"
        assert reopened.finish_run(run.id, caught.value.result_ref).execution_context == original
    elif failure == "failure_metadata":
        assert run.status == "running"
        assert reopened.fail_run(run.id, OSError("confirmed unavailable")).execution_context == original
    else:
        assert run.status == "failed"
        assert isinstance(caught.value.__cause__, (OSError, TypeError))


def test_context_collection_failure_is_preflight(env, monkeypatch):
    import importlib.metadata

    make, identity, path, store = env
    repo = make({"default": lambda uri: pytest.fail("must not load")})
    data = repo.register_input(identity, location=str(path))

    def broken_version(name):
        raise RuntimeError("metadata collector unavailable")

    monkeypatch.setattr(importlib.metadata, "version", broken_version)
    with pytest.raises(RuntimeError, match="metadata collector unavailable"):
        repo.run_input(ParamSum("mass"), data.id, store_name="results")
    assert repo.runs() == [] and store.list_refs() == []
