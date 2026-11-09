"""Explicit selection definitions reproduce real pynbody views and tracked runs."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pynbody
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.calculate import LogicalSnapshotIdentity, PropertyBase
from pynbodyext.core.calculate import FilterBase, RunOptions
from pynbodyext.core.calculate.store import RecordRef, SQLAlchemyResultStore
from pynbodyext.filters import Sphere
from pynbodyext.properties import ParamSum


def snapshot(uri, count=4, edited=False):
    sim = pynbody.new(star=2, dm=count - 2)
    sim["mass"] = np.arange(1, count + 1, dtype=float) * (100 if edited else 10)
    sim["pos"] = np.column_stack([np.arange(count), np.zeros((count, 2))])
    sim._filename = str(uri)
    return sim


@FilterBase.dataclass
class CountingFilter(FilterBase):
    calls = 0
    def calculate(self, sim, params=None):
        type(self).calls += 1
        return np.arange(len(sim)) % 2 == 0


@FilterBase.dataclass
class BadMask(FilterBase):
    mode: str = "short"
    def calculate(self, sim, params=None):
        if self.mode == "short":
            return np.ones(len(sim) - 1, dtype=bool)
        return np.ones((len(sim), 1), dtype=bool)


@PropertyBase.dataclass
class Explodes(PropertyBase):
    def calculate(self, sim, params=None):
        raise RuntimeError("analysis failure")


@pytest.fixture
def env(tmp_path):
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'payload.sqlite'}")
    repos = []
    identity = LogicalSnapshotIdentity("project", "sim", "099", revision="r1")
    path = tmp_path / "snap"
    def make(loader=snapshot):
        repo = orm.AnalysisRepository(
            f"sqlite:///{tmp_path / 'metadata.sqlite'}", stores={"results": store}, loaders={"default": loader}
        )
        repos.append(repo)
        return repo
    repo = make()
    root = repo.register_input(identity, location=str(path))
    yield make, repo, repo.load_input(root.id), store
    for repo in repos:
        repo.engine.dispose()
    store.engine.dispose()


def selection(env, indices=None):
    return env[1].register_selection(env[2], orm.SelectionSpec.indices([1, 0, 1, 0] if indices is None else indices))


def row_counts(repo):
    with repo.session() as session:
        return tuple(session.scalar(sa.select(sa.func.count()).select_from(cls))
                     for cls in (orm.DataObject, orm.Selection, orm.Run))


@pytest.mark.parametrize("indices", [[3, 1, 3, 0], [], [0, 1, 2, 3]])
def test_spec_indices_copy_and_roundtrip(indices):
    original = list(indices)
    spec = orm.SelectionSpec.indices(indices)
    indices[:] = [99]
    assert json.loads(spec.to_json())["indices"] == original
    assert orm.SelectionSpec.from_json(spec.to_json()).to_json() == spec.to_json()


@pytest.mark.parametrize("bad", [[True], [1.0], [np.int64(1)], [-1], [[1]], ["1"], (1,), np.array([1]), np.array([True])])
def test_spec_rejects_noninteger_indices(bad):
    with pytest.raises((TypeError, ValueError)):
        orm.SelectionSpec.indices(bad)


@pytest.mark.parametrize("bad", [[], {}, {"schema": "unknown", "kind": "indices", "indices": []},
    {"schema": "pynbodyext.orm.selection/v1", "kind": "indices", "indices": [], "extra": 1},
    {"schema": "pynbodyext.orm.selection/v1", "kind": "bad"},
    {"schema": "pynbodyext.orm.selection/v1", "kind": "family", "family": ""}])
def test_spec_rejects_malformed_json(bad):
    with pytest.raises((TypeError, ValueError)):
        orm.SelectionSpec.from_json(json.dumps(bad))


def test_spec_rejects_duplicate_json_keys():
    with pytest.raises(ValueError):
        orm.SelectionSpec.from_json('{"schema":"x","schema":"pynbodyext.orm.selection/v1","kind":"indices","indices":[]}')


@pytest.mark.parametrize("bad", [lambda sim: True, ParamSum("mass"), Sphere(lambda sim: 1)])
def test_spec_requires_stable_constructible_filter(bad):
    with pytest.raises((TypeError, ValueError)):
        orm.SelectionSpec.filter(bad)


@pytest.mark.parametrize("indices", [[1, 0, 1, 3, 2, 3], [], [0, 1, 2, 3], [2, 2]])
def test_reopen_replays_order_and_typed_identity(env, indices):
    make, repo, root, _ = env
    chosen = selection(env, indices)
    assert chosen.expected_count == 4 and chosen.root_input_id == root.input.id
    assert chosen.view_input.signature == root.sim_identity(root.sim[np.array(indices, dtype=np.int64)])
    reopened = make()
    found = reopened.get_selection(chosen.id)
    assert sa.inspect(found).detached
    assert found.root_input.timestep.simulation.project_id == "project"
    assert found.view_input.parent_id == root.input.id
    loaded = reopened.load_selection(chosen.id)
    assert loaded.input.id == chosen.view_input_id
    assert loaded.location.id == root.location.id
    assert loaded.sim.get_index_list(loaded.sim.ancestor).tolist() == indices
    assert loaded.sim_identity(loaded.sim) == chosen.view_input.signature
    assert reopened.runs() == []
    with pytest.raises(ValueError, match="root"):
        reopened.load_input(chosen.view_input_id)


def test_definitions_share_view_without_overwriting(env):
    make, repo, root, _ = env
    star_indices = root.sim.star.get_index_list(root.sim).tolist()
    a = selection(env, star_indices)
    b = repo.register_selection(root, orm.SelectionSpec.family("star"))
    assert a.id != b.id and a.descriptor_key != b.descriptor_key
    assert a.view_input_id == b.view_input_id
    assert selection(env, star_indices).id == a.id
    assert json.loads(make().get_selection(a.id).definition_json)["kind"] == "indices"
    assert json.loads(make().get_selection(b.id).definition_json)["kind"] == "family"
    for chosen in (a, b):
        loaded = repo.load_selection(chosen.id)
        assert loaded.sim.get_index_list(loaded.sim.ancestor).size == len(star_indices)


def test_filter_materialized_once_and_replay_uses_signature(env):
    _, repo, root, _ = env
    CountingFilter.calls = 0
    chosen = repo.register_selection(root, orm.SelectionSpec.filter(CountingFilter()))
    assert CountingFilter.calls == 1
    assert chosen.expected_count == 4
    loaded = repo.load_selection(chosen.id)
    assert loaded.sim.get_index_list(loaded.sim.ancestor).tolist() == [0, 2]
    assert CountingFilter.calls == 2
    sphere = repo.register_selection(root, orm.SelectionSpec.filter(Sphere(1.5)))
    loaded = repo.load_selection(sphere.id)
    assert loaded.sim.get_index_list(loaded.sim.ancestor).tolist() == [0, 1]


@pytest.mark.parametrize("mode", ["short", "2d"])
def test_wrong_filter_mask_is_explicit_and_atomic(env, mode):
    _, repo, root, _ = env
    with pytest.raises(ValueError, match="mask"):
        repo.register_selection(root, orm.SelectionSpec.filter(BadMask(mode)))
    assert row_counts(repo) == (1, 0, 0)


@pytest.mark.parametrize("problem", ["bounds", "family", "view", "metadata", "provider", "location", "transformed"])
def test_registration_rejects_invalid_root_or_definition(env, problem):
    _, repo, root, _ = env
    spec = orm.SelectionSpec.indices([0])
    if problem == "bounds":
        spec = orm.SelectionSpec.indices([4])
    elif problem == "family":
        spec = orm.SelectionSpec.family("gas")
    elif problem == "view":
        root = replace(root, sim=root.sim[[0]])
    elif problem == "metadata":
        other = repo.register_input(LogicalSnapshotIdentity("project", "sim", "100"))
        root = replace(root, input=other)
    elif problem == "provider":
        from pynbodyext.calculate import ConfiguredSnapshotIdentity
        root = replace(root, sim_identity=ConfiguredSnapshotIdentity({root.sim.filename: LogicalSnapshotIdentity("wrong", "sim", "099")}))
    elif problem == "location":
        other = repo.register_input(LogicalSnapshotIdentity("project", "sim", "100"), location=root.location.uri)
        root = replace(root, location=repo.get_input(other.id).locations[0])
    if problem == "transformed":
        with root.sim.translate([1., 0., 0.]):
            with pytest.raises(ValueError):
                repo.register_selection(root, spec)
    else:
        with pytest.raises((ValueError, TypeError)):
            repo.register_selection(root, spec)
    assert row_counts(repo)[1:] == (0, 0)


def test_selection_insert_rolls_back_view(env):
    _, repo, root, _ = env
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_selections"):
            raise RuntimeError("injected selection insert")
    sa.event.listen(repo.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            selection(env)
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", fail)
    assert row_counts(repo) == (1, 0, 0)
    assert selection(env).id


def test_running_commit_view_identity_root_location_and_context(env):
    make, repo, root, _ = env
    chosen = selection(env)
    observed = []
    def loader(uri):
        running = make().runs(status="running")
        assert len(running) == 1
        run = running[0]
        assert run.input_id == chosen.view_input_id and run.selection_id == chosen.id
        assert run.selection.root_input.id == root.input.id
        assert run.input_location.data_object_id == root.input.id
        assert run.execution_context["input"] == {"uri": uri, "loader_id": "default", "revision": "r1"}
        assert repo.engine.pool.checkedout() == 0
        observed.append(uri)
        return snapshot(uri)
    active = make(loader)
    recorded = active.run_selection(ParamSum("mass"), chosen.id, store_name="results", execution_metadata={"random_seed": 7})
    assert observed == [root.location.uri]
    assert recorded.result.value == 60
    assert recorded.result.store_ref.sim_signature == chosen.view_input.signature
    assert recorded.run.execution_context["random_seed"] == 7
    assert make().fetch_result(recorded.run.id).value == 60
    assert active.audit() == []


def test_alias_cache_hit_and_refresh_keep_definitions(env):
    make, repo, root, _ = env
    indices = root.sim.star.get_index_list(root.sim).tolist()
    a = selection(env, indices)
    b = repo.register_selection(root, orm.SelectionSpec.family("star"))
    repo.register_input(root.input.signature, location=root.location.uri + ".alias")
    first = repo.run_selection(ParamSum("mass"), a.id, store_name="results", location_id=root.location.id)
    alias = repo.get_input(root.input.id).locations[-1]
    hit = repo.run_selection(ParamSum("mass"), b.id, store_name="results", location_id=alias.id)
    fresh = repo.run_selection(ParamSum("mass"), a.id, store_name="results", location_id=alias.id, reuse=False)
    assert hit.run.selection_id == b.id and first.run.selection_id == a.id
    assert hit.result.from_store and hit.result.store_ref == first.result.store_ref
    assert fresh.result.store_ref != first.result.store_ref
    assert make().fetch_result(first.run.id).store_ref == first.result.store_ref


@pytest.mark.parametrize("problem", ["loader", "count", "digest", "family", "root"])
def test_replay_failure_records_failed_run_before_analysis(env, problem):
    make, repo, root, store = env
    chosen = repo.register_selection(root, orm.SelectionSpec.family("star") if problem == "family" else orm.SelectionSpec.filter(Sphere(1.5)))
    def loader(uri):
        if problem == "loader":
            raise OSError("unavailable")
        if problem == "root":
            return snapshot(uri)[[0]]
        sim = snapshot(uri, count=5 if problem == "count" else 4)
        if problem == "digest":
            sim["pos"] *= 100
        if problem == "family":
            sim = pynbody.new(dm=4)
            sim._filename = uri
        return sim
    active = make(loader)
    with pytest.raises(orm.RunExecutionError) as caught:
        active.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    run = active.get_run(caught.value.run_id)
    assert caught.value.__cause__ is not None
    assert run.status == "failed" and run.selection_id == chosen.id
    assert run.input_location_id == root.location.id and run.result_link is None
    assert store.list_refs() == []


def test_preflight_rejects_metadata_and_locations_without_loading(env):
    make, repo, root, _ = env
    chosen = selection(env)
    active = make(lambda uri: pytest.fail("preflight must not load"))
    other = repo.register_input(LogicalSnapshotIdentity("project", "sim", "100"), location=root.location.uri)
    for kwargs in ({"location_id": repo.get_input(other.id).locations[0].id}, {"execution_metadata": {"random_seed": True}}):
        with pytest.raises(ValueError):
            active.run_selection(ParamSum("mass"), chosen.id, store_name="results", **kwargs)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    with pytest.raises(ValueError, match="location"):
        repo.begin_run(chosen.view_input_id, recipe.hash, store_name="results", input_location_id=root.location.id)
    assert repo.runs() == []


def test_finalization_receipt_recovery_revalidates_selection(env):
    make, repo, root, _ = env
    chosen = selection(env)
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_result_links"):
            raise RuntimeError("injected link failure")
    sa.event.listen(repo.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(orm.RunPersistenceError) as caught:
            repo.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    finally:
        sa.event.remove(repo.engine, "before_cursor_execute", fail)
    err = caught.value
    pending = make().get_run(err.run_id)
    assert pending.status == "running" and pending.selection_id == chosen.id
    assert pending.input_location_id == root.location.id
    assert err.result.value == 60 and err.result_ref is not None
    assert repo.unlinked_versions(pending.id) == [err.result_ref]
    with pytest.raises(ValueError):
        repo.finish_run(pending.id, RecordRef(root.input.signature, pending.recipe_hash, err.result_ref.version_id))
    repo.finish_run(pending.id, err.result_ref)
    assert repo.finish_run(pending.id, err.result_ref).status == "succeeded"
    assert repo.fetch_result(pending.id).store_ref == err.result_ref and repo.audit() == []
    assert repo.get_run(pending.id).execution_context == pending.execution_context


@pytest.mark.parametrize("problem", ["definition", "count", "schema", "key", "root", "view", "revision", "location", "run_selection"])
def test_corruption_rejects_fetch_finish_candidates_and_audit(env, problem):
    _, repo, root, _ = env
    chosen = selection(env)
    other = repo.register_input(LogicalSnapshotIdentity("project", "sim", "100"), location=root.location.uri)
    other_selection = selection(env, [0])
    recorded = repo.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    with repo.session() as session, session.begin():
        row = session.get(orm.Selection, chosen.id)
        if problem == "definition":
            row.definition_json = orm.SelectionSpec.indices([0]).to_json()
        elif problem == "count":
            row.expected_count = 5
        elif problem == "schema":
            row.schema = "unknown"
        elif problem == "key":
            row.descriptor_key = "wrong"
        elif problem == "root":
            row.root_input_id = other.id
        elif problem == "view":
            row.view_input_id = other_selection.view_input_id
        elif problem == "revision":
            data = session.get(orm.DataObject, root.input.id)
            signature = list(data.signature)
            signature[5] = "changed"
            data.identity_json = json.dumps(signature)
        elif problem == "location":
            session.get(orm.Run, recorded.run.id).input_location_id = repo.get_input(other.id).locations[0].id
        else:
            session.get(orm.Run, recorded.run.id).selection_id = other_selection.id
    for operation in (lambda: repo.fetch_result(recorded.run.id),
                      lambda: repo.finish_run(recorded.run.id, recorded.result.store_ref),
                      lambda: repo.unlinked_versions(recorded.run.id)):
        with pytest.raises((TypeError, ValueError)):
            operation()
    assert repo.audit()[0].code == "invalid_link"


def test_same_count_edits_need_refresh_or_new_revision(env):
    make, repo, root, _ = env
    chosen = selection(env)
    first = repo.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    changed = make(lambda uri: snapshot(uri, edited=True))
    assert changed.load_selection(chosen.id).sim_identity(changed.load_selection(chosen.id).sim) == chosen.view_input.signature
    hit = changed.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    fresh = changed.run_selection(ParamSum("mass"), chosen.id, store_name="results", reuse=False)
    assert hit.result.store_ref == first.result.store_ref and hit.result.value == 60
    assert fresh.result.value == 600 and fresh.result.store_ref != first.result.store_ref
    new = repo.register_input(replace(LogicalSnapshotIdentity(*root.input.signature[2:7]), revision="r2"), location=root.location.uri)
    revised = repo.register_selection(repo.load_input(new.id), orm.SelectionSpec.indices([1, 0, 1, 0]))
    assert revised.id != chosen.id and revised.view_input.signature != chosen.view_input.signature


@pytest.mark.data
def test_real_file_selection_replays_after_reopen(env, _test_data):
    make, repo, _, _ = env
    path = (_test_data / "gadget3/data/snapshot_103/snap_103.hdf5").resolve()
    root = repo.register_input(LogicalSnapshotIdentity("project", "data", "103"), location=str(path))
    active = make(pynbody.load)
    loaded = active.load_input(root.id)
    chosen = active.register_selection(loaded, orm.SelectionSpec.indices([3, 1, 3, 0]))
    replayed = make(pynbody.load).load_selection(chosen.id)
    assert replayed.sim.get_index_list(replayed.sim.ancestor).tolist() == [3, 1, 3, 0]
    assert replayed.sim_identity(replayed.sim) == chosen.view_input.signature
    assert float(ParamSum("mass").run(replayed.sim).value) == pytest.approx(float(loaded.sim["mass"][[3, 1, 3, 0]].sum()))


@FilterBase.dataclass
class ArrayFilter(FilterBase):
    values: object
    def calculate(self, sim, params=None):
        return self.values


@FilterBase.dataclass
class MutatingFilter(FilterBase):
    active = None
    def calculate(self, sim, params=None):
        type(self).active = sim.translate([1., 0., 0.])
        return np.ones(len(sim), dtype=bool)


def test_nonconstructible_filter_is_rejected():
    calc = ArrayFilter(np.ones(1024, dtype=bool))
    assert calc.to_signature().stable and not calc.to_signature().constructible
    with pytest.raises(ValueError, match="constructible"):
        orm.SelectionSpec.filter(calc)


def test_scoped_transform_filter_is_rejected_before_execution(env):
    from pynbodyext.transforms import ShiftPosTo
    _, repo, root, _ = env
    calc = Sphere(1).transform(ShiftPosTo(np.zeros(3)))
    with pytest.raises(ValueError, match="original coordinates"):
        repo.register_selection(root, orm.SelectionSpec.filter(calc))
    assert row_counts(repo) == (1, 0, 0)
    assert root.sim_identity(root.sim) == root.input.signature


def test_filter_external_transform_is_detected_after_materialization(env):
    _, repo, root, _ = env
    try:
        with pytest.raises(ValueError, match="transformation"):
            repo.register_selection(root, orm.SelectionSpec.filter(MutatingFilter()))
        assert row_counts(repo) == (1, 0, 0)
    finally:
        if MutatingFilter.active is not None:
            MutatingFilter.active.revert()
            MutatingFilter.active = None


@pytest.mark.parametrize("problem", ["extra", "missing", "flag", "unknown"])
def test_filter_json_envelope_is_strict(problem):
    spec = json.loads(orm.SelectionSpec.filter(Sphere(1)).to_json())
    signature = spec["calculator_signature"]
    if problem == "extra":
        signature["other"] = 1
    elif problem == "missing":
        del signature["schema"]
    elif problem == "flag":
        signature["stable"] = "yes"
    else:
        signature["schema"] = "unknown"
    with pytest.raises((TypeError, ValueError)):
        orm.SelectionSpec.from_json(json.dumps(spec))


def test_begin_failure_prevents_loading_and_selection(env):
    make, repo, _, _ = env
    chosen = selection(env)
    active = make(lambda uri: pytest.fail("begin failure must not load"))
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_runs"):
            raise RuntimeError("begin unavailable")
    sa.event.listen(active.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="begin unavailable"):
            active.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    finally:
        sa.event.remove(active.engine, "before_cursor_execute", fail)
    assert repo.runs() == []


@pytest.mark.parametrize("problem", ["raised", "collect", "payload", "failure_write"])
def test_analysis_failure_and_failure_write_recovery(env, monkeypatch, problem):
    _, repo, root, store = env
    chosen = selection(env)
    if problem == "payload":
        def broken(*args, **kwargs):
            raise OSError("payload unavailable")
        monkeypatch.setattr(store, "store", broken)
    def fail(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE pynbodyext_orm_runs"):
            raise RuntimeError("status unavailable")
    if problem == "failure_write":
        sa.event.listen(repo.engine, "before_cursor_execute", fail)
    try:
        if problem == "collect":
            recorded = repo.run_selection(Explodes(), chosen.id, store_name="results", options=RunOptions(errors="collect"))
            assert recorded.run.status == "failed" and recorded.result.errors
            run_id = recorded.run.id
        else:
            error = orm.RunPersistenceError if problem == "failure_write" else orm.RunExecutionError
            with pytest.raises(error) as caught:
                repo.run_selection(ParamSum("mass") if problem == "payload" else Explodes(), chosen.id, store_name="results")
            run_id = caught.value.run_id
            assert caught.value.__cause__ is not None
    finally:
        if problem == "failure_write":
            sa.event.remove(repo.engine, "before_cursor_execute", fail)
    run = repo.get_run(run_id)
    assert run.selection_id == chosen.id and run.input_location_id == root.location.id
    assert run.result_link is None
    if problem == "failure_write":
        assert run.status == "running"
        assert repo.fail_run(run.id, RuntimeError("analysis failure")).status == "failed"
    else:
        assert run.status == "failed"


def test_selection_recovery_checks_corrupt_descriptor_and_root_location(env):
    _, repo, root, _ = env
    chosen = selection(env)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    pending = repo.begin_run(chosen.view_input_id, recipe.hash, store_name="results",
                             selection_id=chosen.id, input_location_id=root.location.id)
    loaded = repo.load_selection(chosen.id)
    result = ParamSum("mass").run(loaded.sim, sim_identity=loaded.sim_identity, store=env[3])
    with repo.session() as session, session.begin():
        session.get(orm.Selection, chosen.id).definition_json = orm.SelectionSpec.indices([0]).to_json()
    with pytest.raises(ValueError):
        repo.finish_run(pending.id, result.store_ref)
    assert repo.get_run(pending.id).status == "running"


def test_selection_audit_never_loads_or_decodes(env, monkeypatch):
    make, repo, _, store = env
    chosen = selection(env)
    repo.run_selection(ParamSum("mass"), chosen.id, store_name="results")
    monkeypatch.setattr(store.codec, "decode", lambda value: pytest.fail("audit must not decode"))
    assert make(lambda uri: pytest.fail("audit must not load")).audit() == []


@pytest.mark.parametrize("indices", [[3, 1, 3, 0], [3, 1]])
def test_cross_family_reordering_is_explicitly_rejected_without_writes(env, indices):
    _, repo, root, _ = env
    with pytest.raises(ValueError, match="root family-block order"):
        repo.register_selection(root, orm.SelectionSpec.indices(indices))
    assert row_counts(repo) == (1, 0, 0)


def test_changed_family_layout_fails_selection_run_before_analysis(env):
    make, repo, root, store = env
    chosen = selection(env, [1, 0, 1, 3, 2, 3])
    def changed(uri):
        sim = pynbody.new(dm=1, star=3)
        sim["mass"] = [10., 20., 30., 40.]
        sim._filename = uri
        return sim
    active = make(changed)
    with pytest.raises(orm.RunExecutionError) as caught:
        active.run_selection(Explodes(), chosen.id, store_name="results")
    assert isinstance(caught.value.__cause__, ValueError)
    assert "root family-block order" in str(caught.value.__cause__)
    run = active.get_run(caught.value.run_id)
    assert run.status == "failed" and run.selection_id == chosen.id
    assert run.result_link is None and store.list_refs() == []


@FilterBase.dataclass
class MaskFromRootProperty(FilterBase):
    def calculate(self, sim, params=None):
        return np.ones(len(sim) + int(sim.properties.get("wrong_mask", False)), dtype=bool)


def test_wrong_mask_after_run_creation_keeps_failure_and_no_analysis(env):
    make, repo, root, store = env
    chosen = repo.register_selection(root, orm.SelectionSpec.filter(MaskFromRootProperty()))
    def loader(uri):
        sim = snapshot(uri)
        sim.properties["wrong_mask"] = True
        return sim
    active = make(loader)
    with pytest.raises(ValueError, match="mask"):
        active.load_selection(chosen.id)
    assert active.runs() == []
    with pytest.raises(orm.RunExecutionError) as caught:
        active.run_selection(Explodes(), chosen.id, store_name="results")
    assert isinstance(caught.value.__cause__, ValueError) and "mask" in str(caught.value.__cause__)
    assert active.get_run(caught.value.run_id).status == "failed" and store.list_refs() == []


@pytest.mark.parametrize("problem", ["missing_selection", "missing_binding", "missing_store", "unstable_recipe", "aliases", "descriptor"])
def test_selection_metadata_preflight_does_no_io(env, problem):
    make, repo, root, _ = env
    chosen = selection(env)
    active = make(lambda uri: pytest.fail("preflight must not load"))
    selection_id, store_name, calc = chosen.id, "results", ParamSum("mass")
    if problem == "missing_selection":
        selection_id = 99999
    elif problem == "missing_binding":
        active = orm.AnalysisRepository(repo.engine, stores={"results": env[3]})
    elif problem == "missing_store":
        store_name = "missing"
    elif problem == "unstable_recipe":
        from pynbodyext.core.calculate import LambdaProperty
        calc = LambdaProperty(lambda sim: 1)
    elif problem == "aliases":
        repo.register_input(root.input.signature, location=root.location.uri + ".alias")
    else:
        with repo.session() as session, session.begin():
            session.get(orm.Selection, chosen.id).descriptor_key = "corrupt"
    with pytest.raises((KeyError, TypeError, ValueError)):
        active.run_selection(calc, selection_id, store_name=store_name)
    assert repo.runs() == []


@pytest.mark.parametrize("problem", ["unique", "root_fk", "view_fk", "count"])
def test_selection_database_constraints(env, problem):
    _, repo, _, _ = env
    chosen = selection(env)
    values = {name: getattr(chosen, name) for name in ("root_input_id", "view_input_id", "schema", "definition_json", "expected_count", "descriptor_key")}
    if problem != "unique":
        values["descriptor_key"] = "other"
    if problem == "root_fk":
        values["root_input_id"] = 99999
    elif problem == "view_fk":
        values["view_input_id"] = 99999
    elif problem == "count":
        values["expected_count"] = -1
    with pytest.raises(sa.exc.IntegrityError):
        with repo.session() as session, session.begin():
            session.add(orm.Selection(**values))
    assert row_counts(repo) == (2, 1, 0)


def test_malformed_filter_scope_is_an_explicit_definition_error():
    data = json.loads(orm.SelectionSpec.filter(Sphere(1)).to_json())
    data["calculator_signature"]["payload"] = {"node": "scoped", "scope": None, "base": {}}
    with pytest.raises(ValueError, match="scope"):
        orm.SelectionSpec.from_json(json.dumps(data))
