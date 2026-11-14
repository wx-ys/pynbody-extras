"""Database-owned repositories can register path identities without fake simulations."""
from __future__ import annotations

import json

import numpy as np
import pynbody
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.core.calculate.runtime.sim_identity import snapshot_identity
from pynbodyext.core.calculate.store import SQLAlchemyResultStore
from pynbodyext.properties import ParamSum


def snapshot(path):
    sim = pynbody.new(dm=2, star=2)
    sim["mass"] = np.arange(1., 5.)
    sim["pos"] = np.zeros((4, 3))
    sim._filename = str(path)
    return sim


@pytest.fixture
def env(tmp_path):
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'data.sqlite'}")
    repo = orm.AnalysisRepository(store.engine, stores={"results": store}, loaders={"default": snapshot}, allow_path_inputs=True)
    yield repo, tmp_path / "snapshot_099", store
    repo.engine.dispose()


def test_path_registration_does_not_invent_simulation_or_timestep(env):
    repo, path, _ = env
    signature = ("sim", str(path))
    data = repo.register_input(signature, location=str(path))
    assert repo.register_input(signature, location=str(path)).id == data.id
    loaded = repo.get_input(data.id)
    assert loaded.signature == signature
    assert loaded.timestep is None and loaded.timestep_id is None
    assert loaded.locations[0].uri == str(path)
    with repo.session() as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(orm.Simulation)) == 0
        assert session.scalar(sa.select(sa.func.count()).select_from(orm.Timestep)) == 0


def test_path_run_reopen_versions_and_audit(env):
    repo, path, store = env
    data = repo.register_input(("sim", str(path)), location=str(path))
    first = repo.run_input(ParamSum("mass"), data.id, store_name="results")
    second = repo.run_input(ParamSum("mass"), data.id, store_name="results")
    assert second.result.from_store
    assert first.result.store_ref == second.result.store_ref
    assert first.run.execution_context["input"] == {"uri": str(path), "loader_id": "default", "revision": None}
    assert repo.audit() == []
    reopened = orm.AnalysisRepository(store.engine, stores={"results": store}, allow_path_inputs=True)
    assert reopened.fetch_result(first.run.id).value == 10
    assert reopened.get_run(first.run.id).input.timestep is None
    assert reopened.audit() == []


def test_path_loaded_identity_and_selection_replay(env):
    repo, path, _ = env
    root = repo.register_input(("sim", str(path)), location=str(path))
    loaded = repo.load_input(root.id)
    assert loaded.sim_identity(loaded.sim) == root.signature
    selected = repo.register_selection(loaded, orm.SelectionSpec.indices([1, 0, 1, 3]))
    replayed = repo.load_selection(selected.id)
    np.testing.assert_array_equal(replayed.sim["mass"], [2., 1., 2., 4.])
    assert replayed.input.parent_id == root.id
    assert replayed.input.timestep is None
    run = repo.run_selection(ParamSum("mass"), selected.id, store_name="results")
    assert run.result.value == 9
    assert repo.audit() == []


def test_path_catalogue_metadata_and_typed_objects(env):
    repo, path, _ = env
    root = repo.register_input(("sim", str(path)), location=str(path))
    cat = repo.register_catalogue(root.id, "catalogue", finder="test")
    one = repo.register_catalogue_object(cat.id, 1)
    text = repo.register_catalogue_object(cat.id, "1")
    assert one.id != text.id
    assert repo.get_catalogue(cat.id).root_input.timestep is None
    assert len(repo.catalogue_objects(cat.id)) == 2


def test_path_direct_run_and_manual_completion(env):
    repo, path, store = env
    sim = snapshot(path)
    run = repo.run(ParamSum("mass"), sim, sim_identity=snapshot_identity, store_name="results")
    assert run.result.value == 10
    calc = ParamSum("mass")
    recipe = repo.register_recipe(calc.to_signature())
    started = repo.begin_run(run.run.input_id, recipe.hash, store_name="results")
    result = calc.run(sim, store=store)
    done = repo.finish_run(started.id, result.store_ref)
    assert done.status == "succeeded"
    assert repo.audit() == []


@pytest.mark.parametrize("identity", [("session", 2), ("sim", "relative"), ("sim", "/a/../b"), ("sim", "/a", "subset", "bad"), ("sim", "/a", "halo", True, "bad"), ("sim", "/a", "other")])
def test_path_invalid_identity_rejected_without_rows(env, identity):
    repo, _, _ = env
    with pytest.raises((ValueError, TypeError)):
        repo.register_input(identity)
    with repo.session() as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(orm.DataObject)) == 0


def test_path_wrong_loader_root_fails_and_is_auditable(env):
    repo, path, _ = env
    root = repo.register_input(("sim", str(path)), location=str(path) + "-wrong")
    with pytest.raises(orm.RunExecutionError):
        repo.run_input(ParamSum("mass"), root.id, store_name="results")
    assert repo.runs()[0].status == "failed"
    assert repo.audit() == []


def test_path_corrupt_hierarchy_rejected_before_run(env):
    repo, path, _ = env
    root = repo.register_input(("sim", str(path)), location=str(path))
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    with repo.session() as session, session.begin():
        row = session.get(orm.DataObject, root.id)
        row.identity_json = json.dumps(("sim", str(path) + "-other"))
    with pytest.raises(ValueError):
        repo.begin_run(root.id, recipe.hash, store_name="results")
    assert repo.runs() == []


def test_halo_replay_descriptors_distinguish_subset_and_halo_numbers(env):
    from pynbodyext.orm.selections import SELECTION_SCHEMA, descriptor_key
    repo, path, _ = env
    root = repo.register_input(('sim', str(path)), location=str(path))
    loaded = repo.load_input(root.id)
    spec = orm.SelectionSpec.indices([1, 0, 1])
    subset = repo.register_selection(loaded, spec)
    ids = [subset.id]
    keys = [subset.descriptor_key]
    for number in [5, 8]:
        view = spec.materialize(loaded)
        view._halo_number = number
        signature = loaded.sim_identity(view)
        data = repo.register_input(signature)
        key = descriptor_key(root.id, len(loaded.sim), spec, view_signature=signature)
        with repo.session() as session, session.begin():
            selection = orm.Selection(root_input_id=root.id, view_input_id=data.id, schema=SELECTION_SCHEMA,
                                      definition_json=spec.to_json(), expected_count=len(loaded.sim), descriptor_key=key)
            session.add(selection)
            session.flush()
            ids.append(selection.id)
            keys.append(key)
        replayed = repo.load_selection(ids[-1])
        assert replayed.sim._halo_number == number
        assert replayed.sim_identity(replayed.sim) == signature
        np.testing.assert_array_equal(replayed.sim['mass'], [2., 1., 2.])
        run = repo.run_selection(ParamSum('mass'), ids[-1], store_name='results')
        assert run.result.store_ref.sim_signature == signature
    assert len(set(keys)) == 3
    assert len(set(ids)) == 3
    assert repo.load_selection(subset.id).input.kind == 'subset'
    assert repo.audit() == []
