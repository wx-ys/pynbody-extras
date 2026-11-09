"""Invocation context is copied once, before execution, independently of payloads."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import subprocess
import sys
import types

import numpy as np
import pynbody
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.calculate import ConfiguredSnapshotIdentity, LogicalSnapshotIdentity, PropertyBase
from pynbodyext.core.calculate import RunOptions
from pynbodyext.core.calculate.store import SQLAlchemyResultStore
from pynbodyext.properties import ParamSum


def snapshot(path):
    sim = pynbody.new(2)
    sim["mass"] = [6.0, 0.0]
    sim._filename = str(path)
    return sim


@PropertyBase.dataclass
class Explodes(PropertyBase):
    def calculate(self, sim, params=None):
        raise RuntimeError("scientific failure")


@pytest.fixture
def env(tmp_path):
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'payload.sqlite'}")
    repos = []

    def make(loaders=None):
        repo = orm.AnalysisRepository(
            f"sqlite:///{tmp_path / 'metadata.sqlite'}", stores={"results": store}, loaders=loaders
        )
        repos.append(repo)
        return repo

    yield make, LogicalSnapshotIdentity("project", "sim", "099", revision="r3", loader_id="custom"), tmp_path / "snap", store
    for repo in repos:
        repo.engine.dispose()
    store.engine.dispose()


def test_begin_run_context_round_trips_and_copies_metadata(env):
    make, identity, path, _ = env
    repo = make()
    data = repo.register_input(identity, location=str(path))
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    metadata = {"tool_versions": {"script": "r1"}, "random_seed": 42}
    run = repo.begin_run(data.id, recipe.hash, store_name="results", execution_metadata=metadata)
    original = run.execution_context
    assert original["schema"] == "pynbodyext.orm.execution-context/v1"
    assert original["scope"] == "invocation" and original["python"] == platform.python_version()
    assert set(original["packages"]) == {"pynbodyext-core", "pynbodyext", "numpy", "pynbody"}
    assert original["input"] == {"uri": None, "loader_id": "custom", "revision": "r3"}
    assert original["tool_versions"] == {"script": "r1"} and original["random_seed"] == 42
    metadata["tool_versions"]["script"] = "edited"
    run.execution_context["tool_versions"]["script"] = "edited decoded copy"
    with pytest.raises(AttributeError):
        run.execution_context = {}
    assert run.execution_context == original
    repo.engine.dispose()
    assert make().get_run(run.id).execution_context == original


@pytest.mark.parametrize("entrypoint", ["run", "run_input", "begin_run"])
@pytest.mark.parametrize("metadata", [
    [], {"unknown": 1}, {"tool_versions": None}, {"tool_versions": []},
    {"tool_versions": {1: "v"}}, {"tool_versions": {"a": 1}},
    {"random_seed": True}, {"random_seed": 1.0}, {"random_seed": "1"},
    {"random_seed": float("nan")}, {"random_seed": np.int64(1)},
])
def test_invalid_execution_metadata_is_preflight(env, entrypoint, metadata):
    make, identity, path, store = env
    repo = make({"custom": lambda uri: pytest.fail("must not load")})
    calc = ParamSum("mass")
    if entrypoint == "run":
        invoke = lambda: repo.run(
            calc, snapshot(path), sim_identity=ConfiguredSnapshotIdentity({path: identity}),
            store_name="results", execution_metadata=metadata,
        )
    else:
        data = repo.register_input(identity, location=str(path))
        if entrypoint == "begin_run":
            recipe = repo.register_recipe(calc.to_signature())
            invoke = lambda: repo.begin_run(
                data.id, recipe.hash, store_name="results", execution_metadata=metadata
            )
        else:
            invoke = lambda: repo.run_input(calc, data.id, store_name="results", execution_metadata=metadata)
    with pytest.raises(ValueError, match="execution_metadata"):
        invoke()
    assert repo.runs() == [] and store.list_refs() == []
    if entrypoint == "run":
        with repo.session() as session:
            assert session.scalars(sa.select(orm.DataObject)).all() == []
            assert session.scalars(sa.select(orm.Recipe)).all() == []


@pytest.mark.parametrize("metadata,seed,tools", [
    (None, None, {}), ({}, None, {}), ({"random_seed": None}, None, {}),
    ({"random_seed": -1}, -1, {}), ({"random_seed": 0}, 0, {}),
    ({"tool_versions": {"script": ""}}, None, {"script": ""}),
])
def test_execution_metadata_defaults_and_accepted_values(env, metadata, seed, tools):
    make, identity, _, _ = env
    repo = make()
    data = repo.register_input(identity)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    run = repo.begin_run(data.id, recipe.hash, store_name="results", execution_metadata=metadata)
    assert run.execution_context["random_seed"] == seed
    assert run.execution_context["tool_versions"] == tools


def test_missing_package_versions_are_null_and_snapshotted(env, monkeypatch):
    make, identity, _, _ = env
    repo = make()
    data = repo.register_input(identity)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    versions = {"pynbodyext": "a", "numpy": "b"}

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    first = repo.begin_run(data.id, recipe.hash, store_name="results")
    versions["numpy"] = "new"
    second = repo.begin_run(data.id, recipe.hash, store_name="results")
    assert first.execution_context["packages"] == {"pynbodyext-core": None, "pynbodyext": "a", "numpy": "b", "pynbody": None}
    assert second.execution_context["packages"]["numpy"] == "new"
    assert make().get_run(first.id).execution_context == first.execution_context


@pytest.mark.parametrize("full_installed", [False, True])
def test_code_owner_version_is_recorded_independently_of_full_metadata(env, monkeypatch, full_installed):
    make, identity, _, _ = env
    versions = {"pynbodyext-core": "1.2.3", "numpy": "2.3.5", "pynbody": "2.8.0"}
    if full_installed:
        versions["pynbodyext"] = "1.2.3"

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", version)
    repo = make()
    data = repo.register_input(identity)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    run = repo.begin_run(data.id, recipe.hash, store_name="results")
    assert run.execution_context["packages"]["pynbodyext-core"] == "1.2.3"
    assert run.execution_context["packages"]["pynbodyext"] == ("1.2.3" if full_installed else None)
    versions["pynbodyext-core"] = "next"
    assert make().get_run(run.id).execution_context == run.execution_context


def test_context_reads_only_already_loaded_native(env, monkeypatch):
    make, identity, _, _ = env
    repo = make()
    data = repo.register_input(identity)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    monkeypatch.delitem(sys.modules, "pynbodyext._native", raising=False)

    class BlockNative:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "pynbodyext._native":
                pytest.fail("context collection attempted native import")

    monkeypatch.setattr(sys, "meta_path", [BlockNative(), *sys.meta_path])
    run = repo.begin_run(data.id, recipe.hash, store_name="results")
    assert run.execution_context["native"] is None
    module = types.ModuleType("pynbodyext._native")
    module.__version__ = "native-v1"
    monkeypatch.setitem(sys.modules, "pynbodyext._native", module)
    loaded = repo.begin_run(data.id, recipe.hash, store_name="results")
    assert loaded.execution_context["native"] == {"module": "pynbodyext._native", "version": "native-v1"}
    del module.__version__
    unversioned = repo.begin_run(data.id, recipe.hash, store_name="results")
    assert unversioned.execution_context["native"]["version"] is None
    assert loaded.execution_context["native"]["version"] == "native-v1"


def test_success_failure_hit_contexts_survive_reopen(env, monkeypatch):
    make, identity, path, _ = env
    repo = make()
    provider = ConfiguredSnapshotIdentity({path: identity})
    sim = snapshot(path)
    options = {"sim_identity": provider, "store_name": "results"}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "environment-A")
    first = repo.run(ParamSum("mass"), sim, execution_metadata={"random_seed": 1}, **options)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "environment-B")
    hit = repo.run(ParamSum("mass"), sim, execution_metadata={"random_seed": 2}, **options)
    with pytest.raises(orm.RunExecutionError) as caught:
        repo.run(Explodes(), sim, execution_metadata={"random_seed": 3}, **options)
    failed = repo.get_run(caught.value.run_id)
    collected = repo.run(Explodes(), sim, options=RunOptions(errors="collect"), **options)
    assert failed.status == collected.run.status == "failed"
    assert hit.run.reused and first.result.store_ref == hit.result.store_ref
    assert first.run.execution_context["packages"]["numpy"] == "environment-A"
    assert hit.run.execution_context["packages"]["numpy"] == "environment-B"
    assert [x.execution_context["random_seed"] for x in (first.run, hit.run, failed)] == [1, 2, 3]
    repo.engine.dispose()
    reopened = make()
    for run in (first.run, hit.run, failed, collected.run):
        assert reopened.get_run(run.id).execution_context == run.execution_context
    assert reopened.fetch_result(hit.run.id).store_ref == first.result.store_ref


def test_cache_hit_without_known_producer_context(env):
    make, identity, path, store = env
    sim = snapshot(path)
    provider = ConfiguredSnapshotIdentity({path: identity})
    saved = ParamSum("mass").run(sim, sim_identity=provider, store=store)
    repo = make()
    hit = repo.run(
        ParamSum("mass"), sim, sim_identity=provider, store_name="results",
        execution_metadata={"tool_versions": {"consumer": "v2"}},
    )
    assert hit.run.reused and hit.result.store_ref == saved.store_ref
    assert hit.run.execution_context["scope"] == "invocation"
    assert hit.run.execution_context["tool_versions"] == {"consumer": "v2"}
    assert len(repo.runs()) == 1


def test_finalization_and_recovery_do_not_replace_context(env, monkeypatch):
    make, identity, path, store = env
    repo = make()
    data = repo.register_input(identity)
    recipe = repo.register_recipe(ParamSum("mass").to_signature())
    first = repo.begin_run(data.id, recipe.hash, store_name="results")
    original = first.execution_context
    saved = ParamSum("mass").run(snapshot(path), sim_identity=ConfiguredSnapshotIdentity({path: identity}), store=store)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "later")
    assert repo.finish_run(first.id, saved.store_ref).execution_context == original
    assert repo.finish_run(first.id, saved.store_ref).execution_context == original
    second = repo.begin_run(data.id, recipe.hash, store_name="results")
    second_original = second.execution_context
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "latest")
    assert repo.fail_run(second.id, RuntimeError("failed")).execution_context == second_original
    assert make().get_run(first.id).execution_context == original


def test_direct_run_context_has_no_selected_uri_and_does_not_seed(env, monkeypatch):
    make, identity, path, _ = env
    repo = make()
    repo.register_input(identity, location=str(path))
    repo.register_input(identity, location=str(path) + ".alias")
    sim = snapshot(path)
    provider = ConfiguredSnapshotIdentity({path: identity})
    monkeypatch.setattr(random, "seed", lambda *args, **kwargs: pytest.fail("must not seed RNG"))
    monkeypatch.setattr(np.random, "seed", lambda *args, **kwargs: pytest.fail("must not seed RNG"))
    for selected in (sim, sim[[0]]):
        recorded = repo.run(
            ParamSum("mass"), selected, sim_identity=provider, store_name="results",
            execution_metadata={"random_seed": 42},
        )
        assert recorded.run.input_location_id is None
        assert recorded.run.execution_context["input"] == {"uri": None, "loader_id": "custom", "revision": "r3"}


def test_pure_calculation_still_excludes_orm():
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
assert not any(name == "pynbodyext.orm" or name.startswith("pynbodyext.orm.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
