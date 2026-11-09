"""Root jobs replay through trusted bindings and the existing tracked runner."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import pynbody
import pytest

pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.calculate import LogicalSnapshotIdentity, Pipeline, PropertyBase
from pynbodyext.core.calculate import LambdaProperty, RunOptions
from pynbodyext.core.calculate import interfaces
from pynbodyext.core.calculate.store import RecordRef, SQLAlchemyResultStore
from pynbodyext.properties import ParamSum


def snapshot(uri, mass=(2., 4.)):
    sim = pynbody.new(2)
    sim["mass"] = mass
    sim["pos"] = [[0., 0., 0.], [1., 0., 0.]]
    sim._filename = str(uri)
    return sim


@PropertyBase.dataclass
class RegisteredTotal(PropertyBase):
    signature_name = "jobtests.RegisteredTotal"
    signature_aliases = ("jobtests.OldTotal",)
    extra: object = None
    calls = 0

    def calculate(self, sim, params=None):
        type(self).calls += 1
        return float(sim["mass"].sum())


@PropertyBase.dataclass
class AliasOnly(PropertyBase):
    signature_aliases = ("jobtests.AliasOnly",)
    amount: float = 2.0

    def calculate(self, sim, params=None):
        return float(sim["mass"].sum()) * self.amount


@PropertyBase.dataclass
class PathOnly(PropertyBase):
    def calculate(self, sim, params=None):
        pytest.fail("must not calculate")


class Choice(Enum):
    FIRST = "first"


@pytest.fixture
def registry():
    saved = dict(interfaces._SIGNATURE_TYPES)
    yield interfaces._SIGNATURE_TYPES
    interfaces._SIGNATURE_TYPES.clear()
    interfaces._SIGNATURE_TYPES.update(saved)


@pytest.fixture
def env(tmp_path):
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'payload.sqlite'}")
    repos = []
    calls = []
    RegisteredTotal.calls = 0
    def loader(uri):
        calls.append(uri)
        return snapshot(uri)
    def make(*, loaders=None, stores=None):
        repo = orm.AnalysisRepository(
            f"sqlite:///{tmp_path / 'metadata.sqlite'}",
            stores={"results": store} if stores is None else stores,
            loaders={"default": loader} if loaders is None else loaders,
        )
        repos.append(repo)
        return repo
    repo = make()
    root = repo.register_input(LogicalSnapshotIdentity("project", "sim", "099"), location=str(tmp_path / "snap"))
    yield make, repo, root, store, calls
    for item in repos:
        item.engine.dispose()
    store.engine.dispose()


def job(**kwargs):
    return orm.JobSpec.from_calculator(RegisteredTotal(), 1, store_name="results", **kwargs)


def test_manifest_is_frozen_and_has_exactly_six_fields():
    spec = job()
    fields = json.loads(spec.to_json())
    assert set(fields) == {"schema", "calculator_signature_text", "input_id", "location_id", "store_name", "reuse"}
    assert fields["schema"] == "pynbodyext.orm.job/v1"
    assert fields["location_id"] is None and fields["reuse"] is True
    assert orm.JobSpec.from_json(spec.to_json()) == spec
    with pytest.raises(FrozenInstanceError):
        spec.reuse = False


@pytest.mark.parametrize("field,values", [
    ("input_id", [True, False, 0, -1, 1., "1", np.int64(1), None]),
    ("location_id", [True, False, 0, -1, 1., "1", np.int64(1)]),
    ("store_name", ["", " ", None, 1, True, []]),
    ("reuse", [0, 1, "yes", None, np.bool_(True)]),
    ("schema", [None, 1, "", "unknown"]),
    ("calculator_signature_text", [None, {}, 1, "", "[]"]),
])
def test_strict_constructor_fields(field, values):
    spec = job()
    for value in values:
        with pytest.raises((TypeError, ValueError)):
            replace(spec, **{field: value})


@pytest.mark.parametrize("field", ["input_id", "location_id", "store_name", "reuse", "schema", "calculator_signature_text"])
def test_field_subclasses_are_not_coerced(field):
    class Int(int):
        pass
    class Str(str):
        pass
    value = Int(1) if field in ("input_id", "location_id", "reuse") else Str(getattr(job(), field))
    with pytest.raises((TypeError, ValueError)):
        replace(job(), **{field: value})


@pytest.mark.parametrize("problem", ["missing", "extra", "schema", "bool_id", "float_id", "reuse", "notobject"])
def test_json_rejects_malformed_contract(problem):
    data = json.loads(job().to_json())
    if problem == "missing":
        del data["location_id"]
    elif problem == "extra":
        data["credentials"] = "must-not-be-accepted"
    elif problem == "schema":
        data["schema"] = "pynbodyext.orm.job/v9"
    elif problem == "bool_id":
        data["input_id"] = True
    elif problem == "float_id":
        data["input_id"] = 1.
    elif problem == "reuse":
        data["reuse"] = 1
    else:
        data = []
    with pytest.raises((TypeError, ValueError)):
        orm.JobSpec.from_json(json.dumps(data))


@pytest.mark.parametrize("text", ['{"schema":1,"schema":2}', '{"input_id":NaN}', '{"reuse":Infinity}', "not JSON"])
def test_json_rejects_duplicate_and_nonfinite_constants(text):
    with pytest.raises(ValueError):
        orm.JobSpec.from_json(text)


def test_manifest_contains_recipe_and_addresses_but_no_bindings(env):
    _, repo, root, _, _ = env
    spec = orm.JobSpec.from_calculator(ParamSum("mass"), root.id, store_name="results")
    data = json.loads(spec.to_json())
    assert set(data) == {"schema", "calculator_signature_text", "input_id", "location_id", "store_name", "reuse"}
    assert "sqlite:" not in spec.to_json() and str(repo.engine.url) not in spec.to_json()
    assert "loader" not in data and "pickle" not in data and "sim" not in data
    assert json.loads(data["calculator_signature_text"])["payload"] == ParamSum("mass").to_signature().payload


@pytest.mark.parametrize("calculator", [LambdaProperty(lambda sim: 1), PathOnly(), RegisteredTotal(np.arange(500))])
def test_refuses_unstable_nonconstructible_and_unregistered_recipes(calculator):
    with pytest.raises((TypeError, ValueError)):
        orm.JobSpec.from_calculator(calculator, 1, store_name="results")


@pytest.mark.parametrize("calculator", [
    RegisteredTotal().options(cache=False),
    RegisteredTotal().options(progress=lambda event: None),
    Pipeline({"total": RegisteredTotal().options(perf_memory=True)}),
])
def test_nondefault_options_are_rejected_before_signature_discards_them(calculator):
    with pytest.raises(ValueError, match="default.*RunOptions"):
        orm.JobSpec.from_calculator(calculator, 1, store_name="results")


def test_constructor_created_nondefault_options_are_rejected(registry):
    class CustomOptions(PropertyBase):
        signature_name = "jobtests.CustomOptions"
        def __init__(self):
            super().__init__(default_options=RunOptions(cache=False))
        def signature_payload(self):
            return {}
        @classmethod
        def from_signature_payload(cls, payload):
            return cls()
    text = CustomOptions().to_signature().to_json()
    with pytest.raises(ValueError, match="default.*RunOptions"):
        orm.JobSpec(text, 1, "results")


def test_registered_alias_and_loaded_enum_are_accepted():
    original = RegisteredTotal(Choice.FIRST)
    sig = original.to_signature().as_dict(full=True)
    sig["payload"]["class"] = "jobtests.OldTotal"
    spec = orm.JobSpec(json.dumps(sig), 1, "results")
    assert orm.JobSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("nested", [False, True])
def test_alias_only_class_roundtrips_and_executes_without_global_registration(env, nested):
    calculator = Pipeline({"total": AliasOnly(3.0)}) if nested else AliasOnly(3.0)
    data = json.loads(calculator.to_signature().to_json(full=True))
    target = data["payload"]["outputs"][0]["value"] if nested else data["payload"]
    canonical = target["class"]
    assert canonical not in interfaces._SIGNATURE_TYPES
    target["class"] = "jobtests.AliasOnly"
    spec = orm.JobSpec(json.dumps(data), env[2].id, "results")
    restored = orm.JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.execute(env[1]).result.value == ({"total": 18.0} if nested else 18.0)
    assert canonical not in interfaces._SIGNATURE_TYPES
    with pytest.raises(ValueError, match="registered"):
        orm.JobSpec.from_calculator(calculator, env[2].id, store_name="results")


def test_rebuilder_cannot_return_an_unvetted_registered_type(registry):
    class Switch(PropertyBase):
        signature_name = "jobtests.Switch"
        def signature_payload(self):
            return {}
        @classmethod
        def from_signature_payload(cls, payload):
            return RegisteredTotal()
    with pytest.raises(ValueError, match="registered|vetted|identity"):
        orm.JobSpec(Switch().to_signature().to_json(), 1, "results")


def test_loaded_non_enum_is_not_a_constructor_escape(tmp_path, monkeypatch):
    import types
    calls = []
    module = types.ModuleType("job_escape")
    module.construct = lambda value: calls.append(value)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sig = RegisteredTotal().to_signature().as_dict(full=True)
    sig["payload"]["init"] = {"extra": {"type": "enum", "class": "job_escape.construct", "value": "payload"}}
    with pytest.raises(ValueError, match="loaded enum"):
        orm.JobSpec(json.dumps(sig), 1, "results")
    assert calls == []


def test_nested_signature_wrapper_flags_are_not_trusted():
    nested = RegisteredTotal().to_signature().as_dict(full=True)
    nested["constructible"] = "true"
    sig = RegisteredTotal().to_signature().as_dict(full=True)
    sig["payload"]["init"] = {"extra": {"type": "calculator", "value": nested}}
    with pytest.raises(ValueError, match="boolean flags"):
        orm.JobSpec(json.dumps(sig), 1, "results")


@pytest.mark.parametrize("kind", ["root", "nested", "enum"])
def test_manifest_cannot_import_unknown_payload_types(tmp_path, monkeypatch, env, kind):
    marker = tmp_path / "imported"
    (tmp_path / "untrusted_job_module.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    if kind == "nested":
        sig = Pipeline({"total": RegisteredTotal()}).to_signature().as_dict(full=True)
        sig["payload"]["outputs"][0]["value"]["class"] = "untrusted_job_module.Total"
    else:
        sig = RegisteredTotal().to_signature().as_dict(full=True)
        if kind == "root":
            sig["payload"]["class"] = "untrusted_job_module.Total"
        else:
            sig["payload"]["init"] = {"extra": {"type": "enum", "class": "untrusted_job_module.Total", "value": 1}}
    with pytest.raises(ValueError, match="registered|loaded.*enum"):
        orm.JobSpec(json.dumps(sig), env[2].id, "results")
    assert not marker.exists() and "untrusted_job_module" not in sys.modules
    assert env[4] == [] and env[1].runs() == [] and RegisteredTotal.calls == 0


def test_all_nested_types_are_preflighted_before_any_constructor(registry):
    constructions = []
    class Constructor(PropertyBase):
        signature_name = "jobtests.Constructor"
        def __init__(self):
            super().__init__()
            constructions.append(True)
        def signature_payload(self):
            return {}
        @classmethod
        def from_signature_payload(cls, payload):
            return cls()
    sig = Pipeline({"first": Constructor(), "last": RegisteredTotal()}).to_signature().as_dict(full=True)
    sig["payload"]["outputs"][1]["value"]["class"] = "missing.Type"
    constructions.clear()
    with pytest.raises(ValueError, match="registered"):
        orm.JobSpec(json.dumps(sig), 1, "results")
    assert constructions == []


@pytest.mark.parametrize("nested", [False, True])
def test_signature_version_mismatch_is_rejected_before_constructors(registry, nested):
    constructions = []
    class Constructor(PropertyBase):
        signature_name = "jobtests.VersionConstructor"
        def __init__(self):
            super().__init__()
            constructions.append(True)
        def signature_payload(self):
            return {}
        @classmethod
        def from_signature_payload(cls, payload):
            return cls()
    calculator = Pipeline({"first": Constructor(), "last": ParamSum("mass")}) if nested else Constructor()
    sig = calculator.to_signature().as_dict(full=True)
    target = sig["payload"]["outputs"][1]["value"] if nested else sig["payload"]
    target["version"] = 999
    constructions.clear()
    with pytest.raises(ValueError, match="version"):
        orm.JobSpec(json.dumps(sig), 1, "results")
    assert constructions == []


@pytest.mark.parametrize("kind", ["dataclass", "generic"])
def test_parameter_drift_during_reconstruction_is_rejected(registry, kind):
    if kind == "dataclass":
        @dataclass
        class Drifting(PropertyBase):
            signature_name = "jobtests.DriftingDataclass"
            value: float = 2
            def __post_init__(self):
                super().__init__()
                self.value += 1
    else:
        class Drifting(PropertyBase):
            signature_name = "jobtests.DriftingGeneric"
            def __init__(self, value):
                super().__init__()
                self.value = value
            def signature_payload(self):
                return {"value": self.value}
            @classmethod
            def from_signature_payload(cls, payload):
                return cls(payload["value"] + 1)
    sig = Drifting(3).to_signature()
    with pytest.raises(ValueError, match="identity|recipe"):
        orm.JobSpec(sig.to_json(), 1, "results")


def test_numeric_equivalence_and_nested_alias_preserve_recipe_identity():
    sig = Pipeline({"total": RegisteredTotal(2)}).to_signature().as_dict(full=True)
    child = sig["payload"]["outputs"][0]["value"]
    child["class"] = "jobtests.OldTotal"
    child["init"]["extra"] = 2.0
    spec = orm.JobSpec(json.dumps(sig), 1, "results")
    assert orm.JobSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("flag,value", [("stable", False), ("constructible", False), ("stable", "true"), ("constructible", 1)])
def test_signature_flags_are_strict_and_cannot_be_coerced(flag, value):
    sig = RegisteredTotal().to_signature().as_dict(full=True)
    sig[flag] = value
    with pytest.raises(ValueError):
        orm.JobSpec(json.dumps(sig), 1, "results")


@pytest.mark.parametrize("calculator", [LambdaProperty(lambda sim: 1), RegisteredTotal(np.arange(500))])
def test_forged_optimistic_flags_do_not_enable_unsupported_payload(calculator):
    sig = calculator.to_signature().as_dict(full=True)
    sig["stable"] = sig["constructible"] = True
    with pytest.raises(ValueError):
        orm.JobSpec(json.dumps(sig), 1, "results")


def test_execute_rechecks_type_registration(env, registry):
    spec = job()
    registry.pop("jobtests.RegisteredTotal")
    with pytest.raises(ValueError, match="registered"):
        spec.execute(env[1])
    assert env[4] == [] and env[1].runs() == [] and RegisteredTotal.calls == 0


@pytest.mark.parametrize("problem", ["store", "loader", "input", "location", "ambiguous", "remote"])
def test_execution_preflight_does_not_load_calculate_or_record_run(env, problem):
    make, repo, root, _, calls = env
    spec = job()
    if problem == "store":
        repo = make(stores={})
    elif problem == "loader":
        repo = make(loaders={})
    elif problem == "input":
        spec = replace(spec, input_id=99999)
    elif problem == "location":
        other = repo.register_input(LogicalSnapshotIdentity("project", "sim", "100"), location="/other/snap")
        spec = replace(spec, location_id=repo.get_input(other.id).locations[0].id)
    elif problem == "ambiguous":
        repo.register_input(root.signature, location="/other/snap")
    else:
        remote = repo.register_input(LogicalSnapshotIdentity("project", "sim", "100"), location="s3://bucket/snap")
        spec = replace(spec, input_id=remote.id)
    with pytest.raises((KeyError, ValueError)):
        spec.execute(repo)
    assert calls == [] and repo.runs() == [] and RegisteredTotal.calls == 0


def test_jobs_reject_reconstructible_selections_and_legacy_views(env):
    _, repo, root, _, calls = env
    loaded = repo.load_input(root.id)
    selection = repo.register_selection(loaded, orm.SelectionSpec.indices([0]))
    halo = repo.register_input(root.signature + ("halo", 1, "a" * 16))
    calls.clear()
    for input_id in (selection.view_input_id, halo.id):
        with pytest.raises(ValueError, match="root"):
            replace(job(), input_id=input_id).execute(repo)
    assert calls == [] and repo.runs() == [] and RegisteredTotal.calls == 0
    assert repo.run_selection(ParamSum("mass"), selection.id, store_name="results").result.value == 2


def test_selected_location_and_root_job_receipt(env):
    _, repo, root, _, calls = env
    repo.register_input(root.signature, location="/relocated/snap")
    location = next(item for item in repo.get_input(root.id).locations if item.uri == "/relocated/snap")
    spec = replace(job(), location_id=location.id)
    recorded = spec.execute(repo)
    assert recorded.result.value == 6 and RegisteredTotal.calls == 1
    assert calls == [location.uri]
    assert recorded.run.input_id == root.id and recorded.run.input_location_id == location.id
    assert recorded.run.recipe.hash == RegisteredTotal().to_signature().hash
    assert repo.fetch_result(recorded.run.id).store_ref == recorded.result.store_ref


@pytest.mark.parametrize("failure", ["loader", "calculation"])
def test_post_preflight_failures_keep_tracked_runner_errors(env, failure):
    make, repo, root, _, _ = env
    if failure == "loader":
        def broken(uri):
            raise OSError("unavailable snapshot")
        repo = make(loaders={"default": broken})
        spec = job()
    else:
        spec = orm.JobSpec.from_calculator(ParamSum("absent"), root.id, store_name="results")
    with pytest.raises(orm.RunExecutionError) as caught:
        spec.execute(repo)
    run = repo.get_run(caught.value.run_id)
    assert run.status == "failed" and run.result_link is None
    assert isinstance(caught.value.__cause__, OSError if failure == "loader" else KeyError)


def test_new_python_process_replays_science_cache_refresh_and_history(env, tmp_path):
    _, repo, root, store, _ = env
    spec = orm.JobSpec.from_calculator(ParamSum("mass"), root.id, store_name="results")
    first = spec.execute(repo)
    assert first.result.value == 6
    manifest = tmp_path / "job.json"
    manifest.write_text(spec.to_json())
    expected = {
        "run": first.run.id,
        "ref": first.result.store_ref.to_json(),
        "signature": first.result.calculator.to_signature().as_dict(full=True),
    }
    script = r'''
import json, sys
from dataclasses import replace
import pynbody
from pynbodyext import orm
from pynbodyext.properties import ParamSum
from pynbodyext.core.calculate.store import SQLAlchemyResultStore

calls = []
def loader(uri):
    calls.append(uri)
    sim = pynbody.new(2)
    sim["mass"] = [10., 20.]
    sim["pos"] = [[0., 0., 0.], [1., 0., 0.]]
    sim._filename = uri
    return sim
payloads = SQLAlchemyResultStore(sys.argv[2])
repo = orm.AnalysisRepository(sys.argv[1], stores={"results": payloads}, loaders={"default": loader})
try:
    spec = orm.JobSpec.from_json(open(sys.argv[3]).read())
    expected = json.loads(sys.argv[4])
    hit = spec.execute(repo)
    fresh = replace(spec, reuse=False).execute(repo)
    old = repo.fetch_result(expected["run"])
    current = repo.fetch_result(fresh.run.id)
    print(json.dumps({
        "hit": float(hit.result.value), "fresh": float(fresh.result.value),
        "old": float(old.value), "current": float(current.value),
        "hit_signature": hit.result.calculator.to_signature().as_dict(full=True),
        "fresh_signature": fresh.result.calculator.to_signature().as_dict(full=True),
        "hit_ref": hit.result.store_ref.to_json(), "fresh_ref": fresh.result.store_ref.to_json(),
        "old_ref": old.store_ref.to_json(), "hit_run": hit.run.id, "fresh_run": fresh.run.id,
        "reused": [hit.run.reused, fresh.run.reused], "loads": len(calls),
        "runs": len(repo.runs()), "audit": [issue.code for issue in repo.audit()],
    }))
finally:
    repo.engine.dispose()
    payloads.engine.dispose()
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(repo.engine.url), str(store.engine.url), str(manifest), json.dumps(expected)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    observed = json.loads(completed.stdout)
    assert [observed[key] for key in ("hit", "fresh", "old", "current")] == [6., 30., 6., 30.]
    assert observed["hit_signature"] == observed["fresh_signature"] == expected["signature"]
    assert observed["hit_ref"] == observed["old_ref"] == expected["ref"]
    first_ref = RecordRef.from_json(expected["ref"])
    fresh_ref = RecordRef.from_json(observed["fresh_ref"])
    assert first_ref.cache_key == fresh_ref.cache_key and first_ref.version_id != fresh_ref.version_id
    assert len({first.run.id, observed["hit_run"], observed["fresh_run"]}) == 3
    assert observed["reused"] == [True, False] and observed["loads"] == 2
    assert observed["runs"] == 3 and observed["audit"] == []
