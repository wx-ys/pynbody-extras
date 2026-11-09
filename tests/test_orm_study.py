"""Serial studies preserve real Run, scientific value and immutable reference contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pynbody
import pytest

sa = pytest.importorskip("sqlalchemy")

from pynbodyext import orm
from pynbodyext.calculate import LogicalSnapshotIdentity, PropertyBase
from pynbodyext.core.calculate import LambdaProperty
from pynbodyext.core.calculate.result.result import ErrorInfo
from pynbodyext.core.calculate.store import RecordRef, SQLAlchemyResultStore
from pynbodyext.core.calculate.store.codec import ValueCodec
from pynbodyext.properties import ParamSum


def api():
    from pynbodyext.orm import StudyCell, StudyReport, StudyPersistenceError, run_study
    return SimpleNamespace(Cell=StudyCell, Report=StudyReport, Persistence=StudyPersistenceError, run=run_study)


@pytest.fixture
def env(tmp_path):
    payload_url = f"sqlite:///{tmp_path / 'payload.sqlite'}"
    metadata_url = f"sqlite:///{tmp_path / 'metadata.sqlite'}"
    store = SQLAlchemyResultStore(payload_url)
    calls, multiplier = [], [1]

    def loader(uri):
        calls.append(uri)
        index = 1 if uri.endswith("snap1") else 2
        sim = pynbody.new(2)
        sim["mass"] = pynbody.array.SimArray([2. * index * multiplier[0], 4. * index * multiplier[0]], "Msol")
        sim["pos"] = [[0., 0., 0.], [float(index), 0., 0.]]
        sim._filename = uri
        return sim

    repo = orm.AnalysisRepository(metadata_url, stores={"results": store}, loaders={"default": loader})
    identities = [LogicalSnapshotIdentity("project", "sim", str(i)) for i in (1, 2)]
    inputs = [repo.register_input(identity, location=str(tmp_path / f"snap{i}"))
              for i, identity in enumerate(identities, 1)]
    yield SimpleNamespace(repo=repo, store=store, inputs=inputs, calls=calls, multiplier=multiplier,
                          loader=loader, identities=identities, payload_url=payload_url, metadata_url=metadata_url)
    repo.engine.dispose()
    store.engine.dispose()


def run(env, calculators=None, **kwargs):
    return api().run(env.repo, [item.id for item in env.inputs],
                     calculators if calculators is not None else {"mass": ParamSum("mass"), "x": ParamSum("x")},
                     "results", **kwargs)


def assert_no_work(env):
    assert env.calls == []
    assert env.repo.runs() == []
    assert env.store.list_refs() == []
    with env.repo.session() as session:
        assert session.scalars(sa.select(orm.Recipe)).all() == []


def test_real_two_by_two_order_run_mapping_and_actual_values(env):
    report = run(env)
    ids = tuple(item.id for item in env.inputs)
    assert report.input_ids == ids
    assert report.identities == tuple(item.signature() for item in env.identities)
    assert report.store_name == "results"
    assert [label for label, _ in report.columns] == ["mass", "x"]
    assert [(cell.input_id, cell.label) for cell in report.cells] == [(i, label) for i in ids for label in ("mass", "x")]
    assert [float(cell.value) for cell in report.cells] == [6, 1, 12, 2]
    assert len(env.repo.runs()) == len(report.cells) == len(env.calls) == 4
    for cell in report.cells:
        recorded = env.repo.get_run(cell.run_id)
        assert cell is report.cell(cell.input_id, cell.label)
        assert cell.status == recorded.status == "succeeded" and not cell.errors
        assert recorded.input_id == cell.input_id and recorded.store_name == report.store_name
        assert recorded.input_location_id == env.repo.get_input(cell.input_id).locations[0].id
        assert recorded.recipe_hash == dict(report.columns)[cell.label]
        assert cell.ref == RecordRef.from_json(recorded.result_link.ref_json)
        assert cell.ref.version_id and env.repo.fetch_result(cell.run_id).store_ref == cell.ref
    assert env.calls == [env.repo.get_input(i).locations[0].uri for i in ids for _ in range(2)]


def test_repeat_refresh_report_reopen_and_current_invalidation_preserve_history(env, tmp_path):
    original = run(env)
    hit = run(env)
    assert len(env.repo.runs()) == 8
    assert {c.run_id for c in original.cells}.isdisjoint({c.run_id for c in hit.cells})
    assert [c.ref for c in original.cells] == [c.ref for c in hit.cells]
    assert all(env.repo.get_run(c.run_id).reused for c in hit.cells)
    env.multiplier[0] = 3
    fresh = run(env, reuse=False)
    assert [float(c.value) for c in fresh.cells] == [18, 1, 36, 2]
    assert all(a.ref != b.ref for a, b in zip(original.cells, fresh.cells))
    path = tmp_path / "report.json"
    original.save(path)
    env.repo.engine.dispose()
    env.store.engine.dispose()
    store = SQLAlchemyResultStore(env.payload_url)
    repo = orm.AnalysisRepository(env.metadata_url, stores={"results": store})
    try:
        loaded = api().Report.load(path)
        for old, new in zip(loaded.cells, fresh.cells):
            assert store.delete(new.ref)
            assert float(store.fetch(old.ref).value) == float(old.value)
            assert repo.fetch_result(old.run_id).store_ref == old.ref
            assert repo.fetch_result(new.run_id).store_ref == new.ref
        assert repo.audit() == []
    finally:
        repo.engine.dispose()
        store.engine.dispose()


def test_alias_labels_are_distinct_runs_and_configured_live_options_survive(env):
    calc = ParamSum("mass").options(cache=False)
    report = run(env, {"one": calc, "two": calc})
    assert len(report.cells) == 4
    assert report.columns[0][1] == report.columns[1][1]
    assert len({cell.run_id for cell in report.cells}) == 4
    assert report.cells[0].ref == report.cells[1].ref


class StableLiveOnly(PropertyBase[float]):
    signature_name = "tests.orm_study.StableLiveOnly"

    def signature_payload(self):
        return {"constant": 7}

    def calculate(self, sim, params=None):
        return 7.


def test_stable_nonconstructible_calculator_does_not_inherit_jobs_restrictions(env):
    calc = StableLiveOnly()
    assert calc.to_signature().stable and not calc.to_signature().constructible
    report = run(env, {"live": calc})
    assert [cell.value for cell in report.cells] == [7., 7.]


class RepeatedLabels(Mapping):
    def __iter__(self):
        return iter(["same", "same"])
    def __len__(self):
        return 2
    def __getitem__(self, key):
        return ParamSum("mass")
    def items(self):
        return [("same", ParamSum("mass")), ("same", ParamSum("x"))]


@pytest.mark.parametrize("problem", ["duplicate", "missing", "bool", "float", "zero", "negative", "numpy",
                                       "view", "selection", "hierarchy", "kind", "identity", "parent"])
def test_full_list_invalid_late_input_has_no_side_effects(env, problem):
    ids = [item.id for item in env.inputs]
    if problem == "duplicate":
        ids[1] = ids[0]
    elif problem in ("missing", "bool", "float", "zero", "negative", "numpy"):
        ids[1] = {"missing": 999999, "bool": True, "float": float(ids[1]), "zero": 0,
                  "negative": -1, "numpy": np.int64(ids[1])}[problem]
    elif problem in ("view", "selection"):
        identity = env.identities[1].signature() + ("subset", "a" * 16)
        ids[1] = env.repo.register_input(identity).id
        if problem == "selection":
            loaded = env.repo.load_input(env.inputs[1].id)
            selected = env.repo.register_selection(loaded, orm.SelectionSpec.indices([0]))
            ids[1] = selected.view_input_id
            env.calls.clear()
    else:
        with env.repo.session() as session, session.begin():
            row = session.get(orm.DataObject, ids[1])
            if problem == "hierarchy":
                row.timestep.logical_id = "changed"
            elif problem == "kind":
                row.kind = "halo"
            elif problem == "identity":
                row.identity_key = "bad"
            else:
                row.parent_id = ids[0]
    with pytest.raises((ValueError, KeyError, TypeError)):
        api().run(env.repo, ids, {"mass": ParamSum("mass")}, "results")
    assert_no_work(env)


@pytest.mark.parametrize("calculators", [[("mass", ParamSum("mass"))], RepeatedLabels(),
                                          {"good": ParamSum("mass"), "": ParamSum("x")},
                                          {"good": ParamSum("mass"), "  ": ParamSum("x")},
                                          {"good": ParamSum("mass"), 1: ParamSum("x")},
                                          {"good": ParamSum("mass"), "bad": object()},
                                          {"good": ParamSum("mass"), "unstable": LambdaProperty(lambda sim: 1)}])
def test_all_calculators_preflight_before_recipe_writes(env, calculators):
    with pytest.raises((TypeError, ValueError)):
        run(env, calculators)
    assert_no_work(env)


def test_existing_late_recipe_conflict_prevents_any_work(env):
    target, other = ParamSum("x"), ParamSum("mass")
    env.repo.register_recipe(target.to_signature())
    with env.repo.session() as session, session.begin():
        session.get(orm.Recipe, target.to_signature().hash).signature_text = other.to_signature().to_json()
    with pytest.raises(ValueError, match="recipe"):
        run(env)
    assert env.calls == [] and env.repo.runs() == [] and env.store.list_refs() == []
    with env.repo.session() as session:
        assert len(session.scalars(sa.select(orm.Recipe)).all()) == 1


@pytest.mark.parametrize("problem", ["store", "storetype", "loader", "nolocations", "ambiguous", "remote", "relative",
                                       "foreign", "missinglocation", "unknownkey", "boolkey", "floatkey", "badvalue",
                                       "boolvalue", "nonmapping", "nonevalue", "reuse", "errors"])
def test_binding_and_location_preflight_rejects_entire_job_list(env, problem):
    second = env.inputs[1].id
    store_name, options = "results", {}
    if problem == "store":
        store_name = "absent"
    elif problem == "storetype":
        store_name = 42
    elif problem == "loader":
        with env.repo.session() as session, session.begin():
            row = session.get(orm.DataObject, second)
            identity = LogicalSnapshotIdentity("project", "sim", "2", loader_id="absent")
            from pynbodyext.core.calculate.store.identity import sim_identity_key
            row.identity_json = json.dumps(identity.signature())
            row.identity_key = sim_identity_key(identity.signature())
    elif problem in ("nolocations", "remote", "relative"):
        with env.repo.session() as session, session.begin():
            location = session.scalar(sa.select(orm.DataLocation).where(orm.DataLocation.data_object_id == second))
            if problem == "nolocations":
                session.delete(location)
            else:
                location.uri = "s3://bucket/snap" if problem == "remote" else "relative/snap"
    elif problem == "ambiguous":
        env.repo.register_input(env.identities[1], location="/data/second-alias")
    elif problem in ("foreign", "missinglocation", "unknownkey", "boolkey", "floatkey", "badvalue", "boolvalue", "nonevalue"):
        own = env.repo.get_input(second).locations[0].id
        options["location_ids"] = {
            "foreign": {second: env.repo.get_input(env.inputs[0].id).locations[0].id},
            "missinglocation": {second: 999999}, "unknownkey": {999999: own}, "boolkey": {True: own},
            "floatkey": {float(second): own}, "badvalue": {second: 1.0},
            "boolvalue": {second: True}, "nonevalue": {second: None},
        }[problem]
    elif problem == "nonmapping":
        options["location_ids"] = [(second, 1)]
    else:
        options[problem] = 1 if problem == "reuse" else "ignore"
    with pytest.raises((ValueError, TypeError, KeyError)):
        api().run(env.repo, [i.id for i in env.inputs], {"mass": ParamSum("mass")}, store_name, **options)
    assert_no_work(env)


def test_selected_locations_owned_by_each_root_are_recorded(env):
    env.repo.register_input(env.identities[1], location="/data/alias")
    selected = env.repo.get_input(env.inputs[1].id).locations[-1].id
    report = run(env, {"mass": ParamSum("mass")}, location_ids={env.inputs[1].id: selected})
    assert env.repo.get_run(report.cells[1].run_id).input_location_id == selected


@pytest.mark.parametrize("empty", ["inputs", "calculators", "both"])
def test_empty_axes_are_valid_readonly_reports(env, empty):
    ids = [] if empty in ("inputs", "both") else [i.id for i in env.inputs]
    calculators = {} if empty in ("calculators", "both") else {"mass": ParamSum("mass")}
    report = api().run(env.repo, ids, calculators, "results")
    assert report.cells == () and report.input_ids == tuple(ids)
    assert_no_work(env)


@pytest.mark.parametrize("collected", [False, True])
def test_scientific_errors_record_failed_run_and_continue_later_cells(env, collected):
    bad = ParamSum("absent")
    if collected:
        bad = bad.options(errors="collect")
    report = run(env, {"mass": ParamSum("mass"), "bad": bad, "x": ParamSum("x")}, errors="record")
    assert len(report.cells) == len(env.repo.runs()) == 6
    for input in env.inputs:
        failure = report.cell(input.id, "bad")
        assert failure.status == "failed" and failure.ref is None and failure.errors
        assert env.repo.get_run(failure.run_id).status == "failed"
        assert env.repo.get_run(failure.run_id).result_link is None
        assert "absent" in failure.errors[0].message
        assert float(report.cell(input.id, "mass").value) == 6 * (env.inputs.index(input) + 1)
        assert report.cell(input.id, "x").status == "succeeded"


@pytest.mark.parametrize("collected", [False, True])
def test_raise_policy_stops_after_failed_run(env, collected):
    bad = ParamSum("absent").options(errors="collect") if collected else ParamSum("absent")
    with pytest.raises(orm.RunExecutionError) as caught:
        run(env, {"bad": bad, "later": ParamSum("mass")})
    assert len(env.repo.runs()) == len(env.calls) == 1
    assert env.repo.get_run(caught.value.run_id).status == "failed"
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("policy", ["raise", "record"])
def test_metadata_finalization_halts_with_exact_receipt_partial_report_and_manual_recovery(env, policy, monkeypatch):
    observed, original = [], env.repo.run_input

    def capture(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except orm.RunPersistenceError as receipt:
            observed.append(receipt)
            raise

    monkeypatch.setattr(env.repo, "run_input", capture)
    linked = [0]
    cause = RuntimeError("metadata unavailable")
    def block_link(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_result_links"):
            linked[0] += 1
            if linked[0] == 2:
                raise cause
    sa.event.listen(env.repo.engine, "before_cursor_execute", block_link)
    try:
        with pytest.raises(api().Persistence) as caught:
            run(env, errors=policy)
    finally:
        sa.event.remove(env.repo.engine, "before_cursor_execute", block_link)
    error = caught.value
    assert error.receipt is observed[0] and error.__cause__ is error.receipt
    assert error.receipt.__cause__ is cause
    assert error.result is error.receipt.result
    assert error.result_ref is error.receipt.result_ref
    assert error.run_id == error.receipt.run_id
    assert error.input_id == env.inputs[0].id and error.label == "x"
    assert len(error.report.cells) == 1 and len(error.report.input_ids) == 2
    assert len(env.repo.runs()) == len(env.calls) == 2
    pending = env.repo.get_run(error.run_id)
    assert pending.status == "running" and pending.result_link is None
    assert float(env.store.fetch(error.result_ref).value) == 1
    assert len(env.store.list_refs()) == 2
    checkpoint = api().Report.from_json(error.report.to_json())
    with pytest.raises(KeyError):
        checkpoint.cell(error.input_id, error.label)
    # Only caller code attaches the saved receipt after removing the failure.
    assert env.repo.finish_run(error.run_id, error.result_ref).status == "succeeded"
    assert env.repo.audit() == []
    resumed = api().run(env.repo, [env.inputs[0].id], {"mass": ParamSum("mass"), "x": ParamSum("x")}, "results")
    assert len(env.repo.runs()) == 4 and len(checkpoint.cells) == 1
    assert resumed.cells[0].run_id != checkpoint.cells[0].run_id
    assert resumed.cells[0].ref == checkpoint.cells[0].ref
    assert resumed.cells[1].ref == error.result_ref


def test_fail_run_persistence_error_stops_even_without_saved_payload(env):
    cause = RuntimeError("failure metadata unavailable")
    def block_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE pynbodyext_orm_runs"):
            raise cause
    sa.event.listen(env.repo.engine, "before_cursor_execute", block_update)
    try:
        with pytest.raises(api().Persistence) as caught:
            run(env, {"bad": ParamSum("absent"), "later": ParamSum("mass")}, errors="record")
    finally:
        sa.event.remove(env.repo.engine, "before_cursor_execute", block_update)
    error = caught.value
    assert error.receipt.__cause__ is cause
    assert error.result_ref is error.result is None
    assert len(error.report.cells) == 0 and len(env.calls) == len(env.repo.runs()) == 1
    assert env.repo.get_run(error.run_id).status == "running"


def test_raw_begin_metadata_failure_never_continues_in_record_mode(env):
    cause = RuntimeError("begin unavailable")
    def block_begin(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO pynbodyext_orm_runs"):
            raise cause
    sa.event.listen(env.repo.engine, "before_cursor_execute", block_begin)
    try:
        with pytest.raises(RuntimeError, match="begin unavailable") as caught:
            run(env, errors="record")
    finally:
        sa.event.remove(env.repo.engine, "before_cursor_execute", block_begin)
    assert caught.value is cause
    assert env.calls == [] and env.repo.runs() == []


def portable_report():
    a = api()
    identities = tuple(("sim", value) for value in (1, True, 1.0, "1"))
    key = "a" * 64
    cells = tuple(a.Cell(i + 1, "mass", f"run-{i}", "succeeded",
                         pynbody.array.SimArray([i, i + 1], "kpc"), RecordRef(identity, key, version_id=f"version-{i}"))
                  for i, identity in enumerate(identities))
    failure = a.Cell(1, "bad", "failed-run", "failed", errors=(ErrorInfo("ValueError", "bad input", "compute", "trace"),))
    return a.Report("results", (1, 2, 3, 4), identities, (("mass", key), ("bad", "b" * 64)), cells + (failure,))


def test_report_json_units_errors_typed_refs_and_load_is_readonly(tmp_path, monkeypatch):
    report = portable_report()
    path = tmp_path / "portable.json"
    report.save(path)
    def blocked(*args, **kwargs):
        pytest.fail("report loading must be read-only without backend lookup or calculator reconstruction")
    monkeypatch.setattr(orm.AnalysisRepository, "session", blocked)
    monkeypatch.setattr(SQLAlchemyResultStore, "fetch", blocked)
    monkeypatch.setattr(orm.AnalysisRepository, "load_input", blocked)
    monkeypatch.setattr(ParamSum, "from_signature_payload", blocked, raising=False)
    restored = api().Report.load(path)
    assert [type(identity[1]) for identity in restored.identities] == [int, bool, float, str]
    assert [cell.ref for cell in restored.cells] == [cell.ref for cell in report.cells]
    for cell in restored.cells[:4]:
        np.testing.assert_array_equal(cell.value, report.cell(cell.input_id, cell.label).value)
        assert str(cell.value.units) == "kpc"
    error = restored.cell(1, "bad").errors[0]
    assert (error.error_type, error.message, error.phase, error.traceback_text) == ("ValueError", "bad input", "compute", "trace")
    assert restored.to_dict() == api().Report.from_json(restored.to_json()).to_dict()


@pytest.mark.parametrize("problem", ["schema", "unknown", "missing", "duplicateinput", "boolinput", "floatinput",
                                       "identitylength", "identitytype", "duplicatecolumn", "emptylabel", "badhash",
                                       "duplicatecell", "duplicaterun", "foreigninput", "foreigncolumn", "badstatus",
                                       "noref", "unpinned", "foreignref", "typedref", "hashref", "failureref",
                                       "emptyerrors", "successerrors", "errortype", "unknownerror", "unknowncell"])
def test_malformed_report_rejected_instead_of_silently_changing_cells(problem):
    data = portable_report().to_dict()
    cell = data["cells"][0]
    if problem == "schema":
        data["schema"] = "unknown"
    elif problem == "unknown":
        data["credentials"] = "secret"
    elif problem == "missing":
        del data["columns"]
    elif problem == "duplicateinput":
        data["input_ids"][1] = data["input_ids"][0]
    elif problem in ("boolinput", "floatinput"):
        data["input_ids"][0] = True if problem == "boolinput" else 1.0
    elif problem == "identitylength":
        data["identities"].pop()
    elif problem == "identitytype":
        data["identities"][0] = ValueCodec().encode(["sim", 1])
    elif problem == "duplicatecolumn":
        data["columns"].append(dict(data["columns"][0]))
    elif problem == "emptylabel":
        data["columns"][0]["label"] = " "
    elif problem == "badhash":
        data["columns"][0]["key"] = "short"
    elif problem == "duplicatecell":
        data["cells"].append(dict(cell))
    elif problem == "duplicaterun":
        data["cells"][1]["run_id"] = cell["run_id"]
    elif problem == "foreigninput":
        cell["input_id"] = 99
    elif problem == "foreigncolumn":
        cell["label"] = "unknown"
    elif problem == "badstatus":
        cell["status"] = "running"
    elif problem == "noref":
        cell["ref"] = None
    elif problem in ("unpinned", "foreignref", "typedref", "hashref"):
        ref = RecordRef.from_json(cell["ref"])
        if problem == "unpinned":
            ref = replace(ref, version_id=None)
        elif problem == "foreignref":
            ref = replace(ref, sim_signature=("sim", "unknown"))
        elif problem == "typedref":
            ref = replace(ref, sim_signature=("sim", True))
        else:
            ref = replace(ref, calculator_signature_hash="c" * 64)
        cell["ref"] = ref.to_json()
    elif problem == "failureref":
        data["cells"][-1]["ref"] = cell["ref"]
    elif problem == "emptyerrors":
        data["cells"][-1]["errors"] = []
    elif problem == "successerrors":
        cell["errors"] = data["cells"][-1]["errors"]
    elif problem == "errortype":
        data["cells"][-1]["errors"][0]["message"] = 42
    elif problem == "unknownerror":
        data["cells"][-1]["errors"][0]["extra"] = True
    else:
        cell["extra"] = True
    with pytest.raises((ValueError, TypeError)):
        api().Report.from_dict(data)


@pytest.mark.parametrize("text", ['[]', '{}', '{"schema":"one","schema":"two"}', '{"schema":NaN}',
                                  '{"schema":Infinity}', '{"schema":-Infinity}', 'null'])
def test_invalid_report_json_is_rejected(text):
    with pytest.raises((TypeError, ValueError)):
        api().Report.from_json(text)


@pytest.mark.parametrize("problem", ["duplicate", "unknown", "missing", "nonfinite"])
def test_nested_reference_json_does_not_silently_accept_malformed_fields(problem):
    data = portable_report().to_dict()
    raw = json.loads(data["cells"][0]["ref"])
    if problem == "duplicate":
        text = data["cells"][0]["ref"][:-1] + ',"version_id":"silently-replaced"}'
    else:
        if problem == "unknown":
            raw["extra"] = True
        elif problem == "missing":
            del raw["version_id"]
        else:
            raw["extra"] = float("nan")
        text = json.dumps(raw)
    data["cells"][0]["ref"] = text
    with pytest.raises((TypeError, ValueError)):
        api().Report.from_dict(data)


def test_successful_none_and_numpy_nested_values_roundtrip():
    report = portable_report()
    values = [None, {"array": np.array([1, 2], dtype=np.int16), "scalar": np.float32(2)}, (1, None), np.nan]
    cells = tuple(replace(cell, value=value) for cell, value in zip(report.cells[:4], values)) + report.cells[4:]
    restored = api().Report.from_json(replace(report, cells=cells).to_json())
    assert restored.cells[0].status == "succeeded" and restored.cells[0].value is None and restored.cells[0].ref
    np.testing.assert_array_equal(restored.cells[1].value["array"], np.array([1, 2], dtype=np.int16))
    assert type(restored.cells[1].value["scalar"]) is np.float32
    assert restored.cells[2].value == (1, None) and np.isnan(restored.cells[3].value)
