"""Historical results remain addressable while the current cache is refreshed."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pynbody
import pytest

from pynbodyext.core.calculate import PropertyBase
from pynbodyext.core.calculate.store import InMemoryResultStore, RecordRef, ResultRecord, SignatureMismatchError
from pynbodyext.properties import ParamSum


IDENTITY = ("sim", "versions", (7, True, "7"))


def snapshot(value):
    sim = pynbody.new(2)
    sim["mass"] = [float(value), 0.0]
    return sim


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryResultStore()
    else:
        pytest.importorskip("sqlalchemy")
        from pynbodyext.core.calculate.store import SQLAlchemyResultStore

        adapter = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'versions.sqlite'}")
        try:
            yield adapter
        finally:
            adapter.engine.dispose()


def save(store, value, calc=None):
    calc = ParamSum("mass") if calc is None else calc
    return store.store(calc.run(snapshot(value)), sim_signature=IDENTITY, calculator_signature=calc.to_signature())


def test_saved_reference_is_not_retargeted_by_refresh(store):
    first = save(store, 6)
    second = save(store, 12)
    assert float(store.fetch(first).value) == 6
    assert float(store.fetch(second).value) == 12
    assert first != second and len({first, second}) == 2
    assert first.version_id and second.version_id
    assert first.cache_key == second.cache_key
    assert first.key != second.key
    calc = ParamSum("mass")
    assert float(store.get(sim_signature=IDENTITY, calculator_signature=calc.to_signature()).value) == 12
    assert store.list_refs() == [RecordRef(IDENTITY, calc.to_signature().hash)]
    assert store.list_versions(first) == [first, second]


def test_identical_saves_have_distinct_receipts(store):
    first, second = save(store, 6), save(store, 6)
    assert first != second
    assert len(store.list_versions(first)) == 2
    assert float(store.fetch(first).value) == float(store.fetch(second).value) == 6


def test_invalidation_preserves_history_and_never_promotes_old_results(store):
    first, second = save(store, 6), save(store, 12)
    calc = ParamSum("mass")
    address = RecordRef(IDENTITY, calc.to_signature().hash)
    assert not store.delete(first)  # a stale receipt cannot invalidate a newer result
    assert float(store.fetch(address).value) == 12
    assert store.delete(second)
    assert not store.delete(second)
    assert not store.has(sim_signature=IDENTITY, calculator_signature=calc.to_signature())
    assert address not in store
    assert store.list_refs() == []
    assert first in store and second in store
    assert float(store.fetch(first).value) == 6
    assert float(store.fetch(second).value) == 12
    with pytest.raises(KeyError):
        store.fetch(address)
    assert store.list_versions(address) == [first, second]
    third = save(store, 18)
    assert store.delete(address)
    assert third in store
    assert store.list_versions(address) == [first, second, third]


def test_engine_exposes_exact_receipt_for_write_hit_and_refresh(store):
    calc = ParamSum("mass")
    opts = {"store": store, "sim_identity": lambda sim: IDENTITY}
    first = calc.run(snapshot(6), **opts)
    second = calc.run(snapshot(12), **opts)
    assert first.store_ref is not None
    assert second.from_store and second.store_ref == first.store_ref
    assert len(store.list_versions(first.store_ref)) == 1
    third = calc.run(snapshot(12), reuse=False, **opts)
    assert not third.from_store and third.store_ref != first.store_ref
    assert store.fetch(first.store_ref).store_ref == first.store_ref
    assert float(store.fetch(first.store_ref).value) == 6
    assert float(store.fetch(third.store_ref).value) == 12
    loaded = store.load(sim_signature=IDENTITY, calculator_signature_text=calc.to_signature().to_json())
    assert loaded.store_ref == third.store_ref
    assert calc.run(snapshot(6)).store_ref is None


def test_reference_json_roundtrip_preserves_version_and_typed_identity(store):
    ref = save(store, 6)
    restored = RecordRef.from_json(ref.to_json())
    assert restored == ref
    assert restored.version_id == ref.version_id
    assert restored.sim_signature == IDENTITY
    assert type(restored.sim_signature[2][1]) is bool
    assert float(store.fetch(restored).value) == 6
    address = RecordRef(IDENTITY, ref.calculator_signature_hash)
    assert RecordRef.from_json(address.to_json()) == address
    payload = json.loads(ref.to_json())
    payload["schema"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        RecordRef.from_json(json.dumps(payload))


@PropertyBase.dataclass
class MassArray(PropertyBase):
    def calculate(self, sim, params=None):
        return {"array": np.asarray(sim["mass"]).copy(), "nested": [1, 2]}


def test_returned_values_and_original_results_cannot_mutate_saved_history(store):
    calc = MassArray()
    original = calc.run(snapshot(6))
    ref = store.store(original, sim_signature=IDENTITY, calculator_signature=calc.to_signature())
    original.value["array"][0] = 111
    original.value["nested"].append(3)
    fetched = store.fetch(ref)
    fetched.value["array"][0] = 222
    fetched.value["nested"].clear()
    save(store, 12, calc)
    again = store.fetch(ref)
    np.testing.assert_array_equal(again.value["array"], [6, 0])
    assert again.value["nested"] == [1, 2]


@pytest.mark.parametrize("changes", [{"sim_signature": ("sim", "other")}, {"calculator_signature_hash": "0" * 64}])
def test_version_references_are_checked_against_record_identity(store, changes):
    ref = save(store, 6)
    assert ref.version_id
    with pytest.raises(SignatureMismatchError):
        store.fetch(replace(ref, **changes))
    assert not store.delete(replace(ref, **changes))
    assert float(store.fetch(ref).value) == 6


def test_missing_version_never_falls_back_to_latest(store):
    ref = save(store, 6)
    missing = replace(ref, version_id="not-a-saved-version")
    assert missing not in store
    with pytest.raises(KeyError):
        store.fetch(missing)


@pytest.mark.parametrize(
    "field", ["sim_signature", "calculator_signature_hash", "calculator_signature_text", "version_id"]
)
def test_invalid_record_identity_is_rejected_before_decode(store, monkeypatch, field):
    ref = save(store, 6)
    calc = ParamSum("mass")
    other = ResultRecord.from_result(
        calc.run(snapshot(6)), sim_signature=IDENTITY, calculator_signature=calc.to_signature(), codec=store.codec
    )
    other.version_id = ref.version_id
    changes = {
        "sim_signature": ("sim", "wrong"),
        "calculator_signature_hash": "0" * 64,
        "calculator_signature_text": ParamSum("pos").to_signature().to_json(),
        "version_id": "wrong-version",
    }
    setattr(other, field, changes[field])
    monkeypatch.setattr(store, "_get_version", lambda version_id: other)
    monkeypatch.setattr(store.codec, "decode", lambda value: pytest.fail("must validate before decoding"))
    with pytest.raises(SignatureMismatchError):
        store.fetch(ref)


@PropertyBase.dataclass
class FailingMass(PropertyBase):
    def calculate(self, sim, params=None):
        if float(sim["mass"][0]) < 0:
            raise RuntimeError("failed recomputation")
        return float(sim["mass"].sum())


def test_failed_refresh_preserves_previous_version_without_receipt(store):
    from pynbodyext.core.calculate.result.enums import ErrorPolicy
    from pynbodyext.core.calculate.runtime.options import RunOptions

    calc = FailingMass()
    opts = {"store": store, "sim_identity": lambda sim: IDENTITY}
    first = calc.run(snapshot(6), **opts)
    failed = calc.run(snapshot(-1), reuse=False, options=RunOptions(errors=ErrorPolicy.COLLECT), **opts)
    assert failed.errors and failed.store_ref is None
    assert store.list_versions(first.store_ref) == [first.store_ref]
    assert float(store.get(sim_signature=IDENTITY, calculator_signature=calc.to_signature()).value) == 6


def test_session_only_identity_has_no_persistent_receipt(tmp_path):
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    adapter = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'session.sqlite'}")
    try:
        calc, sim = ParamSum("mass"), snapshot(6)
        with pytest.warns(UserWarning, match="session-only"):
            result = calc.run(sim, store=adapter)
        assert result.store_ref is None
        assert adapter.list_refs() == []
        assert adapter.list_versions(RecordRef(("session", id(sim)), calc.to_signature().hash)) == []
    finally:
        adapter.engine.dispose()


@pytest.mark.parametrize("change", ["missing", "empty", "old-schema"])
def test_records_require_a_version_and_current_schema(change):
    payload = json.loads(ResultRecord().to_json())
    if change == "missing":
        del payload["version_id"]
    elif change == "empty":
        payload["version_id"] = ""
    else:
        payload["schema"] = "pynbodyext.result.store/v3"
    with pytest.raises(ValueError, match="version_id|schema"):
        ResultRecord.from_json(json.dumps(payload))


def test_failed_value_encoding_leaves_cache_and_history_unchanged(store):
    ref = save(store, 6)
    calc = ParamSum("mass")
    result = calc.run(snapshot(12))
    result.value = lambda: None  # cannot be pickled by the existing codec
    with pytest.raises(TypeError):
        store.store(result, sim_signature=IDENTITY, calculator_signature=calc.to_signature())
    assert store.list_versions(ref) == [ref]
    assert float(store.get(sim_signature=IDENTITY, calculator_signature=calc.to_signature()).value) == 6


def test_sqlite_reopen_reads_all_versions_and_receipts(tmp_path):
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    url = f"sqlite:///{tmp_path / 'reopen.sqlite'}"
    first_store = SQLAlchemyResultStore(url)
    try:
        first, second = save(first_store, 6), save(first_store, 12)
        saved_ref = first.to_json()
    finally:
        first_store.engine.dispose()
    reopened = SQLAlchemyResultStore(url)
    try:
        restored = RecordRef.from_json(saved_ref)
        assert float(reopened.fetch(restored).value) == 6
        assert reopened.list_versions(restored) == [first, second]
        assert reopened.fetch(restored).store_ref == first
    finally:
        reopened.engine.dispose()


def test_sqlite_cache_failure_rolls_back_new_version(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'rollback.sqlite'}")
    try:
        first = save(store, 6)

        def fail_cache_insert(conn, cursor, statement, parameters, context, executemany):
            if statement.startswith("INSERT INTO pynbodyext_stored_results"):
                raise RuntimeError("injected cache failure")

        sqlalchemy.event.listen(store.engine, "before_cursor_execute", fail_cache_insert)
        try:
            with pytest.raises(RuntimeError, match="injected"):
                save(store, 12)
        finally:
            sqlalchemy.event.remove(store.engine, "before_cursor_execute", fail_cache_insert)
        assert store.list_versions(first) == [first]
        assert float(store.fetch(RecordRef(IDENTITY, first.calculator_signature_hash)).value) == 6
        assert float(store.fetch(first).value) == 6
    finally:
        store.engine.dispose()


def test_concurrent_sqlite_saves_keep_both_returned_versions(tmp_path):
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    url = f"sqlite:///{tmp_path / 'concurrent.sqlite'}"
    first, second = SQLAlchemyResultStore(url), SQLAlchemyResultStore(url)
    barrier = Barrier(2)
    calc = ParamSum("mass")
    results = [calc.run(snapshot(6)), calc.run(snapshot(12))]

    def write(index, adapter):
        barrier.wait(timeout=10)
        return adapter.store(results[index], sim_signature=IDENTITY, calculator_signature=calc.to_signature())

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, 0, first), pool.submit(write, 1, second)]
            refs = [f.result(timeout=20) for f in futures]
        assert len(set(refs)) == 2
        assert float(first.fetch(refs[0]).value) == 6
        assert float(first.fetch(refs[1]).value) == 12
        assert len(first.list_versions(refs[0])) == 2
        assert len(first.list_refs()) == 1
    finally:
        first.engine.dispose()
        second.engine.dispose()
