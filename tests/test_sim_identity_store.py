"""Distinct typed simulation inputs must never share results or sample rows."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pynbody
import pytest

from pynbodyext.core.calculate import SampleTable
from pynbodyext.core.calculate.store import InMemoryResultStore, RecordRef, ResultRecord
from pynbodyext.properties import ParamSum


IDENTITIES = [
    ("sim", 1), ("sim", "1"), ("sim", True), ("sim", 1.0),
    ("sim", None), ("sim", "None"), ("sim", (1, "a,b")),
    ("sim", "(1, 'a,b')"), ("sim", ("1", "a,b")),
]


def snapshot(value):
    sim = pynbody.new(1)
    sim["mass"] = [float(value)]
    return sim


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryResultStore()
    else:
        pytest.importorskip("sqlalchemy")
        from pynbodyext.core.calculate.store import SQLAlchemyResultStore

        adapter = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'typed.sqlite'}")
        try:
            yield adapter
        finally:
            adapter.engine.dispose()


def test_refs_distinguish_identity_types():
    refs = [RecordRef(identity, "a" * 64) for identity in IDENTITIES]
    assert len({ref.key for ref in refs}) == len(IDENTITIES)
    assert len(set(refs)) == len(IDENTITIES)


def test_stores_isolate_readthrough_refresh_lists_and_deletion(store):
    calc = ParamSum("mass")
    refs = []
    for index, identity in enumerate(IDENTITIES):
        result = calc.run(snapshot(index + 1), store=store, sim_identity=lambda sim: identity)
        assert not result.from_store
        assert float(result.value) == index + 1
        refs.append(RecordRef(identity, calc.to_signature().hash))

    assert len(store.list_refs()) == len(IDENTITIES)
    assert [type(sig[1]) for sig in store.sim_signatures()] == [type(sig[1]) for sig in IDENTITIES]
    for index, ref in enumerate(refs):
        assert store.has(sim_signature=ref.sim_signature, calculator_signature=calc.to_signature())
        assert ref in store
        assert store.list_refs(sim_signature=ref.sim_signature) == [ref]
        assert float(store.fetch(ref).value) == index + 1
        loaded = store.load(sim_signature=ref.sim_signature, calculator_signature_text=calc.to_signature().to_json())
        assert float(loaded.value) == index + 1
        hit = calc.run(snapshot(100), store=store, sim_identity=lambda sim: ref.sim_signature)
        assert hit.from_store and float(hit.value) == index + 1

    calc.run(snapshot(99), store=store, sim_identity=lambda sim: IDENTITIES[0], reuse=False)
    assert float(store.fetch(refs[0]).value) == 99
    assert float(store.fetch(refs[1]).value) == 2
    assert store.delete(refs[0])
    assert not store.delete(refs[0])
    assert refs[0] not in store
    assert store.list_refs(sim_signature=IDENTITIES[0]) == []
    assert len(store.list_refs()) == len(IDENTITIES) - 1
    assert float(store.fetch(refs[2]).value) == 3


def test_sqlite_reopen_preserves_typed_identities(tmp_path):
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    url = f"sqlite:///{tmp_path / 'reopen.sqlite'}"
    calc = ParamSum("mass")
    original = SQLAlchemyResultStore(url)
    try:
        for index, identity in enumerate(IDENTITIES):
            original.store(calc.run(snapshot(index)), sim_signature=identity, calculator_signature=calc.to_signature())
    finally:
        original.engine.dispose()
    reopened = SQLAlchemyResultStore(url)
    try:
        assert len(reopened.list_refs()) == len(IDENTITIES)
        for index, identity in enumerate(IDENTITIES):
            refs = reopened.list_refs(sim_signature=identity)
            assert len(refs) == 1
            assert type(refs[0].sim_signature[1]) is type(identity[1])
            assert float(reopened.fetch(refs[0]).value) == index
    finally:
        reopened.engine.dispose()


def test_sample_table_keeps_typed_rows_across_all_entry_points(store, tmp_path):
    objects = [snapshot(i + 1) for i in range(len(IDENTITIES))]
    identities = {id(obj): identity for obj, identity in zip(objects, IDENTITIES)}
    table = SampleTable({"mass": ParamSum("mass")}, store=store, sim_identity=lambda obj: identities[id(obj)])
    table.compute(objects, names=[f"object-{i}" for i in range(len(objects))])
    assert len(table) == len(IDENTITIES)
    np.testing.assert_array_equal(table.column("mass"), np.arange(1, len(IDENTITIES) + 1))
    table.add(objects[2])
    assert len(table) == len(IDENTITIES)
    assert len(table.where(lambda row: type(row.identity[1]) is bool)) == 1

    path = tmp_path / "table.json"
    table.save(path)
    recovered = SampleTable.load(path)
    assert len(recovered) == len(IDENTITIES)
    assert [type(row.identity[1]) for row in recovered] == [type(identity[1]) for identity in IDENTITIES]
    np.testing.assert_array_equal(recovered.column("mass"), table.column("mass"))
    from_store = SampleTable.from_store(store, {"mass": ParamSum("mass")})
    assert len(from_store) == len(IDENTITIES)
    np.testing.assert_array_equal(from_store.column("mass"), table.column("mass"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_identity_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        _ = RecordRef(("sim", (value,)), "a" * 64).key


@pytest.mark.parametrize("value", [[1], {"x": 1}, {1}, b"id", object(), np.int64(1)])
def test_unsupported_identity_is_rejected(value):
    with pytest.raises(TypeError, match="identity"):
        _ = RecordRef(("sim", value), "a" * 64).key


def test_signed_float_zero_is_one_identity():
    assert RecordRef(("sim", 0.0), "a" * 64) == RecordRef(("sim", -0.0), "a" * 64)


@pytest.mark.parametrize("schema", [None, "pynbodyext.result.store/v2", "future"])
def test_unknown_or_missing_record_schema_is_rejected(schema):
    data = json.loads(ResultRecord().to_json())
    if schema is None:
        del data["schema"]
    else:
        data["schema"] = schema
    with pytest.raises(ValueError, match="schema"):
        ResultRecord.from_json(json.dumps(data))


def test_identity_keys_are_stable_in_another_process():
    identity = ("sim", "银河", (None, True, 17, 1.25, "a::b"))
    expected = RecordRef(identity, "a" * 64).key
    code = (
        "from pynbodyext.core.calculate.store import RecordRef; "
        f"print(RecordRef({identity!r}, 'a' * 64).key)"
    )
    output = subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": "89"}, text=True)
    assert output.strip() == expected
