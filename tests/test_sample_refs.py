"""Sample cells retain exact saved versions and explicit backend ownership."""

from __future__ import annotations

import json
from dataclasses import replace

import pynbody
import pytest

from pynbodyext.core.calculate import InMemoryResultStore, LambdaProperty, PropertyBase, SampleTable
from pynbodyext.core.calculate.sample import SAMPLE_SCHEMA, SampleRow
from pynbodyext.core.calculate.store import RecordRef
from pynbodyext.core.calculate.store.codec import ValueCodec
from pynbodyext.core.calculate.store.identity import sim_identity_key
from pynbodyext.properties import ParamSum

IDENTITY = ("sim", "sample-reference-fixture")


def identity(sim):
    return IDENTITY


def snapshot(value=6):
    sim = pynbody.new(2)
    sim["mass"] = [float(value), 0.0]
    return sim


@PropertyBase.dataclass
class PositiveMass(PropertyBase[float]):
    signature_name = "tests.sample_refs.PositiveMass"

    def calculate(self, sim, params=None):
        value = float(sim["mass"].sum())
        if value < 0:
            raise RuntimeError("negative mass")
        return value


class PersistentMemoryStore(InMemoryResultStore):
    persistent = True


def test_sample_row_refs_are_final_defaulted_independent_field():
    first = SampleRow("first", IDENTITY, {"mass": 6}, {})
    second = SampleRow("second", IDENTITY)
    assert first.refs == second.refs == {}
    first.refs["mass"] = RecordRef(IDENTITY, "a" * 64, version_id="saved")
    assert second.refs == {}


def test_sample_keeps_original_receipt_after_refresh_reopen_and_invalidation(tmp_path):
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    url = f"sqlite:///{tmp_path / 'sample.sqlite'}"
    store = SQLAlchemyResultStore(url)
    calc = ParamSum("mass")
    table = SampleTable({"mass": calc}, store=store, store_name="results", sim_identity=identity)
    try:
        row = table.add(snapshot(6))
        pinned = row.refs["mass"]
        refreshed = calc.run(snapshot(12), store=store, sim_identity=identity, reuse=False)
        table.save(tmp_path / "sample.json")
    finally:
        store.engine.dispose()
    reopened = SQLAlchemyResultStore(url)
    try:
        assert reopened.delete(refreshed.store_ref)
        loaded = SampleTable.load(tmp_path / "sample.json", store=reopened)
        assert loaded.store_name == "results"
        assert loaded.rows[0].refs["mass"] == pinned
        assert float(loaded.rows[0]["mass"]) == 6
        assert float(reopened.fetch(pinned).value) == 6
        assert not reopened.has(sim_signature=IDENTITY, calculator_signature=calc.to_signature())
    finally:
        reopened.engine.dispose()


def test_cache_hit_and_fill_capture_exact_actual_receipt():
    store, calc = InMemoryResultStore(), ParamSum("mass")
    table = SampleTable({"mass": calc}, store=store, sim_identity=identity)
    row = table.add(snapshot())
    pinned = row.refs["mass"]
    table.add(snapshot(12))
    assert row.refs["mass"] == pinned and float(row["mass"]) == 6
    filled = SampleTable.from_store(store, {"mass": calc}, store_name="results")
    assert filled.rows[0].refs["mass"] == pinned
    assert filled.store_name == "results"
    row.values["mass"] = None
    row.errors["mass"] = "previous failure"
    refreshed = calc.run(snapshot(12), store=store, sim_identity=identity, reuse=False)
    table.fill_from_store()
    assert row.refs["mass"] == refreshed.store_ref
    assert float(row["mass"]) == 12 and row.errors == {}
    calc.run(snapshot(18), store=store, sim_identity=identity, reuse=False)
    table.fill_from_store()  # a populated historical cell is left intact
    assert row.refs["mass"] == refreshed.store_ref
    assert float(row["mass"]) == 12


@pytest.mark.parametrize("errors", ["record", "raise"])
def test_failed_recompute_clears_old_receipt_even_when_raising(errors):
    table = SampleTable({"mass": PositiveMass()}, store=InMemoryResultStore(), sim_identity=identity)
    row = table.add(snapshot())
    assert row.refs["mass"].version_id
    if errors == "raise":
        with pytest.raises(RuntimeError, match="negative mass"):
            table.add(snapshot(-1), reuse=False, errors=errors)
    else:
        table.add(snapshot(-1), reuse=False, errors=errors)
        assert row["mass"] is None and "mass" in row.errors
    assert row.refs == {}


def test_without_store_recompute_clears_previous_receipt():
    table = SampleTable({"mass": ParamSum("mass")}, store=InMemoryResultStore(), sim_identity=identity)
    row = table.add(snapshot())
    assert row.refs
    table.store = None
    table.add(snapshot(12), reuse=False)
    assert float(row["mass"]) == 12 and row.refs == {}


def test_session_only_persistent_recompute_clears_previous_receipt():
    sim = snapshot()
    table = SampleTable({"mass": ParamSum("mass")}, store=InMemoryResultStore())
    row = table.add(sim)
    assert row.refs  # the memory backend can store session identities
    table.store = PersistentMemoryStore()
    table.add(sim, reuse=False)
    assert float(row["mass"]) == 6 and row.refs == {}


def test_unstable_calculator_persistent_recompute_clears_previous_receipt():
    table = SampleTable({"mass": LambdaProperty(lambda sim: 6.0)}, store=InMemoryResultStore(), sim_identity=identity)
    row = table.add(snapshot())
    assert row.refs
    table.store = PersistentMemoryStore()
    with pytest.warns(UserWarning, match="not stored"):
        table.add(snapshot(), reuse=False)
    assert row.refs == {}


def test_filter_and_exports_preserve_distinct_typed_rows_with_duplicate_names(tmp_path):
    pytest.importorskip("pandas")
    objects = [snapshot(6), snapshot(12), snapshot(18), snapshot(24)]
    identities = [("sim", part) for part in (1, True, 1.0, "1")]
    lookup = {id(sim): given for sim, given in zip(objects, identities, strict=True)}
    table = SampleTable(
        {"mass": ParamSum("mass")},
        store=InMemoryResultStore(),
        store_name="science",
        sim_identity=lambda sim: lookup[id(sim)],
    ).compute(objects, names=["duplicate"] * 4)
    selected = table.where(lambda row: True)
    assert selected.store_name == "science" and selected.store is table.store
    assert [row.refs for row in selected] == [row.refs for row in table]
    path = tmp_path / "typed.json"
    selected.save(path)
    loaded = SampleTable.load(path)
    assert loaded.store is None and loaded.store_name == "science"
    assert len(loaded) == 4
    assert [type(row.identity[1]) for row in loaded] == [int, bool, float, str]
    assert [row.refs for row in loaded] == [row.refs for row in table]
    df = loaded.to_dataframe()
    assert list(df.index) == ["duplicate"] * 4
    meta = df.attrs["pynbodyext"]
    keys = [sim_identity_key(given) for given in identities]
    assert meta["store_name"] == "science"
    assert meta["identity_keys"] == keys
    assert set(meta["refs"]) == set(meta["identities"]) == set(keys)
    for key, row in zip(keys, table, strict=True):
        assert RecordRef.from_json(meta["refs"][key]["mass"]) == row.refs["mass"]
        assert sim_identity_key(ValueCodec().decode(meta["identities"][key])) == key
    json.dumps(df.attrs)


@pytest.mark.parametrize("change", ["identity", "hash", "unpinned", "column"])
def test_import_rejects_reference_not_pinned_to_the_cell(change):
    table = SampleTable({"mass": ParamSum("mass")}, store=InMemoryResultStore(), sim_identity=identity)
    row = table.add(snapshot())
    data = table.to_dict()
    ref = row.refs["mass"]
    label = "mass"
    if change == "identity":
        ref = replace(ref, sim_signature=("sim", True))
        data["rows"][0]["identity"] = ValueCodec().encode(("sim", 1))
    elif change == "hash":
        ref = replace(ref, calculator_signature_hash="0" * 64)
    elif change == "unpinned":
        ref = replace(ref, version_id=None)
    else:
        label = "absent"
    data["rows"][0]["refs"] = {label: ref.to_json()}
    with pytest.raises(ValueError, match="reference"):
        SampleTable.from_dict(data)


class UnavailableStore(InMemoryResultStore):
    def get(self, **kwargs):
        pytest.fail("loading must not query a backend")

    def fetch(self, *args, **kwargs):
        pytest.fail("loading must not fetch a result")


def test_failed_fill_clears_stale_receipt_and_preserves_existing_diagnostics():
    table = SampleTable({"mass": ParamSum("mass")}, store=InMemoryResultStore(), sim_identity=identity)
    row = table.add(snapshot())
    assert row.refs
    row.values["mass"] = None
    row.errors["mass"] = "existing diagnostic"

    class FailedStore(InMemoryResultStore):
        def get(self, **kwargs):
            raise OSError("backend unavailable")

    table.store = FailedStore()
    with pytest.raises(OSError, match="backend unavailable"):
        table.fill_from_store()
    assert row.refs == {}
    assert row.values == {"mass": None}
    assert row.errors == {"mass": "existing diagnostic"}


def test_loading_restores_receipts_without_backend_access_or_implicit_binding(tmp_path):
    table = SampleTable({"mass": ParamSum("mass")}, store=InMemoryResultStore(), sim_identity=identity)
    table.add(snapshot())
    table.save(tmp_path / "sample.json")
    loaded = SampleTable.load(tmp_path / "sample.json")
    assert loaded.store is None and loaded.store_name is None
    assert loaded.rows[0].refs == table.rows[0].refs
    unavailable = UnavailableStore()
    explicitly_bound = SampleTable.load(tmp_path / "sample.json", store=unavailable)
    assert explicitly_bound.store is unavailable
    assert explicitly_bound.rows[0].refs == table.rows[0].refs


def test_legacy_v1_load_retains_values_without_inventing_receipts():
    table = SampleTable({"mass": ParamSum("mass")}, store=InMemoryResultStore(), sim_identity=identity)
    table.add(snapshot())
    data = table.to_dict()
    assert SAMPLE_SCHEMA == "pynbodyext.sample.table/v2"
    data["schema"] = "pynbodyext.sample.table/v1"
    data.pop("store_name")
    for item in data["rows"]:
        item.pop("refs")
    loaded = SampleTable.from_dict(data, store=table.store)
    assert float(loaded.rows[0]["mass"]) == 6
    assert loaded.rows[0].refs == {} and loaded.store_name is None
