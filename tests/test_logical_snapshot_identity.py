"""Logical dataset IDs survive relocation; revisions and loading semantics are explicit."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pynbody
import pytest

from pynbodyext.calculate import SampleTable
from pynbodyext.core.calculate import EvalEngine, snapshot_identity
from pynbodyext.core.calculate.store import InMemoryResultStore
from pynbodyext.properties import ParamSum


def api():
    from pynbodyext.calculate import ConfiguredSnapshotIdentity, LogicalSnapshotIdentity

    return LogicalSnapshotIdentity, ConfiguredSnapshotIdentity


def snapshot(path, masses=(1, 2, 3)):
    sim = pynbody.new(len(masses))
    sim["mass"] = np.asarray(masses, dtype=float)
    sim["pos"] = np.zeros((len(masses), 3))
    sim._filename = str(path)
    return sim


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryResultStore()
    else:
        pytest.importorskip("sqlalchemy")
        from pynbodyext.core.calculate.store import SQLAlchemyResultStore

        adapter = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'logical.sqlite'}")
        try:
            yield adapter
        finally:
            adapter.engine.dispose()


def test_relocated_snapshot_reuses_same_saved_version(store, tmp_path):
    Identity, Provider = api()
    identity = Identity("project", "simulation", "099")
    old_path, new_path = tmp_path / "old/snap", tmp_path / "new/snap"
    before = Provider({old_path: identity})
    after = Provider({new_path: identity})
    old, new = snapshot(old_path), snapshot(new_path)
    assert before(old) == after(new) == identity.signature()
    assert before(old) != snapshot_identity(old)
    calc = ParamSum("mass")
    first = calc.run(old, store=store, sim_identity=before)
    second = calc.run(new, store=store, sim_identity=after)
    assert second.from_store and second.value == first.value == 6
    assert first.store_ref == second.store_ref
    assert store.list_versions(first.store_ref) == [first.store_ref]


@pytest.mark.parametrize("field", ["project_id", "simulation_id", "timestep_id", "revision", "loader_id"])
def test_every_logical_field_separates_cache_keys(field, store, tmp_path):
    Identity, Provider = api()
    first_id = Identity("project", "simulation", "099")
    second_id = replace(first_id, **{field: "different"})
    path = tmp_path / "snap"
    sim = snapshot(path)
    calc = ParamSum("mass")
    first = calc.run(sim, store=store, sim_identity=Provider({path: first_id}))
    sim["mass"] *= 2
    second = calc.run(sim, store=store, sim_identity=Provider({path: second_id}))
    assert not second.from_store and second.value == 12
    assert first.store_ref.cache_key != second.store_ref.cache_key
    assert store.fetch(first.store_ref).value == 6


def test_revision_can_be_omitted_and_refresh_is_explicit(store, tmp_path):
    Identity, Provider = api()
    path = tmp_path / "snap"
    identity = Identity("project", "simulation", "099")
    assert identity.revision is None
    provider, sim, calc = Provider({path: identity}), snapshot(path), ParamSum("mass")
    first = calc.run(sim, store=store, sim_identity=provider)
    original_key = provider(sim)
    sim["mass"] *= 2
    assert provider(sim) == original_key
    cached = calc.run(sim, store=store, sim_identity=provider)
    assert cached.from_store and cached.value == 6
    refreshed = calc.run(sim, store=store, sim_identity=provider, reuse=False)
    assert not refreshed.from_store and refreshed.value == 12
    assert refreshed.store_ref != first.store_ref
    assert store.fetch(first.store_ref).value == 6


def test_subsets_order_and_halo_numbers_are_preserved_across_locations(tmp_path):
    Identity, Provider = api()
    identity = Identity("project", "simulation", "099", revision="r1")
    a, b = tmp_path / "a", tmp_path / "b"
    provider = Provider({a: identity, b: identity})
    first, replica = snapshot(a), snapshot(b)
    assert provider(first[[0, 2]]) == provider(replica[[0, 2]])
    assert provider(first[[0, 2]]) != provider(first[[2, 0]])
    assert provider(first[[0, 2]]) != provider(first[[0, 1]])
    assert provider(first[[0, 2]]) != provider(first)
    halo = first[[0, 2]]
    halo._halo_number = np.int64(7)
    assert provider(halo) == identity.signature() + ("halo", 7, snapshot_identity(halo)[-1])
    previous = provider(halo)
    halo._halo_number = np.int64(8)
    assert provider(halo) != previous


def test_external_transform_falls_back_until_reverted(store, tmp_path):
    Identity, Provider = api()
    path = tmp_path / "snap"
    provider = Provider({path: Identity("project", "simulation", "099")})
    sim = snapshot(path)
    initial = provider(sim)
    transformation = sim.translate([1, 0, 0])
    try:
        assert provider(sim) == ("session", id(sim))
        view = sim[[0, 1]]
        assert provider(view) == ("session", id(view))
        if store.persistent:
            with pytest.warns(UserWarning, match="session-only"):
                result = ParamSum("mass").run(sim, store=store, sim_identity=provider)
            assert result.store_ref is None and store.list_refs() == []
    finally:
        transformation.revert()
    assert provider(sim) == initial


def test_unconfigured_objects_retain_existing_fallback(tmp_path):
    Identity, Provider = api()
    provider = Provider({tmp_path / "registered": Identity("p", "s", "t")})
    unknown = snapshot(tmp_path / "unregistered")
    assert provider(unknown) == snapshot_identity(unknown)
    memory = pynbody.new(1)
    assert provider(memory) == ("session", id(memory))


def test_config_is_detached_and_identity_is_frozen(tmp_path):
    Identity, Provider = api()
    identity = Identity("project", "simulation", "099")
    path = tmp_path / "snap"
    config = {path: identity}
    provider = Provider(config)
    config.clear()
    assert provider(snapshot(path)) == identity.signature()
    with pytest.raises(FrozenInstanceError):
        identity.revision = "mutated"


def test_normalized_paths_cannot_have_conflicting_bindings(tmp_path, monkeypatch):
    Identity, Provider = api()
    monkeypatch.chdir(tmp_path)
    identity = Identity("p", "s", "t")
    provider = Provider({"snap": identity, tmp_path / "snap": identity})
    assert provider(snapshot(tmp_path / "snap")) == identity.signature()
    with pytest.raises(ValueError, match="conflict"):
        Provider({"snap": identity, tmp_path / "snap": replace(identity, timestep_id="other")})


@pytest.mark.parametrize("path", ["", "<created>", b"snap", "https://example.com/snap", "s3://bucket/snap"])
def test_invalid_locations_are_not_silently_registered(path):
    Identity, Provider = api()
    with pytest.raises((TypeError, ValueError)):
        Provider({path: Identity("p", "s", "t")})


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_id", ""),
        ("simulation_id", "   "),
        ("timestep_id", 99),
        ("revision", ""),
        ("revision", 1),
        ("loader_id", None),
        ("loader_id", ""),
    ],
)
def test_invalid_identity_fields_are_rejected(field, value):
    Identity, _ = api()
    kwargs = dict(project_id="p", simulation_id="s", timestep_id="t")
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        Identity(**kwargs)


def test_literal_ids_are_preserved_and_revision_none_is_not_string_none():
    Identity, _ = api()
    identity = Identity(" P ", "s", "001")
    assert identity.signature() == ("sim", "pynbodyext.logical-snapshot/v1", " P ", "s", "001", None, "default")
    assert replace(identity, revision="None").signature() != identity.signature()


def test_config_roundtrip_and_fresh_process_stability(tmp_path):
    Identity, Provider = api()
    path = tmp_path / "snap"
    provider = Provider({path: Identity("项目", "simulation", "099", loader_id="physical-v1")})
    saved = provider.to_json()
    assert Provider.from_json(saved).to_json() == saved
    assert Provider.from_json(saved)(snapshot(path)) == provider(snapshot(path))
    program = """
import json, sys
from types import SimpleNamespace
from pynbodyext.calculate import ConfiguredSnapshotIdentity
provider = ConfiguredSnapshotIdentity.from_json(sys.argv[1])
print(json.dumps(provider(SimpleNamespace(filename=sys.argv[2]))))
"""
    fresh = subprocess.check_output([sys.executable, "-c", program, saved, str(path)], text=True)
    assert tuple(json.loads(fresh)) == provider(snapshot(path))


@pytest.mark.parametrize("change", ["schema", "missing", "extra", "relative", "conflict", "field", "identity-extra"])
def test_invalid_json_is_rejected_before_use(tmp_path, change):
    Identity, Provider = api()
    data = json.loads(Provider({tmp_path / "snap": Identity("p", "s", "t")}).to_json())
    if change == "schema":
        data["schema"] = "unknown"
    elif change == "missing":
        del data["entries"]
    elif change == "extra":
        data["extra"] = True
    elif change == "relative":
        data["entries"][0]["path"] = "relative/snap"
    elif change == "conflict":
        extra = json.loads(json.dumps(data["entries"][0]))
        extra["identity"]["timestep_id"] = "different"
        data["entries"].append(extra)
    elif change == "field":
        data["entries"][0]["identity"]["revision"] = 1
    else:
        data["entries"][0]["identity"]["revizion"] = "typo"
    with pytest.raises((TypeError, ValueError)):
        Provider.from_json(json.dumps(data))


def test_identity_lookup_does_not_read_snapshot_files(tmp_path, monkeypatch):
    Identity, Provider = api()
    sim = snapshot(tmp_path / "does-not-exist")

    def forbidden(*args, **kwargs):
        pytest.fail("identity lookup must not open files")

    monkeypatch.setattr("builtins.open", forbidden)
    identity = Identity("p", "s", "t")
    provider = Provider({tmp_path / "does-not-exist": identity})
    assert provider(sim) == identity.signature()
    assert Provider.from_json(provider.to_json())(sim) == identity.signature()


def test_engine_and_sample_table_use_existing_identity_seam(store, tmp_path):
    Identity, Provider = api()
    a, b = tmp_path / "a", tmp_path / "b"
    provider = Provider({a: Identity("p", "s", "t"), b: Identity("p", "s", "t")})
    calc = ParamSum("mass")
    first = EvalEngine(sim_identity=provider).run(calc, snapshot(a), store=store)
    table = SampleTable({"mass": calc}, store=store, sim_identity=provider)
    table.compute([snapshot(b)])
    assert table.column("mass")[0] == first.value
    assert store.list_versions(first.store_ref) == [first.store_ref]


def test_sqlite_reopen_with_relocated_config(tmp_path):
    Identity, Provider = api()
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    url = f"sqlite:///{tmp_path / 'reopen.sqlite'}"
    a, b = tmp_path / "a", tmp_path / "b"
    identity = Identity("p", "s", "t")
    saved_config = Provider({b: identity}).to_json()
    first_store = SQLAlchemyResultStore(url)
    try:
        first = ParamSum("mass").run(snapshot(a), store=first_store, sim_identity=Provider({a: identity}))
    finally:
        first_store.engine.dispose()
    reopened = SQLAlchemyResultStore(url)
    try:
        second = ParamSum("mass").run(snapshot(b), store=reopened, sim_identity=Provider.from_json(saved_config))
        assert second.from_store and second.store_ref == first.store_ref
    finally:
        reopened.engine.dispose()


def test_sample_table_default_labels_identify_logical_rows(tmp_path):
    Identity, Provider = api()
    a, b, alias = tmp_path / "a", tmp_path / "b", tmp_path / "alias"
    first_id = Identity("project", "simulation", "001")
    second_id = replace(first_id, timestep_id="002")
    table = SampleTable({"mass": ParamSum("mass")}, sim_identity=Provider({a: first_id, b: second_id, alias: first_id}))
    first = table.add(snapshot(a))
    second = table.add(snapshot(b, masses=(10, 20, 30)))
    assert first.name != second.name
    assert "001" in first.name and "002" in second.name
    assert table.row(first.name)["mass"] == 6
    assert table.row(second.name)["mass"] == 60
    table.add(snapshot(alias))
    assert len(table.rows) == 2


def test_sample_table_labels_include_logical_fields_and_view_suffixes(tmp_path):
    Identity, Provider = api()
    identity = Identity("halo", "s", "t")  # project label must not be parsed as a view tag
    variants = [identity, replace(identity, revision="None"), replace(identity, loader_id="physical")]
    paths = [tmp_path / str(index) for index in range(len(variants))]
    provider = Provider(dict(zip(paths, variants)))
    store = InMemoryResultStore()
    table = SampleTable({"mass": ParamSum("mass")}, store=store, sim_identity=provider)
    root = snapshot(paths[0])
    rows = [table.add(root)] + [table.add(snapshot(path)) for path in paths[1:]]
    subset = root[[0, 1]]
    rows.append(table.add(subset))
    halo = root[[0, 1]]
    halo._halo_number = 7
    rows.append(table.add(halo))
    assert len({row.name for row in rows}) == 5
    assert ":subset " in rows[-2].name
    assert ":halo 7" in rows[-1].name
    restored = SampleTable.from_store(store, {"mass": ParamSum("mass")})
    assert [row.name for row in restored.rows] == [row.name for row in rows]
