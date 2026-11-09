"""Tests for the SQLAlchemy-backed result store (the tangos-like relational adapter).

The adapter implements the same four primitives (``_put``, ``_get_by_key``,
``_delete``, ``_list``) as the in-memory reference, so the guaranteed behaviours
— round-trip on the structured signature, persistence across engine reopen,
delete, and simulation-filtered listing — must hold identically.
"""

from __future__ import annotations

import numpy as np
import pynbody
import pytest

pytest.importorskip("sqlalchemy")

from pynbodyext.core.calculate import Pipeline, PropertyBase
from pynbodyext.core.calculate.result.signature import calculator_to_signature
from pynbodyext.core.calculate.store import ValueCodec


def make_sim() -> pynbody.SimSnap:
    sim = pynbody.new(6)
    sim["mass"] = np.arange(6.0)
    sim["r"] = np.arange(6.0)
    return sim


@PropertyBase.dataclass
class MassSum(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(sim["mass"].sum())


def make_result():
    calc = Pipeline({"m": MassSum()}, name="p")
    sim = make_sim()
    return calc, sim, calc.run(sim)


def test_sql_store_roundtrips_result():
    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    calc, sim, result = make_result()
    structured = calculator_to_signature(calc)
    store = SQLAlchemyResultStore("sqlite:///:memory:", codec=ValueCodec())

    ref = store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)
    assert store.has(sim_signature=("sim", "snap_103"), calculator_signature=structured)

    fetched = store.fetch(ref)
    np.testing.assert_allclose(dict(fetched.value)["m"], dict(result.value)["m"])
    assert fetched.provenance is not None
    assert fetched.provenance.calculator_signature_hash == structured.hash

    # a different sim identity is a miss
    assert not store.has(sim_signature=("sim", "other"), calculator_signature=structured)


def test_sql_store_persists_across_engine_reopen(tmp_path):
    """A result survives closing the engine and reopening the same database file."""
    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    calc, sim, result = make_result()
    structured = calculator_to_signature(calc)
    db = tmp_path / "results.sqlite"

    store = SQLAlchemyResultStore(f"sqlite:///{db}", codec=ValueCodec())
    store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)

    store2 = SQLAlchemyResultStore(f"sqlite:///{db}", codec=ValueCodec())
    found = store2.get(sim_signature=("sim", "snap_103"), calculator_signature=structured)
    assert found is not None
    np.testing.assert_allclose(dict(found.value)["m"], dict(result.value)["m"])


def test_sql_store_delete():
    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    calc, sim, result = make_result()
    structured = calculator_to_signature(calc)
    store = SQLAlchemyResultStore("sqlite:///:memory:", codec=ValueCodec())

    ref = store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)
    assert store.has(sim_signature=("sim", "snap_103"), calculator_signature=structured)
    assert store.delete(ref)
    assert not store.has(sim_signature=("sim", "snap_103"), calculator_signature=structured)


def test_engine_run_persists_to_sql_store_and_reloads(tmp_path):
    """Full flow: run through the engine, persist to SQLite, reload in a new process.

    This is the tangos-like end-to-end path — the calculator runs, the result is
    stored keyed on the pluggable sim identity + content-addressed signature, and
    a fresh engine reading the same database file reconstructs it.
    """
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    db = tmp_path / "results.sqlite"
    engine = EvalEngine(sim_identity=path_identity)

    store = SQLAlchemyResultStore(f"sqlite:///{db}", codec=ValueCodec())
    result = engine.run(MassSum(), make_sim(), store=store)
    structured = calculator_to_signature(MassSum())

    # A fresh store over the same file recovers the stored result.
    store2 = SQLAlchemyResultStore(f"sqlite:///{db}", codec=ValueCodec())
    found = store2.get(sim_signature=("sim", "snapshot_103", "halo_0"), calculator_signature=structured)
    assert found is not None
    assert found.value == result.value
    assert found.provenance is not None
    assert found.provenance.calculator_signature_hash == structured.hash


def test_sql_store_load_reconstructs_calculator_from_text(tmp_path):
    """End-to-end load: run, persist, then re-materialize the calculator from its text.

    The stored signature text round-trips through :meth:`ResultStore.load`, which
    rebuilds a runnable calculator and pairs it with the stored result — the full
    tangos-like save/load cycle.
    """
    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    db = tmp_path / "results.sqlite"
    store = SQLAlchemyResultStore(f"sqlite:///{db}", codec=ValueCodec())
    result = MassSum().run(make_sim(), store=store, sim_identity=path_identity)
    structured = calculator_to_signature(MassSum())

    store2 = SQLAlchemyResultStore(f"sqlite:///{db}", codec=ValueCodec())
    loaded = store2.load(
        sim_signature=("sim", "snapshot_103", "halo_0"),
        calculator_signature_text=structured.to_json(),
    )
    assert loaded is not None
    assert isinstance(loaded.calculator, MassSum)
    assert loaded.value == result.value
    assert loaded.calculator.run(make_sim()).value == result.value

    assert store2.load(
        sim_signature=("sim", "other"),
        calculator_signature_text=structured.to_json(),
    ) is None


def test_sql_store_lists_by_sim():
    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    calc, sim, result = make_result()
    structured = calculator_to_signature(calc)
    store = SQLAlchemyResultStore("sqlite:///:memory:", codec=ValueCodec())

    store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)
    store.store(result, sim_signature=("sim", "snap_104"), calculator_signature=structured)

    refs = store.list_refs(sim_signature=("sim", "snap_103"))
    assert len(refs) == 1
    assert refs[0].sim_signature == ("sim", "snap_103")

    all_refs = store.list_refs()
    assert len(all_refs) == 2


@PropertyBase.dataclass
class TypedPayload(PropertyBase[dict]):
    def calculate(self, sim, params=None):
        return {
            None: np.int64(3),
            "None": np.float32(1.5),
            (1, "star", (False, None)): np.uint64(2**64 - 1),
            ("1", "star", (False, None)): np.bool_(False),
        }


def test_sql_store_reopen_preserves_scalar_dtypes_and_dictionary_keys(tmp_path):
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    url = f"sqlite:///{tmp_path / 'typed.sqlite'}"
    calc = TypedPayload()
    sim = make_sim()
    expected = calc.run(sim).value
    store = SQLAlchemyResultStore(url)
    try:
        ref = store.store(
            calc.run(sim), sim_signature=("sim", "typed-snapshot"),
            calculator_signature=calc.to_signature(),
        )
    finally:
        store.engine.dispose()

    reopened = SQLAlchemyResultStore(url)
    try:
        actual = reopened.fetch(ref).value
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert type(actual[key]) is type(value)
            assert actual[key].dtype == value.dtype
            np.testing.assert_equal(actual[key], value)
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize("shape", [(2,), (), (0, 2), (2, 0, 3)])
def test_object_arrays_survive_sqlite_reopen(tmp_path, shape):
    from pynbody.array import SimArray

    from pynbodyext.core.calculate.store.sqlalchemy_store import SQLAlchemyResultStore

    calc, sim, result = make_result()
    value = np.empty(shape, dtype=object)
    for index in range(value.size):
        value.flat[index] = {"nested": (index, [1, 2])}
    value = SimArray(value, units="Msol")
    result.value = value
    url = f"sqlite:///{tmp_path / 'objects.sqlite'}"
    store = SQLAlchemyResultStore(url)
    ref = store.store(result, sim_signature=("sim", "object-array"), calculator_signature=calculator_to_signature(calc))
    store.engine.dispose()
    reopened = SQLAlchemyResultStore(url)
    restored = reopened.fetch(ref).value
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert restored.tolist() == value.tolist()
    assert str(restored.units) == str(value.units)
    reopened.engine.dispose()
