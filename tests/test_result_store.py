"""Tests for the result-store seam (codec + store interface + in-memory adapter).

This is the persistence boundary the future database-backed store plugs into:
a :class:`ValueCodec` turns arbitrary result values (including pynbody arrays
with units) into a portable dict, and a :class:`ResultStore` persists a
:class:`~pynbodyext.core.calculate.result.result.Result` keyed by the
structured calculator signature.
"""

from __future__ import annotations

import numpy as np
import pynbody

from pynbodyext.core.calculate import Param, Pipeline, PropertyBase
from pynbodyext.core.calculate.result.signature import calculator_to_signature


def make_sim() -> pynbody.SimSnap:
    sim = pynbody.new(6)
    sim["mass"] = np.arange(6.0)
    sim["temp"] = np.arange(6.0) * 10.0
    sim["r"] = np.arange(6.0)
    return sim


@PropertyBase.dataclass
class MassSum(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(sim["mass"].sum())


def make_result():
    from pynbodyext.core.calculate import FilterBase

    @FilterBase.dataclass
    class RBelow(FilterBase):
        radius: Param[float] = Param(field_name="r")

        def calculate(self, sim, params=None) -> bool:
            return sim["r"] < self.radius

    calc = Pipeline({"m": MassSum().filter(RBelow(5.0))}, name="p")
    sim = make_sim()
    return calc, sim, calc.run(sim)


def test_value_codec_roundtrips_scalars_and_nested():
    from pynbodyext.core.calculate.store.codec import ValueCodec

    codec = ValueCodec()
    for value in (None, True, 3, 3.14, "hello", b"\x00\x01"):
        assert codec.decode(codec.encode(value)) == value
    nested = {"a": [1, 2, 3], "b": ("x", "y"), "c": None, "d": {1: "one"}}
    assert codec.decode(codec.encode(nested)) == nested


def test_value_codec_roundtrips_numpy_and_simarray_with_units():
    from pynbody.array import SimArray

    from pynbodyext.core.calculate.store.codec import ValueCodec

    codec = ValueCodec()
    arr = np.arange(6.0)
    recovered = codec.decode(codec.encode(arr))
    assert isinstance(recovered, np.ndarray)
    np.testing.assert_array_equal(recovered, arr)

    # A SimArray carrying a genuine, named unit preserves that unit.
    mass = SimArray(np.arange(6.0), units="Msol")
    recovered = codec.decode(codec.encode(mass))
    assert isinstance(recovered, np.ndarray)
    np.testing.assert_array_equal(recovered, np.asarray(mass))
    assert str(recovered.units) == "Msol"

    # Dimensionless (NoUnit) arrays normalise to a plain ndarray: the unit is
    # meaningless, so the codec drops it rather than emitting an unrereconstructable
    # units string.
    plain = make_sim()["mass"]
    recovered = codec.decode(codec.encode(plain))
    assert isinstance(recovered, np.ndarray) and not hasattr(recovered, "units")


def test_value_codec_content_hash_is_stable():
    from pynbodyext.core.calculate.store.codec import ValueCodec

    codec = ValueCodec()
    arr = np.arange(6.0)
    a = codec.content_hash(arr)
    b = codec.content_hash(np.arange(6.0))
    assert a == b
    assert codec.content_hash(np.arange(6.0) + 1) != a


def test_store_roundtrips_result_keyed_on_signature():
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec

    calc, sim, result = make_result()
    sig = calc.signature()
    assert sig  # non-empty fake identity is fine; structured signature is the key

    structured = calculator_to_signature(calc)
    store = InMemoryResultStore(codec=ValueCodec())
    ref = store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)

    assert store.has(sim_signature=("sim", "snap_103"), calculator_signature=structured)
    fetched = store.fetch(ref)
    np.testing.assert_allclose(dict(fetched.value)["m"], dict(result.value)["m"])
    assert fetched.provenance is not None
    assert fetched.provenance.calculator_signature_hash == structured.short_hash()

    # lookup without a ref works too
    found = store.get(sim_signature=("sim", "snap_103"), calculator_signature=structured)
    assert found is not None
    np.testing.assert_allclose(dict(found.value)["m"], dict(result.value)["m"])

    # different sim identity or calculator -> miss
    assert not store.has(sim_signature=("sim", "other"), calculator_signature=structured)
    assert store.get(sim_signature=("sim", "snap_103"), calculator_signature=calculator_to_signature(MassSum())) is None


def test_store_is_injected_with_identity_from_engine():
    """A store keyed on a stable (address-based) sim identity survives reloads."""
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    engine = EvalEngine(sim_identity=path_identity)
    sim = make_sim()
    calc = MassSum()
    result = engine.run(calc, sim)
    structured = calculator_to_signature(calc)

    store = InMemoryResultStore(codec=ValueCodec())
    store.store(result, sim_signature=engine.make_sim_signature(sim), calculator_signature=structured)
    assert store.has(sim_signature=("sim", "snapshot_103", "halo_0"), calculator_signature=structured)
    assert not store.has(sim_signature=("sim", id(sim)), calculator_signature=structured)


def test_stored_key_is_content_addressed_from_signature():
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec, compute_calculator_key

    calc, sim, result = make_result()
    structured = calculator_to_signature(calc)
    store = InMemoryResultStore(codec=ValueCodec())
    ref = store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)

    key_hash = compute_calculator_key(structured)[0]
    assert ref.calculator_signature_hash == key_hash == structured.short_hash()
    assert store.get(sim_signature=("sim", "snap_103"), calculator_signature=structured) is not None


def test_engine_run_stores_result_when_store_provided():
    """``engine.run(calc, sim, store=...)`` persists the result automatically.

    The store seam must be reachable from the actual execution path: the engine
    derives ``sim_signature`` from the injected identity provider and the
    structured ``CalculatorSignature`` from the run's root node, then stores the
    assembled result keyed on that composite identity.
    """
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    engine = EvalEngine(sim_identity=path_identity)
    sim = make_sim()
    calc = MassSum()
    store = InMemoryResultStore(codec=ValueCodec())

    result = engine.run(calc, sim, store=store)

    structured = calculator_to_signature(calc)
    sim_sig = ("sim", "snapshot_103", "halo_0")
    assert store.has(sim_signature=sim_sig, calculator_signature=structured)

    refs = store.list_refs(sim_signature=sim_sig)
    assert len(refs) == 1
    fetched = store.fetch(refs[0])
    assert fetched.value == result.value
    assert fetched.provenance is not None
    assert fetched.provenance.calculator_signature_hash == structured.short_hash()


def test_calculator_run_threads_store_through() -> None:
    """``calc.run(sim, store=..., sim_identity=...)`` reaches the store seam.

    The calculator-level entry point must accept the persistence hook plus the
    identity provider so a database caller gets restart-stable identity without
    constructing an :class:`EvalEngine` by hand.
    """
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec

    def path_identity(sim) -> tuple:
        return ("sim", "snapshot_103", "halo_0")

    calc = MassSum()
    store = InMemoryResultStore(codec=ValueCodec())

    result = calc.run(make_sim(), store=store, sim_identity=path_identity)

    structured = calculator_to_signature(calc)
    assert store.has(sim_signature=("sim", "snapshot_103", "halo_0"), calculator_signature=structured)
    assert store.fetch(store.list_refs()[0]).value == result.value


def test_engine_run_skips_store_on_error() -> None:
    """A run that completes with errors must not silently persist partial results."""
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec

    @PropertyBase.dataclass
    class Exploding(PropertyBase[float]):
        def calculate(self, sim, params=None) -> float:
            raise RuntimeError("boom")

    from pynbodyext.core.calculate.result.enums import ErrorPolicy
    from pynbodyext.core.calculate.runtime.options import RunOptions

    store = InMemoryResultStore(codec=ValueCodec())
    engine = EvalEngine(sim_identity=lambda sim: ("sim", "snapshot_103", "halo_0"))
    result = engine.run(
        Exploding(), make_sim(), options=RunOptions(errors=ErrorPolicy.COLLECT), store=store
    )

    assert store.list_refs(sim_signature=("sim", "snapshot_103", "halo_0")) == []
    assert result.errors, "expected the run to carry an error"


def test_result_carries_calculator_backreference() -> None:
    """A live :class:`Result` holds a back-reference to the calculator that ran.

    This is the ``result.calculator`` convenience: re-run or inspect the exact
    recipe without needing to keep the original calculator around.
    """
    calc = MassSum()
    sim = make_sim()
    result = calc.run(sim)
    assert result.calculator is calc
    assert isinstance(result.calculator, MassSum)
    assert result.calculator.run(sim).value == result.value


def test_store_load_returns_result_with_calculator_ref() -> None:
    """``store.load(sim_signature, calculator_signature_text)`` rebuilds the recipe onto the Result.

    The stored signature text must round-trip: ``load`` derives the same
    content-addressed hash from the text to locate the record, and the returned
    ``Result`` carries a reconstructed, runnable calculator in ``.calculator``.
    A different sim identity is a miss.
    """
    from pynbodyext.core.calculate.result.signature import calculator_to_signature
    from pynbodyext.core.calculate.store.base import InMemoryResultStore, ValueCodec

    calc = MassSum()
    structured = calculator_to_signature(calc)
    store = InMemoryResultStore(codec=ValueCodec())
    result = calc.run(make_sim())
    store.store(result, sim_signature=("sim", "snap_103"), calculator_signature=structured)

    loaded = store.load(
        sim_signature=("sim", "snap_103"),
        calculator_signature_text=structured.to_json(),
    )
    assert loaded is not None
    assert isinstance(loaded.calculator, MassSum)  # recipe rebuilt onto the Result
    assert loaded.value == result.value
    assert loaded.calculator.run(make_sim()).value == result.value

    assert store.load(
        sim_signature=("sim", "other"),
        calculator_signature_text=structured.to_json(),
    ) is None


def test_record_json_roundtrips_named_provenance_value_faithfully():
    """The whole-record JSON serialization must not degrade tuples to lists.

    The codec-encoded ``value``/``named`` pair keys with values as tuples, and the
    provenance carries a nested ``cache_key()`` structure of tuples.  A save/load
    keyed on the signature must restore those exactly so a loaded result is
    indistinguishable from the live one.
    """
    from pynbodyext.core.calculate.store.base import ResultRecord, ValueCodec

    calc, sim, result = make_result()
    structured = calculator_to_signature(calc)
    record = ResultRecord.from_result(
        result, sim_signature=("sim", "snap_103"), calculator_signature=structured, codec=ValueCodec()
    )

    recovered = ResultRecord.from_json(record.to_json())
    assert recovered.named == record.named
    assert recovered.provenance == record.provenance
    assert recovered.value == record.value
    assert recovered.sim_signature == record.sim_signature

    # And the round-tripped record still reconstructs the value through the codec.
    np.testing.assert_allclose(dict(recovered.to_result(ValueCodec()).value)["m"], dict(result.value)["m"])
