"""A signature is a key that has to mean the same thing tomorrow.

The content hash of a calculator signature keys stored results and, later, rows of
large sample tables, so it must be equal for equal calculators in every Python
session.  A signature that can only tell objects apart within one run (by ``id()``
or a function ``repr``) must say so — ``stable`` is false — and a persistent store
must refuse it rather than write a key no later session can match.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import textwrap
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from calculate_helpers import MassSum, OpaqueProperty, RBelow, TempMean, kind_samples, make_pipeline, make_sim

from pynbodyext.core.calculate import LambdaProperty, Param, Pipeline, PropertyBase
from pynbodyext.core.calculate.result.signature import (
    SIGNATURE_SCHEMA,
    CalculatorSignature,
    calculator_from_signature,
    calculator_to_signature,
)
from pynbodyext.core.calculate.store import (
    InMemoryResultStore,
    SignatureMismatchError,
    UnstableSignatureError,
)

# ---------------------------------------------------------------------------
# Stability across sessions
# ---------------------------------------------------------------------------


def test_stable_hashes_are_equal_in_a_fresh_interpreter() -> None:
    """The hash a stable calculator gets here is the hash it gets in a new process."""
    here = {i: calculator_to_signature(s) for i, s in enumerate(kind_samples())}
    stable = {i: sig.hash for i, sig in here.items() if sig.stable}
    assert len(stable) >= 8, "the kind samples went stale: too few stable kinds"

    script = textwrap.dedent(
        """
        import json
        from calculate_helpers import kind_samples
        from pynbodyext.core.calculate.result.signature import calculator_to_signature
        print(json.dumps({i: calculator_to_signature(s).hash for i, s in enumerate(kind_samples())}))
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parent, env=env, capture_output=True, text=True, check=True
    )
    there = {int(k): v for k, v in json.loads(out.stdout.strip().splitlines()[-1]).items()}

    for index, digest in stable.items():
        assert there[index] == digest, type(kind_samples()[index]).__name__


@pytest.mark.parametrize(
    ("make", "constructible", "stable"),
    [
        (lambda: MassSum(), True, True),
        (lambda: make_pipeline(), True, True),
        (lambda: MassSum().filter(RBelow(5.0)), True, True),
        (lambda: LambdaProperty(lambda sim: 1.0), False, False),
        (lambda: OpaqueProperty(), False, False),
    ],
    ids=["dataclass", "pipeline", "scoped", "lambda", "opaque"],
)
def test_constructible_and_stable_are_reported(make, constructible: bool, stable: bool) -> None:
    sig = make().to_signature()
    assert sig.constructible is constructible
    assert sig.stable is stable
    assert bool(sig.unstable_paths) is (not stable)


def test_instability_propagates_to_the_enclosing_graph() -> None:
    """One session-specific leaf makes the whole key session-specific, and says where."""
    sig = Pipeline({"ok": MassSum(), "bad": LambdaProperty(lambda sim: 1.0)}).to_signature()
    assert not sig.stable
    assert sig.unstable_paths == ("calculator.outputs['bad'].func",)


def test_two_opaque_nodes_stay_apart_within_a_run() -> None:
    """Unstable is not ambiguous: the in-run cache must still tell them apart."""
    assert OpaqueProperty().signature() != OpaqueProperty().signature()
    assert LambdaProperty(lambda s: 1).signature() != LambdaProperty(lambda s: 1).signature()


def test_a_hash_only_array_is_stable_but_not_constructible() -> None:
    """A large array is identified by its content hash: durable, but not rebuildable."""

    @PropertyBase.dataclass
    class Weighted(PropertyBase[float]):
        weights: Param[object]

        def calculate(self, sim, params=None) -> float:
            return 0.0

    sig = Weighted(weights=np.arange(1000.0)).to_signature()
    assert sig.stable
    assert not sig.constructible
    assert sig.hash != Weighted(weights=np.arange(1000.0) + 1).to_signature().hash


# ---------------------------------------------------------------------------
# What the hash covers
# ---------------------------------------------------------------------------


def test_the_hash_covers_schema_and_payload_only() -> None:
    sig = make_pipeline().to_signature()
    identity = {"schema": SIGNATURE_SCHEMA, "payload": sig.identity()}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert sig.hash == expected
    assert len(sig.hash) == 64
    # descriptive flags do not move the key
    relabelled = CalculatorSignature(payload=sig.payload, constructible=False, stable=False)
    assert relabelled.hash == sig.hash


def test_full_json_carries_the_flags_and_round_trips() -> None:
    sig = LambdaProperty(lambda sim: 1.0).to_signature()
    data = json.loads(sig.to_json(full=True))
    assert data["schema"] == SIGNATURE_SCHEMA
    assert data["stable"] is False
    again = CalculatorSignature.from_json(sig.to_json(full=True))
    assert (again.stable, again.unstable_paths) == (sig.stable, sig.unstable_paths)
    assert again.hash == sig.hash


def test_a_v1_signature_text_is_still_readable() -> None:
    payload = MassSum().to_signature().payload
    old = json.dumps({"schema": "pynbodyext.calculator.signature/v1", "payload": payload})
    assert calculator_from_signature(old).to_signature().payload == payload


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_a_pipeline_round_trips_and_runs_the_same() -> None:
    pipeline = make_pipeline()
    rebuilt = calculator_from_signature(pipeline.to_signature().to_json())
    assert isinstance(rebuilt, Pipeline)
    assert list(rebuilt.outputs) == ["m", "t"]
    assert rebuilt.to_signature().hash == pipeline.to_signature().hash
    assert rebuilt(make_sim()) == pipeline(make_sim())


def test_pipeline_identity_is_its_outputs_not_its_label() -> None:
    def build(keys=("m", "t"), name="p"):
        return Pipeline({keys[0]: MassSum(), keys[1]: TempMean()}, name=name)

    base = build().to_signature().hash
    assert build(name="another label").to_signature().hash == base
    assert build(keys=("mass", "t")).to_signature().hash != base  # output names key the value dict
    reordered = Pipeline({"t": TempMean(), "m": MassSum()}).to_signature().hash
    assert reordered != base  # and their order is the order of that dict


def test_pipeline_expression_reads_like_its_constructor() -> None:
    text = Pipeline({"m": MassSum(), "t": TempMean()}).to_signature().pretty()
    assert text == 'Pipeline({"m": MassSum(), "t": TempMean()})'


def test_a_pipeline_subclass_with_its_own_constructor_stays_generic() -> None:
    class Summary(Pipeline):
        def __init__(self) -> None:
            super().__init__({"m": MassSum()})

    payload = Summary().to_signature().payload
    assert payload["node"] == "generic"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class _PersistentMemoryStore(InMemoryResultStore):
    """An in-memory store that behaves like a persistent one, for the refusal path."""

    persistent = True


def _result(calc):
    return calc.run(make_sim())


def test_a_persistent_store_refuses_a_session_specific_signature() -> None:
    calc = LambdaProperty(lambda sim: 1.0)
    store = _PersistentMemoryStore()
    with pytest.raises(UnstableSignatureError, match="session-specific"):
        store.store(_result(calc), sim_signature=("sim", "a"), calculator_signature=calc.to_signature())
    assert store.list_refs() == []


def test_an_in_memory_store_accepts_a_session_specific_signature() -> None:
    calc = LambdaProperty(lambda sim: 1.0)
    store = InMemoryResultStore()
    store.store(_result(calc), sim_signature=("sim", "a"), calculator_signature=calc.to_signature())
    assert store.has(sim_signature=("sim", "a"), calculator_signature=calc.to_signature())


def test_a_run_with_an_unstable_calculator_warns_and_skips_a_persistent_store() -> None:
    store = _PersistentMemoryStore()
    with pytest.warns(UserWarning, match="not stored"):
        result = LambdaProperty(lambda sim: 2.0).run(make_sim(), store=store)
    assert result.value == 2.0  # the run itself is fine
    assert store.list_refs() == []


def test_records_are_keyed_on_the_full_hash() -> None:
    calc = make_pipeline()
    store = InMemoryResultStore()
    ref = store.store(_result(calc), sim_signature=("sim", "a"), calculator_signature=calc.to_signature())
    assert ref.calculator_signature_hash == calc.to_signature().hash
    assert len(ref.calculator_signature_hash) == 64
    assert _result(calc).provenance.calculator_signature_hash == calc.to_signature().hash


def test_a_record_for_another_calculator_under_the_same_key_is_refused() -> None:
    """A hit is checked against the signature that asked for it."""
    store = InMemoryResultStore()
    mass, temp = MassSum(), TempMean()
    store.store(_result(mass), sim_signature=("sim", "a"), calculator_signature=mass.to_signature())

    # Simulate a collision / corrupted row: the temperature key now holds the mass record.
    (key, record) = next(iter(store._records.items()))
    record.calculator_signature_hash = temp.to_signature().hash
    store._records[key.replace(mass.to_signature().hash, temp.to_signature().hash)] = record

    with pytest.raises(SignatureMismatchError):
        store.get(sim_signature=("sim", "a"), calculator_signature=temp.to_signature())
    with pytest.raises(SignatureMismatchError):
        store.has(sim_signature=("sim", "a"), calculator_signature=temp.to_signature())
    assert store.get(sim_signature=("sim", "a"), calculator_signature=mass.to_signature()) is not None


def test_the_sql_store_is_persistent_and_refuses_unstable_keys(tmp_path) -> None:
    pytest.importorskip("sqlalchemy")
    from pynbodyext.core.calculate.store import SQLAlchemyResultStore

    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'r.sqlite'}")
    assert store.persistent
    calc = LambdaProperty(lambda sim: 1.0)
    with pytest.raises(UnstableSignatureError):
        store.store(_result(calc), sim_signature=("sim", "a"), calculator_signature=calc.to_signature())

    stable = make_pipeline()
    store.store(_result(stable), sim_signature=("sim", "a"), calculator_signature=stable.to_signature())
    store.engine.dispose()
    reopened = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'r.sqlite'}")
    try:
        loaded = reopened.load(sim_signature=("sim", "a"), calculator_signature_text=stable.to_signature().to_json())
    finally:
        reopened.engine.dispose()
    assert loaded is not None
    assert loaded.calculator is not None
    assert loaded.calculator.to_signature().hash == stable.to_signature().hash


def test_opaque_node_identity_outlives_the_node_in_an_in_memory_store():
    store = InMemoryResultStore()
    previous = OpaqueProperty()
    original = calculator_to_signature(previous)
    assert original.hash == calculator_to_signature(previous).hash
    store.store(previous.run(make_sim()), sim_signature=("sim", "a"), calculator_signature=original)
    del previous

    # Old object addresses can be recycled while the store still holds their keys.
    keys = {calculator_to_signature(OpaqueProperty()).hash for _ in range(100)}
    assert len(keys) == 100
    assert original.hash not in keys


def test_object_array_signatures_cover_every_element_without_disclosing_values():
    from pynbodyext.core.calculate.nodes.expr import ConstantProperty

    values = np.full(2000, "private text", dtype=object)
    other = values.copy()
    other[1000] = "another value"
    first = calculator_to_signature(ConstantProperty(values))
    assert first.hash != calculator_to_signature(ConstantProperty(other)).hash
    assert first.hash == calculator_to_signature(ConstantProperty(values.copy())).hash
    assert not first.constructible
    assert "private text" not in first.to_json(full=True)


@pytest.mark.parametrize("mode", ["opaque", "nested", "overlap"])
def test_arrays_without_portable_contents_still_run_with_independent_session_keys(mode):
    from pynbodyext.core.calculate.nodes.expr import ConstantProperty

    def sample():
        if mode == "overlap":
            dtype = np.dtype({"names": ["a", "b"], "formats": ["i4", "i4"], "offsets": [0, 0]})
            return np.zeros(1, dtype=dtype)
        leaf = np.array([Decimal("1.1")], dtype=object)
        if mode == "opaque":
            return leaf
        outer = np.empty(1, dtype=object)
        outer[0] = leaf
        return outer

    values = sample()
    calc = ConstantProperty(values)
    signature = calculator_to_signature(calc)
    assert not signature.constructible and not signature.stable
    assert signature.hash == calculator_to_signature(calc).hash
    assert signature.hash != calculator_to_signature(ConstantProperty(sample())).hash
    assert "Decimal" not in signature.to_json()
    np.testing.assert_array_equal(calc.run(make_sim()).value, values)


@pytest.mark.parametrize("schema", ["pynbodyext.calculator.signature/v1", None])
def test_legacy_signed_and_compact_opaque_payloads_cannot_enter_a_persistent_store(schema):
    calc = make_pipeline()
    sig = Pipeline({"child": OpaqueProperty()}).to_signature().as_dict()
    sig.pop("stable")
    sig.pop("unstable_paths")
    if schema:
        sig["schema"] = schema
    legacy = CalculatorSignature.from_json(json.dumps(sig))
    with pytest.raises(UnstableSignatureError):
        _PersistentMemoryStore().store(_result(calc), sim_signature=("sim", "a"), calculator_signature=legacy)
    assert CalculatorSignature.from_json(calc.to_signature().to_json()).stable


def test_pipeline_output_mapping_preserves_cached_parent_identity():
    outputs = {"m": MassSum()}
    pipeline = Pipeline(outputs)
    parent = Pipeline({"child": pipeline})
    signature = parent.to_signature()
    outputs.clear()
    with pytest.raises(TypeError):
        pipeline.outputs["t"] = TempMean()
    with pytest.raises(AttributeError):
        pipeline.outputs = {"t": TempMean()}
    assert parent.to_signature().hash == signature.hash
    assert list(parent(make_sim())["child"]) == ["m"]
    assert pickle.loads(pickle.dumps(pipeline)).to_signature().hash == pipeline.to_signature().hash
    assert list(pipeline.named("copy").outputs) == ["m"]


def test_pipeline_decoder_requires_a_pipeline_class():
    payload = make_pipeline().to_signature().as_dict()
    payload["payload"]["class"] = "builtins.dict"
    with pytest.raises(ValueError, match="Pipeline"):
        calculator_from_signature(payload)


def test_pipeline_subclass_inheriting_init_preserves_its_declared_state():
    class Summary(Pipeline):
        signature_name = "test_signature_stability.InheritedSummary"

        def signature_payload(self):
            return {"label": self.label, "outputs": dict(self.outputs)}

        @classmethod
        def from_signature_payload(cls, identity):
            node = cls(identity["outputs"])
            node.label = identity["label"]
            return node

    first = Summary({"z": TempMean(), "a": MassSum()})
    second = Summary({"z": TempMean(), "a": MassSum()})
    first.label, second.label = "first", "second"
    assert first.to_signature().hash != second.to_signature().hash
    rebuilt = calculator_from_signature(first.to_signature().to_json())
    assert rebuilt.label == "first"
    assert list(rebuilt.outputs) == ["z", "a"]
    assert rebuilt.to_signature().hash == first.to_signature().hash


def test_inherited_pipeline_state_and_ordered_output_bindings_both_key_the_store():
    class Summary(Pipeline):
        signature_name = "test_signature_stability.SummaryOmittingOutputs"

        def signature_payload(self):
            return {"tag": "stars"}

        @classmethod
        def from_signature_payload(cls, identity):
            return cls({"mass": MassSum()})

    sim = make_sim()
    store = InMemoryResultStore()
    first = Summary({"mass": MassSum()})
    second = Summary({"old-mass": MassSum()})
    saved = first.run(sim, store=store)
    independent = second.run(sim, store=store)
    assert independent.value == {"old-mass": 15.0} and not independent.from_store
    assert independent.store_ref.cache_key != saved.store_ref.cache_key
    with pytest.raises(ValueError, match="ordered outputs"):
        calculator_from_signature(second.to_signature())


@pytest.mark.parametrize("format", ["pairs", "nested mapping"])
def test_inherited_pipeline_hooks_can_represent_ordered_outputs_in_their_own_fields(format):
    class Summary(Pipeline):
        signature_name = "test_signature_stability.CustomOutputRecipe"

        def signature_payload(self):
            recipe = list(self.outputs.items()) if format == "pairs" else {"columns": dict(self.outputs)}
            return {"recipe": recipe}

        @classmethod
        def from_signature_payload(cls, identity):
            recipe = identity["recipe"]
            return cls(dict(recipe) if isinstance(recipe, list) else recipe["columns"])

    node = Summary({"z": TempMean(), "a": MassSum()})
    signature = node.to_signature()
    pretty = signature.pretty()
    assert 'outputs["z"]' in pretty and 'outputs["a"]' in pretty
    rebuilt = calculator_from_signature(signature.to_json())
    assert list(rebuilt.outputs) == ["z", "a"]
    assert signature.hash == rebuilt.to_signature().hash


def test_short_lived_functions_cannot_share_a_lambda_property_session_key():
    keys = {calculator_to_signature(LambdaProperty(lambda sim: 1)).hash for _ in range(100)}
    assert len(keys) == 100

    def shared(sim):
        return 1

    assert LambdaProperty(shared).to_signature().hash == LambdaProperty(shared).to_signature().hash


def test_checking_restored_pipeline_outputs_does_not_run_a_child_constructor_twice():
    builds = []

    class Child(OpaqueProperty):
        signature_name = "test_signature_stability.CountedChild"

        def signature_payload(self):
            return {}

        @classmethod
        def from_signature_payload(cls, identity):
            builds.append(1)
            return cls()

    class Summary(Pipeline):
        signature_name = "test_signature_stability.CountedSummary"

        def signature_payload(self):
            return {"outputs": self.outputs}

        @classmethod
        def from_signature_payload(cls, identity):
            return cls(identity["outputs"])

    calculator_from_signature(Summary({"score": Child()}).to_signature())
    assert builds == [1]


def test_linear_inherited_pipeline_recipes_visit_each_layer_once_and_stay_compact():
    visits = []

    class Layer(Pipeline):
        signature_name = "test_signature_stability.LinearLayer"

        def signature_payload(self):
            visits.append(1)
            return {"pairs": list(self.outputs.items())}

        @classmethod
        def from_signature_payload(cls, identity):
            return cls(dict(identity["pairs"]))

    sizes = []
    for depth in (4, 8):
        node = MassSum()
        for _ in range(depth):
            node = Layer({"child": node})
        visits.clear()
        sig = calculator_to_signature(node)
        text = sig.to_json()
        sizes.append(len(text))
        assert len(visits) == depth, (depth, len(visits), len(text))
        assert calculator_from_signature(text).to_signature().hash == sig.hash
    assert sizes[1] < 2.5 * sizes[0] and sizes[1] < 10_000


def test_inherited_pipeline_keeps_extra_dependencies_and_output_instability():
    class Summary(Pipeline):
        def signature_payload(self):
            return {"tag": "test"}

        def declared_dependencies(self):
            return [*super().declared_dependencies(), self.extra]

    first, second = Summary({"m": MassSum()}), Summary({"m": MassSum()})
    first.extra, second.extra = MassSum(), TempMean()
    assert first.to_signature().hash != second.to_signature().hash
    opaque = Summary({"unbound": OpaqueProperty()})
    opaque.extra = MassSum()
    assert not opaque.to_signature().stable and not opaque.to_signature().constructible


def test_pipeline_recovers_an_old_pickled_outputs_attribute():
    pipeline = Pipeline({"z": TempMean(), "a": MassSum()})
    state = vars(pipeline)
    state["outputs"] = dict(pipeline.outputs)
    state.pop("_outputs", None)
    state["_signature_cache"] = MassSum().to_signature()
    rebuilt = pickle.loads(pickle.dumps(pipeline))
    assert list(rebuilt.outputs) == ["z", "a"]
    assert rebuilt.to_signature().hash == Pipeline(rebuilt.outputs).to_signature().hash


def test_compact_stability_checks_encoded_slots_and_leaves_user_parameter_names_alone():
    for node, slot in [("generic", "identity"), ("dataclass", "init")]:
        signature = CalculatorSignature({"node": node, "class": "lab.Science", slot: {"type": "array", "axis": 0}})
        assert CalculatorSignature.from_json(signature.to_json()).stable


@pytest.mark.parametrize("root_record", [True, False])
def test_object_array_signatures_distinguish_metadata_on_nested_fields(root_record):
    from pynbodyext.core.calculate.nodes.expr import ConstantProperty

    def signature(unit):
        quantity = np.dtype("f8", metadata={"unit": unit})
        dtype = np.dtype([("measurement", [("value", quantity, (1,))])])
        if root_record:
            value = np.zeros(1, dtype=[("record", dtype), ("label", "O")])
        else:
            value = np.empty(1, dtype=object)
            value[0] = np.zeros(1, dtype=dtype)
        return calculator_to_signature(ConstantProperty(value))

    assert signature("kpc").hash != signature("Mpc").hash
