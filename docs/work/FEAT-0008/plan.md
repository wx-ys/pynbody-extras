# FEAT-0008 value-consuming postprocessors implementation plan

> **For agentic workers:** Execute this plan inline using the executing-plans skill, task by task. Steps use checkbox syntax for tracking. Implementation is deferred until the coordinating agent finishes FEAT-0007 and explicitly resumes this item. Do not commit, push, or dispatch subagents under this preparation task.

**Goal:** Add `PostprocessorBase` so authors implement `process(value)` over one upstream Calculator while retaining the existing Calculator DAG, signatures, error handling, cleanup, and result-store receipts.

**Architecture:** Follow accepted ADR-014 and both FEAT-0008 specifications. A property role adapter declares its required dynamic `source` through `Param`, receives the resolved public value through `ParamView`, and calls `process(params.source)`. Keep scheduling, serialization, and persistence in their current owners.

**Tech stack:** Python 3.11+, existing dataclass calculator/Param machinery, pynbody `SimArray`, pytest, repository `.venv`; no new dependencies.

## Preparation evidence and boundaries

This is a plan, not implementation or a test result. Preparation read `AGENTS.md`, `CONTEXT.md`, the work index, FEAT-0008 metadata/both locales/handoff, ADR-014, the role and Param sources, and signature/runtime/store/parameter/layering contract tests. The approved local predecessors may be used without claiming remote merge. Preserve the shared dirty checkout and all unrelated work.

The existing implementation already supplies the necessary extension points:

- `nodes/declarative.py::_materialise_base_init_fields` copies `Param` declarations from undecorated role bases. It appends fields after the subclass's own fields and preserves `kw_only`. A required positional source appended after `factor: float = 2.0` would fail dataclass construction; declare the role source **keyword-only**.
- The decorator invokes `_init_dataclass_base` after collecting dynamic metadata. `TransformBase` demonstrates using that hook; no decorator or base-class rewrite is needed.
- `PropertyBase.compute` ultimately calls `calculate(sim, params)`. The standard runtime resolves dynamic calculators and supplies `ParamView`; the adapter never calls `source.run()`.
- The dataclass serializer already includes init fields, nested calculator values, class signature version, and independent stable/constructible flags. The postprocessor stays an ordinary property/dataclass node; it needs no new node kind, serializer dispatch, or registry.
- `pynbodyext/calculate.py` is the actual short public facade. Add it to implementation scope even though the work item's enumerated file list mentions only the core facade: this is required by its explicit facade contract.

## File map

| File | Responsibility |
| --- | --- |
| `pynbodyext/core/calculate/nodes/postprocessors.py` (new) | Role declaration, initialization validation, adapter, process authoring hook |
| `pynbodyext/core/calculate/nodes/__init__.py` | Node export and `__all__` |
| `pynbodyext/core/calculate/__init__.py` | Core export and role documentation |
| `pynbodyext/calculate.py` | Short public export and authoring documentation |
| `tests/test_postprocessors.py` (new) | Module-level examples and behavioral acceptance matrix |
| `CONTEXT.md` | Glossary/authoring contract for the new role |
| `docs/work/FEAT-0008/{en.md,zh-CN.md,meta.json,handoff.md}` | Paired completion evidence and honest review status during implementation |

No intended changes to runtime engine/context, Param collection/resolution, signature encoding, stores, ORM, fitting packages, or installation dependencies.

Implementation refinement recorded in ADR-014: the generic engine currently does
not revert a property scope on failure. The approved process-exception clause is
implemented with a narrow `PostprocessorBase.execute` wrapper around the ordinary
runtime. It cleans only its own successfully applied bound scope, honors both
scope/transform NEVER and nonrevertible handles, and preserves the original error
if cleanup fails. Test single/chained cleanup, NEVER/nonrevertible cases, cleanup
failure, and inherited ancestor scope ownership. Do not change the generic engine;
pre-adapter transform/filter application failures are outside this local path.
Ownership here means which cleanup callback the adapter invokes. Native pynbody
compound handles may cascade into earlier transforms, and retained native NEVER
steps can block older-handle cleanup. Characterize these existing semantics and
preserve/log the original failure; do not add private native-handle manipulation.
Test `collect_partial` failure followed by a scoped successful sibling so science
values/cache invalidation after cleanup are exercised in the same run.

## Task 1: Public API, inherited source, and value delivery

- [ ] Create `tests/test_postprocessors.py` with module-level examples (module scope is necessary for honest signature reconstruction):

```python
from typing import Any, ClassVar

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray

from pynbodyext.calculate import PostprocessorBase, PropertyBase
from pynbodyext.core.calculate import Param, Pipeline, calculator_from_signature


@PropertyBase.dataclass
class ArrayValues(PropertyBase[SimArray]):
    offset: float = 0.0
    calls: ClassVar[int] = 0

    def calculate(self, sim, params=None):
        type(self).calls += 1
        return sim["mass"].copy() + SimArray(self.offset, sim["mass"].units)


@PostprocessorBase.dataclass
class ScaleValues(PostprocessorBase[SimArray]):
    factor: float = 2.0

    def process(self, value):
        return value * self.factor


def mass_sim():
    sim = pynbody.new(3)
    sim["mass"] = SimArray([1.0, 2.0, 4.0], "Msol")
    sim["pos"] = SimArray(np.zeros((3, 3)), "kpc")
    return sim


def test_units_and_signature_roundtrip():
    calc = ScaleValues(3.0, source=ArrayValues())
    original = calc.run(mass_sim()).value
    rebuilt = calculator_from_signature(calc.to_signature().to_json())
    repeated = rebuilt.run(mass_sim()).value
    assert isinstance(original, SimArray)
    assert original.units == repeated.units == SimArray([], "Msol").units
    np.testing.assert_array_equal(original, [3.0, 6.0, 12.0])
    np.testing.assert_array_equal(repeated, original)
    assert type(rebuilt) is ScaleValues
    assert rebuilt.to_signature().payload == calc.to_signature().payload
    assert rebuilt.to_signature().hash == calc.to_signature().hash


def test_two_outputs_share_the_upstream_once():
    ArrayValues.calls = 0
    source = ArrayValues()
    result = Pipeline({
        "two": ScaleValues(source=source),
        "three": ScaleValues(3.0, source=source),
    }).run(mass_sim())
    assert ArrayValues.calls == 1
    np.testing.assert_array_equal(result.value["two"], [2.0, 4.0, 8.0])
    np.testing.assert_array_equal(result.value["three"], [3.0, 6.0, 12.0])
```

- [ ] Add public-surface identity assertions for the node package, core facade, and `pynbodyext.calculate`: all three export the same `PostprocessorBase` and list it in `__all__`.
- [ ] Assert `inspect.signature(ScaleValues).parameters["source"].kind` is `KEYWORD_ONLY`, omitting source raises `TypeError`, and `ScaleValues(3.0, source=ArrayValues())` works without redeclaring source. Define a decorated `ScaleValues` subclass with one further default static parameter; verify source still materializes/inherits and signature reconstruction retains the extra field.
- [ ] Add an identity-value processor whose `process` returns the object unchanged, and a source returning a known `SimArray`. Assert the processor receives the same object and units, including a one-element array; no implicit conversion, flattening, copy, or extraction to scalar belongs in this adapter.
- [ ] Run `.venv/bin/python -m pytest tests/test_postprocessors.py -q`. Record the expected missing `PostprocessorBase` public API import failure. Verify `.venv`/pynbody imports work first; environment errors are not red evidence.

## Task 2: Implement the narrow role and construction guards

- [ ] Add `nodes/postprocessors.py` with `PostprocessorBase(PropertyBase[T], Generic[T], ABC)`. Keep the role undecorated, like `TransformBase`. Use this structure as the implementation skeleton:

```python
class PostprocessorBase(PropertyBase[T], Generic[T], ABC):
    source: Param[CalculatorBase[Any, Any]] = Param(kw_only=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if cls.calculate is not PostprocessorBase.calculate:
            raise TypeError("PostprocessorBase owns calculate(); override process(value) instead")
        super().__init_subclass__(**kwargs)

    def _init_dataclass_base(self) -> None:
        super()._init_dataclass_base()
        if not isinstance(self.source, CalculatorBase):
            raise TypeError("source must be a CalculatorBase; Result and RecordRef are not supported")
        specs = {spec.name: spec for spec in collect_param_specs(type(self))}
        source_spec = specs.get("source")
        if source_spec is None or source_spec.kind != "dynamic" or not source_spec.signature:
            raise TypeError("source must be a signature-bearing dynamic Param")
        if set(self.dynamic_param_names()) != {"source"}:
            raise TypeError("source is the only supported dynamic postprocessor parameter")
        if (
            source_spec.field_name is not None
            or source_spec.target_units is not None
            or not source_spec.optional_units
        ):
            raise TypeError("source must pass its value without unit conversion")

    def calculate(self, sim: SimSnap, params: Any = None) -> T:
        return self.process(params.source)

    @abstractmethod
    def process(self, value: Any) -> T:
        """Process the resolved public source value with static configuration."""
        raise NotImplementedError
```

Import `ABC`, `abstractmethod`, `Any`, `Generic`, `TypeVar`, `Param`, `collect_param_specs`, `CalculatorBase`, and `PropertyBase` from their owning modules; defer annotation-only `SimSnap` under `TYPE_CHECKING` according to lint conventions. Define `T = TypeVar("T")`. Do not add a hand-written source constructor, scheduling method, or serialization hook for this first dataclass-authoring API. Ordinary unannotated/static fields remain static and are read through `self` in `process`.

- [ ] Tests must reject raw `Result` (obtained by running `ArrayValues`), a real `RecordRef`, `None`, numbers, callables, and containers of calculators as `source`, each with `TypeError` naming source. The general Param resolver accepts more values; this role intentionally narrows construction before runtime.
- [ ] Test a scientific subclass declaring `extra: Param[float] = 1.0`, and a subclass declaring `dynamic_param_specs = {"extra": None}` with an ordinary static `extra` field. Both must fail construction explicitly. Test a source redeclared static or with `signature=False`; neither may lose its upstream identity silently.
- [ ] Test class creation rejects both a direct `calculate` override and an override introduced on a derived processor; error names `process`. A subclass implementing only `process` must work. Do not silently execute subclass `calculate` while bypassing the adapter.
- [ ] Test redefined source unit metadata is refused so the value-delivery test remains true. Keep the guard local to the postprocessor; no change to general Param semantics.
- [ ] Wire the same class through all three export surfaces and their `__all__` lists. Describe the additional role in the two facade module docstrings with `ScaleValues(source=...)` authoring; no independent runtime kind is introduced.
- [ ] Run the focused new tests to green; retain the initial failure output for handoff evidence.

## Task 3: Signature identity and independent flags

- [ ] Add assertions that changing `factor` changes the hash, changing `ArrayValues.offset` changes the enclosing hash, and equal parameter/source recipes retain equal hashes. Changing node labels must retain the existing label-only meaning.
- [ ] Test the same class with `signature_version` temporarily changed using `monkeypatch.setattr(ScaleValues, "signature_version", 1)`; its hash must change and its payload must contain version 1. Compare signatures captured before/after this one class mutation, not two different class names. Let monkeypatch restore state.
- [ ] Test `ScaleValues(source=ArrayValues()).to_signature()` has `stable=True`, `constructible=True`; the module-level roundtrip from Task 1 proves reconstruction behavior rather than only flags.
- [ ] Test `ScaleValues(source=LambdaProperty(lambda sim: sim["mass"]))` propagates `stable=False`, `constructible=False` and an instability path under source. Reconstruction must explicitly refuse the signature.
- [ ] Add module-level `WeightedValues(PostprocessorBase[Any])` with a required static `weights` ndarray field and `process(value)` returning value. `WeightedValues(weights=np.arange(1000.0), source=ArrayValues())` must be stable but not constructible because existing signature machinery hashes large arrays. Changing its array content must change the hash. This case proves the two flags are independent.
- [ ] Run `.venv/bin/python -m pytest tests/test_postprocessors.py tests/test_signature_roundtrip.py tests/test_signature_stability.py tests/test_signature_type_identity.py tests/test_calculate_params.py -q`. Do not alter the serializer or existing tests to invent new flags for the role.

## Task 4: Errors, transform cleanup, and exact cached receipts

- [ ] Define a module-level `FailValues(PostprocessorBase[Any])` whose `process` raises `RuntimeError("postprocessor failed")`. Parameterize `errors` over `"collect"`/`"raise"` and use a fresh snapshot/store per case. Use existing `ShiftX` in `tests/test_calculate_runtime.py` as the pattern for a local transform; do not import another test module's test functions.
- [ ] Define a local test-helper `ShiftPositions(TransformBase[object])` dataclass with `dx: float = 1.0`; its `build_handle` returns `GenericTranslation(target, "pos", np.array([self.dx, 0.0, 0.0]), description="postprocessor-test-shift")`. Bind it on the failing processor using `.transform(...)`, so source and process execute in the ordinary transformed scope. The original position array must be restored after failure under both policies.
- [ ] Under `collect`, assert `not result.ok`, root error status, recorded message includes `postprocessor failed`, `value is None`, and `store_ref is None`. Under `raise`, assert the RuntimeError propagates with that message. In both cases assert `store.list_refs() == []` and the failed processor's address has no historical versions. Let the engine own cleanup and persistence; add no adapter `try/except` and no manual save.
- [ ] Test a successful `ScaleValues` run twice with an `InMemoryResultStore` and explicit `sim_identity=lambda sim: ("sim", "postprocessor-test")`. Assert first receipt is non-None/pinned, second `from_store` is true, receipts equal exactly, `store.list_versions(first.store_ref)` has just the first receipt, values/units agree, and the counted source ran once across both runs.
- [ ] Refresh the same successful processor with `reuse=False`: new pinned receipt must differ while `store.fetch(first.store_ref)` still has the original value/receipt. This reuses the existing immutable-result contract without adding a postprocessor persistence API.
- [ ] Run `.venv/bin/python -m pytest tests/test_postprocessors.py tests/test_calculate_runtime.py tests/test_calculate_errors.py tests/test_store_reuse.py tests/test_result_versions.py -q`.

## Task 5: Documentation, complete checks, and handoff

- [ ] Add `PostprocessorBase` to the Calculator DAG glossary in `CONTEXT.md` and describe its single Calculator source, static configuration, `process(value)` authoring, adapter-owned `calculate`, public import, units/value delivery, and existing property/runtime lifecycle. Link ADR-014.
- [ ] Run the specification's focused command:

```bash
.venv/bin/python -m pytest tests/test_postprocessors.py tests/test_calculate_contract.py tests/test_signature_roundtrip.py tests/test_layering_contract.py -q
```

- [ ] Run `make check`, `make test-all`, and `make docs-check`; public behavior requires the full suite under repository policy. Inspect actual counts and any skips/failures. If environment/data setup blocks a check, record the limitation precisely without treating it as a product regression or claiming success.
- [ ] Update both `en.md` and `zh-CN.md` acceptance evidence in sync; set `meta.json.status` to the actual review state, not merged/done; append a handoff with initial red reason, commands/exact green counts, verified local prerequisites, changed file list, and limits. Regenerate the work index using the repository's established script if metadata changes require it.
- [ ] Self-review against ADR-014 and the acceptance matrix: value/units+roundtrip (Task 1), one shared upstream run (Task 1), signature source/parameters/version and separate flags (Task 3), errors/cleanup/no failed saves and pinned hit (Task 4). All extra dynamic/source/method guards stay in Task 2. Confirm no unrelated shared-checkout changes were overwritten.
- [ ] Return the completed local change and red/green evidence to the coordinator. No commit, push, PR publication, or merge unless separately instructed by the active user/coordinator.
