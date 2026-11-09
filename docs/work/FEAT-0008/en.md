# FEAT-0008: Add a value-consuming postprocessor on the existing DAG

## Context and goal

Provide process(value) authoring without a second executor or repeated upstream runs.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: BUG-0010, BUG-0011

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-014](../../adr/0014-dag-value-postprocessors.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §6.1. PostprocessorBase subclasses the existing property/runtime contract with one source calculator dependency declared through existing Param mechanisms. The reusable adapter resolves source then calls process(value). Signature identity includes source and dataclass parameters, honors signature_version and reconstructibility. Initial source accepts a Calculator only; reject raw Result/RecordRef inputs explicitly. Expose through calculate facade. No fitting dependency, new global registry or custom scheduling.

The initial processor supports one dynamic field, source; other processor configuration fields are static parameters. The adapter owns calculate(sim, params) and calls process(params.source); scientific subclasses override process only. Do not silently accept additional dynamic configuration unless its resolved-value delivery contract is designed and tested in this same reviewed increment.

The adapter's exception path invokes cleanup only for its own successfully applied
bound transform, respecting scope/transform `NEVER` and nonrevertible handles. Cleanup
failure is logged without replacing the original exception. Inherited ancestor
scopes and generic property success behavior stay with their existing owners;
transform/filter application failures before adapter execution are outside this
narrow process-exception cleanup path. See ADR-014's implementation boundary.
The engine reports the last recorded phase, so cleanup can leave error phase
`revert`; the original exception type, message and traceback remain preserved.
Cleanup delegates to the existing transform callback: a native compound pynbody
handle may also unwind earlier transforms, and a retained native `NEVER` step may
block earlier cleanup. These handle semantics remain unchanged; cleanup failures
are logged with the original process exception preserved.

## File scope

- `pynbodyext/core/calculate/nodes/postprocessors.py`
- `pynbodyext/core/calculate/nodes/__init__.py`
- `pynbodyext/core/calculate/__init__.py`
- `pynbodyext/calculate.py` (the actual short public facade)
- `tests/test_postprocessors.py`
- `CONTEXT.md`

## Acceptance criteria

- [x] A module-level ScaleValues example consumes an array with units, scales it, and round-trips its signature.
- [x] Two downstream processors in one Pipeline share a counted upstream evaluation once.
- [x] Parameter/source/version changes alter signature; stable and constructible retain their separate meanings.
- [x] process exception follows errors=collect/raise, cleans transformed input and never persists failed output; cached success returns exact pinned ref.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [ ] Update both locales, meta.json and handoff; prepare one scoped PR using the template and red/green evidence.

```bash
.venv/bin/python -m pytest tests/test_postprocessors.py tests/test_calculate_contract.py tests/test_signature_roundtrip.py tests/test_layering_contract.py -q
```

## Implemented API and evidence

```python
from pynbodyext.calculate import PostprocessorBase

@PostprocessorBase.dataclass
class ScaleValues(PostprocessorBase[object]):
    factor: float = 2.0

    def process(self, value):
        return value * self.factor

scaled = ScaleValues(3.0, source=upstream)
result = scaled.run(sim)
```

`source` is inherited, required and keyword-only so default static subclass fields
remain ordinary positional parameters. Construction rejects raw values, Result,
RecordRef, extra dynamic fields/specs, unsigned/static source declarations, and
source unit conversion metadata. Overriding `calculate` fails before signature
registration. The generic dataclass/property signature route remains unchanged;
large static arrays can be stable but nonconstructible, independently of unstable
source callables. Existing recipe signature memoization remains unchanged: an
algorithm version change is tested on a fresh recipe.

Red evidence: missing short-facade API caused one expected ImportError during
collection (`/tmp/feat0008-red.log`). Scoped exception acceptance then exposed
missing generic property cleanup; expanded cleanup tests recorded six failures
with thirty passing (`/tmp/feat0008-cleanup-red.log`). The role-local wrapper
passes these cases without changing generic engine behavior. Exploratory native
callback tests were corrected to reflect the existing compound-handle semantics,
not to promise per-callback reversal or ancestor effect isolation.

Focused validation covering all acceptance cases plus signature, Param, runtime,
error, store, receipt and layering regressions: **164 passed** in 2.77s
(`/tmp/feat0008-focused.log`). This includes 39 postprocessor cases and a real
`collect_partial` Pipeline whose successful sibling observes correct position
values after the failed sibling's cleanup, without persisting the failed pipeline.

Full validation: `make check` passed lint, mypy (120 source files), **1355 passed,
2 skipped, 4 deselected** in 40.00s, and docs validation (27 work items).
`make test-all` passed **1359 tests, 2 skipped** in 41.00s. The two skips require
optional pandas. Logs: `/tmp/feat0008-check.log`, `/tmp/feat0008-test-all.log`.
Both locales, metadata and handoff are updated; status is review, and the work
index is regenerated. Local predecessors are verified, not claimed merged. No
Git commands, commit, push, PR or release publication occurred. Linux/SQLite is
verified; other platforms/SQL dialects and generic successful-scope cleanup are
not certified by this increment. PR publication remains with the parent workflow.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.
