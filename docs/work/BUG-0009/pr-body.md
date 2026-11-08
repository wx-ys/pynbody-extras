## What changed

Simulation identities such as `("sim", 1)` and `("sim", "1")` previously shared
one result-store key, allowing cached results to overwrite or answer another
input. Use one versioned typed encoder for stores and SampleTable row indexing,
so `True`, `1` and `1.0` also remain distinct. Result records use v3 and reject
unknown or missing schemas. CalculatorSignature semantics are unchanged.

The maintainer confirmed there are no existing databases/caches: no legacy
compatibility or migration is included. Identity providers use built-in scalar
values and recursive tuples; unsupported/non-finite values are rejected.

Work item: [BUG-0009](https://github.com/wx-ys/pynbody-extras/blob/55b12db/docs/work/BUG-0009/en.md).

## Evidence

- [x] New identity regressions: **19 failed / 1 passed → 20 passed**.
- [x] Focused identity/store/reuse/sample tests: **80 passed, 1 skipped**.
- [x] `make check`: **936 passed, 1 skipped, 3 deselected**; lint, mypy and docs pass.
- [x] `make test-all`: **939 passed, 1 skipped**.
- [x] `git diff --check` passes.

## Architecture

- [x] ADR-006 added under `docs/adr/`; maintainer authorized publication and merge.
- New encoding is a leaf inside the existing store layer; import contracts pass.
- Historical result versions, ORM and logical snapshot identity remain separate work.

## Checklist

- [x] Existing tests were not weakened.
- [x] Public identity constraints and persistence examples documented in CONTEXT.md.
- [x] Bilingual work item, status and handoff updated.
- [x] No testdata, virtual environment, build outputs, secrets or unrelated local notes included.
