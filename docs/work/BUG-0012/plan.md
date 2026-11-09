# BUG-0012 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkboxes. This plan does not request sub-agent delegation.

**Goal:** Satisfy the bilingual work-item contract without changing unrelated runtime behavior.

**Architecture:** Preserve Calculator DAG, ResultStore and the existing import layers. Correct the existing codec/cell state boundary only. The code below is a proposed minimal patch, not code already installed or verified green.

**Tech Stack:** Python >=3.11, NumPy, pynbody >=2.8.0, pytest; SQLite integration via existing store extra.

## Files and prerequisites

- Modify: `pynbodyext/core/calculate/sample.py`.
- Add regression code to: `tests/test_sample_recovery.py` (reuse existing imports when present).
- Read all paths and prerequisite IDs from [the specification](en.md).
- Update both locale files, metadata, handoff and generated index after implementation.

## Task 1: Reproduce the behavior

- [x] Add the following regression to `tests/test_sample_recovery.py`. In an existing test file merge imports rather than duplicating definitions.

```python
from typing import ClassVar
import pynbody
from pynbodyext.core.calculate import InMemoryResultStore, PropertyBase, SampleTable


@PropertyBase.dataclass
class RecoverableForReview(PropertyBase[float]):
    signature_name = "tests.review.RecoverableForReview"
    failing: ClassVar[bool] = True

    def calculate(self, sim, params=None):
        if type(self).failing:
            raise RuntimeError("temporary failure")
        return 7.0


def test_store_fill_clears_only_recovered_error():
    sim = pynbody.new(1)
    identity = lambda sim: ("sim", "recovery-fixture")
    store = InMemoryResultStore()
    calc = RecoverableForReview()
    table = SampleTable({"value": calc}, store=store, sim_identity=identity)
    RecoverableForReview.failing = True
    try:
        row = table.add(sim, errors="record")
        assert "value" in row.errors
        RecoverableForReview.failing = False
        calc.run(sim, store=store, sim_identity=identity)
        table.fill_from_store()
        assert row.values["value"] == 7.0
        assert "value" not in row.errors
    finally:
        RecoverableForReview.failing = True
```

- [x] Run `.venv/bin/python -m pytest tests/test_sample_recovery.py -q`. Expected: the new scenario fails for the behavior described in the spec, while unrelated existing cases retain their baseline. Object-array cases currently produce shape mismatches and a scalar TypeError; structured arrays currently fail dtype parsing; recovered cells currently retain their error.

## Task 2: Correct the owning boundary

- [x] Apply this scoped change at the indicated method. Preserve all branches not shown.

```python
# In SampleTable.fill_from_store, keep the existing loop/miss behavior:
if hit is not None:
    row.values[label] = hit.value
    row.errors.pop(label, None)
```

- [x] Add the remaining spec boundary cases before broad verification. In particular, exercise SQLite reopen for codec fixes; for cell recovery test cache miss and a store read exception. Assert public values/state, not internal call choreography.
- [x] Run `.venv/bin/python -m pytest tests/test_sample_recovery.py -q`; expected all tests in the file pass.

## Task 3: Verify and hand off

- [x] Run the exact focused command in [the specification](en.md); expected exit 0.
- [x] Run `make check` and `make test-all`; expected exit 0. A source-only happy path is insufficient for serialization/public behavior.
- [x] Review dtype metadata/unsupported-type policy against the spec before claiming universal NumPy coverage. Do not silently discard unsupported metadata. If the minimal patch needs a correction to satisfy a listed case, update it narrowly and repeat that failed case.
- [x] Update en.md and zh-CN.md outcomes together, append actual red/green logs to handoff, set metadata according to the real implementation/review state, and run `make docs-index` then `make docs-check`.
- [x] Prepare one reviewable diff. If commits are authorized, stage only the listed code/tests and this work-item documentation; use commit message `fix: address bug-0012`. Publishing/merging follows the active user's authorization, not an assumed permission in this document.

## Actual implementation / 实际实施

Implemented locally and verified; status review, not merged or published. See handoff.md for exact red/green counts and additional review corrections. Proposed snippets above are planning history; codec implementation additionally validates NPY header readability and supports payload-sized headers. No architecture expansion or database migration.
