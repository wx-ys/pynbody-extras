# BUG-0012: Clear stale cell errors after a successful store fill

## Context and goal

Keep cell value/error state consistent when recovering from a prior failure using a stored result.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: No new work-item dependency; start from current main.

## Scope and contract

After a successful fill_from_store hit, set the value and remove only that column error. A miss preserves its error. A store exception propagates under the existing contract and must not clear an error before a value is available. Do not redesign the None/missing sentinel or change error policies.

## File scope

- `pynbodyext/core/calculate/sample.py`
- `tests/test_sample_recovery.py`
- `tests/test_sample_table.py`

## Acceptance criteria

- [x] Recorded failure followed by separately seeded cache becomes a value without an error.
- [x] An unfilled column retains its own error; a nonempty column is unchanged.
- [x] A failed backend read leaves existing diagnostic state untouched.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [ ] Publish a scoped PR after authorization; local bilingual handoff and red/green evidence are ready.

```bash
.venv/bin/python -m pytest tests/test_sample_recovery.py tests/test_sample_table.py -q
```

For implementation: new regression/API tests fail first and pass after the change. This is a future validation requirement, not an execution claim from this review.

Detailed steps: [Implementation plan](plan.md).

## First regression scenario

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

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.

## Local implementation — 2026-11-09

After a successful cached value assignment, clear only that column error. Added real recorded-failure recovery, missing-column, nonempty-cell and backend-read-error regressions; existing missing sentinel and error propagation stay intact.

- Red: 1 failed, 26 passed, 1 skipped: successful cache fill retained the old error.
- Focused green: 27 passed, 1 skipped; independent review no findings.
- `make check`: 1155 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- `make test-all`: 1159 passed, 1 skipped.

Status: review; locally implemented and verified, not committed, published or merged. No migration or unrelated architecture change. SQLite/Linux verified; other platforms/backends not certified.
