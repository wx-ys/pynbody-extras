# BUG-0010: Preserve object-array shape and element boundaries

## Context and goal

Fix demonstrated data loss in ValueCodec JSON round trips. This is a correction of array semantics, not a new store architecture.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: No new work-item dependency; start from current main.

## Scope and contract

Use a versioned object-array tag carrying shape, optional units and C-order flat encoded elements. Allocate an object array of the stored shape, then assign elements individually; never let np.array infer shape from nested element values. Validate nonnegative integral dimensions and exact element count. Preserve existing readable tags; no database migration. Keep ordinary numeric arrays unchanged. Old object-array payloads remain readable with their existing limitations.

## File scope

- `pynbodyext/core/calculate/store/codec.py`
- `tests/test_value_codec.py`
- `tests/test_result_store.py`
- `tests/test_sql_store.py`

## Acceptance criteria

- [x] A vector containing two lists remains shape (2,), with each list still one element.
- [x] Zero-dimensional object scalar and empty shapes (0,2), (2,0,3) round-trip.
- [x] Nested dict/tuple elements, non-contiguous views and object SimArray units survive.
- [x] Invalid dimensions or mismatched flat length fail explicitly; memory and reopened SQLite return the same shape/dtype/values.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Publish a scoped PR after authorization; local bilingual handoff and red/green evidence are ready.

```bash
.venv/bin/python -m pytest tests/test_value_codec.py tests/test_result_store.py tests/test_sql_store.py -q
```

For implementation: new regression/API tests fail first and pass after the change. This is a future validation requirement, not an execution claim from this review.

Detailed steps: [Implementation plan](plan.md).

## First regression scenario

```python
import json
import numpy as np
import pytest
from pynbodyext.core.calculate.store import ValueCodec


def object_vector():
    value = np.empty(2, dtype=object)
    value[0], value[1] = [1, 2], [3, 4]
    return value


@pytest.mark.parametrize("value", [object_vector(), np.array(7, dtype=object),
                                  np.empty((0, 2), dtype=object)])
def test_object_array_preserves_element_boundaries(value):
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.shape == value.shape
    assert restored.dtype == value.dtype
    assert restored.tolist() == value.tolist()
```

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.

## Local implementation — 2026-11-09

Versioned object-array payload stores shape, C-order elements and units; reconstruction preallocates and assigns each element. Added JSON, in-memory and reopened SQLite boundary regressions. Independent code review: no actionable findings.

- Red: 19 failed, 56 passed: shape inference, scalar iteration, lost units and missing new-tag validation.
- Focused green: 75 passed.
- `make check`: 1103 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- `make test-all`: 1107 passed, 1 skipped.

Status: review; locally implemented and verified, not committed, published or merged. No migration or unrelated architecture change. SQLite/Linux verified; other platforms/backends not certified.

## Integration outcome — 2026-11-09

Merged via [BUG-0010 PR](https://github.com/wx-ys/pynbody-extras/pull/42), merge commit `1b4f1b8f7c3187df2d7f3ac18e6ecdcfe8749e37`. Current status: done; this supersedes the historical local-only/review notes above. Exact scoped integration checks are recorded in handoff.md and the [integration record](../../design/integration-2026-11-09.md). No release was published.
