# BUG-0011: Round-trip structured NumPy arrays without dtype string parsing

## Context and goal

Fix structured numeric arrays that encode successfully but cannot decode.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: BUG-0010

## Scope and contract

Add a dedicated versioned structured-array NPY tag using BytesIO and np.save/np.load with allow_pickle=False on both sides. Detect dtype.fields before the ordinary raw-byte path. Preserve shape, dtype field names/formats/offsets/itemsize, byte order and units when applicable. Reject structured dtypes with hasobject before serialization; never persist pointer bytes. Reject dtype forms that cannot be represented losslessly with a clear TypeError at encode time. Do not broaden opaque pickle behavior.

Dtype metadata, including metadata on nested fields, is explicitly unsupported and must raise before NPY serialization rather than being silently discarded. Overlapping/out-of-order field layouts rejected by NPY also fail during encoding.

## File scope

- `pynbodyext/core/calculate/store/codec.py`
- `tests/test_value_codec.py`
- `tests/test_sql_store.py`

## Acceptance criteria

- [x] Simple records, nested records, subarray fields and aligned offsets/itemsize survive JSON and SQLite reopen.
- [x] Zero-dimensional and empty structured arrays round-trip.
- [x] Object-bearing fields reject before any record is committed.
- [x] Truncated NPY data and an object-bearing NPY payload fail without invoking pickle; existing numerical-array/SimArray tests stay intact.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [ ] Publish a scoped PR after authorization; local bilingual handoff and red/green evidence are ready.

```bash
.venv/bin/python -m pytest tests/test_value_codec.py tests/test_sql_store.py tests/test_result_versions.py -q
```

For implementation: new regression/API tests fail first and pass after the change. This is a future validation requirement, not an execution claim from this review.

Detailed steps: [Implementation plan](plan.md).

## First regression scenario

```python
import json
import numpy as np
from pynbodyext.core.calculate.store import ValueCodec


def test_structured_array_roundtrip():
    value = np.array([(1, 2.0)], dtype=[("id", "i4"), ("mass", "f8")])
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)
```

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.

## Local implementation — 2026-11-09

Added structured_array_v1 using NPY with pickle disabled, preserving shape, field layouts, byte order and units. Reject object fields, nested metadata and unreadable headers during encoding; validate dtype headers using a zero-element array and support wide headers within payload size. JSON, SQLite reopen, no-commit rejection and malformed payload regressions pass.

- Red: 40 failed, 103 passed; reviewer regressions 3 failed, 1 passed (unreadable titles and wide header).
- Focused green: 147 passed; re-review no actionable findings.
- `make check`: 1151 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- `make test-all`: 1155 passed, 1 skipped.

Status: review; locally implemented and verified, not committed, published or merged. No migration or unrelated architecture change. SQLite/Linux verified; other platforms/backends not certified.
