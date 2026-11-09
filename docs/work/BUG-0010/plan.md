# BUG-0010 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkboxes. This plan does not request sub-agent delegation.

**Goal:** Satisfy the bilingual work-item contract without changing unrelated runtime behavior.

**Architecture:** Preserve Calculator DAG, ResultStore and the existing import layers. Correct the existing codec/cell state boundary only. The code below is a proposed minimal patch, not code already installed or verified green.

**Tech Stack:** Python >=3.11, NumPy, pynbody >=2.8.0, pytest; SQLite integration via existing store extra.

## Files and prerequisites

- Modify: `pynbodyext/core/calculate/store/codec.py`.
- Add regression code to: `tests/test_value_codec.py` (reuse existing imports when present).
- Read all paths and prerequisite IDs from [the specification](en.md).
- Update both locale files, metadata, handoff and generated index after implementation.

## Task 1: Reproduce the behavior

- [x] Add the following regression to `tests/test_value_codec.py`. In an existing test file merge imports rather than duplicating definitions.

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

- [x] Run `.venv/bin/python -m pytest tests/test_value_codec.py -q`. Expected: the new scenario fails for the behavior described in the spec, while unrelated existing cases retain their baseline. Object-array cases currently produce shape mismatches and a scalar TypeError; structured arrays currently fail dtype parsing; recovered cells currently retain their error.

## Task 2: Correct the owning boundary

- [x] Apply this scoped change at the indicated method. Preserve all branches not shown.

```python
# In ValueCodec._encode_array, replace only the dtype.kind == "O" branch.
if value.dtype.kind == "O":
    payload = {
        "__kind": "array_objects_v2",
        "shape": list(value.shape),
        "items": [self._encode(item) for item in value.flat],
    }
    if units is not None:
        payload["units"] = units
    return payload

# In ValueCodec._decode, add before the existing array_objects branch.
if kind == "array_objects_v2":
    return self._decode_object_array(data)

# Add this private method; existing imports include math and Any.
def _decode_object_array(self, data: dict[str, Any]) -> Any:
    shape = data["shape"]
    items = data["items"]
    if (not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(items, list)
            or math.prod(shape) != len(items)):
        raise ValueError("invalid object-array shape or element count")
    array = np.empty(tuple(shape), dtype=object)
    for index, item in enumerate(items):
        array.flat[index] = self._decode(item)
    if "units" in data:
        from pynbody.array import SimArray
        return SimArray(array, units=data["units"])
    return array
```

- [x] Add the remaining spec boundary cases before broad verification. In particular, exercise SQLite reopen for codec fixes; for cell recovery test cache miss and a store read exception. Assert public values/state, not internal call choreography.
- [x] Run `.venv/bin/python -m pytest tests/test_value_codec.py -q`; expected all tests in the file pass.

## Task 3: Verify and hand off

- [x] Run the exact focused command in [the specification](en.md); expected exit 0.
- [x] Run `make check` and `make test-all`; expected exit 0. A source-only happy path is insufficient for serialization/public behavior.
- [x] Review dtype metadata/unsupported-type policy against the spec before claiming universal NumPy coverage. Do not silently discard unsupported metadata. If the minimal patch needs a correction to satisfy a listed case, update it narrowly and repeat that failed case.
- [x] Update en.md and zh-CN.md outcomes together, append actual red/green logs to handoff, set metadata according to the real implementation/review state, and run `make docs-index` then `make docs-check`.
- [x] Prepare one reviewable diff. If commits are authorized, stage only the listed code/tests and this work-item documentation; use commit message `fix: address bug-0010`. Publishing/merging follows the active user's authorization, not an assumed permission in this document.

## Actual implementation / 实际实施

Implemented locally and verified; status review, not merged or published. See handoff.md for exact red/green counts and additional review corrections. Proposed snippets above are planning history; codec implementation additionally validates NPY header readability and supports payload-sized headers. No architecture expansion or database migration.
