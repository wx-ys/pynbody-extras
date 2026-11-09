# BUG-0011 Implementation Plan

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
from pynbodyext.core.calculate.store import ValueCodec


def test_structured_array_roundtrip():
    value = np.array([(1, 2.0)], dtype=[("id", "i4"), ("mass", "f8")])
    codec = ValueCodec()
    restored = codec.decode(json.loads(json.dumps(codec.encode(value))))
    assert restored.dtype == value.dtype
    np.testing.assert_array_equal(restored, value)
```

- [x] Run `.venv/bin/python -m pytest tests/test_value_codec.py -q`. Expected: the new scenario fails for the behavior described in the spec, while unrelated existing cases retain their baseline. Object-array cases currently produce shape mismatches and a scalar TypeError; structured arrays currently fail dtype parsing; recovered cells currently retain their error.

## Task 2: Correct the owning boundary

- [x] Apply this scoped change at the indicated method. Preserve all branches not shown.

```python
# Add `import io`. In _encode_array, before the ordinary raw-byte branch:
if value.dtype.fields is not None:
    if value.dtype.hasobject:
        raise TypeError("structured arrays containing object fields are unsupported")
    _reject_dtype_metadata(value.dtype)
    buffer = io.BytesIO()
    try:
        np.save(buffer, np.asarray(value), allow_pickle=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("structured dtype cannot be represented losslessly") from exc
    encoded = buffer.getvalue()
    payload = {"__kind": "structured_array_v1", "data_b64": _b64(encoded)}
    if units is not None:
        payload["units"] = units
    return payload

# Add a module-level helper; nested metadata would otherwise be lost by NPY.
def _reject_dtype_metadata(dtype: np.dtype[Any]) -> None:
    if dtype.metadata is not None:
        raise TypeError("structured dtype metadata is unsupported")
    if dtype.subdtype is not None:
        _reject_dtype_metadata(dtype.subdtype[0])
    if dtype.fields is not None:
        for name in dtype.names or ():
            _reject_dtype_metadata(dtype.fields[name][0])

# In _decode, add a dedicated branch:
if kind == "structured_array_v1":
    array = np.load(io.BytesIO(_unb64(data["data_b64"])), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.fields is None or array.dtype.hasobject:
        raise ValueError("invalid structured-array payload")
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
- [x] Prepare one reviewable diff. If commits are authorized, stage only the listed code/tests and this work-item documentation; use commit message `fix: address bug-0011`. Publishing/merging follows the active user's authorization, not an assumed permission in this document.

## Actual implementation / 实际实施

Implemented locally and verified; status review, not merged or published. See handoff.md for exact red/green counts and additional review corrections. Proposed snippets above are planning history; codec implementation additionally validates NPY header readability and supports payload-sized headers. No architecture expansion or database migration.
