# BUG-0007: Preserve NumPy scalars and dictionary keys in ValueCodec

## Context

The 2026-11-08 design audit reproduced data loss at the ResultStore codec seam.
The user approved starting the V1 plan on a new branch, retaining push, PR and
merge operations. This is the first P0 checkpoint.

## Goal

Decode existing NumPy scalar and literal tuple/None dictionary-key payloads
without changing their types or values. Keep the encoded representation intact.

## Actual behavior

NumPy numeric scalars raise TypeError because the decoder casts the encoded
dictionary. False NumPy booleans become true. None keys become strings; tuple
keys are split on commas, losing element types, nesting and quoting. Distinct
keys can collapse and overwrite values.

## Expected behavior

JSON and SQLite round trips preserve scalar dtypes and supported key identities.

## Reproduction

```python
import numpy as np
from pynbodyext.core.calculate.store import ValueCodec
codec = ValueCodec()
codec.decode(codec.encode(np.int64(3)))
codec.decode(codec.encode({None: 1, "None": 2, (1, 2): 3, ("1", "2"): 4}))
```

## Non-goals

No identity-key migration, immutable result versions, ORM tables, new codec
format, broader arbitrary-object support, SPH changes, or remote GitHub actions.

## Acceptance criteria

- [x] NumPy integer/unsigned/float/bool/string/bytes scalars retain type, dtype
  and value through JSON, including NaN/Inf.
- [x] None and literal tuple keys retain identity, including singleton/nested
  tuples and strings containing commas or quotes; distinct keys do not collapse.
- [x] Existing NoneType/none and literal tuple key records remain readable.
- [x] Invalid/nonliteral tuple key text is rejected without executing it.
- [x] Encoding and insertion-order-independent content hashes are unchanged.
- [x] SQLite close/reopen preserves mixed scalar values and key identities.
- [x] Run focused tests, make check and make test-all; record baseline failures.

## Design notes

Decision: a local decoder correction; no import-layer, ResultStore schema,
calculator-signature, simulation-key or pickle-policy changes. Human review is
left to the user's PR. No architecture migration is performed.

Recursively decode np_scalar.value, then obtain a NumPy scalar with [()] instead
of .item(); restore U/S scalar NUL padding from the stored dtype width. Recognize both NoneType and none tags. Parse tuple text using
ast.literal_eval, require a hashable tuple, and normalize invalid text to a
descriptive ValueError. Never use eval.

Supported tuple elements are Python literal values, including nested tuples.
Text such as (nan,) or (np.int64(1),) is rejected; arbitrary legacy object reprs
need a separate typed-key format proposal. Already decoded/resaved data that
lost its types cannot be automatically repaired.

## Test plan

Use writing-plans / TDD inline on codex/p0-value-codec-roundtrip.

- [x] Add tests/test_value_codec.py and a SQLite reopen regression in
  tests/test_sql_store.py, leaving existing assertions intact.
- [x] Run `.venv/bin/python -m pytest tests/test_value_codec.py tests/test_sql_store.py -q`;
  confirm scalar TypeError and key mismatch failures.
- [x] Change the relevant decoder branches in store/codec.py:

  ```python
  # np_scalar
  return np.asarray(self._decode(data["value"]), dtype=data["dtype"])[()]
  # None key
  if kind in ("none", "NoneType"):
      return None
  # tuple key: literal_eval(text), require tuple and call hash(value);
  # convert SyntaxError/ValueError/TypeError to descriptive ValueError.
  ```

- [x] Run focused codec/store/sample tests, make check and make test-all.
- [x] Review the diff, sync both languages and handoff, set review status,
  and prepare the PR description and user-operated Git commands.

## Outcome

Red: 29 failed, 9 passed across the new codec tests and SQL store tests.
Green: focused codec/store/reuse/sample tests: 95 passed, 1 skipped.
`make check`: ruff and mypy passed; fast tests: 850 passed, 3 failed,
1 skipped, 3 deselected. `make test-all`: 853 passed, 3 failed, 1 skipped.
All three failures are the pre-existing BUG-0002 SPH/pynbody 2.3.3 failures,
also reproduced before this patch (814 passed, 3 failed, 1 skipped in the fast
suite). No failing tests were removed or skipped. The docs check is run
separately because make check stops at the failing pytest target.

### Combined baseline verification — 2026-11-08

PR #34's image/SPH baseline and its floor-contract follow-up were merged into
this branch for integration validation. Only the generated work-item index
conflicted; regenerating it retained all eight items. No codec implementation or
existing numerical assertion changed during integration.

`make check` now passes: **866 passed, 1 skipped, 3 deselected**, with lint,
mypy and docs validation passing. `make test-all`: **869 passed, 1 skipped**.
Environment: Linux / Python 3.11.9 / pynbody 2.8.0. The earlier failing counts
above are historical evidence from the old dependency baseline.
The maintainer now explicitly authorizes push, PR updates and merge.

### Merge verification — 2026-11-08

Merged in [PR #33](https://github.com/wx-ys/pynbody-extras/pull/33)
(commit `88ae3b4`). Both checkpoints were validated together: make check passed
with 866 passed / 1 skipped / 3 deselected; make test-all passed with
869 passed / 1 skipped. Local main was synchronized to 88ae3b4 and matched the
validated integration tree. Retain the work item until the next release or 30 days.

## Open questions

None for this bounded repair. Typed-key format evolution is deferred.

Independent review found a trailing-NUL string-width issue. Four regressions
failed before its correction (4 failed / 31 passed); restoring padding from the
stored dtype fixed them. The reviewer verified 35 codec cases and eight
additional string edge probes, with no remaining findings.
