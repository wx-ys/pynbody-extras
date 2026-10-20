# TASK-0001: Fix the existing mypy errors so the pre-commit hook passes

## Context

The pre-commit `mypy` hook runs on every staged file under `pynbodyext/` and
follows its imports. `main` already has mypy errors, so any commit touching
those modules fails the hook. BUG-0003 (PR #25) had to commit with
`--no-verify`; BUG-0001 noted 22 errors from `make typecheck`, all in files
it did not touch.

Errors the hook reported for the BUG-0003 commit (`bins/result.py` staged):

```text
pynbodyext/gravity/base.py:193: error: Argument 1 to "full" has incompatible type "tuple[Any]"; expected "tuple[()]"  [arg-type]
pynbodyext/gravity/base.py:193: error: Argument 1 to "float" has incompatible type "generic[Any] | complex | str | bytes | memoryview[int]"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
pynbodyext/core/calculate/store/sqlalchemy_store.py:68: error: Incompatible types in assignment (expression has type "str | Engine", variable has type "Engine")  [assignment]
pynbodyext/core/calculate/store/sqlalchemy_store.py:107: error: Value of type "Any | None" is not indexable  [index]
pynbodyext/core/calculate/bins/result.py:353: error: Incompatible return value type (got "ndarray[tuple[Any, ...], dtype[Any]] | None", expected "ndarray[tuple[Any, ...], dtype[Any]]")  [return-value]
```

## Goal

`make typecheck` (`mypy pynbodyext`) reports no errors, so commits pass the
hook without `--no-verify`.

## Non-goals

- Changing runtime behaviour. Each fix is a type-level change (narrowing,
  annotation, `cast`) unless a real bug is found; a real bug gets its own item.
- Raising mypy strictness.

## Acceptance criteria

- [ ] The full `make typecheck` output is recorded in this item (the list
      above is only part of it).
- [ ] `make typecheck` passes on the locked environment (`uv.lock`).
- [ ] No new `# type: ignore` without an error code and a reason.
- [ ] `make check` passes; no runtime behaviour changes.

## Design notes

Likely fixes for the five known errors:

- `gravity/base.py:193` — `np.isscalar` does not narrow `softening`. Use
  `isinstance(softening, (int, float, np.floating, np.integer))` or
  `np.ndim(softening) == 0` plus `float(np.asarray(softening))`, and pass the
  shape as `(int(pos.shape[0]),)`.
- `store/sqlalchemy_store.py:68` — `hasattr(..., "connect")` does not narrow
  `str | Engine`; use `isinstance(url_or_engine, str)` (or `cast`).
- `store/sqlalchemy_store.py:107` — `inserted_primary_key` is `Row | None` in
  the stubs; check for `None` (raise if missing) before indexing.
- `bins/result.py:353` — `BinResultModel.valid_mask` is
  `np.ndarray | None`, while `BinNDResult.valid_mask` promises `np.ndarray`.
  Decide whether `None` can reach it; if not, narrow at assignment (line 270),
  otherwise build the all-true mask.

Some errors may depend on the numpy / sqlalchemy stub versions; record the
versions with the output.

## Test plan

- `make typecheck` before (record count) and after (zero).
- `make check`; `make test-all` if `gravity` or `store` code paths change.

## Outcome

## Open questions

- Is the error count the same with the versions in `uv.lock` and on the
  user's conda Python? The hook runs `uv run mypy`.
