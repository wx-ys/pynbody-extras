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

### Full output

`make typecheck` on the user's Mac (2026-10-20): **22 errors in 10 files**
(110 files checked). mypy 1.19.0, numpy 2.3.5, matplotlib 3.10.7,
SQLAlchemy 2.0.52. `note:` lines omitted.

```text
pynbodyext/gravity/base.py:193: error: Argument 1 to "full" has incompatible type "tuple[Any]"; expected "tuple[()]"  [arg-type]
pynbodyext/gravity/base.py:193: error: Argument 1 to "float" has incompatible type "generic[Any] | complex | str | bytes | memoryview[int]"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
pynbodyext/core/calculate/store/sqlalchemy_store.py:68: error: Incompatible types in assignment (expression has type "str | Engine", variable has type "Engine")  [assignment]
pynbodyext/core/calculate/store/sqlalchemy_store.py:107: error: Value of type "Any | None" is not indexable  [index]
pynbodyext/plot/image/smooth.py:173: error: No overload variant of "pad" matches argument types "ndarray[tuple[Any, ...], dtype[Any]]", "tuple[tuple[int, int], tuple[int, int]]", "str"  [call-overload]
pynbodyext/plot/image/smooth.py:338: error: Cannot call function of unknown type  [operator]
pynbodyext/plot/image/noise.py:93: error: Argument 1 to "_broadcast" has incompatible type "float | ... | None"; expected "ImageData | ..."  [arg-type]
pynbodyext/plot/image/cmaps.py:323: error: Argument "bad" to "with_extremes" of "Colormap" has incompatible type "str | tuple[float, ...]"; expected "tuple[float, float, float] | str | ... | None"  [arg-type]
pynbodyext/plot/image/cmaps.py:352: error: Argument 2 to "LinearSegmentedColormap" has incompatible type "dict[str, ndarray[...]]"; expected "dict[Literal['red', 'green', 'blue', 'alpha'], Sequence[tuple[float, ...]]]"  [arg-type]
pynbodyext/plot/image/cmaps.py:355: error: (same as 352)  [arg-type]
pynbodyext/plot/image/display.py:200: error: Incompatible types in assignment (expression has type "_ScalarMappable", variable has type "AxesImage | QuadMesh | ContourSet | PathCollection | Colorbar | None")  [assignment]
pynbodyext/plot/image/display.py:207: error: Argument 1 to "colorbar" of "FigureBase" has incompatible type "AxesImage | ... | None"; expected "_ScalarMappable | ColorizingArtist"  [arg-type]
pynbodyext/plot/image/display.py:242: error: Incompatible return value type (got "tuple[... | ImageData | AdaptiveMap | None, None]", expected "tuple[... | None, ImageData | None]")  [return-value]
pynbodyext/plot/image/display.py:272: error: Argument 1 to "set_aspect" of "_AxesBase" has incompatible type "str | float"; expected "Literal['auto', 'equal'] | float"  [arg-type]
pynbodyext/plot/image/display.py:456: error: Argument "shading" to "pcolormesh" of "Axes" has incompatible type "str"; expected "Literal['flat', 'nearest', 'gouraud', 'auto'] | None"  [arg-type]
pynbodyext/plot/image/display.py:526: error: Incompatible types in assignment (expression has type "ndarray[..., dtype[float64]]", variable has type "int | Sequence[float]")  [assignment]
pynbodyext/plot/image/display.py:1044: error: (same as 526)  [assignment]
pynbodyext/plot/image/compose.py:419: error: Argument 1 to "add_colorbar" has incompatible type "_ScalarMappable"; expected "AxesImage | ... | AdaptiveMap | None"  [arg-type]
pynbodyext/core/calculate/bins/_query_engines.py:216: error: Incompatible types in assignment (expression has type "ndarray[...]", variable has type "BinsArray")  [assignment]
pynbodyext/core/calculate/bins/result.py:353: error: Incompatible return value type (got "ndarray[...] | None", expected "ndarray[...]")  [return-value]
pynbodyext/transforms/wrap.py:212: error: Incompatible types in assignment (expression has type "ndarray[...]", variable has type "int")  [assignment]
pynbodyext/transforms/wrap.py:213: error: No overload variant of "__call__" of "_UFunc_Nin2_Nout1" matches argument types "int", "ndarray[..., dtype[float64]]", "int"  [call-overload]
```

By area: `plot/image` 14 (display 7, cmaps 3, smooth 2, noise 1, compose 1),
`core` 4 (store 2, bins 2), `gravity` 2, `transforms` 2. Most are stub
precision (matplotlib `Literal` parameters, numpy shape types) rather than
logic errors; `display.py:200/207/242` and `compose.py:419` share one
mistyped "mappable" alias and should be fixed together.

## Goal

`make typecheck` (`mypy pynbodyext`) reports no errors, so commits pass the
hook without `--no-verify`.

## Non-goals

- Changing runtime behaviour. Each fix is a type-level change (narrowing,
  annotation, `cast`) unless a real bug is found; a real bug gets its own item.
- Raising mypy strictness.

## Acceptance criteria

- [x] The full `make typecheck` output is recorded in this item
      (see Full output).
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

Some errors may depend on the numpy / matplotlib / sqlalchemy stub versions
(recorded with the output). The pre-commit hook uses `uv run mypy`; check it
agrees with `make typecheck` (`.venv/bin/python -m mypy`).

Suggested order (smallest blast radius first, one commit per area):
`gravity` + `store` + `bins` + `transforms` (8 errors, the ones that block
core commits), then `plot/image` (14).

## Test plan

- `make typecheck` before (record count) and after (zero).
- `make check`; `make test-all` if `gravity` or `store` code paths change.

## Outcome

## Open questions

- Is the error count the same with the versions in `uv.lock` and on the
  user's conda Python? The hook runs `uv run mypy`.
