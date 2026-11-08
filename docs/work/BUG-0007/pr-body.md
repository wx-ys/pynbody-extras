## What changed

Fix ValueCodec decoding so NumPy scalars keep their dtype and value, None keys
remain None, and literal tuple keys preserve element types, nesting and quoting.
Previously a result such as `{None: 1, "None": 2}` could lose a value on reload,
and NumPy numeric scalars raised TypeError. The encoded format is unchanged;
nonliteral tuple representations now raise ValueError instead of silently
changing identity. Includes a SQLite close/reopen regression and trailing-NUL string-width cases.

Work item: [BUG-0007](docs/work/BUG-0007/en.md), first P0 checkpoint of the V1 plan.

## Evidence

- [x] Failing tests first: codec + SQL tests produced **29 failed, 9 passed**.
  Failures included casting an encoded dictionary to a NumPy scalar, lost key
  identities and missing rejection of nonliteral tuple text.
- [x] Focused codec/store/reuse/sample suite: **95 passed, 1 skipped**.
- [x] `make check`: ruff, mypy and docs pass; **866 passed, 1 skipped, 3 deselected**.
- [x] `make test-all`: **869 passed, 1 skipped**.
- [x] `make docs-check` passes (8 work items); `git diff --check` passes.

Validated together with the image/SPH baseline in PR #34, on Linux / Python
3.11.9 / pynbody 2.8.0. The previous three SPH failures are resolved by that
baseline. Only the generated work-item index conflicted during integration;
regeneration preserved all items. Merge #34 first; this PR then adds only the
codec correction and its tests/documentation.

## Architecture

- [x] No ADR needed: localized decoder correction; no import-layer, schema,
  simulation-key, calculator-signature or pickle-policy change. Compatibility
  decision and limits are recorded in BUG-0007.
- [ ] ADR added or updated under `docs/adr/`.

## Checklist

- [x] Existing tests were not weakened to get a green run.
- [x] ValueCodec documentation and bilingual work-item records are updated.
- [x] No testdata, .venv, build output, unrelated local files or secrets included.

Nonliteral historical tuple keys cannot be reconstructed by this format and now
fail explicitly. Already decoded/resaved data that lost key types cannot be
automatically recovered. A future typed-key format is outside this checkpoint.

Independent review found a trailing-NUL string-width edge case, reproduced with
four failing tests and fixed by restoring padding from the stored dtype.
Follow-up review found no remaining issues.
