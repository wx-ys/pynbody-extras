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
- [ ] `make check` passes locally: ruff and mypy passed; pytest reported
  **850 passed, 3 failed, 1 skipped, 3 deselected**.
- [ ] `make test-all` passes: **853 passed, 3 failed, 1 skipped**.
- [x] Separate `make docs-check` passed (8 work items); `git diff --check` passed.

The three full-suite failures are pre-existing [BUG-0002](docs/work/BUG-0002/en.md)
SPH compatibility failures with pynbody 2.3.3 (two missing `Kernel.value` calls and
one unequal-z grid placement). Before this patch, the fast suite reported
814 passed / the same 3 failed / 1 skipped. These failures remain visible;
this PR does not claim a green full suite or resolve BUG-0002.

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
