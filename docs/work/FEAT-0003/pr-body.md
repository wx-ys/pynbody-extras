## What changed

Implement [FEAT-0003](https://github.com/wx-ys/pynbody-extras/blob/8af0f94/docs/work/FEAT-0003/en.md): a new explicitly imported `pynbodyext.orm` package connects Simulation → Timestep → DataObject → Run → Recipe/ResultLink using SQLAlchemy mappings. Locations remain separate metadata, recipes reuse existing stable CalculatorSignature DAGs, and result links contain pinned RecordRef JSON. Scientific arrays and named pipeline outputs remain in ResultStore.

`AnalysisRepository.run` commits a running row, delegates to the existing calculator, then atomically finalizes metadata with the exact saved/reused version. Cache hits receive their own run records; refreshing a cache does not retarget historical run results. Query helpers and ordinary SQLAlchemy sessions expose the relationships.

The user explicitly selected separate metadata/payload stores. Final metadata failures expose run ID and the saved receipt for `finish_run` recovery. Audit checks associations without decoding scientific values; unlinked versions are recovery candidates, never automatic ownership guesses or deletion targets. Raised/collected errors, payload write failures and metadata interruptions remain inspectable. Conflicting terminal transitions are rejected with conditional updates.

SQLite 3.35+ is tested; no other dialect is claimed validated. This increment has one root input/output per run, no distributed retries/GC, no old-database migration, and no packaging redesign. It uses the existing `store` extra; ADR-005 full/lightweight distribution work remains separate.

## Evidence

- [x] Failing tests first: 20 missing-ORM API failures, with the pure-calculation boundary test already passing; then 21 passed.
- [x] Two additional malformed-reference/hierarchy tests failed before integrity fixes.
- [x] Final ORM suite: 26 passed, including SQLite reopen, pinned history, real concurrent completion, metadata rollback/recovery and payload/failure-recording errors.
- [x] Independent review: no actionable blockers; 31 ORM/package-layer tests passed.
- [x] `make check`: ruff, mypy (116 modules), docs; 1046 passed, 1 skipped, 3 deselected.
- [x] `make test-all`: 1049 passed, 1 skipped.

## Architecture

- [x] [ADR-009](https://github.com/wx-ys/pynbody-extras/blob/8af0f94/docs/adr/0009-orm-analysis-associations.md) records separate-store lifecycle, identity rules, explicit recovery and limits.
- [x] Declared `orm` above `calculate` in package layering; no exceptions or reverse imports.
- [x] CONTEXT and bilingual FEAT-0003 docs updated. No dependency or payload schema changes.

## Checklist

- [x] Existing tests were not weakened; the package-layer registry intentionally adds the new layer per ADR-009
- [x] Public behavior and docs updated
- [x] No testdata, virtual environments, build output, secrets or unrelated local files included
