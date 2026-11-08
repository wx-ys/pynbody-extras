## What changed

Implement FEAT-0001: a reference returned by `store()` now stays pinned to the saved result. If a calculation saves 6 and later recomputes 12 under the same key, the first reference still fetches 6 while `get`/`load` and read-through return 12. `Result.store_ref` exposes the version saved or reused by a run; `list_versions` and reference JSON support historical lookup, including after reopening SQLite.

Each save appends a UUID version and updates the current cache atomically. `delete` only invalidates the current cache and never deletes history; an older pinned reference cannot invalidate a newer result. Both adapters resolve current reads through saved versions, and validate identities/signatures before value decoding. SQLite cache-update failures roll back the new version too.

The record format is v4. The project has no existing databases/caches and requires no migration or compatibility layer. History is retained without automatic GC; references must be paired with their owning backend. SQLite is tested; other SQL dialects need separate validation.

## Evidence

- [x] Wrote failing tests first: initial `tests/test_result_versions.py` run had 25 failures, including old references returning a refreshed value and missing version APIs.
- [x] Expanded version suite: 37 passed, covering write/hit/refresh receipts, failed refresh, invalidation, mutable-value isolation, typed JSON, mismatched records, reopen, rollback and simultaneous SQLite writers.
- [x] Existing signature-corruption regression plus related suites: 77 passed. All existing tests remain intact.
- [x] `make check`: ruff/mypy/docs pass; 973 passed, 1 skipped, 3 deselected.
- [x] `make test-all`: 976 passed, 1 skipped.

## Architecture

- [x] ADR added: `docs/adr/0007-immutable-result-versions.md`; the user approved continuation and publication/merge.
- [x] `CONTEXT.md`, both FEAT-0001 locales, metadata and handoff document the new contract.
- [x] Backends implement two additional primitives: `_get_version` and `_list_versions`.

## Checklist

- [x] Existing tests were not weakened to get a green run
- [x] Public behavior and user-facing docs are updated
- [x] No `testdata/`, `.venv/`, build output, secrets or unrelated local files are included
