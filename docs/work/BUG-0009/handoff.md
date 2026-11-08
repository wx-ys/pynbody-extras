## 2026-11-08 — Codex / local checkout

- State: in-progress; branch codex/p0-sim-identity-contract, base 3a30cff.
- User decision: development project has no old databases/caches; no compatibility
  or migration is required. ADR-006 records the direct typed-format change.
- Baseline: fast suite 916 passed / 1 skipped / 3 deselected, pynbody 2.8.0.
- Next: red-to-green typed identity regressions, full checks and human code review.
- Existing untracked .develop_notes/, .github/, .planning/, .test_code/ untouched.

## 2026-11-08 — Typed identity implementation complete

- State: review; user confirmed no old DB/cache compatibility is required.
- Added a shared typed JSON encoder, v3 record-schema validation, and typed
  identity equality for store listing and SampleTable indexing. SQL adapter
  inherits the encoding through the existing helper; no migration/dual read.
- Red: tests/test_sim_identity_store.py: 19 failed / 1 passed.
- Green: focused identity/store/reuse/sample: 80 passed / 1 skipped.
- Final make check: lint and mypy (111 files) pass; 936 passed / 1 skipped /
  3 deselected; docs 10 items. make test-all: 939 passed / 1 skipped.
- Baseline index check ran during work-item creation and was stale; regenerated
  and validated successfully in the final check. No production failure remains.
- Self-reviewed types, nested tuple serialization, invalid IDs, reopen, CRUD,
  SampleTable persistence and import boundaries. Human review remains required.
- ADR-006 is intentionally included despite docs/adr/ being ignored by default;
  only the new decision file is included, no unrelated local architecture notes.
- Next: review this PR, then immutable result references and data registry.
- No changes to pre-existing untracked directories; no dependency changes.

## 2026-11-08 — Publication and maintainer authorization

- State: done (implementation and local validation); PR #36 created.
- User explicitly authorized pushing to wx-ys/pynbody-extras, PR creation and merge.
- Implementation commit: 55b12db. Final tests: make check 936 passed /
  1 skipped / 3 deselected; make test-all 939 passed / 1 skipped.
- This follow-up changes only completion metadata/docs; no production/test edits.
- Merge after remote checks permit it; do not bypass branch protection.
- Retain work item until the next release or 30 days before archival.
