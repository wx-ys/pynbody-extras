## 2026-11-09 — Codex / design and implementation start

- Base main aaf9b92; branch codex/logical-snapshot-identity.
- User requested next step and explicitly selected optional revision, with user
  controlled cache refresh. ADR-003/004 logical identity/location separation retained.
- ADR-008 records the concrete provider and configuration contract.
- Prior push/PR/merge authorization persists. Preserve pre-existing untracked
  .develop_notes/, .github/, .planning/, .test_code/.
- Next: red tests, small provider implementation, focused/full checks, publication.

## 2026-11-09 — Codex / implementation verified

- State: done implementation; publication pending. User's optional revision choice
  implemented with explicit reuse=False refresh and no automatic content checks.
- RED: 45 missing API failures; GREEN: all 45 passed. Additional 2 default-label
  regressions failed (v1 names), then passed with logical JSON labels/view suffixes.
- Baseline relevant suites: 75 passed. Focused integration/layering: 150 passed,
  1 skipped before label follow-up. Final identity/sample suites: 70 passed, 1 skipped.
- make check exited 0: ruff, mypy (112 modules), docs; 1020 passed, 1 skipped,
  3 deselected. make test-all exited 0: 1023 passed, 1 skipped.
- Independent reviewer rechecked public typing, transformations and row naming;
  final verdict approved pending full checks (now passed). Existing tests unchanged.
- JSON from_json is a static factory for the concrete schema provider, avoiding a
  generic constructor/classmethod inference error. No subclass factory promise.
- No DB schema, dependencies or automatic data-loading changes. No migration.
- Next: publish, record PR, verify head/checks, merge under existing authorization,
  fast-forward local main. Next roadmap layer: ORM logical object/Run associations.
