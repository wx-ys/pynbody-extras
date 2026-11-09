## 2026-11-09 — Codex / ORM association design

- Base main a29081c; branch codex/orm-analysis-associations.
- User explicitly approved separate metadata/payload writes with validation and
  explicit recovery, rather than requiring a shared SQL transaction.
- Preserve optional revision and manual refresh; no legacy migration.
- Existing push/PR/merge authorization persists. Preserve unrelated untracked dirs.
- New orm package above calculate; no core reverse import or new dependency.
- Next: red tests, implementation, independent review, full checks and publication.

## 2026-11-09 — Codex / implementation verified

- Implementation complete, source based on a29081c. New orm package with seven
  SQLAlchemy models; no duplicate scientific payload or replacement executor.
- Baseline: 91 passed. Initial RED: 20 failed + 1 pure-calc passed; GREEN: 21 passed.
- Integrity RED: 2 failed (malformed ref shape and mismatched input hierarchy),
  then validation fixed. Expanded ORM tests: 26 passed.
- Focused integration/layering before final 3 tests: 121 passed. Independent final
  review: no actionable blockers; 31 ORM/package-layer tests passed.
- make check exited 0: ruff, mypy 116 modules, docs; 1046 passed, 1 skipped,
  3 deselected. make test-all exited 0: 1049 passed, 1 skipped.
- SQLite foreign keys active; terminal updates use UPDATE RETURNING (SQLite 3.35+).
- New declared orm package layer above calculate; existing tests not weakened.
- Explicit recovery requires a known pinned ref; unlinked versions are candidates
  scoped to this repository/store name. No global orphan deletion or automatic retry.
- Next: publish PR, check base/head/checks, merge and synchronize local main under
  existing authorization. Keep unrelated pre-existing untracked directories intact.

## 2026-11-09 — Codex / published

- PR: https://github.com/wx-ys/pynbody-extras/pull/39
- Validated implementation commit: 8af0f94; commit hooks ruff/mypy passed.
- This follow-up only records PR metadata; source and tests are unchanged.
- Merge already authorized. Verify final head/base/checks, mark ready, merge
  without bypassing protections and fast-forward local main.
