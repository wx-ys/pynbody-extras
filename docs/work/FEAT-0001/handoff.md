## 2026-11-08 — Codex / immutable result versions

- State: in-progress; base 8f191b2; branch codex/immutable-result-references.
- User asked to continue immutable-history milestone; push/PR/merge authorized.
- ADR-007 records versioned saves vs current cache, deletion and receipt semantics.
- No old database/cache migration; no ORM or automatic historical deletion.
- Next: failing public-contract tests, implementation, full validation, PR.
- Preserve pre-existing untracked .develop_notes/, .github/, .planning/, .test_code/.

## 2026-11-08 — Codex / implementation verified

- State: done (implementation); publication/merge authorized and pending execution.
- Added immutable versions, current-cache lookup/invalidation, reference JSON and
  Result.store_ref in memory/SQLite; schema v4; no legacy migration.
- RED: original version contract suite 25 failed before implementation.
- GREEN: expanded version suite 37 passed; related regression suite 77 passed.
- make check: 973 passed, 1 skipped, 3 deselected; ruff/mypy/docs pass.
- make test-all: 976 passed, 1 skipped (both full commands exited 0).
- Existing signature-corruption test remains unchanged: memory cache reads now
  resolve the immutable version instead of trusting mutable cached payload metadata.
- SQL rollback, reopen and two simultaneous successful writers verified.
- History is retained indefinitely pending explicit GC; references require backend
  location; SQL dialects other than SQLite are not validated in this milestone.
- Next: publish this branch, record PR, merge without bypassing protections and
  synchronize local main. Later roadmap work: logical data identity/ORM Run links.
