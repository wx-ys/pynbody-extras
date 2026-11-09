# FEAT-0004 handoff

## 2026-11-09 — Codex / proposed next-stage design

- Started from merged FEAT-0003 / PR #39, main 2d32336; user asked to continue.
- Audited ORM, identity providers, tests and original V1 phase C/D requirements.
- Proposed explicit runtime loader bindings and root-only load/run APIs.
- Key unresolved choice: runtime bindings vs persisted declarative parameters.
- No implementation or runtime contract change yet. User review of en/zh-CN spec
  and proposed ADR-010 is required before implementation.
- Existing no-migration, optional-revision, separate-store/recovery and publication
  decisions persist. Preserve unrelated untracked user directories.
- Next after design approval: red tests, implementation plan, implementation,
  independent review, make check/test-all, scoped PR and merge under authorization.

- Baseline before implementation: ORM, package-layer and logical-identity suites
  passed (78 tests, 6.57 s); log /tmp/pynbody-loading-baseline.log.
- docs-check passed with 14 work items. Clarified preflight rejects unstable
  calculator signatures; constructibility is not a new run requirement.

## 2026-11-09 — design approved

- User selected: implement the draft with explicit loader/configuration bindings
  in project code. ADR-010 accepted. Execute the plan inline on the current branch.

## 2026-11-09 — implemented and verified

- Added orm/loading.py with LoadedInput/SnapshotLoader and local root validation.
- Repository copies loader bindings; get_input/load_input/run_input expose the
  accepted flow. Run.input_location records the selected source. Shared execution
  helper retains previous error/result/recovery semantics. No existing tests edited.
- RED: 30 missing-API failures plus 2 tightened binding-message failures. GREEN:
  initial loading+ORM 58 passed; expanded focused 133 passed (37 loading tests).
- Independent review approved: 68 loading/ORM/layer tests passed, no blockers.
- make check exit 0: ruff, mypy 117 modules, docs; 1082 passed, 1 skipped, 4 deselected.
- make test-all exit 0: 1086 passed, 1 skipped, including real local pynbody load.
- Logs: /tmp/pynbody-loading-{red,bindings-red,green,focused,check,all}.log.
- Remote main still 2d32336; next publish PR and merge under persistent user
  authorization. Keep unrelated untracked user directories untouched.

- First commit hook hit a transient mypy cache-deserialization KeyError
  ('is_type_form'). No code/typing-rule change was made. Re-running the exact
  file-scoped check with both original and fresh cache passed (4 source files).
  Commit uses MYPY_CACHE_DIR=/tmp/pynbody-loading-mypy-isolated-20261109; no hook bypass.

## 2026-11-09 — published

- PR: https://github.com/wx-ys/pynbody-extras/pull/40
- Validated source commit: 6eff07446c143cbe7d198d4bc17cead40d261792.
- Commit hooks passed with isolated mypy cache; no checks bypassed.
- This follow-up records PR metadata only. Verify final head/checks, mark ready,
  merge under existing authorization and synchronize main. PR status is the
  authoritative publication record. No unrelated user directories staged.
