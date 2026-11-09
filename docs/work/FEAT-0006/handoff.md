# FEAT-0006 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-012 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.

## 2026-11-09 — local implementation and verification

- Approved ADR-012 implemented on the locally verified FEAT-0005 prerequisite;
  this does not claim a remote merge. Source is limited to the ORM owning seam.
- Added `execution_metadata` keyword to `run`, `run_input`, `begin_run`. Only
  tool_versions string maps and integer/null random_seed are accepted; bool,
  float, NumPy integer, unknown keys and invalid maps fail preflight with
  ValueError. Validation/JSON copying occurs before registration/execution.
- Run stores non-null `pynbodyext.orm.execution-context/v1` JSON Text with
  `scope="invocation"`. A getter-only property returns fresh decoded copies.
  Captures Python and three distribution versions, selected URI and logical
  loader_id/revision, loaded native info and caller metadata before loader/work.
  Missing distributions/native version information are null; native inspection
  uses only sys.modules and never imports the extension.
- Selected loader URI/prefix is preserved even when the returned filename is
  normalized or a later direct session edit changes DataLocation. Direct run and
  manual begin_run without selected location preserve URI null and no inferred
  alias/FK selection. Known logical loader/revision remain available.
- Cache-hit context belongs to this invocation/consumer; existing pinned payload
  receipt and keys are unchanged, and payload producer environment stays unknown.
  Context survives raised/collected errors, loader errors/invalid roots, failure
  recording interruption and result-link interruption/recovery after SQLite
  reopen. Repository finish/fail/recovery never recollect or replace it.
- RED: `.venv/bin/python -m pytest tests/test_orm_environment.py
  tests/test_orm_loading.py -q` -> **52 failed, 38 passed**. Failures showed missing
  metadata/context APIs or missing collector behavior, not an unavailable test
  environment. An initial test-only RunOptions import typo was corrected before
  capturing this red run. Log: `/tmp/feat0006-red.log`.
- GREEN: the same two-module command -> **90 passed**
  (`/tmp/feat0006-green-initial.log`). Exact spec command including associations
  and package layers -> **121 passed** (`/tmp/feat0006-focused.log`).
- `make check` -> lint/typecheck/docs clean; **1223 passed, 2 skipped,
  4 deselected**, **118 source files** typechecked. `/tmp/feat0006-check.log`.
- `make test-all` -> **1227 passed, 2 skipped**.
  `/tmp/feat0006-test-all.log`. No new skips were added.
- Files: new `pynbodyext/orm/environment.py`, ORM models/repository,
  new `tests/test_orm_environment.py`, appended loading regression tests;
  relevant ORM paragraphs only in CONTEXT.md; this item's bilingual specs,
  metadata/handoff and plan; regenerated INDEX for this item's review status.
  Existing tests and unrelated dirty source/docs were preserved.
- Status: **review**, not done/merged. No Git mutations, PR, release, dependency
  installation or subsequent feature work performed.
- Parent confirmed independent spec and code-quality review both passed without
  findings. Final `make docs-check` passed: **27 work items, 0 archived**
  (`/tmp/feat0006-docs-check.log`).
- Limits: Linux/SQLite verified; other platforms/dialects unverified. Direct
  caller-owned session writes remain caller responsibility. No migration,
  automatic seeding, cache invalidation, payload producer snapshot or concurrency
  retry/reconciliation was introduced. Native null is unknown loaded-state
  information, not an availability check.

## 2026-11-09 — Final distribution integration review

- Final reviewer reproduced that core/store knew its installed core version but
  Run context queried only full's distribution name. Recorded the ADR-019
  integration in ADR-012 and both specs before changing assertions.
- Preserve `packages.pynbodyext` as the independently observed full metadata
  version and add `packages.pynbodyext-core` for the code owner. Missing
  distributions remain null; execution keys and payload provenance are unchanged.
- Red: four selected checks fail because the core key is absent
  (`/tmp/final-code-owner-red.log`). Green: all 49 environment tests pass
  (`/tmp/final-code-owner-green.log`), including core-only/full variants and
  immutable reopen behavior. Whole-source and rebuilt artifact verification are
  recorded in the final TASK-0002 handoff and installation matrix.
- 最终组合审查补齐代码拥有者版本：保留 full 独立字段，新增 core 字段；先改双语
  合同与 ADR，再确认四项红灯和环境模块 49 项绿灯。未扩大缓存或环境采集范围。
