# FEAT-0011 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-017 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.

## 2026-11-09 — local JobSpec implementation and review correction

- Implemented frozen six-field `JobSpec` (`pynbodyext.orm.job/v1`) with exact built-in scalar types, positive int IDs excluding bool, nullable location, nonempty backend label, reuse bool, strict missing/extra/duplicate/schema JSON rejection and non-finite constant rejection. API: from_calculator/to_json/from_json/execute(repo). No live bindings, snapshots, credentials, backend URLs or pickle enter manifest fields; scientific parameter strings remain caller-controlled and are not redacted.
- Full nested payload preflight checks explicit registered calculator names/aliases and already-loaded Enum types before any constructor. Class-path-only custom calculators are excluded even if importable; projects explicitly import/register types first. No automatic plugin import, global monkeypatch, registry policy or generic decoder change. Registered constructors/rebuilders remain trusted application code.
- Reject non-default RunOptions in the original dependency graph before signatures discard them, and in reconstructed graphs. Execute rechecks type availability, accepts root snapshots only (rejects valid FEAT-0010 selections as well as digest-only views), and calls existing run_input without a second loading/execution stack. Existing RunExecutionError/RunPersistenceError and receipt/recovery semantics remain the owning behavior.
- Initial expected missing-API red: 1 failed (`/tmp/feat0011-red.log`); first new-module green: 55 passed (`/tmp/feat0011-jobs.log`). Added loaded-non-Enum callable and nested signature-wrapper flag regressions.
- Specification review found silent recipe drift: generic reconstruction warned about stale versions and could rebuild different scientific parameters. Parent confirmed correction within approved exact-recipe/version scope. Added four red regressions for root/nested stale versions and dataclass/generic parameter drift (`/tmp/feat0011-drift-red.log`). The job boundary now checks every class-bearing node's version before any constructor, then compares reconstructed canonical identity hashes after recursively normalizing registered aliases and omitted zero versions. Numerical equivalence is retained, booleans remain distinct; generic decoder behavior outside jobs is unchanged. All 62 new-module cases pass (`/tmp/feat0011-drift-green.log`), including nested aliases/numeric equivalence and loaded Enum support.
- Real new-process test reopens metadata/payload SQLite, explicitly imports ParamSum and binds loader/store: initial 6, child loads changed masses totaling 30 but cache hit returns 6 with new Run/same signature/ref, refresh returns 30/new version, original Run remains fetchable as 6. Asserts actual scientific values, signature envelopes, cache keys/version IDs, run IDs/reuse flags, loader counts, metadata links and clean audit. Preflight tests assert no loader calls, scientific calculate calls or Runs.
- Pre-correction focused/check/all logs were successful but are preserved as `/tmp/feat0011-{focused,check,all}-pre-drift.log`; they are not final-source evidence. Corrected runtime/tests are frozen for independent re-review while the renewed focused/check/all commands run. Final verification counts follow in a separate entry.
- Files: pynbodyext/orm/jobs.py, orm/__init__.py, tests/test_orm_jobs.py, owning CONTEXT section, this item's EN/ZH/meta/handoff and generated docs/work/INDEX.md. Existing shared dirty state and FEAT-0006/0009/0010 source were preserved. No dependency, migration, core/provider rewrite, Git command, commit/push/PR/release or subagent. Local prerequisites were verified; no remote merge is claimed. SQLite/Linux only; other dialects/platforms remain unverified.

## 2026-11-09 — corrected-source verification before alias-only probe

- Focused jobs/loading/signature/type-identity/selections/package/core-layering/calculator-contract suite: **226 passed** in 22.94s (`/tmp/feat0011-focused-pre-alias.log`).
- `make check`: lint and mypy passed; **1572 passed, 2 skipped, 5 deselected** in 62.95s; docs check passed with 27 work items (`/tmp/feat0011-check-pre-alias.log`). `make test-all`: **1577 passed, 2 skipped** in 64.29s (`/tmp/feat0011-all-pre-alias.log`). Both skips require optional pandas. These logs are after the recipe-drift correction but before the alias-only correction below; they are not final-source evidence.
- Standalone docs validation passed (`/tmp/feat0011-docs.log`). Both locales, owning CONTEXT section, metadata status=review and generated index are updated. Runtime/tests remain frozen; only documentation/status changed after final-source verification.
- Next: parent obtains independent specification re-review and quality review, records their actual results, then manages the authorized integration workflow. No Git/subagent/remote actions were used in this implementation subtask.

## 2026-11-09 — alias-only registered type correction

- Parent probe identified a valid registered alias without a declared signature_name: incoming alias passed, but rebuilt canonical class name was rejected. Root/nested module-level alias-only regressions went **2 red** (`/tmp/feat0011-alias-red.log`) then green. The test copies signature JSON rather than mutating the calculator's shared signature payload, verifies JSON roundtrip and tracked scientific value 18, verifies no canonical global registration, and retains initial class-path-only/from_calculator rejection.
- Jobs-local trusted-type mapping is filled only from the complete incoming explicit-registry preflight, recording vetted types and their canonical names. Rebuilt validation and recipe normalization use that mapping. No global registration, imports, generic decoder change or initial arbitrary-class-path expansion. A separate regression verifies a rebuilder cannot switch to another registered-but-unvetted type; prior parameter-drift/version/type/Enum guards remain green.
- New-module suite **65 passed** (`/tmp/feat0011-alias-green.log`). Runtime/tests renewed freeze sent to parent for independent review. Previous successful broad logs renamed `*-pre-alias.log`; final focused/check/all are repeated on the corrected source and their actual counts follow.

## 2026-11-09 — final alias-corrected source verification

- Focused jobs/loading/signature/type identity/selections/layering/calculator contracts: **229 passed** in 24.15s (`/tmp/feat0011-focused.log`).
- `make check`: lint, mypy (123 source files), docs27 passed; **1575 passed, 2 skipped, 5 deselected** in 67.25s (`/tmp/feat0011-check.log`). `make test-all`: **1580 passed, 2 skipped** in 67.09s (`/tmp/feat0011-all.log`). Both skips require optional pandas. These are final alias-corrected frozen-source logs; earlier pre-alias/pre-drift successes are explicitly historical.
- Both locales and owning CONTEXT explain the local vetted-type mapping and unchanged initial explicit-registration boundary; status remains review. Standalone final docs validation passes (`/tmp/feat0011-docs.log`). Runtime/tests unchanged after renewal of freeze; only documentation counts updated after final runs.
- Next: parent records renewed independent review results and decides subsequent authorized integration. FEAT-0012 not started here. No Git/subagent/dependency/global registry/decoder/import policy action.

## 2026-11-09 — renewed independent review confirmed

- Parent reported alias-corrected specification and quality re-reviews both passed; targeted reviewer verification: **10 passed, 55 deselected**. Parent independently verified final focused229/check1575+2skip+5deselected/all1580+2skip logs. No source/test changes or broad reruns followed this confirmation.
- Both locales record final review results and completed documentation work. Metadata remains status=review; generated index is refreshed from current metadata and final docs-check remains passing. No commit/PR/remote merge is claimed.
- Parent is proceeding to FEAT-0012. This subtask did not start FEAT-0012 or perform Git/subagent/remote actions; shared dirty state is preserved.

## 2026-11-09 — scoped integration verification

- User authorized commit, PR and merge. This increment was reconstructed in isolation after verified predecessors.
- `make check`: 1577 passed, 2 skipped, 5 deselected in 55.11s; `make test-all`: 1582 passed, 2 skipped in 54.59s. Log: `/tmp/pynbody-publish-test-10.log`.
- Independent complete-source review found no actionable findings; slice tests protect extraction boundaries. Publication/merge URL and final status are recorded after the remote operation is confirmed.

## 2026-11-09 — remote integration confirmed / 远端集成确认

- Merged: https://github.com/wx-ys/pynbody-extras/pull/51; merge commit `a7d6f6a053ed41a439a55ce3e88e3c6b2f2f46cb`.
- Actual remote state confirmed MERGED before status=done. User authorized autonomous PR/merge; predecessors were merged in dependency order.
- Checks: 1577 passed, 2 skipped, 5 deselected in 55.11s; 1582 passed, 2 skipped in 54.59s.
- Local main is synchronized separately and verified against the complete final source. Platform/backend limitations from the specification remain.
