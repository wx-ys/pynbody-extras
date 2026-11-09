# FEAT-0012 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-018 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.


## 2026-11-09 — local serial study implementation and frozen-source verification

- Implemented ORM-only `run_study`, frozen StudyCell/StudyReport and StudyPersistenceError under ADR-018; exports, 95 behavioral study cases, owning CONTEXT section and paired work-item evidence updated. Full-list preflight is read-only before any Run/loader/recipe write, validates exact positive unique root IDs/labels, bindings/location ownership and existing recipe conflicts, and retains configured options/stable nonconstructible live calculators. Views replayed by run_selection remain excluded. No lower-layer ORM import or new executor.
- Real SQLite/pynbody 2×2 numerical cases cover ordered runs/values/locations/refs, repeated new Run IDs on cache reuse, refresh and historical values after both databases reopen/current invalidation. Recorded thrown and collected science failures continue; default raise stops. Real finalization/fail_run metadata failures halt with exact original receipt/result/ref/cause and completed partial report, with caller-only recovery and explicit reinvocation. Raw Run-creation metadata failures propagate.
- Strict versioned report JSON preserves full axes, values/units/nested NumPy/None, typed identities/pinned refs and ErrorInfo fields without backend access or calculator reconstruction. Malformed fields, duplicates, unknown locations and mismatched refs reject. Nested RecordRef JSON has its own strict duplicate/field/non-finite boundary before the existing typed-ref decoder.
- Red: **91 failed**, all missing API (`/tmp/feat0012-red.log`); first green **91 passed** in 7.09s (`/tmp/feat0012-green-attempt.log`). Nested-ref regressions **4 red**, then green (`/tmp/feat0012-ref-red.log`, final focused evidence below).
- Final frozen-source focused study/associations/loading/SampleTable refs/layering/calculator contracts: **193 passed, 1 skipped** in 17.76s (`/tmp/feat0012-focused.log`). `make check`: lint/mypy124/docs27 passed, **1670 passed, 2 skipped, 5 deselected** in 70.78s (`/tmp/feat0012-check.log`). `make test-all`: **1675 passed, 2 skipped** in 72.39s (`/tmp/feat0012-all.log`). Optional pandas accounts for skips; other SQL dialects/platforms are unverified.
- Status=review, generated index refreshed; source/tests frozen before broad checks and parent reviews. Only documentation/status updated afterward. Verified local prerequisites do not imply remote merge; no Git/subagent/dependency/migration/remote actions in this increment.
- Next: parent completes independent specification/quality reviews and records results, then manages any separately authorized integration. No automatic attach/delete/retry/resume or exactly-once promise.

## 2026-11-09 — scoped integration verification

- User authorized commit, PR and merge. This increment was reconstructed in isolation after verified predecessors.
- `make check`: 1672 passed, 2 skipped, 5 deselected in 60.39s (0:01:00); `make test-all`: 1677 passed, 2 skipped in 61.61s (0:01:01). Log: `/tmp/pynbody-publish-test-11.log`.
- Independent complete-source review found no actionable findings; slice tests protect extraction boundaries. Publication/merge URL and final status are recorded after the remote operation is confirmed.
