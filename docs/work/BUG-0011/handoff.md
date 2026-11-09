# BUG-0011 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: review reproduced the defect; copy the regression scenario to the indicated tests before implementation.
- Blockers: none for the scoped existing-contract correction.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — regression examples verified

- Done: executed the spec regression code from temporary test files against unchanged source.
- Commands: PYTHONPATH=. .venv/bin/python -m pytest /tmp/pynbody-review-red -q
- Result: five expected failures across the three items (3 object-array, 1 structured-array, 1 stale-cell-error). No source/test files added to the repository.
- Next: follow plan.md and the complete bilingual acceptance matrix; implement red to green.

## 2026-11-09 — Codex / local implementation

- Change: Added structured_array_v1 using NPY with pickle disabled, preserving shape, field layouts, byte order and units. Reject object fields, nested metadata and unreadable headers during encoding; validate dtype headers using a zero-element array and support wide headers within payload size. JSON, SQLite reopen, no-commit rejection and malformed payload regressions pass.
- Baseline: 5df15dd; branch codex/sequential-review-fixes; pre-existing documentation retained.
- Red: 40 failed, 103 passed; reviewer regressions 3 failed, 1 passed (unreadable titles and wide header). Logs: /tmp/bug0011-red.log.
- Focused green: 147 passed; re-review no actionable findings.
- make check: 1151 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- make test-all: 1155 passed, 1 skipped.
- Status: review, not committed/published/merged. No schema migration or architectural expansion.
- Next: human review and separately authorized Git publication. Later local items use the verified preceding diff; no claim that predecessors merged.

## 2026-11-09 — scoped integration verification

- User authorized commit, PR and merge. This increment was reconstructed in isolation after verified predecessors.
- `make check`: 1151 passed, 1 skipped, 4 deselected in 30.58s; `make test-all`: 1155 passed, 1 skipped in 30.26s. Log: `/tmp/pynbody-publish-test-2.log`.
- Independent complete-source review found no actionable findings; slice tests protect extraction boundaries. Publication/merge URL and final status are recorded after the remote operation is confirmed.

## 2026-11-09 — remote integration confirmed / 远端集成确认

- Merged: https://github.com/wx-ys/pynbody-extras/pull/43; merge commit `8c34db56a53f4dfc2185d9398c2a19266cd71800`.
- Actual remote state confirmed MERGED before status=done. User authorized autonomous PR/merge; predecessors were merged in dependency order.
- Checks: 1151 passed, 1 skipped, 4 deselected in 30.58s; 1155 passed, 1 skipped in 30.26s.
- Local main is synchronized separately and verified against the complete final source. Platform/backend limitations from the specification remain.
