# BUG-0010 handoff

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

- Change: Versioned object-array payload stores shape, C-order elements and units; reconstruction preallocates and assigns each element. Added JSON, in-memory and reopened SQLite boundary regressions. Independent code review: no actionable findings.
- Baseline: 5df15dd; branch codex/sequential-review-fixes; pre-existing documentation retained.
- Red: 19 failed, 56 passed: shape inference, scalar iteration, lost units and missing new-tag validation. Logs: /tmp/bug0010-red.log.
- Focused green: 75 passed.
- make check: 1103 passed, 1 skipped, 4 deselected; lint/typecheck/docs passed.
- make test-all: 1107 passed, 1 skipped.
- Status: review, not committed/published/merged. No schema migration or architectural expansion.
- Next: human review and separately authorized Git publication. Later local items use the verified preceding diff; no claim that predecessors merged.
