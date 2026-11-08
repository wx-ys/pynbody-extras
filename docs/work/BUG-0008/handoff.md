## 2026-11-08 — Codex / local checkout

- State: in-progress; branch codex/fix-median-constant-boundary, from main 88ae3b4.
- Scope: follow-up from PR #34, honoring the documented constant=NaN boundary.
- Reproduced: 2x2 positive image returns zeros, with and without invalid pixels.
- Plan: red cropped-window regression, minimal padding/path fix, focused/full
  validation, independent review, push/PR/merge under continuing user authorization.
- Non-goals: other modes, even sizes, performance redesign and architecture.

## 2026-11-08 — Implementation and validation

- State: review. Both all-valid and masked/nonfinite constant inputs use NaN
  padding and arithmetic nanmedian. Other modes retain their previous behavior.
- Even constant windows use k//2 before and k-1-k//2 after, preserving origin=0
  alignment and output shape; this prevents a regression when leaving SciPy's
  all-valid fast path. General even-window behavior for other modes is deferred.
- Red: 34 failed / 2 passed / 40 deselected before implementation.
- Green: 126 smooth/ops/methods tests passed (36 new cases).
- Checks: make check passed (902 passed / 1 skipped / 3 deselected plus lint,
  mypy and docs); make test-all passed (905 passed / 1 skipped).
- Closed previously merged BUG-0002/0006/0007 records, retaining history.
- Next: independent review, commit, push, PR and merge under existing user authorization.
- Env: Linux / Python 3.11.9 / pynbody 2.8.0 / repository .venv.

## 2026-11-08 — Review correction and final validation

- Independent review found the new nonempty window route broke empty input.
  Six empty-axis regressions failed first; preserving the original SciPy path
  for array.size == 0 fixed them. Focused suites now report 132 passed.
- Final make check: 908 passed / 1 skipped / 3 deselected, plus lint/mypy/docs.
  Final make test-all: 911 passed / 1 skipped. Earlier counts record the
  pre-review checkpoint.
- Review verification: 490 nonempty constant reference cases, 60 nonconstant
  baseline comparisons, and 150 empty-input/exception comparisons passed.
  Final reviewer reran 82 smooth tests, with no remaining findings.
