## What changed

Fix per-axis padding in the masked/NaN-aware median filter. A `(3, 5)` or
`(5, 3)` window previously produced a wrongly shaped median grid and raised
ValueError on a `(6, 7)` image. Each axis now uses its own symmetric radius.
Ten regressions cover masks, nonfinite pixels, all-masked images, square windows,
and ImageData geometry/units/history against an explicit neighbour reference.

Work item: BUG-0006. This branch is based on main and independent of codec PR #33.

## Evidence

- [x] Red: median tests before the fix: **8 failed, 4 passed, 28 deselected**.
- [x] Green: smooth/ops/methods suites: **90 passed**.
- [ ] `make check` passes locally: lint/typecheck passed; pytest returned
  **824 passed, 3 failed, 1 skipped, 3 deselected**.
- [ ] `make test-all` passes: **827 passed, 3 failed, 1 skipped**.
- [x] `make docs-check` passes; `git diff --check` passes.

All three failures are pre-existing BUG-0002 SPH failures under pynbody 2.3.3:
two missing Kernel.value calls and one incorrect unequal-z grid placement.
This change does not claim a green full suite or resolve BUG-0002.

## Architecture

- [x] No ADR needed: one-line numeric padding correction, no interface or layer change.
- [ ] ADR added or updated under `docs/adr/`.

## Checklist

- [x] Existing tests were not weakened or skipped.
- [x] Bilingual work-item records and handoff are updated.
- [x] No testdata, .venv, build output, unrelated local files or secrets included.

Boundary-mode semantics and even window sizes remain outside this work item.
