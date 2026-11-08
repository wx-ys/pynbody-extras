## 2026-10-20 — Claude @ cloud session

- State: ready
- Done: found while fixing TASK-0001; confirmed with the same NumPy steps
  that size (3, 5) gives a (7, 6) median array for a (6, 7) image and
  `np.where` raises.
- Next: write the failing test, then fix the pad widths.
- Commands: `.venv/bin/python -m pytest tests -k median -rs`.
- Failing test: none yet.
- Blockers: none.
- Files: `pynbodyext/plot/image/smooth.py`, tests for `plot.image.smooth`.
- Env: Linux VM / NumPy 2.2.6 (reproduction only).

## 2026-11-08 — Codex / local checkout

- State: review
- Branch: codex/p0-median-filter-padding, from origin/main d76c479.
  Codec PR #33 was still open at synchronization; this fix is independent.
- Done: one-line correction to symmetric per-axis padding; ten regression cases
  using an independently enumerated/clamped nearest-neighbour reference. Covered
  mask, NaN/Inf, all-masked images, both rectangular orientations, square windows,
  and ImageData geometry/units/history. Existing tests remain intact.
- Red: `.venv/bin/python -m pytest tests/test_plot_image_smooth.py -k median -q`:
  8 failed / 4 passed / 28 deselected before the implementation.
- Green: `.venv/bin/python -m pytest tests/test_plot_image_smooth.py
  tests/test_plot_image_ops.py tests/test_plot_image_methods.py -q`: 90 passed.
- Checks: make check passed ruff and mypy, then pytest reported 824 passed /
  3 failed / 1 skipped / 3 deselected. make test-all: 827 passed / 3 failed /
  1 skipped. Separate make docs-check passed (7 work items).
- Baseline failures: BUG-0002 tests for volume kernel placement, direct kernel
  sum and unequal z-resolution in tests/test_bins_sph_render.py. These were
  present before this patch; none were skipped or weakened.
- Next: user reviews/pushes/opens PR/merges. Continue BUG-0002 only after the
  supported-pynbody compatibility decision; edge-mode semantics and even-sized
  windows are outside this bounded fix.
- Files: pynbodyext/plot/image/smooth.py, tests/test_plot_image_smooth.py,
  docs/work/BUG-0006/, docs/work/INDEX.md.
- Env: Linux, Python 3.11.9, pynbody 2.3.3, repository .venv.
- Independent review: no actionable findings; reviewer separately ran the smooth
  suite (40 passed). No existing tests were changed, weakened or skipped.

## 2026-11-08 — Codex / final baseline verification

- State: review. Renamed branch to codex/p0-image-sph-baseline.
- Maintainer approved BUG-0002's Python >=3.11 / pynbody >=2.8.0 floor.
- After that upgrade, make check passes (830 passed / 1 skipped / 3 deselected,
  lint, mypy, docs) and make test-all passes (833 passed / 1 skipped).
- Earlier failing counts remain historical median-only evidence. All BUG-0006
  acceptance criteria are now satisfied. The median implementation is unchanged.
- Next: user publishes and reviews the combined BUG-0006 + BUG-0002 branch;
  use docs/work/BUG-0002/pr-body.md. No push, PR creation or merge performed here.
