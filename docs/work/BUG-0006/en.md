# BUG-0006: median_filter pads the wrong amount for a non-square size and fails on masked images

## Context

Found while fixing the mypy errors in TASK-0001 (`plot/image/smooth.py:173`).
`median_filter` pads the NaN-filled image before taking sliding windows:

```python
pad_y, pad_x = kernel_y // 2, kernel_x // 2
padded = np.pad(filled, ((pad_y, pad_x), (pad_y, pad_x)), mode=...)
```

Axis 0 gets `(pad_y, pad_x)` and axis 1 gets `(pad_y, pad_x)`; it should be
`((pad_y, pad_y), (pad_x, pad_x))`. For a square size the two agree, so the
bug only shows with a non-square `size` and a mask (or non-finite pixels),
which is the path that pads (a fully valid image goes to
`scipy.ndimage.median_filter` instead).

## Goal

`median_filter(data, size=(ky, kx), mask=...)` returns an image of the same
shape as `data` for any odd `ky`, `kx`, with each pixel the median of its
`ky x kx` neighbourhood over valid pixels.

## Actual behavior

Reproduced with the same NumPy steps (image 6 x 7, one invalid pixel):

```text
size (3, 3): medians (6, 7)            # correct
size (3, 5): medians (7, 6)            # wrong shape
ValueError: operands could not be broadcast together with shapes (6,7) (7,6) ()
```

For some shapes (e.g. a square image) the broadcast can succeed and return
medians from shifted windows instead of raising.

## Expected behavior

Output shape equals input shape; values match a direct per-pixel `nanmedian`
over the `ky x kx` window (with the same edge mode).

## Reproduction

```python
import numpy as np
from pynbodyext.plot.image.smooth import median_filter

data = np.random.default_rng(0).random((6, 7))
mask = np.ones_like(data, bool); mask[0, 0] = False
median_filter(data, size=(3, 5), mask=mask)  # ValueError
```

## Non-goals

- Even sizes, other filters.

## Acceptance criteria

- [x] A test with a non-square `size` and a mask fails before the fix.
- [x] The pad widths are `((pad_y, pad_y), (pad_x, pad_x))`.
- [x] The result matches a brute-force `nanmedian` reference on a small image
      for `size=(3, 5)` and `(5, 3)`, and still matches for `size=3`.
- [x] `make check` passes.

## Design notes

One-line fix in `pynbodyext/plot/image/smooth.py`. Check `ImageData`'s
`median_filter` op and any other caller that forwards `size`.

## Test plan

- Red: the reproduction above as a test.
- Green: the fix; reference comparison; `make check`.

## Outcome

Implemented on `codex/p0-median-filter-padding`, independently of codec PR #33.
The masked/NaN path now pads each axis symmetrically using its own radius.
Ten regression cases cover `(3, 5)`, `(5, 3)` and square windows, masks,
NaN/Inf, all-masked images and the ImageData wrapper. A reference enumerates
and clamps neighbour coordinates without using padding or sliding windows.
Image geometry, units, operation history and input immutability are checked.

- Red: `pytest tests/test_plot_image_smooth.py -k median -q`:
  **8 failed, 4 passed, 28 deselected**, before the one-line fix.
- Green: smooth/ops/methods tests: **90 passed**.
- `make check`: ruff and mypy passed; fast tests: **824 passed, 3 failed,
  1 skipped, 3 deselected**.
- `make test-all`: **827 passed, 3 failed, 1 skipped**.
- The three failures are unchanged BUG-0002 SPH failures on pynbody 2.3.3;
  `make check` is therefore intentionally not checked off above.
- Docs validation runs separately because make check stops at pytest.
- No dependency, architecture, edge-mode, or even-window contract was changed.


### Final baseline verification — 2026-11-08

The branch was renamed to `codex/p0-image-sph-baseline`. After the maintainer
approved BUG-0002's dependency upgrade, `make check` passes (830 passed,
1 skipped, 3 deselected) and `make test-all` passes (833 passed, 1 skipped).
The earlier failing counts above record the median-only checkpoint; its remaining
acceptance criterion is now satisfied. See [combined PR body](../BUG-0002/pr-body.md).

## Open questions

Known follow-up from PR #34 review: constant padding in the masked/nonfinite
median path currently adds zeros despite the documented NaN padding. The all-valid
fast path also needs its boundary semantics checked. This behavior predates the
per-axis correction and is outside this PR's explicit boundary-mode scope; handle
it with dedicated boundary tests instead of changing it incidentally here.
