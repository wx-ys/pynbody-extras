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

- [ ] A test with a non-square `size` and a mask fails before the fix.
- [ ] The pad widths are `((pad_y, pad_y), (pad_x, pad_x))`.
- [ ] The result matches a brute-force `nanmedian` reference on a small image
      for `size=(3, 5)` and `(5, 3)`, and still matches for `size=3`.
- [ ] `make check` passes.

## Design notes

One-line fix in `pynbodyext/plot/image/smooth.py`. Check `ImageData`'s
`median_filter` op and any other caller that forwards `size`.

## Test plan

- Red: the reproduction above as a test.
- Green: the fix; reference comparison; `make check`.

## Outcome

## Open questions
