# BUG-0008: Constant median boundaries include artificial zeros

## Context

PR #34's review identified a pre-existing boundary bug left outside BUG-0006's
per-axis padding correction. `median_filter` documents constant padding with NaN,
so values beyond the image must not contribute to the median. Both the all-valid
SciPy fast path and the masked/nonfinite NumPy path currently add zeros.

## Goal

Honor the existing constant-mode contract for both all-valid and invalid images.

## Actual behavior

For `[[10, 20], [30, 40]]` with size=3, constant mode returns four zeros.
After changing the bottom-right pixel to NaN, all three valid outputs remain zero.

## Expected behavior

The first input produces four values of 25. The second produces 20 at each valid
pixel and retains NaN at the invalid center. Only finite, unmasked pixels inside
the image are samples; outside pixels are absent, not zero-valued samples.

## Reproduction

```python
median_filter([[10., 20.], [30., 40.]], size=3, mode="constant")
```

## Non-goals

Other boundary modes (including their even-window behavior), general size validation, large-kernel
performance redesign, dependencies, and calculator/store architecture.

## Acceptance criteria

- [x] Positive and negative images match an independent cropped-window median
      reference at corners, edges and interior pixels.
- [x] All-valid, masked and nonfinite inputs obey the same constant semantics;
      invalid output centers remain NaN and inputs are unchanged.
- [x] Odd/even square/rectangular windows, singleton/empty axes and all-invalid images work.
- [x] ImageData forwards constant mode, preserves geometry/units and records it.
- [x] Existing tests, make check and make test-all pass without weaker tolerances.

## Design notes

Keep the SciPy fast path for all-valid images in other modes. Constant mode always
uses the NaN-aware window path for nonempty inputs, explicitly passing `constant_values=np.nan` to
`np.pad`. Do not pass that argument to other padding modes. No public signature
or documented requirement changes. This corrects existing contract violations.
For constant padding, use `k//2` cells before and `k-1-k//2` after the image,
matching SciPy's default origin=0 footprint for even windows and retaining the
input shape. Take the arithmetic median of the remaining real samples; when
there are two middle samples, average them as np.nanmedian already does.

## Test plan

1. Write cropped-window references with no np.pad or sliding-window reuse.
2. Run new regressions on the current implementation and record expected failures.
3. Apply the small routing/padding correction; rerun focused tests.
4. Run make check and make test-all; independent review; publish and merge under
   the maintainer's continuing authorization.

## Outcome

Constant mode now always uses the NaN-aware window path with explicit NaN
padding. Before/after padding widths retain origin=0 alignment and the original
shape for odd and even sizes. Empty inputs retain the original SciPy path;
other modes keep their previous paths.

- Red: new constant regressions on the original implementation: **34 failed,
  2 passed, 40 deselected**.
- Review regression: empty arrays failed in six new tests before preserving
  their original path.
- Green: smooth/ops/methods suites: **132 passed** (42 new cases).
- `make check`: lint/mypy/docs pass; **908 passed, 1 skipped, 3 deselected**.
- `make test-all`: **911 passed, 1 skipped**.
- The 2x2 positive example now yields 25 everywhere; with one invalid pixel,
  valid pixels yield 20 and the invalid center remains NaN.
- Both public docstrings describe the constant-mode sample exclusion and
  arithmetic median. All-valid constant inputs now use the window path too,
  so its existing large-window memory warning applies to them.
- Closed the metadata/handoffs for already merged BUG-0002/0006/0007, retaining
  their historical red/green evidence and the required retention window.

Validation: Linux / Python 3.11.9 / pynbody 2.8.0, repository .venv.


## Open questions

None for the existing constant-padding contract. Other modes remain separate.
