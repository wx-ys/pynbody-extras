## What changed

Constant-mode median filtering now excludes samples outside the image, as its
public NaN-padding contract promises. Previously `[[10, 20], [30, 40]]` with a
3x3 constant window returned four zeros; it now returns four values of 25. With
one invalid pixel, the valid centers return 20 and the invalid center remains NaN.

Both all-valid and masked/nonfinite constant inputs use explicit NaN padding.
Asymmetric before/after padding preserves origin=0 alignment and output shape for
even windows. Empty arrays retain their existing SciPy path. Other modes retain
their existing behavior. Public docstrings explain sample exclusion and averaging
the two central values when a neighbourhood has an even number of valid samples.

42 regression cases cover signed values, masks, NaN/Inf, odd/even rectangular
windows, singleton and empty axes, all-invalid images, input immutability, and
ImageData metadata/history. Also finish the records for already merged BUG-0002,
BUG-0006 and BUG-0007, preserving their historical validation evidence.

## Evidence

- [x] Red: initial constant regressions: **34 failed, 2 passed, 40 deselected**.
- [x] Red: independent review found an empty-input regression; **6 failed** before
  retaining the original empty-array path.
- [x] Green: smooth/ops/methods suites: **132 passed**.
- [x] `make check`: lint, mypy and docs pass; **908 passed, 1 skipped, 3 deselected**.
- [x] `make test-all`: **911 passed, 1 skipped**.
- [x] `git diff --check` passes.

Environment: Linux / Python 3.11.9 / pynbody 2.8.0, project .venv.

## Architecture

- [x] No ADR needed: local numerical correction to the existing documented contract;
  no public signature, dependency, calculator/store or import-layer changes.
- [ ] ADR added or updated under `docs/adr/`.

## Checklist

- [x] Existing tests were not weakened or skipped.
- [x] Bilingual work item BUG-0008, metadata, index and handoff updated.
- [x] No testdata, environments, build output or unrelated local files included.

Constant mode now always uses the existing window-based NaN path for nonempty
images, so its documented large-window memory cost also applies to all-valid
inputs. Other boundary modes and their even-window behavior remain outside scope.

Independent review found and verified the empty-input correction. Final review
reported no remaining findings, with 490 constant reference comparisons,
60 nonconstant baseline comparisons and 150 empty-input/exception comparisons.
