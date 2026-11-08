## What changed

Fix two numeric baseline failures (BUG-0006 and BUG-0002):

- A masked/nonfinite image filtered with a `(3, 5)` or `(5, 3)` median window
  previously raised ValueError or used displaced windows. Pad each axis with its
  own symmetric radius; ten regression cases cover reference values and ImageData
  metadata/history.
- The locked pynbody 2.3.3 silently misplaced particles when `nz != ny` and lacked
  the kernel API expected by two tests. Require Python >=3.11 and pynbody >=2.8.0,
  lock 2.8.0, and synchronize tool targets, typing imports and installation docs.
  This intentionally ends Python 3.10 support, approved by the maintainer.
- Share one kernel-reference helper. Kernel2D.value is still unimplemented in
  2.8.0, so sample get_value(r/h) at unit h and apply the dimensional scaling.
  Retain all existing numeric assertions/tolerances and add h=0.7/1.4 projection
  cases plus a package/environment floor contract.

Upstream grid fix:
https://github.com/pynbody/pynbody/commit/040ade1eaf58a5fa9ac22194d7b4be8b26c536aa
(included in v2.8.0). This is a tested baseline, not a claim about the first fixed
release. The lock retains the existing TUNA index and adds pynbody's osfclient/HTTP
transitive dependencies. This branch is based on main and independent of codec PR #33.

## Evidence

- [x] Red: median regressions before correction: **8 failed, 4 passed, 28 deselected**.
- [x] Red: SPH + renderer contracts on 2.3.3: **4 failed, 42 passed**.
- [x] Green: median smooth/ops/methods suites: **90 passed**.
- [x] Green: SPH + renderer contracts on 2.8.0: **48 passed**.
- [x] `make check`: lint/typecheck/docs pass; **830 passed, 1 skipped, 3 deselected**.
- [x] `make test-all`: **833 passed, 1 skipped**.
- [x] Locked all-extras sync, offline lock validation, and `git diff --check` pass.
- [x] Independent review: no actionable findings; kernel helper also checked
  against 3-D value and direct line-of-sight integrals at five smoothing lengths.

Validation environment: Linux, Python 3.11.9, pynbody 2.8.0, project .venv.
Other Python/platform combinations were not run.

## Architecture

- [x] No ADR needed: no calculator/store interface, renderer signature or layer change.
  Maintainer-approved compatibility decision is recorded in BUG-0002's bilingual spec.
- [ ] ADR added or updated under `docs/adr/`.

## Checklist

- [x] Existing tests were not weakened or skipped.
- [x] Bilingual work-item records, status, index and handoffs are updated.
- [x] No testdata, .venv, build output, unrelated local files or secrets included.

Median boundary-mode semantics and even window sizes are outside this fix.
The pre-existing untracked .github workflow is not included; its Python 3.10 matrix
entry needs removing before that separate workflow is published.
