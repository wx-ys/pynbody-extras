# BUG-0002: SPH 3-D render tests fail against the locked pynbody 2.3.3

## Context

`make test` on a fresh `make setup` environment (macOS, Python 3.11.7) fails
three tests in `tests/test_bins_sph_render.py`. `uv.lock` resolves `pynbody`
to **2.3.3** for Python >= 3.11 (and 2.0.0 for 3.10), while `pyproject.toml`
only asks for `pynbody>=2.0`. The tests, and one production behaviour, assume a
newer pynbody: the BUG-0001 handoff recorded pynbody 2.8.0 on the machine where
the suite passed.

## Goal

The locked environment that `make setup` creates passes `make test`, and a
3-D `sph_render` with `nz != ny` is never silently wrong.

## Actual behavior

```text
FAILED tests/test_bins_sph_render.py::test_a_volume_render_places_the_kernel_on_the_cell_grid
  AttributeError: 'CubicSplineKernel' object has no attribute 'value'
FAILED tests/test_bins_sph_render.py::test_a_volume_render_matches_a_direct_kernel_sum
  AttributeError: 'CubicSplineKernel' object has no attribute 'value'
FAILED tests/test_bins_sph_render.py::test_an_unequal_z_resolution_renders_the_grid_it_was_asked_for
  assert (np.int64(0), ..., np.int64(0)) == (2, 3, 2)
```

Two separate causes:

1. **Kernel API (tests only).** pynbody 2.3.3 kernels have
   `get_value(d, h=1)`, where `d` is already `r / h` and the input is a
   scalar. Lines 341, 369 and 403 call `kernel.value(distance, h)` with a
   physical distance (line 369 passes an array). Line 260 of the same file still
   uses `get_value(r / h)`, so the file mixes the two APIs.
2. **`to_3d_grid` z sizing (production).** In pynbody 2.3.3 `to_3d_grid` sizes
   z pixels with `ny`. For a grid of `(4, 6, 3)` cells, the particle in cell
   `(2, 3, 2)` lands at `(0, 0, 0)`. The test says the warning for this
   upstream bug was removed because it "is fixed", which is true only for newer
   pynbody. On 2.3.3, `bins.sph_render[...]` with `nz != ny` returns a wrong
   grid **without a warning**.

## Expected behavior

- `make setup && make test` passes with the versions in `uv.lock`.
- With any pynbody that `pyproject.toml` allows, a 3-D render with `nz != ny`
  is either correct or warns (or raises) that it is not.

## Reproduction

```bash
make setup
.venv/bin/python -c "import pynbody; print(pynbody.__version__)"   # 2.3.3
.venv/bin/python -m pytest tests/test_bins_sph_render.py -k "volume_render or unequal_z"
```

## Non-goals

- Changing the SPH view's conventions or the `_pynbody_render` adapter's
  positional signatures (pinned by `tests/test_pynbody_render_contract.py`).
- Supporting pynbody 1.x.

## Acceptance criteria

- [x] `pyproject.toml` and `uv.lock` agree on a pynbody floor that the suite
      actually passes on, or the suite passes on every pynbody the floor admits.
- [x] The three tests above pass in the environment `make setup` creates.
- [x] `tests/test_bins_sph_render.py` uses one kernel-evaluation helper, not a
      mix of `get_value` and `value`.
- [x] With a pynbody whose `to_3d_grid` has the z-sizing bug, a 3-D render
      with `nz != ny` warns or raises; a test covers this (or the floor
      excludes such versions and a contract test pins it).
- [x] `make check` and `make test-all` pass.

## Design notes

### Maintainer decision — 2026-11-08

Option A is approved: require **Python >=3.11 and pynbody >=2.8.0**, and
lock/test pynbody 2.8.0. Python 3.10 support is deliberately removed.
2.8.0 is the chosen known-good baseline, not a claim about the earliest fixed release.
Upstream [040ade1](https://github.com/pynbody/pynbody/commit/040ade1eaf58a5fa9ac22194d7b4be8b26c536aa)
fixes z pixel sizing (`ny` -> `nz`); the commit is an ancestor of
[v2.8.0](https://github.com/pynbody/pynbody/releases/tag/v2.8.0), whose package
metadata requires Python >=3.11. Keep the adapter signatures and numerical tolerances.

Inspection of the installed 2.8.0 kernels found that `Kernel2D.value` still raises
`NotImplementedError`. Numerical references therefore share one helper that samples
`get_value(r / h)` at unit h and divides by `h**kernel.h_power`. This preserves
both 2-D projection and 3-D normalization, with no compatibility branch in production.
Projection tests also cover h=0.7 and h=1.4, besides the original h=1 case.

The rejected alternative was to keep the old floor and add a version-dependent
warning/workaround. The approved floor avoids that extra production behavior.

## Test plan

- Reproduce the three SPH failures and the new floor-contract failure on 2.3.3.
- Re-lock and sync all extras; test the SPH renderer signatures, numerical references,
  and non-cubic grid placement on the exact supported floor, 2.8.0.
- Run `make check`, `make test-all`, and verify lock consistency.

## Outcome

Implemented on `codex/p0-image-sph-baseline` after the BUG-0006 median fix.
The branch starts from main and does not include codec PR #33.

- Updated Python/package metadata, Ruff/mypy targets, the lock and both READMEs.
  Removed the obsolete Python 3.10 typing fallback and its direct dependency.
- Locked pynbody 2.8.0 using the existing TUNA index. Other retained packages keep
  their versions; the lock removes Python 3.10-only resolutions and adds pynbody's
  required `osfclient`/HTTP dependencies.
- Red: renderer contract + SPH suites on 2.3.3: **4 failed, 42 passed** (the new
  floor contract plus the three original failures).
- Green: the same suites, including two new projection scales: **48 passed**.
- `make check`: lint and mypy pass; **830 passed, 1 skipped, 3 deselected**;
  documentation validation passes.
- `make test-all`: **833 passed, 1 skipped**. Existing assertions and numerical
  tolerances are intact; no failing tests were skipped or deleted.
- `uv sync --locked --all-extras` and offline lock validation pass on Linux /
  Python 3.11.9 / pynbody 2.8.0. Other Python/platform combinations were not run.
- Local `.github/` files were already untracked and are outside this commit.
  Their draft CI matrix still includes Python 3.10; remove that entry before
  publishing that separate workflow.

Review/push/PR/merge remain with the maintainer. See [PR body](pr-body.md).

## Open questions

None for this fix; the maintainer approved the support-floor change above.
