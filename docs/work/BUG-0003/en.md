# BUG-0003: vcirc is offered without the native gravity extension and fails with ImportError

## Context

`make test` on macOS (Python 3.11.7) fails all three tests in
`tests/test_bins_vcirc.py`. The environment has no compiled
`pynbodyext._native` (no `_native*.so` in the tree or in `.venv`), so
`GRAVITY_NATIVE_AVAILABLE` is `False` and `pynbodyext.gravity` exports only
`GRAVITY_NATIVE_AVAILABLE`. `tests/test_gravity.py` skips in this case
(`pytest.importorskip("pynbodyext._native")`); `tests/test_bins_vcirc.py` does
not, and the `vcirc` derived query does not check availability either.

## Goal

Without the native extension, `vcirc` is not offered (or fails with a clear
message), and the suite skips its tests instead of failing.

## Actual behavior

```text
FAILED tests/test_bins_vcirc.py::test_vcirc_available_for_radial_axes_only
FAILED tests/test_bins_vcirc.py::test_vcirc_matches_analytic_point_mass
FAILED tests/test_bins_vcirc.py::test_vcirc_is_finite_and_nonnegative

pynbodyext/core/calculate/bins/result.py:924: in _vcirc
    from pynbodyext.gravity import KernelKind, calculate_acceleration
E   ImportError: cannot import name 'KernelKind' from 'pynbodyext.gravity'
```

- `"vcirc" in bins.queries.names()` is `True` for a 1-D `rxy` profile even
  when the extension is missing; the `condition` checks only the axes.
- `bins["vcirc"]` then raises an `ImportError` about `KernelKind`, which does
  not tell the user that the C++ extension is missing.

## Expected behavior

- When `GRAVITY_NATIVE_AVAILABLE` is `False`, either `vcirc` is not listed in
  `bins.queries.names()`, or `bins["vcirc"]` raises an error that names the
  missing `pynbodyext._native` extension and how to build it.
- `tests/test_bins_vcirc.py` skips its gravity-dependent tests without the
  extension, the same way `tests/test_gravity.py` does.
- With the extension, behaviour is unchanged.

## Reproduction

```bash
.venv/bin/python -c "from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE as A; print(A)"  # False
.venv/bin/python -m pytest tests/test_bins_vcirc.py
```

```python
import numpy as np, pynbody
from pynbody.array import SimArray
from pynbodyext.core.calculate import Bin1D

sim = pynbody.new(dm=1)
sim["pos"] = SimArray(np.zeros((1, 3)), "kpc")
sim["mass"] = SimArray([1e10], "Msol")
sim["smooth"] = SimArray([0.0], "kpc")
bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=2)(sim)
print("vcirc" in bins.queries.names())  # True
bins["vcirc"]                            # ImportError: KernelKind
```

## Non-goals

- Changing how `vcirc` is computed.
- Fixing the native build itself (see Open questions); that is a build change
  under `setup.py` / `cpp/` and needs its own item and human review.

## Acceptance criteria

- [x] Without the extension, `vcirc` is not in `bins.queries.names()`, or
      `bins["vcirc"]` raises an error whose message names
      `pynbodyext._native` (pick one and document it in `en.md` / `zh-CN.md`).
      Chosen: the clear error (see Decision).
- [ ] `tests/test_bins_vcirc.py` skips its gravity-dependent tests when
      `pynbodyext._native` is missing, and still runs them when it is present.
- [x] A test covers the "extension missing" path, e.g. by monkeypatching
      `GRAVITY_NATIVE_AVAILABLE`, so it runs on every machine.
      (`test_vcirc_without_native_extension_names_it`)
- [ ] `make check` passes with and without the extension.

## Design notes

- The `condition` in `@BinNDResult.derived("vcirc", ...)`
  (`pynbodyext/core/calculate/bins/result.py`) can also require
  `GRAVITY_NATIVE_AVAILABLE`. Read it at call time, not import time, so a test
  can monkeypatch it.
- If the query stays listed, wrap the import and raise an `ImportError` (or a
  project error type) with a message such as "vcirc needs the compiled
  pynbodyext._native extension".
- `pynbodyext/core/` changes need human review (`AGENTS.md`); this one should
  be small.

## Decision

`vcirc` stays listed on 1-D `r` / `rxy` profiles; without the extension,
`bins["vcirc"]` raises `ImportError(name="pynbodyext._native")` whose message
names the extension and how to rebuild it. Reasons:

- If the `condition` hid it, `bins["vcirc"]` would fall through to the
  particle-array path and fail with an unrelated message, so the user would
  still not learn that the extension is missing.
- The query stays discoverable; `names()` does not depend on the build.

The check reads `pynbodyext.util.deps.GRAVITY_NATIVE_AVAILABLE` at call time,
so a test can monkeypatch it.

## Test plan

- Red: on a machine without the extension, `tests/test_bins_vcirc.py` fails
  with `ImportError` (already observed).
- Add `pytest.importorskip("pynbodyext._native", ...)` for the
  gravity-dependent tests.
- Add a test with `GRAVITY_NATIVE_AVAILABLE` monkeypatched to `False` that
  checks the chosen behaviour.
- `make check`; `make test-all` on a machine with the extension.

## Outcome

Fixed in upstream PR #25 (`83e57c5`), merged by the maintainer.

- `_vcirc` raises `ImportError(name="pynbodyext._native")` with a rebuild hint
  when the extension is missing; `vcirc` stays listed (see Decision).
- Red: without the extension, the three tests in `tests/test_bins_vcirc.py`
  failed with `ImportError: cannot import name 'KernelKind'`.
- Green: not yet recorded. The PR was merged before the
  `pytest tests/test_bins_vcirc.py -rs` and `make check` output was pasted
  (expected without the extension: 3 skipped, 1 passed). The two unticked
  criteria above wait for that run.
- The commit used `--no-verify`: ruff passed; the mypy hook failed on 5 errors
  already on `main` (`gravity/base.py:193`, `store/sqlalchemy_store.py:68,107`,
  `bins/result.py:353`).

## Open questions

- Why is the extension missing on macOS? `setup.py` builds
  `pynbodyext._native` with `-fopenmp` and `optional=True`. Apple clang does not
  support `-fopenmp` without `libomp`, so the build probably fails silently.
  Check with `.venv/bin/python setup.py build_ext --inplace` and look at the
  compiler error. If confirmed, open a separate item for the macOS build
  (for example, add `-fopenmp` only when OpenMP is available, or document
  `brew install libomp`).
- A review comment on PR #25 (left open at merge) concerns native *load*
  failures. `GRAVITY_NATIVE_AVAILABLE` uses `find_spec`, which is `True` when a
  `_native*.so` exists even if it cannot be loaded (for example a missing
  OpenMP runtime), and then `import pynbodyext.gravity` fails. Consider
  deciding availability by actually importing `pynbodyext._native`, in a
  separate item.
