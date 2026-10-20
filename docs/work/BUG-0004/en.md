# BUG-0004: pynbodyext._native is not built on macOS; the optional build fails silently

## Context

On the user's Mac (Apple Silicon, Python 3.11.7) there is no `_native*.so`
in the tree or in `.venv`, so `GRAVITY_NATIVE_AVAILABLE` is `False`, the
gravity layer is unavailable, `bins["vcirc"]` raises (BUG-0003), and the SPH
scatter falls back to NumPy. Nobody was told: `setup.py` declares the
extension with `optional=True`, so a failed compile only prints a warning.

`setup.py` passes `-fopenmp` in both `extra_compile_args` and
`extra_link_args`. Apple clang does not accept `-fopenmp` without
`-Xpreprocessor` and the `libomp` runtime, so the compile most likely fails
there. This is a hypothesis until the compiler output is seen.

## Goal

`make setup` (or `pip install -e .`) on macOS either builds
`pynbodyext._native`, or fails / warns in a way the user cannot miss, with a
documented fix.

## Actual behavior

```text
$ .venv/bin/python -c "from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE as A; print(A)"
False
```

The compiler error itself has not been captured yet (see Reproduction).

## Expected behavior

- On macOS with Apple clang, the extension builds, with OpenMP when `libomp`
  is available and without it (serial) otherwise; or the build stops with a
  message that says what to install.
- The README states the macOS requirements.

## Reproduction

```bash
.venv/bin/python setup.py build_ext --inplace 2>&1 | tail -40
clang++ --version
brew list libomp 2>/dev/null && echo "libomp present"
```

## Non-goals

- Changing the C++ algorithms.
- Python-side availability checks (BUG-0005).

## Acceptance criteria

- [ ] The compiler error on macOS is captured in this item and the cause is
      confirmed or replaced.
- [ ] On macOS, `make setup` produces an importable `pynbodyext._native`
      (with or without OpenMP), or prints an explicit, actionable error.
- [ ] Linux builds keep `-fopenmp` and their current performance.
- [ ] `make cpp-test` and `tests/test_gravity.py` pass on macOS after the fix.
- [ ] README / README.zh-CN describe the macOS build requirements.

## Design notes

- Options: detect OpenMP support in `setup.py` (try-compile a tiny OpenMP
  program) and add the flags only when it works; on macOS use
  `-Xpreprocessor -fopenmp -lomp` with the Homebrew `libomp` include/lib
  paths; otherwise build serially.
- Check that the C++ code compiles without OpenMP (`#pragma omp` is ignored,
  but `#include <omp.h>` or `omp_*` calls need a guard such as `_OPENMP`).
- `setup.py` and `cpp/` changes need human review (`AGENTS.md`).

## Test plan

- Red: capture the failing `build_ext` output on macOS.
- Green: `build_ext --inplace` succeeds; `import pynbodyext._native` works;
  `make cpp-test`; `pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs`
  runs (not skips) the gravity tests.

## Outcome

## Open questions

- Should OpenMP be required on macOS (document `brew install libomp`), or is
  a serial fallback acceptable?
- Should a silent `optional=True` failure become a visible warning at import
  time (`pynbodyext.gravity` already emits an `ImportWarning`, which is hidden
  by default)?
