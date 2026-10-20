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

Compiler output (2026-10-20, Apple clang, Python 3.11 from `/opt/anaconda3`):

```text
$ uv run --with setuptools --with pybind11 python setup.py build_ext --inplace 2>&1 | tail -60
running build_ext
building 'pynbodyext._native' extension
clang++ ... -c cpp/bindings/module.cpp -o build/temp.macosx-11.1-arm64-cpython-311/cpp/bindings/module.o -fvisibility=hidden -g0 -std=c++17 -mmacosx-version-min=10.14 -O2 -fopenmp
clang++: error: unsupported option '-fopenmp'
warning: build_ext: building extension "pynbodyext._native" failed: Command '['clang++', ..., '-fopenmp']' returned non-zero exit status 1.
```

Confirmed: Apple clang rejects `-fopenmp`; because the extension is
`optional=True`, setuptools turns the failure into a warning and installs the
package without `_native`.

## Expected behavior

- On macOS with Apple clang, the extension builds, with OpenMP when `libomp`
  is available and without it (serial) otherwise; or the build stops with a
  message that says what to install.
- The README states the macOS requirements.

## Reproduction

The project `.venv` (made by uv) has no `setuptools`, so calling `setup.py`
directly fails with `ModuleNotFoundError: No module named 'setuptools'`
(observed 2026-10-20). Supply the build requirements from `pyproject.toml`
without changing `.venv`:

```bash
uv run --with setuptools --with pybind11 python setup.py build_ext --inplace 2>&1 | tail -60
clang++ --version
brew list libomp 2>/dev/null && echo "libomp present"
```

## Non-goals

- Changing the C++ algorithms.
- Python-side availability checks (BUG-0005).

## Acceptance criteria

- [x] The compiler error on macOS is captured in this item and the cause is
      confirmed or replaced. (Confirmed: `unsupported option '-fopenmp'`.)
- [x] On macOS, `make setup` produces an importable `pynbodyext._native`
      (with or without OpenMP), or prints an explicit, actionable error.
      (Verified with `setup.py build_ext --inplace`: OpenMP via libomp.)
- [x] Linux builds keep `-fopenmp` and their current performance.
      (Linux/GCC 11: probe picks `-fopenmp`, `libgomp` is linked.)
- [x] `make cpp-test` and `tests/test_gravity.py` pass on macOS after the fix.
- [x] README / README.zh-CN describe the macOS build requirements.

## Design notes

- Options: detect OpenMP support in `setup.py` (try-compile a tiny OpenMP
  program) and add the flags only when it works; on macOS use
  `-Xpreprocessor -fopenmp -lomp` with the Homebrew `libomp` include/lib
  paths; otherwise build serially.
- The C++ code does not compile without OpenMP today: `#include <omp.h>` is
  unguarded in `cpp/bindings/module.cpp` and `cpp/image/scatter.cpp`;
  `module.cpp` calls `omp_set_num_threads` 8 times and `scatter.cpp` calls
  `omp_get_max_threads`. A serial fallback needs `#ifdef _OPENMP` guards (or
  a small `cpp/` shim header). `#pragma omp` in `gravity/direct.cpp` (2),
  `gravity/traversal.cpp` (4) and `image/scatter.cpp` (1) is ignored without
  OpenMP and needs no change.
- `setup.py` and `cpp/` changes need human review (`AGENTS.md`).

## Decision

Use OpenMP when the build's compiler can compile and link it, otherwise build
single-threaded with a visible warning (answers the first Open question).

- `setup.py`: a `BuildExt` subclass probes the compiler with a small OpenMP
  program before building. On macOS it tries Homebrew libomp first
  (`LIBOMP_PREFIX`, `brew --prefix libomp`, `/opt/homebrew/opt/libomp`,
  `/usr/local/opt/libomp`) with `-Xpreprocessor -fopenmp -I<prefix>/include`
  and `-L<prefix>/lib -lomp -Wl,-rpath,<prefix>/lib`; then plain `-fopenmp`.
  If nothing works it prints a warning (with `brew install libomp` on macOS)
  and builds without OpenMP. `PYNBODYEXT_OPENMP=0` forces a single-threaded
  build. The build always prints which setup it used.
- `cpp/bindings/module.cpp`, `cpp/image/scatter.cpp`: `#include <omp.h>` is
  guarded by `#ifdef _OPENMP`; the 8 `omp_set_num_threads` calls go through
  `set_omp_threads()`, a no-op without OpenMP. `threads` is then ignored.
- `Makefile` `cpp-test`: `OMP_FLAGS` uses Homebrew libomp on macOS and
  `-fopenmp` elsewhere; `make cpp-test OMP_FLAGS=` builds serially.
- README / README.zh-CN: build requirements, `brew install libomp`,
  `PYNBODYEXT_OPENMP`, `LIBOMP_PREFIX`.

## Test plan

- Red: capture the failing `build_ext` output on macOS.
- Green: `build_ext --inplace` succeeds; `import pynbodyext._native` works;
  `make cpp-test`; `pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs`
  runs (not skips) the gravity tests.

## Outcome

Fixed in upstream PR #29 (`4e4480a`), merged by the maintainer.

- Red: on macOS, `setup.py build_ext` failed with
  `clang++: error: unsupported option '-fopenmp'`; `optional=True` turned it
  into a warning, so no `pynbodyext._native` was installed and the gravity
  and `vcirc` tests skipped or failed.
- Green (macOS, Apple clang, Homebrew libomp 23.1.2, Python 3.11.7): the
  build prints `building with OpenMP (libomp at /opt/homebrew/opt/libomp)`;
  `otool -L` shows `libomp.dylib`; `make cpp-test` -> ALL PASS;
  `pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs` -> 11 passed.
- Green (Linux, GCC 11): `-fopenmp` build, `PYNBODYEXT_OPENMP=0` build and a
  compiler that rejects `-fopenmp` all build; direct accelerations identical
  across them; `make cpp-test` passes with and without OpenMP.

## Open questions

- Should OpenMP be required on macOS (document `brew install libomp`), or is
  a serial fallback acceptable?
  - Answer: neither alone; OpenMP when libomp is found, otherwise a
    single-threaded build with a warning (see Decision).
  - Data point: the user's Mac already has Homebrew `libomp` 23.1.2
    (`/opt/homebrew/Cellar/libomp/23.1.2`, `lib/libomp.dylib`), so an
    "OpenMP when libomp is found, serial otherwise" build can be verified
    there. Homebrew's libomp is keg-only: pass its `include/` and `lib/`
    paths explicitly (`brew --prefix libomp`), and link with `-lomp` instead
    of `-fopenmp`.
- Should a silent `optional=True` failure become a visible warning at import
  time (`pynbodyext.gravity` already emits an `ImportWarning`, which is hidden
  by default)?
