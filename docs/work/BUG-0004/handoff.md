## 2026-10-20 — Claude @ cloud session

- State: draft
- Done: opened from BUG-0003's open question. `setup.py` builds
  `pynbodyext._native` with `-fopenmp` and `optional=True`; no `_native*.so`
  exists on the user's Mac.
- Next: on the Mac, run `.venv/bin/python setup.py build_ext --inplace` and
  paste the compiler error into `Actual behavior`; then confirm the cause and
  move to `ready`.
- Commands: `.venv/bin/python setup.py build_ext --inplace 2>&1 | tail -40`;
  `clang++ --version`; `brew list libomp`.
- Failing test: none; gravity tests skip without the extension.
- Blockers: needs the compiler output from macOS.
- Files: `setup.py`, `cpp/`, `README.md`, `README.zh-CN.md`.
- Env: macOS (Apple Silicon) / Python 3.11.7.

## 2026-10-20 — Claude @ cloud session

- State: draft
- Done: first reproduction attempt failed before compiling:
  `.venv/bin/python setup.py build_ext --inplace` ->
  `ModuleNotFoundError: No module named 'setuptools'` (uv venvs do not ship
  setuptools; it is only a `[build-system]` requirement). Updated
  Reproduction to supply setuptools and pybind11 via `uv run --with`.
- Next: run the new command on the Mac and paste the compiler error.
- Commands: `uv run --with setuptools --with pybind11 python setup.py build_ext --inplace 2>&1 | tail -60`.
- Failing test: none.
- Blockers: still needs the compiler output.
- Files: `docs/work/BUG-0004/*`.
- Env: macOS / `.venv` with pybind11 3.1.0, no setuptools.

## 2026-10-20 — Claude @ cloud session

- State: ready
- Done: captured the macOS compiler output:
  `clang++: error: unsupported option '-fopenmp'`, turned into a warning by
  `optional=True`. Cause confirmed; first criterion ticked. Found that a
  serial build also needs `#ifdef _OPENMP` guards: `omp.h` and `omp_*` calls
  are unguarded in `cpp/bindings/module.cpp` (8 calls) and
  `cpp/image/scatter.cpp`.
- Next: decide the Open question (require libomp vs serial fallback), then
  change `setup.py` (OpenMP detection / Homebrew libomp flags) and add the
  `_OPENMP` guards; `cpp/` and `setup.py` need human review.
- Commands: `uv run --with setuptools --with pybind11 python setup.py build_ext --inplace`;
  `brew list libomp`; `make cpp-test`.
- Failing test: none; gravity tests skip without the extension.
- Blockers: the Open question needs a decision from the maintainer.
- Files: `setup.py`, `cpp/bindings/module.cpp`, `cpp/image/scatter.cpp`,
  `README.md`, `README.zh-CN.md`.
- Env: macOS (Apple Silicon) / Apple clang / Python 3.11 (`/opt/anaconda3`);
  Homebrew `libomp` 23.1.2 installed (keg-only).

## 2026-10-20 — Claude @ cloud session

- State: review
- Done: implemented the Decision in `en.md` (OpenMP probe in `setup.py`
  with Homebrew libomp on macOS and a single-threaded fallback; `_OPENMP`
  guards in `module.cpp` / `scatter.cpp`; `OMP_FLAGS` in `make cpp-test`;
  README). Branch `fix/macos-native-openmp`. Verified on Linux (GCC 11,
  Python 3.10, pybind11 3.1.0) in a scratch copy: default build picks
  `-fopenmp` and links `libgomp`; `PYNBODYEXT_OPENMP=0` and a fake compiler
  that rejects `-fopenmp` both build single-threaded with no warnings and the
  expected message; `direct_accelerations_py` (3000 particles, threads
  0/1/4) is identical across the three builds (max rel diff 0.0);
  `make cpp-test` and `make cpp-test OMP_FLAGS=` print ALL PASS.
- Next: verify on the Mac (libomp 23.1.2): build prints
  `building with OpenMP (libomp at ...)`; `otool -L` shows `libomp.dylib`;
  `make cpp-test`; `pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs`
  now runs the gravity tests; `make check`. Then tick the remaining criteria.
- Commands: `rm -rf build && uv run --with setuptools --with pybind11 python setup.py build_ext --inplace`;
  `otool -L pynbodyext/_native*.so | grep omp`; `make cpp-test`;
  `.venv/bin/python -m pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs`.
- Failing test: none known.
- Blockers: the macOS path is not verified yet; `setup.py` and `cpp/` need
  human review.
- Files: `setup.py`, `cpp/bindings/module.cpp`, `cpp/image/scatter.cpp`,
  `Makefile`, `README.md`, `README.zh-CN.md`, `docs/work/BUG-0004/*`.
- Env: Linux aarch64 VM / GCC 11.4 / Python 3.10.12 / pybind11 3.1.0.

## 2026-10-20 — Claude @ cloud session

- State: review
- Done: verified on the user's Mac (Apple clang, libomp 23.1.2,
  Python 3.11.7): the build prints
  `building with OpenMP (libomp at /opt/homebrew/opt/libomp)` and links with
  `-L/opt/homebrew/opt/libomp/lib -lomp -Wl,-rpath,...`; `otool -L` shows
  `/opt/homebrew/opt/libomp/lib/libomp.dylib`; `make cpp-test` -> ALL PASS;
  `pytest tests/test_gravity.py tests/test_bins_vcirc.py -rs` -> 11 passed
  (gravity and vcirc tests now run instead of skipping). All five criteria
  ticked.
- Next: open the PR; after merge set `done` and fill `Outcome`. Optionally
  run `make check`.
- Commands: as above.
- Failing test: none.
- Blockers: none; `setup.py` and `cpp/` need human review.
- Files: unchanged from the previous entry.
- Env: macOS (Apple Silicon) / Apple clang / libomp 23.1.2 / Python 3.11.7 /
  pytest 9.0.2. `make cpp-test` shows 5 pre-existing
  `-Winstantiation-after-specialization` warnings in
  `cpp/gravity/multipole/eval.cpp:588-592` (not touched by this item).

## 2026-10-20 — Claude @ cloud session

- State: done
- Done: upstream PR #29 (`4e4480a`) merged. Set `done`, `pr: 29`, filled
  `Outcome`, answered the first Open question.
- Next: archive after the retention window. Still open: whether a failed
  optional build should warn at import time (second Open question; related
  to BUG-0005).
- Commands: none.
- Failing test: none.
- Blockers: none.
- Files: `docs/work/BUG-0004/*`.
- Env: n/a.
