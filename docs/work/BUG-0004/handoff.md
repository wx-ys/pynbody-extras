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
