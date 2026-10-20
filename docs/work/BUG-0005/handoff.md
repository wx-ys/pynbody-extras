## 2026-10-20 — Claude @ cloud session

- State: ready
- Done: opened from the open review comment on PR #25. Confirmed in code:
  `deps.module_available` uses `find_spec`; `gravity/base.py` raises on a
  failed `_native` import; `_sph_scatter.native_pair_builder()` already
  tries the import and falls back.
- Next: write the "present but unloadable" test (red), then switch `deps` to
  an import-based, cached check and carry the original error into the
  messages.
- Commands: `.venv/bin/python -m pytest tests/test_bins_vcirc.py tests/test_gravity.py tests/test_package_layers_contract.py -rs`;
  `make check`.
- Failing test: none yet.
- Blockers: none; `pynbodyext/util` and `gravity` changes need human review.
- Files: `pynbodyext/util/deps.py`, `pynbodyext/gravity/__init__.py`,
  `pynbodyext/core/calculate/bins/result.py`, `tests/test_bins_vcirc.py`.
- Env: macOS / Python 3.11.7 (user's Mac).
