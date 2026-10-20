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

## 2026-10-20 — Claude @ cloud session

- State: review
- Done: implemented the Decision in `en.md` on branch
  `fix/native-availability-import` (`util/deps.py`, `gravity/__init__.py`,
  `_vcirc` message). New `tests/test_native_availability.py`: subprocess
  probes with a fake meta-path finder for "found but fails to load" and
  "not found", checking the flag, the exports, the warning (recorded and
  visible under default filters) and the `vcirc` error; plus real-environment
  checks. `test_bins_vcirc.py` also checks the reason is in the message.
  Red -> green, verified in a Linux VM with a stub pynbody dist-info (probe
  only, before the vcirc step was added): old code -> `import
  pynbodyext.gravity` raises `ImportError` in the broken case; new code ->
  imports, `NativeExtensionWarning` printed to stderr, reason names the load
  error.
- Next: on the Mac run `.venv/bin/python -m pytest tests/test_native_availability.py tests/test_bins_vcirc.py tests/test_gravity.py tests/test_package_layers_contract.py -rs`
  and `make check`; tick the remaining criteria.
- Commands: as above.
- Failing test: none expected; pytest has not run yet (no pytest/pynbody in
  this session's shell).
- Blockers: `util` and `gravity` changes need human review.
- Files: `pynbodyext/util/deps.py`, `pynbodyext/gravity/__init__.py`,
  `pynbodyext/core/calculate/bins/result.py`,
  `tests/test_native_availability.py`, `tests/test_bins_vcirc.py`,
  `docs/work/BUG-0004/*`, `docs/work/BUG-0005/*`.
- Env: Linux aarch64 VM / Python 3.10.12 (probe only).
