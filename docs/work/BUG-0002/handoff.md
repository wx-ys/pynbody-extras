## 2026-10-20 — Claude @ cloud session

- State: draft
- Done: opened from the user's `make test` run after BUG-0001 merged
  (797 passed / 6 failed / 3 skipped). Traced the three SPH failures to the
  locked pynbody 2.3.3: the kernel API (`get_value(d=r/h)` only) and the
  `to_3d_grid` z sizing that uses `ny`.
- Next: answer the open questions (pynbody floor; option A or B), then set
  `ready`.
- Commands: `.venv/bin/python -m pytest tests/test_bins_sph_render.py -k "volume_render or unequal_z"`.
- Failing test: the three tests named in `en.md`, on pynbody 2.3.3.
- Blockers: maintainer decision on the pynbody floor.
- Files: `tests/test_bins_sph_render.py`, `pyproject.toml`, `uv.lock`,
  `pynbodyext/core/calculate/bins/_pynbody_render.py`,
  `pynbodyext/util/deps.py`.
- Env: macOS / Python 3.11.7 / pytest 9.0.2 / pynbody 2.3.3.

## 2026-11-08 — Codex / local checkout

- State: review; branch codex/p0-image-sph-baseline, based on main d76c479 plus
  BUG-0006 commit e9b6003. Independent of codec PR #33.
- Decision: maintainer explicitly approved Python >=3.11 and pynbody >=2.8.0.
  Upstream 040ade1eaf58a5fa9ac22194d7b4be8b26c536aa is an ancestor of v2.8.0;
  installed source uses `(z2-z1)/nz`. The earliest fixed release is not claimed.
- Done: updated metadata/lock/docs/tool targets, removed obsolete typing fallback,
  added support-floor contract, unified kernel references and added non-unit h
  projection coverage. Kernel2D.value remains unimplemented in 2.8.0, so the
  helper evaluates get_value(r/h) at unit h and applies h**(-h_power).
- Red: renderer contract + SPH tests: 4 failed / 42 passed on 2.3.3.
- Green: same suites plus two projection scales: 48 passed on 2.8.0.
- Checks: make check passed (830 passed / 1 skipped / 3 deselected, lint, mypy,
  docs); make test-all passed (833 passed / 1 skipped); locked sync and offline
  lock validation passed. No original assertion/tolerance weakened.
- Next: maintainer reviews, pushes, opens PR using pr-body.md, and merges.
  Python 3.10 is deliberately unsupported. Other platforms/interpreters untested.
- Local-only caveat: pre-existing untracked .github/workflows/ci.yml still has
  Python 3.10 in its matrix. It is not included; adjust before publishing it.
- Env: Linux / Python 3.11.9 / pynbody 2.8.0 / uv 0.9.17 / project .venv.
- Independent review: no actionable findings; 48 focused tests passed and the
  reference helper agreed with 3-D value / direct projection integrals at five h values.
