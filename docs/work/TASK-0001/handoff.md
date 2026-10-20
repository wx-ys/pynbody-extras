## 2026-10-20 — Claude @ cloud session

- State: draft
- Done: opened after BUG-0003 had to commit with `--no-verify`. Recorded the
  five errors the hook reported and likely fixes.
- Next: on the Mac run `make typecheck` and paste the full output (BUG-0001
  saw 22 errors); then move to `ready`.
- Commands: `make typecheck`; `uv run mypy --config-file=pyproject.toml pynbodyext`.
- Failing test: none (type check only).
- Blockers: needs the full mypy output from the locked environment.
- Files: `pynbodyext/gravity/base.py`,
  `pynbodyext/core/calculate/store/sqlalchemy_store.py`,
  `pynbodyext/core/calculate/bins/result.py`, others from the full output.
- Env: macOS / Python 3.11.7 (user's Mac).

## 2026-10-20 — Claude @ cloud session

- State: ready
- Done: recorded the full `make typecheck` output from the user's Mac
  (22 errors / 10 files; mypy 1.19.0, numpy 2.3.5, matplotlib 3.10.7,
  SQLAlchemy 2.0.52), grouped by area, ticked the first criterion.
- Next: fix `gravity`, `store`, `bins`, `transforms` first (8 errors), then
  `plot/image` (14); one commit per area; `make typecheck` after each.
- Commands: `make typecheck`; `make check`.
- Failing test: none (type check only).
- Blockers: none.
- Files: see Full output in `en.md`.
- Env: macOS / Python 3.11.7 / `.venv`.

## 2026-10-20 — Claude @ cloud session

- State: review
- Done: fixed all 22 errors on branch `fix/mypy-baseline` (see Changes in
  `en.md`). Written without running mypy: this session's shell has no mypy
  and cannot install it. All edited files compile (`py_compile`). Opened
  BUG-0006 for the `median_filter` pad bug found on the way.
- Next: on the Mac run `make typecheck` (target: 0 errors) and paste any
  remaining errors; then `make check` and
  `pytest tests/test_plot_image_typehints.py` (the `Artist` alias changed).
- Commands: `make typecheck`; `make check`;
  `.venv/bin/python -m pytest tests/test_plot_image_typehints.py -rs`.
- Failing test: none expected.
- Blockers: needs the mypy run.
- Files: see Changes.
- Env: Linux VM (no mypy).

## 2026-10-20 — Claude @ cloud session

- State: review
- Done: verified on the Mac: `make typecheck` -> 0 errors (was 22);
  `tests/test_plot_image_typehints.py` passes; `make test` -> 814 passed,
  1 skipped, 3 deselected, 3 failed. The 3 failures are exactly BUG-0002's
  (`test_a_volume_render_places_the_kernel_on_the_cell_grid`,
  `test_a_volume_render_matches_a_direct_kernel_sum`,
  `test_an_unequal_z_resolution_renders_the_grid_it_was_asked_for`).
  All criteria ticked.
- Next: commit without `--no-verify` (the mypy hook should pass now), open
  the PR; after merge set `done` and fill `Outcome`.
- Commands: `make typecheck`; `make test`.
- Failing test: only BUG-0002's three.
- Blockers: none.
- Files: unchanged from the previous entry.
- Env: macOS / Python 3.11.7 / mypy 1.19.0.
