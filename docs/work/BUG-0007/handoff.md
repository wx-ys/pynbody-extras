## 2026-11-08 — Codex / local checkout

- State: review
- Scope: first P0 checkpoint of the user-approved incremental V1 plan. The user
  owns push, PR creation and merge; no remote action was taken.
- Branch: codex/p0-value-codec-roundtrip, based on d76c479 (local main).
- Done: recursively decode NumPy scalar payloads and retain dtype; recognize
  NoneType keys; safely parse hashable literal tuple keys; preserve existing wire
  encoding and reject nonliteral tuples. Added codec and SQLite reopen regressions.
- Red: `.venv/bin/python -m pytest tests/test_value_codec.py tests/test_sql_store.py -q`
  produced 29 failed / 9 passed before the implementation.
- Green: `.venv/bin/python -m pytest tests/test_value_codec.py tests/test_result_store.py
  tests/test_sql_store.py tests/test_store_reuse.py tests/test_sample_table.py -q`
  produced 95 passed / 1 skipped.
- Checks: make check passed ruff and mypy, then failed at pytest with
  850 passed / 3 failed / 1 skipped / 3 deselected. make test-all produced
  853 passed / 3 failed / 1 skipped. Separate make docs-check passed (8 items).
- Baseline failures: BUG-0002 in tests/test_bins_sph_render.py:
  test_a_volume_render_places_the_kernel_on_the_cell_grid,
  test_a_volume_render_matches_a_direct_kernel_sum,
  test_an_unequal_z_resolution_renders_the_grid_it_was_asked_for.
  The first two depend on missing Kernel.value; the third exposes unequal-z
  sizing. They were present before this change. No test was weakened or skipped.
- Compatibility: no schema/key migration. Nonliteral tuple reprs (including nan
  or NumPy constructor reprs) now raise ValueError rather than becoming wrong
  string keys; data already decoded and resaved with lost types is not repairable.
- Next: user reviews this checkpoint. Separate follow-ups: BUG-0002 compatibility
  decision, BUG-0006 median padding, typed simulation identity encoding, and
  immutable historical-result references before the ORM/Workspace milestones.
- Files: pynbodyext/core/calculate/store/codec.py, tests/test_value_codec.py,
  tests/test_sql_store.py, docs/work/BUG-0007/, docs/work/INDEX.md.
- Env: Python 3.11.9; pynbody 2.3.3; SQLAlchemy 2.0.52; repository .venv.
- Existing untracked .develop_notes/, .github/, .planning/, .test_code/ are
  unrelated user files and must not be staged with this checkpoint.

- Independent review: caught U/S scalar trailing-NUL width loss; four new tests
  failed before the follow-up fix. Decoder restores padding from stored dtype.
  Reviewer reran 35 codec tests and eight string probes, with no remaining findings.

## 2026-11-08 — Combined validation before merging #33

- State: review; PR #33. User explicitly authorized push, PR management and merge.
- Integrated PR #34's baseline plus d34680b floor-contract follow-up. Regenerated
  the only conflict, docs/work/INDEX.md, preserving all eight work items.
- No codec implementation or existing assertion changed during integration.
- Final make check: 866 passed / 1 skipped / 3 deselected; lint, mypy and docs pass.
- Final make test-all: 869 passed / 1 skipped on Linux / Python 3.11.9 / pynbody 2.8.0.
- Next: merge #34 first, then mark #33 ready and merge after GitHub checks pass.
- Pre-existing untracked directories remain untouched.
