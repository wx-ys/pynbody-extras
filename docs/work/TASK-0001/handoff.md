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
