## 2026-10-20 — Claude @ cloud session

- State: ready
- Done: found while fixing TASK-0001; confirmed with the same NumPy steps
  that size (3, 5) gives a (7, 6) median array for a (6, 7) image and
  `np.where` raises.
- Next: write the failing test, then fix the pad widths.
- Commands: `.venv/bin/python -m pytest tests -k median -rs`.
- Failing test: none yet.
- Blockers: none.
- Files: `pynbodyext/plot/image/smooth.py`, tests for `plot.image.smooth`.
- Env: Linux VM / NumPy 2.2.6 (reproduction only).
