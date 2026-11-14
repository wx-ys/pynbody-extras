# FEAT-0017 handoff

## 2026-11-14

Implementation authorized. Baseline 41dc264; branch codex/database-user-interface. Tests before implementation; no unrelated files staged.

## Implementation and review

Accepted ADR-020. All 52 new contracts pass; 88 focused regression tests pass; make check passes (1731 passed, 2 skipped, 7 deselected). Independent reviews fixed and rechecked scientific identity, replay, recovery and metadata-only browsing. Public code and new docs ready; complete make test-all before PR merge. Existing unrelated docs/examples and local files remain unstaged.

## Full verification

make test-all passed: 1738 passed, 2 skipped in 81.08 s. The initial sandbox run could not reach PyPI for isolated builds; the authorized network-enabled rerun passed. No test was skipped or weakened to bypass that environment failure. Log: /tmp/database-test-all.log. Ready for publication.
