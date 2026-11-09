# SPIKE-0001 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — Codex / packaging experiment

- Proposal: docs/design/packaging-decision.md, candidate pynbodyext-core owns all code/native/CSS; pynbodyext is metadata-only full package with mandatory SQLAlchemy and exact core version. Names unreserved, approval pending.
- Actual experiments: original metadata, core wheel/sdist, full wheel/sdist all exit 0; core wheel has 119 package files incl new Linux/OpenMP native and CSS; full has zero package files; overlap empty.
- sdist rebuild: exit 0 but native compiler failed missing gravity/octree.hpp; resulting wheel has no .so. Optional build exit alone is inadequate.
- Offline install: initial uv-cache sandbox error (exit 2), normal approved retry exit 1 because registry dependencies absent from cache. No network or global installs. Matrix remains unverified.
- Evidence: /tmp/pynbody-packaging-spike/{commands.json,inventory.json,sdist-rebuild.json,*-build.log,full-install.log}; experiment scripts /tmp/pynbody-packaging-{spike,rebuild}.py. Documentation records exact setup, commands, versions and limits.
- Status: review; TASK-0002 blocked pending human approval. No production packaging edits or release publication.

## 2026-11-09 — remote integration confirmed / 远端集成确认

- Merged: https://github.com/wx-ys/pynbody-extras/pull/41; merge commit `674ea4f21b473df282ea2e67f1ec9f8426596eb8`.
- Actual remote state confirmed MERGED before status=done. User authorized autonomous PR/merge; predecessors were merged in dependency order.
- Checks: Documentation/decision record in PR41; actual installation matrix verified and merged with TASK-0002..
- Local main is synchronized separately and verified against the complete final source. Platform/backend limitations from the specification remain.
