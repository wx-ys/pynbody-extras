# TASK-0002 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — SPIKE evidence

- Temporary sdist rebuild confirmed no headers, missing gravity/octree.hpp, optional native failure swallowed with exit 0. Built wheel has no .so. See docs/design/packaging-decision.md.
- Still blocked on approved distribution names/layout. Include explicit header collection and native presence/import assertions in the eventual implementation plan.

## 2026-11-09 — approval received

- User approved candidate names and full/core ownership. ADR-019 accepted; status ready, no approval blocker remains. Installation matrix remains unverified until actual execution.

## 2026-11-09 — implementation and artifact validation in progress

- ADR-019 layout implemented: root core owns all code/native/CSS; full metadata
  member forwards exact-version extras; one source version and deterministic
  synchronizer; uv.lock resolver regenerated with original mirror and 59 packages.
- Red evidence: six original installation contracts failed, including actual
  sdist headers; adversarial full-wheel checker red; two SPDX backend-floor tests
  red before raising setuptools minimum to 77.0.0. Nine contracts now pass.
- Actual Python 3.11.9/Linux PEP 517 core/full sdist/source wheels, clean sdist
  wheel rebuilds, forced-serial and no-compiler fallback passed. Both wheel and
  sdist resolver installs, four extras, core/full runtime, installed ownership,
  uninstall/repair, pip and uv source recipes passed. Joint 11-Run/4-cell workflow
  passed from full wheel and source without PYTHONPATH.
- Legacy experiment: pip26.3 same-version direct upgrade exits0 but retains old
  full's 119 import RECORD paths; mark that transition failed. Removing old full
  before installing new pair passes, including later full removal preserving core.
- Standard suite passed 1684/2 skipped; native C++ ALL PASS (root verification).
  Initial make check passed 1675/2 skipped/7 deselected before the extra backend
  regressions; final check remains due after final docs/source freeze.
- Root review found package-version provenance only queries the old full name;
  root is correcting that runtime boundary under FEAT-0006. Pause final artifact
  certification until that source is frozen. Then rebuild affected core artifacts,
  refresh matrix hashes, rerun full wheel workflow and core[store] provenance.
- Evidence: docs/design/installation-matrix.md; /tmp/task0002-final, temporary
  drivers /tmp/task0002-build-final.py, /tmp/task0002-install-final.py. Cache for
  offline slow tests: UV_CACHE_DIR=/tmp/task0002-cache UV_OFFLINE=1. Temporary
  optional pandas is never passed to artifact runtime probes. No Git writes,
  release upload, user untracked-file edits, or working .venv mutation.

## 2026-11-09 — final revalidation complete / 最终复验完成

- Final source includes root's FEAT-0006 provenance correction: execution context
  records code-owning `pynbodyext-core` independently, retaining optional full's
  existing field. Refreshed clean source in /tmp/task0002-revalidated, never Git
  archive HEAD; all eight isolated PEP 517 artifact builds reran successfully.
  Final hashes/bytes and source fingerprint are in installation-matrix.md and
  /tmp/task0002-revalidated/{artifact-hashes,source-sha256}.json. Earlier final/
  prototype artifacts are superseded, not silently reused as final certification.
- Normal resolver matrix completed: fresh core/full, four extras, both direct
  sdists, OpenMP, forced serial, actual no-compiler, native warning/error behavior,
  CSS and runtime science. Full wheel's unmodified joint workflow passed (11 Runs,
  4 cells, ordered duplicate selection, Catalogue, Job, StudyReport/SampleTable,
  immutable history and external overwrite). Before/after full-uninstall hashes
  of all installed core files are identical; broken dependency state was detected
  and repaired. Latest driver: /tmp/task0002-install-revalidated.py reports
  INSTALL MATRIX PASSED; native code/flags were not changed.
- Fresh core[store] actual SQLite Run records core=0.0.1/full=null; full actual Run
  records both=0.0.1. Logs core-store-provenance.log/full-provenance.log. Latest
  uv development/light source syncs and probes pass offline; pip full editable
  pair rerun against refreshed independent source copy passes joint workflow and
  package-version Run. The initial pip lightweight source proof remains explicitly
  attributed to the pre-provenance-correction snapshot; final uv light covers the
  corrected source without SQLAlchemy. All probes run outside the checkout with
  PYTHONPATH unset; working .venv untouched.
- Same-version legacy transition revalidated with pip26.3: direct upgrade returns
  0 but retains old full's 119 import RECORD paths. This path is failed, not a
  passing installation. Verified uninstall-old-first migration passes and allows
  later full uninstall without removing core. No version bump/release assumed.
- Final standard make check: 1679 passed, 2 skipped, 7 deselected; lint/typecheck,
  version/lock checks and docs check pass (/tmp/task0002-check-final.log).
  Final standard make test-all: 1686 passed, 2 skipped
  (/tmp/task0002-test-all-final.log). Root's independent optional-pandas-enabled
  all: 1688 passed, zero skipped (/tmp/pynbody-final-pandas-all.log).
  Nine installation contracts pass; unchanged native C++ ALL PASS
  (/tmp/task0002-cpp.log). Windows/macOS/other Python cells remain not-run.
- Updated both task locales, CONTEXT ownership, README pair, matrix and index;
  status review, not merged/done. No Git writes/commit/push/publication, no user
  untracked .github/.develop_notes/.test_code replacement. Root owns final
  integrated review/planning closeout; no required implementation remains here.

## 2026-11-09 — scoped integration verification

- User authorized commit, PR and merge. This increment was reconstructed in isolation after verified predecessors.
- `make check`: 1679 passed, 2 skipped, 7 deselected in 64.12s (0:01:04); `make test-all`: 1686 passed, 2 skipped in 68.65s (0:01:08). Log: `/tmp/pynbody-publish-test-12.log`.
- Independent complete-source review found no actionable findings; slice tests protect extraction boundaries. Publication/merge URL and final status are recorded after the remote operation is confirmed.
