# TASK-0002: Implement and verify the approved installation matrix

## Context and goal

Ship the approved ADR-005 mechanism and verify built artifacts in clean environments.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: SPIKE-0001

## Scope and contract

This is a gated follow-on brief, not authorization to choose a distribution layout. First copy the approved exact package names and file ownership into a focused implementation plan. Update pyproject and uv.lock together; build wheels and sdist, install artifacts rather than editable source. Verify full ORM and lightweight pure calculations, CSS assets, optional native import failure and supported native builds. Discover the repository-owned CI location before editing it; pre-existing untracked .github files are not automatically authorized for replacement. Use local scripts if CI ownership is not established. Do not publish releases as part of validation.

## File scope

- `pyproject.toml`
- `uv.lock`
- `setup.py`
- `README.md`
- `README.zh-CN.md`
- `tests/test_installation_contract.py (create)`
- `docs/design/installation-matrix.md (create)`

## Acceptance criteria

- [x] Linux/macOS/Windows and supported Python combinations have explicit passed/failed/not-run cells with evidence.
- [x] Full install can run ORM SQLite fixture; lightweight install calculates without SQLAlchemy present.
- [x] Uninstalling metapackage does not remove files owned by core; wheel and sdist include required assets.
- [x] Native available/unavailable paths and make check/test-all pass where the platform is claimed supported.

## Execution and validation

- [x] Read the prerequisite decisions/files and satisfy each deliverable criterion.
- [x] Record actual experiment commands and exit codes; do not mark untested platforms as passed.
- [x] Update both languages, metadata and handoff for the next implementation.

```bash
make check && make test-all
```

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.

## 2026-11-09 — evidence update / 实验补充

SPIKE-0001 temporary build verified a concrete prerequisite failure: the core sdist contains no C++ headers; rebuilding exits 0 via optional-native fallback, but no native binary is produced. After packaging approval, include all required headers and assert native presence/import in the claimed supported matrix. See [experiment evidence](../../design/packaging-decision.md).

## 2026-11-09 — approved distribution decision / 发行决定获批

The user explicitly approved names and ownership in [ADR-019](../../adr/0019-full-core-distribution-ownership.md); the decision gate is satisfied. Actual installation and platform validation remain required.

## Implementation contract (ADR-019)

Execute after FEAT-0012. Root `pyproject.toml` becomes `pynbodyext-core` and owns
all import/native/CSS files. New `packages/pynbodyext/pyproject.toml` is a standalone
metadata-only full member, binding the exact source version plus SQLAlchemy and
forwarding image/store/dev/bench extras. Add a deterministic version synchronizer,
MANIFEST.in and artifact validation script; update Makefile and uv.lock together.
Verify actual sdist headers and PEP 517 rebuilds, source install commands, native
OpenMP/serial/no-compiler paths, lifecycle ownership and legacy same-version
transition. Conservative migration removes old full before introducing core.
This extends the original file scope to those necessary member/scripts/manifest,
CONTEXT and local validation documentation; no publication or CI ownership change.

uv 0.9.17 represents the dynamic editable core without a version field in its
lock entry; check its source mapping and actual built version, while full has
the generated static version in uv.lock. This is resolver format, not a relaxed
full/core exact-version requirement.

## Implemented and verified (2026-11-09)

ADR-019 is implemented and TASK-0002 is in review. See
[actual installation matrix](../../design/installation-matrix.md): Linux x86_64 /
CPython 3.11.9 core/full wheel and sdist installs, headers/CSS, PEP 517 rebuilds,
extras, source recipes, OpenMP/serial/no-compiler paths and uninstall ownership
passed. Other platforms/interpreters are explicitly not-run. Same-version pip
direct upgrade failed the ownership contract; verified migration removes the old
single distribution before installing the new pair. Fresh core[store]/full Runs
record core version independently of optional full metadata.

Final make check: 1679 passed, 2 skipped, 7 deselected; standard make test-all:
1686 passed, 2 skipped; optional-pandas-enabled full suite: 1688 passed, zero
skipped. Nine packaging regressions pass; native C++ ALL PASS. Final artifacts
were rebuilt from all current feature source after the provenance correction.
No release upload, commit/push, working .venv mutation or untracked-user-file
replacement occurred.

## Integration outcome — 2026-11-09

Merged via [TASK-0002 PR](https://github.com/wx-ys/pynbody-extras/pull/53), merge commit `7417d949c57e01c2dcc4ac609005eaaf869eddcc`. Current status: done; this supersedes the historical local-only/review notes above. Exact scoped integration checks are recorded in handoff.md and the [integration record](../../design/integration-2026-11-09.md). No release was published.
