# SPIKE-0001: Decide supported full and lightweight distributions

## Context and goal

Prepare a concrete ADR-005 packaging choice for human review before changing package ownership or dependency policy.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: No new work-item dependency; start from current main.

## Scope and contract

Compare a lightweight core distribution plus a full metapackage against the existing optional-store distribution. Recommend the former to preserve ADR-005, but do not choose/publish package names without review. Prototype wheel metadata in a temporary directory, record which distribution owns every pynbodyext path, native binary and CSS file, and show how installing both avoids overlapping ownership. Identify lockfile/release-version synchronization and sdist behavior. The alternative of keeping optional ORM requires explicitly changing ADR-005; extras cannot remove mandatory dependencies. No source packaging edits or external publication in this spike.

## File scope

- `pyproject.toml (read only during spike)`
- `setup.py (read only during spike)`
- `docs/design/packaging-decision.md (create)`
- `tests/test_package_layers_contract.py (read only during spike)`

## Acceptance criteria

- [x] Decision document contains candidate names clearly labelled unreserved, ownership map and install/uninstall matrix.
- [x] Temporary experiment verifies dependency metadata for full vs lightweight without relying on --no-deps as a supported path.
- [x] Record exact experiments, unavailable network/platform limitations, and request review of the final concrete choice.
- [x] TASK-0002 remains gated until the architecture and distribution names are approved.

## Execution and validation

- [x] Read the prerequisite decisions/files and satisfy each deliverable criterion.
- [x] Record actual experiment commands and exit codes; do not mark untested platforms as passed.
- [x] Update both languages, metadata and handoff for the next implementation.

```bash
make docs-check
```

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.

## 2026-11-09 — local spike / 本地实验

Prepared the concrete [packaging proposal](../../design/packaging-decision.md) with unreserved candidate names, complete file ownership, dependency metadata, install/uninstall expectations and actual temporary wheel/sdist experiments. Offline installation could not resolve uncached dependencies. The sdist rebuild exited 0 but omitted native code after missing-header compilation failure; this is an explicit TASK-0002 release-validation requirement. No production packaging changed.

Status: review; human approval pending. Source builds and metadata checks passed, isolated installation and uninstall not certified; TASK-0002 remains blocked. Logs: /tmp/pynbody-packaging-spike/.
