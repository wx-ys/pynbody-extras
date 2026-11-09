# FEAT-0006: Snapshot execution environment and selected input facts on Run

## Context and goal

Record what is known about an invocation without confusing it with the environment that produced a reused payload.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: No new work-item dependency; start from current main.

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-012](../../adr/0012-run-execution-context.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §5.2. Add versioned execution_context JSON to Run and optional keyword execution_metadata to run/run_input/begin_run, accepting only tool_versions (string map) and random_seed (integer or null). Snapshot Python/package versions and selected URI/loader/revision before starting execution; unavailable package versions are null. Native state is recorded without importing the extension. Caller data is copied via JSON validation. Snapshot is immutable through repository APIs; existing direct session writes remain caller responsibility. No automatic random seeding or cache-key changes.

## File scope

- `pynbodyext/orm/models.py`
- `pynbodyext/orm/repository.py`
- `pynbodyext/orm/environment.py`
- `tests/test_orm_environment.py`
- `tests/test_orm_loading.py`
- `CONTEXT.md`

## Acceptance criteria

- [x] Success, raised failure and cache hit each retain their invocation snapshot after SQLite reopen.
- [x] Changing DataLocation.uri or the caller dictionary later does not change existing Run snapshots.
- [x] Cache hit records consumer environment and same pinned payload ref; does not assert an unknown producer environment.
- [x] Invalid execution_metadata fails before loader or Run creation; pure calculator import still excludes ORM.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [ ] Update both locales, meta.json and handoff; prepare one scoped PR using the template and red/green evidence.

```bash
.venv/bin/python -m pytest tests/test_orm_environment.py tests/test_orm_associations.py tests/test_orm_loading.py tests/test_package_layers_contract.py -q
```

The red/green evidence and final local checks are recorded in the handoff below.

## Implemented behavior

`execution_context` uses schema `pynbodyext.orm.execution-context/v1` and
`scope="invocation"`. A getter-only property decodes a fresh copy of the persisted
JSON text; repository completion/failure/recovery never replaces it. Package
versions come from distribution metadata (missing distributions are null), and
native information comes only from an already-loaded module in `sys.modules`.
After ADR-019, `packages` independently records `pynbodyext-core` (code owner)
and `pynbodyext` (full metadata package), in addition to NumPy and pynbody. A
core/store installation records its known core version and null for absent full.

All three entry points validate and JSON-copy `execution_metadata` before Run
creation or execution. Missing metadata normalizes to empty tool_versions and
null random_seed; bool/float seeds, non-string tool maps and unknown keys fail
with ValueError. No RNG state, calculator signatures or cache keys change.

`run_input` snapshots the selected URI/prefix, logical loader_id and revision
before loader IO. Direct `run`, and manual `begin_run` without a selected location,
record URI null: an existing snapshot filename does not assert managed selection.
The invocation context on a cache hit belongs to its consumer; its pinned payload
ref is unchanged and its producer environment remains unknown. Loading/execution
failures retain context; failures while finalizing metadata retain the original
running context for explicit recovery. Invalid metadata or collection/begin-commit
failure performs no loading/calculation and creates no Run.

## Local verification — 2026-11-09

- Red: environment + loading modules, **52 failed, 38 passed**, for missing
  metadata/context behavior (`/tmp/feat0006-red.log`).
- Green: the exact focused command above, **121 passed**
  (`/tmp/feat0006-focused.log`).
- `make check`: lint/typecheck/docs passed; **1223 passed, 2 skipped,
  4 deselected** (`/tmp/feat0006-check.log`).
- `make test-all`: **1227 passed, 2 skipped** (`/tmp/feat0006-test-all.log`).
- Status: **review**. No Git mutations, PR or remote merge. Linux/SQLite verified;
  other platforms/SQL dialects are unverified. No database migration or automatic
  producer-context reconstruction is included.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.
