# FEAT-0012: Record a serial study over registered inputs and calculators

## Context and goal

Produce a value/error/run/ref report per cell through existing tracked runs.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: FEAT-0005, FEAT-0006

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-018](../../adr/0018-serial-recorded-studies.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §7: run_study(repo, input_ids, calculators, store_name, location_ids=None, reuse=True, errors="raise") returns StudyReport. calculators is an ordered label->Calculator mapping; input IDs must be unique roots, location_ids maps input IDs to location IDs. Validate the full job list before work; execute serially via run_input. errors="record" continues only for RunExecutionError or returned failed RecordedRun; RunPersistenceError always stops and carries the partial report plus original receipt/cause in a StudyPersistenceError. StudyReport JSON uses ValueCodec and pinned refs; load is read-only. Resume is an explicit new invocation with new runs, not automatic exactly-once replay. No ORM import from SampleTable.

## File scope

- `pynbodyext/orm/study.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_study.py`
- `CONTEXT.md`

## Acceptance criteria

- [x] Two inputs by two calculators create four Runs and four correctly mapped report cells.
- [x] Repeated study creates four new runs but reuses payload refs; refresh preserves older report refs after DB reopen.
- [x] A scientific failure in record mode leaves an error/run cell and continues later cells.
- [x] Metadata finalization failure stops later cells and exposes partial progress plus receipt; no automatic attach/delete.
- [x] JSON preserves values, units, typed refs and errors; duplicate inputs/labels and unknown location mappings reject preflight.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Update both locales, meta.json and handoff with red/green evidence.
- [ ] Parent integration workflow: prepare a scoped PR when publishing is authorized.

```bash
.venv/bin/python -m pytest tests/test_orm_study.py tests/test_orm_associations.py tests/test_sample_refs.py tests/test_package_layers_contract.py -q
```

## Local implementation and validation

Implemented `run_study`, frozen `StudyCell`/`StudyReport` and `StudyPersistenceError` in the ORM layer. Full-list read-only preflight validates ordered labelled calculators, positive exact-int unique root IDs, loader/store/location bindings and ownership, and existing recipe payload conflicts before any Run, loader call or recipe mutation. Stable nonconstructible live calculators and their configured options remain supported; JobSpec reconstruction restrictions do not apply. Empty axes remain valid after request validation.

Cells execute serially in input-major/mapping order through `run_input`, retaining actual value, structured errors, Run ID and actual pinned Result ref. Repeated invocation creates new Runs while reusing payload refs; refresh and database reopen preserve earlier values and refs. Record mode continues only recorded execution failures or returned failed RecordedRuns. Any RunPersistenceError halts immediately; StudyPersistenceError preserves the exact original receipt, Result/ref aliases, cause chain, interrupted coordinate and prior completed cells. Recovery is exclusively an explicit caller action; the interrupted cell is absent from the completed report.

Versioned report JSON (`pynbodyext.orm.study/v1`) uses ValueCodec and typed RecordRef JSON, preserves units/errors/None/nested NumPy values and full requested axes, rejects malformed/duplicate fields and mismatched/unpinned refs, and loads without backend access or calculator reconstruction. Saved checkpoints do not automatically resume or attach Runs; a new explicit invocation creates new Runs.

Red evidence: **91 failed**, all missing new API (`/tmp/feat0012-red.log`). Initial green: **91 passed** (`/tmp/feat0012-green-attempt.log`). Four additional malformed nested-ref JSON regressions went **4 red** (`/tmp/feat0012-ref-red.log`) then green after strict nested JSON validation. Final-source focused suite is **193 passed, 1 skipped** in 17.76s (`/tmp/feat0012-focused.log`), including all **95 study cases**, existing ORM/loading/SampleTable refs and layering/calculator contracts. The focused skip requires optional pandas. `make check` passed lint/mypy (124 source files)/docs with **1670 passed, 2 skipped, 5 deselected** in 70.78s (`/tmp/feat0012-check.log`); `make test-all` passed **1675 passed, 2 skipped** in 72.39s (`/tmp/feat0012-all.log`). Both full-suite skips require optional pandas. SQLite/Linux verification does not certify other SQL dialects/platforms. Source/tests are frozen for parent review; verified local predecessors do not imply a remote merge. No Git or remote action is claimed.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify local predecessors without claiming remote merge. Commit/push permissions follow the active user authorization, not this template.
