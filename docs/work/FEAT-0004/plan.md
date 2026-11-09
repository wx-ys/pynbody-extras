# Registered snapshot loading implementation plan

> Execute inline using superpowers:executing-plans and test-driven-development.

**Goal:** Reopen registered root snapshots with explicit loader bindings and track
loading plus calculation through the existing Run/ResultStore lifecycle.
**Architecture:** Keep pure loading validation in orm/loading.py. Repository owns
queries and short transactions; both run entry points share completion logic.
**Tech stack:** Existing pynbody, SQLAlchemy 2, SQLite, pytest; no new dependency.

## 1. Accepted decision and red contracts

- [x] Record the user's explicit approval in ADR-010, bilingual spec and handoff.
- [x] Create tests/test_orm_loading.py with a SQLite repository factory and injected
  loaders returning actual pynbody SimSnap objects. Cover detached input inspection,
  explicit/ambiguous/foreign locations, missing binding, root/view/hierarchy checks,
  normalized filenames, copied bindings and actual file load/reopen.
- [x] Run `.venv/bin/pytest -q tests/test_orm_loading.py`; confirm missing loaders
  constructor/API failures, retain the log as red evidence.

Primary contract:
```python
repo = AnalysisRepository(url, stores={"results": store}, loaders={"default": loader})
data = repo.register_input(logical, location=str(path))
loaded = repo.load_input(data.id)
assert loaded.sim_identity(loaded.sim) == logical.signature()
recorded = repo.run_input(ParamSum("mass"), data.id, store_name="results")
assert recorded.run.input_location.uri == str(path)
assert repo.fetch_result(recorded.run.id).store_ref == recorded.result.store_ref
```

## 2. Loading and metadata integration

- [x] Add orm/loading.py: SnapshotLoader callable alias, frozen LoadedInput(sim,
  input, location, sim_identity), local absolute-path/prefix validation and a
  load_snapshot helper validating actual SimSnap/root/filename/provider identity.
  Reuse ConfiguredSnapshotIdentity rather than defining another identity format.
- [x] Extend models.Run with nullable input_location_id FK and input_location
  relationship. Add eager loading in _run_query and validate ownership in begin_run.
- [x] Extract existing hierarchy checks into _validate_input_metadata for both
  input loading and result-reference validation. Do not weaken existing checks.
- [x] Add copied constructor bindings; get_input uses joinedload for scalar parents
  and selectinload for locations; _prepare_load validates before any loader IO.
- [x] Implement load_input as preflight + load_snapshot, without run creation.

## 3. Run integration and error contracts

- [x] Add run_input: preflight input/location/store, register stable recipe, commit
  begin_run(input_location_id=...), load and execute. Refactor existing run's try/
  except/finalization into _execute_run accepting a zero-argument Result producer.
  Its existing failure, collected-error, pinned-receipt and recovery semantics stay.
- [x] Add tests for visible running state before IO, no checked-out metadata
  connection during IO, load failure, failed failure-write, cache hit/refresh with
  aliases, payload/final metadata failures, and exact receipt recovery.
- [x] Run focused loading/ORM/identity/store/import tests; repair causes, not tests.

## 4. Documentation, review and publication

- [x] Document public example, reopen bindings, local/root limits, explicit choice
  and failures in CONTEXT.md; update EN/Chinese spec, metadata and handoff.
- [x] Request independent code review via requesting-code-review skill; address
  actionable findings with focused regression tests.
- [x] Run `make check` and `make test-all`; record actual output and limitations.
- [ ] Commit scoped source/tests/docs, push, create PR with template and red-green
  evidence, inspect final head/checks, merge under existing user authorization,
  fetch/synchronize main. Preserve unrelated untracked user directories.
