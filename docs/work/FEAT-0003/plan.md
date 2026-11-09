# ORM associations implementation plan

Execute inline with TDD on codex/orm-analysis-associations; base a29081c.

- [x] Write tests/test_orm_associations.py for relational traversal, idempotency,
  views/revisions, run/write/hit/refresh, failed execution, metadata failure injection,
  audit, unlinked candidates, explicit recovery, SQLite reopen and FK constraints.
  Run pytest and confirm missing public API failures before implementation.
- [x] Add pynbodyext/orm/models.py (seven typed SQLAlchemy models, relationships,
  constraints) and repository.py (registration, short transaction lifecycle,
  query/fetch/audit/recovery). Add errors/value objects in records.py and public
  exports in __init__.py. Keep payloads out of metadata tables.
- [x] Add explicit orm layer above calculate in package layering contract, per
  ADR-009. Add subprocess test blocking sqlalchemy imports while running pure
  calculation; preserve all existing test behavior.
- [x] Run focused ORM + existing identity/store/layering suites; fix root causes.
- [x] Independent review of transactions, invariants and failure recovery.
- [x] Update CONTEXT.md and bilingual work docs; run make check and make test-all.
- [ ] Record red/green evidence, commit scoped files, push and create PR under
  existing authorization. Verify head/base/checks; merge and synchronize main.

Public example to test:

```python
repo = AnalysisRepository("sqlite:///metadata.sqlite", stores={"results": store})
recorded = repo.run(calc, sim, sim_identity=identity, store_name="results")
assert repo.fetch_result(recorded.run.id).store_ref == recorded.result.store_ref
# On final metadata failure, keep the exception's pinned receipt and explicitly:
repo.finish_run(error.run_id, error.result_ref)
```
