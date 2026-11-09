# Logical snapshot identity implementation plan

Execution: inline in the current feature branch using executing-plans and TDD.
Goal: opt-in logical snapshot identity independent of location, with optional revision.
Architecture: a frozen value object and a configured callable provider adapt to
existing sim_identity; keep engine, storage schema and SampleTable control flow.
Tech stack: Python dataclasses/JSON, existing pynbody snapshots and pytest.

## 1. Contract tests

Create tests/test_logical_snapshot_identity.py. Import new classes inside tests so
missing APIs fail individual assertions/imports without preventing collection.
Test a relocated snapshot and cache hit, field changes, optional revision refresh,
view order/halo/transform behavior, normalized path conflicts, JSON validation,
SQLite reopen, cross-process identity and SampleTable. Representative contract:

```python
identity = LogicalSnapshotIdentity("project", "simulation", "099")
provider = ConfiguredSnapshotIdentity({old_path: identity, new_path: identity})
first = calc.run(old_sim, store=store, sim_identity=provider)
second = calc.run(new_sim, store=store, sim_identity=provider)
assert second.from_store and first.store_ref == second.store_ref
```

- [x] Run `.venv/bin/python -m pytest tests/test_logical_snapshot_identity.py -q`;
  record failures caused by the absent API before implementation.

## 2. Value object and provider

Create pynbodyext/core/calculate/runtime/logical_identity.py. Frozen fields preserve
literal string IDs; validate strings/revision at construction. identity.signature()
returns the ADR tuple. Provider copies a Mapping[path, LogicalSnapshotIdentity],
normalizes using abspath, rejects placeholder paths/URI/bytes and conflicting
normalized entries. Lookup uses ancestor.filename and the existing transformation
and subset rules. Unknown paths call snapshot_identity. No IO or SQL imports.

Extract `_snapshot_view_identity(sim, ancestor, base)` in runtime/sim_identity.py
from the existing final root/halo/subset branch; call it from both providers.
The path provider must produce exactly its existing tuples.

- [x] Implement the value object, provider and shared view helper; rerun identity tests.
- [x] Implement to_json/from_json with the exact schema and entries fields; reject
  invalid schema, unknown/missing fields, relative paths and conflicts on loading.
- [x] Export LogicalSnapshotIdentity and ConfiguredSnapshotIdentity through
  core/calculate/__init__.py and pynbodyext/calculate.py.

## 3. Integration and publication

A separate red regression exposed all default logical SampleTable names as v1.
Update only sample.py::_default_name to recognize the logical schema and derive
a compact JSON label from its five fields, then use the remaining view suffix.
Test two timesteps, revision/loader variations, aliases, subsets, halos and from_store.

- [x] Run new tests with test_store_reuse.py, test_sim_identity_store.py,
  test_result_versions.py, test_sample_table.py and both layering contract suites.
- [x] Update CONTEXT.md with identity/revision/load semantics and public example.
- [x] Run make check and make test-all; diagnose failures without weakening tests.
- [x] Update both work-item locales, metadata, handoff and index with red/green evidence.
- [ ] Stage only scoped files (force-add the ignored ADR by exact path); commit,
  push, create PR, verify required checks, merge under existing authorization and
  fast-forward local main. Keep unrelated untracked directories intact.
