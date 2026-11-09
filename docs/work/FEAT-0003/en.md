# FEAT-0003: Link simulation inputs, calculator recipes and saved results through ORM runs

## Context and goal

Build the first ORM association chain above PR #38 logical identities and PR #37
immutable result references. Preserve accepted V1 ADR-001/002: ORM owns metadata
and references, ResultStore owns payloads, and the current Calculator DAG executes.
The user explicitly selected separate metadata/payload stores with validation and
manual recovery on 2026-11-09. No existing database/cache compatibility is needed.
Revision remains optional; reuse=False is the user's refresh mechanism.

## Scope and entities

New public package pynbodyext.orm contains SQLAlchemy 2 mapped classes and an
AnalysisRepository. It imports the existing calculator layer; the core and public
calculator facade must never import ORM/SQLAlchemy just to calculate.

- Simulation: unique (project_id, logical_id), where logical_id is simulation_id.
- Timestep: unique (simulation PK, logical_id), where logical_id is timestep_id.
- DataObject: unique typed input signature, attached to Timestep. Separate data
  revisions/loading labels and subset/halo selections remain separate objects.
  Derived views link to their root DataObject. Catalogue-specific parents wait.
- DataLocation: URI metadata associated with a root DataObject. Multiple aliases
  and historical reuse of a path by different revisions are allowed; no loading.
- Recipe: full CalculatorSignature hash plus signature JSON; composition/deps
  remain in that existing DAG signature. Labels are not identity.
- Run: UUID, input and recipe FKs, store_name, wall-clock start/end, status,
  error details and nullable reused flag. Every invocation creates a new run.
- ResultLink: one root output per run, pinned RecordRef JSON and store schema.
  Named Pipeline outputs remain in the existing payload. No arrays in ORM.

## Public operations

AnalysisRepository(url_or_engine, stores={name: ResultStore}, create=True).
register_input(LogicalSnapshotIdentity or provider tuple, location=None) registers
hierarchy idempotently. Only configured logical identities and existing validated
subset/halo suffixes are accepted; path/session identities are not silently
converted to scientific objects. register_recipe(calculator signature) checks
stability and validates matching canonical payload on reuse.

begin_run(input_id, recipe_hash, store_name) persists a running record before work.
run(calculator, sim, sim_identity=provider, store_name=..., reuse=True, options=None)
registers the input/recipe, begins a run, delegates to calculator.run, then links
the exact result.store_ref. It returns RecordedRun(run, result). Collected errors
return a failed recorded run without a result link; raised execution errors carry
a run_id and preserve the original exception as cause.

finish_run(run_id, ref, reused=None) validates backend, pinned version, input and
recipe identity before atomically updating run status and inserting the link.
Same-reference completion is idempotent; different completion of a terminal run
is rejected. Manual recovery leaves reused unknown unless explicitly supplied.
fail_run only transitions running to failed. Terminal status transitions use
conditional updates, not an unchecked read-then-write.

get_run and runs filters (input_id, recipe_hash, status) return detached records
with input->timestep->simulation, recipe and result_link loaded. session() exposes
an ordinary SQLAlchemy Session for relationship queries/joins. fetch_result(run_id)
fetches the pinned historical version, with no latest-cache fallback.

## Failure and recovery

Metadata/payload writes are separate even when URLs happen to name the same DB.
If final metadata commit fails, RunPersistenceError exposes run_id, result_ref and
live result; the earlier running row survives. finish_run repairs only after
explicitly choosing and validating a reference. Audit reports unfinished runs,
missing bindings/links/payloads, invalid links and backend read errors without
payload decoding. unlinked_versions(run_id) lists saved matching versions not
referenced by this repository in that named store; these are candidates, not proof
of ownership or permission to delete. It never attaches/deletes automatically.
A crash may occur between steps; no automatic exactly-once execution is claimed.

## Acceptance criteria

- [x] Round-trip Simulation -> Timestep -> DataObject -> Run -> Recipe/ResultLink
  relationships through SQLite and SQLAlchemy joins, including reopened DBs.
- [x] Repeated input/recipe registration is idempotent; revisions, loaders,
  simulations, timesteps and selected views remain isolated; locations are separate.
- [x] Every run including cache hits is recorded; refresh preserves older run results.
- [x] Raised/collected execution errors create failed runs with no links.
- [x] Failures before execution prevent computation; finalization rollback leaves
  an unfinished run, exposes the receipt and supports explicit idempotent recovery.
- [x] Reject unstable recipes, path/session inputs, unpinned/missing/wrong-key refs,
  conflicting terminal transitions and invalid backend names.
- [x] Audit detects missing payload/backend and interrupted association without
  decoding arbitrary payloads. Unlinked versions remain only recovery candidates.
- [x] Foreign keys/unique constraints prevent invalid relations in SQLite.
- [x] New orm import layer is explicit; existing pure calculation works without
  SQLAlchemy. No dependency/schema migration or weakening of existing tests.
- [x] Focused, make check and make test-all pass with red/green evidence.

## Limits and deferred work

One root input/output per run; synchronous local execution; SQLite verified.
No distributed locks/retries, heartbeat, automatic recovery/GC, RunStep tables,
complete catalogue/merger trees, arbitrary recipe-node ORM, data loading or UI.
Registering the same entity concurrently may require caller retry after a unique
constraint conflict. Each SQLAlchemy session belongs to one unit of work.
Persistent metadata may point to an ephemeral store if explicitly configured;
reopening requires the same named store and retained versions. Rebinding a name
does not migrate data. Normal cached values and externally configured stores are
trusted under the existing ValueCodec contract.
ADR-005 default/full vs lightweight distribution remains a separate packaging
work item: this increment uses the existing store extra, not a new packaging policy.

## Plan and outcome

See [ADR-009](../../adr/0009-orm-analysis-associations.md) and [plan](plan.md).
Implemented on base main a29081c. Baseline identity/store suites: 91 passed.
RED: 20 missing-ORM API failures and 1 pure-calculation pass; GREEN: 21 passed.
Two added malformed-reference/hierarchy checks failed before validation fixes.
Final ORM suite: 26 passed, including concurrent conflicting completions and
payload/failure-recording errors. Independent review: no remaining actionable
findings, 31 ORM/package-layer tests passed. Full validation: make check passed
(ruff, mypy 116 modules, docs; 1046 passed, 1 skipped, 3 deselected); make test-all
1049 passed, 1 skipped. Source behavior of existing tests was preserved; the
package-layer declaration deliberately adds orm per ADR-009.

SQLite 3.35+ is needed for conditional UPDATE RETURNING. No other SQL dialect is
claimed validated. Publication and merge are authorized.
