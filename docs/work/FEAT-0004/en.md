# FEAT-0004: Load registered snapshots through explicit loader bindings

Status: proposed design; loading configuration boundary awaits user review.

## Goal and scope

Complete the next usable path after FEAT-0003: reopen metadata, identify a
registered root snapshot, load it from a selected DataLocation, run an existing
Calculator DAG and retain the exact result reference. The ORM currently registers
locations but cannot load them. Existing halo/subset signatures contain only an
index digest, not a reconstructible selection, so loading them is deferred.

Preserve optional revision/manual refresh, location-independent identity,
separate metadata/payload commits and explicit recovery. No old database/cache
migration is required. No new dependency, execution engine or import layer.

## Proposed loading boundary

AnalysisRepository gains loaders={loader_id: callable}. Copy and validate this
process-local mapping, as with named stores. A loader accepts one location string
and returns a pynbody root SimSnap. Parameters are pre-bound in project code
(e.g. functools.partial or an adapter); the database stores the existing loader_id,
not Python code, import paths, credentials or serialized callable objects.
Reopen with equivalent bindings. No implicit default: explicitly bind
loaders={"default": pynbody.load}. Changing interpretation requires a new loader_id;
rebinding does not detect changed semantics or automatically invalidate results.

Alternative: persist a versioned declarative loader specification and parameters.
This can improve configuration portability, but needs a separate allowed-parameter,
codec and reconstruction contract. The recommended first increment uses explicit
runtime bindings consistent with the existing user-managed loader_id.

V1 loads local absolute file paths or snapshot prefixes. Do not require isfile:
split snapshots may use a prefix which is not a file. Remote URIs and relative
locations remain valid metadata but fail loading preflight. Resolve locations by
input ownership; with exactly one location select it, with zero fail, and with
multiple require an explicit location_id. Never probe alternatives or retry another
location silently. Data equivalence remains a user assertion, not a content proof.

## Public API and data flow

- get_input(input_id) eagerly returns the immediate timestep/simulation, parent
  and locations needed to inspect a registration after closing the session.
- load_input(input_id, *, location_id=None) returns LoadedInput containing the
  SimSnap, selected DataObject/DataLocation and a ConfiguredSnapshotIdentity
  provider usable as sim_identity. This operation creates no Run.
- run_input(calculator, input_id, *, store_name, location_id=None, reuse=True,
  options=None) resolves/validates metadata and bindings, registers the recipe,
  commits running, then loads and executes through the existing calculator path.
  Returns the existing RecordedRun. No ORM transaction stays open during loading
  or scientific work. Loading happens even on a cache hit; skipping IO is deferred.
- Run gains nullable input_location_id pointing to the selected DataLocation,
  eagerly available in run queries. run_input sets it; existing run(calculator,
  sim, ...) leaves it null because no repository-selected load was performed.
  The logical input includes the loader_id used. This is selected-location
  provenance, not a historical copy of mutable metadata.

Only load a DataObject whose validated signature/kind/hierarchy describe a root.
Reject views before calling a loader. Require a root SimSnap, a usable local
filename and no active external transformation. A loader may normalize the supplied
prefix to another filename: bind the returned filename to the registered logical
identity and verify the resulting signature exactly equals that registration.
Do not rewrite SimSnap.filename or invent a stable identity for an invalid result.
LoadedInput.sim_identity retains existing subset/halo and transformation rules.

Loader bindings are trusted project code and may perform IO; the repository does
not infer scientific equivalence from a callable's name. Arbitrary custom objects,
created snapshots without a file identity and returned subviews fail validation.

## Failures and recovery

Unknown input/loader/store, invalid hierarchy, unsupported location, ambiguous
location or unconstructible request fail preflight without calling the loader or
creating a Run. A loading/returned-object validation error after run creation
produces failed, with RunExecutionError carrying run_id and the original cause.
A failed metadata status write produces RunPersistenceError and leaves an
inspectable running record, as in FEAT-0003. Load-only calls propagate errors.
Calculation, cache, payload and finalization errors retain existing semantics;
finish_run remains explicit and never chooses a version automatically.

## Acceptance criteria

- [ ] Reopen SQLite; inspect/load a registered root and use its provider with run.
- [ ] run_input records input, selected location, recipe, status and pinned output.
- [ ] Alias relocation preserves cache identity; revision/loader differences split it.
- [ ] Multiple locations require a choice; wrong-owner IDs and invalid local paths
  fail before IO; multipart prefixes are not rejected by file-existence checks.
- [ ] Missing bindings, corrupted hierarchy and registered views fail preflight.
- [ ] Reject non-SimSnap, subview, no-file and externally transformed loader output.
- [ ] Loading errors yield failed Runs; status-write failure remains recoverable.
- [ ] Cache hit/refresh/history and final metadata failure preserve FEAT-0003 rules.
- [ ] No ORM session transaction covers loader execution; no SQLAlchemy import is
  introduced into pure calculation; existing tests remain intact.
- [ ] Focused tests first fail for missing behavior, then pass; make check and
  make test-all pass, including a real local pynbody load/reopen integration test.

## Follow-on scope

Catalogue namespaces and Halo/Subhalo identity; reconstructible selections and
container loading; merger trees; declarative loader config; skip-load cache lookup;
full/lightweight packaging. These are separate increments, not prerequisites to
this root-snapshot path. No directory scanning, downloads, fingerprints, automatic
fallback, global registries or cached live snapshots in this increment.

See [proposed ADR-010](../../adr/0010-explicit-snapshot-loading.md).
