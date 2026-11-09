# FEAT-0010 Replayable Particle Selections Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. This preparation changes only this file. Do not start implementation until the parent resumes this item after verified local FEAT-0006 and FEAT-0009 completion; local verification does not imply remote merge. Do not dispatch subagents or edit other work-item plans.

**Goal:** Persist explicit root-relative particle selection definitions, reconstruct the same ordered view after metadata reopen, and track selection runs through the existing failure and receipt-recovery lifecycle.

**Architecture:** Add an ORM-local SelectionSpec and Selection relation, keeping definition identity separate from the provider's materialized view identity. Replay against the existing validated root-loading seam, then execute through the existing Calculator DAG and `_execute_run`. Permit a root-owned Run location only when an explicitly associated, validated Selection joins that root to the exact Run input.

**Tech stack:** Python 3.11+, pynbody SimSnap and families, existing CalculatorSignature and ConfiguredSnapshotIdentity, NumPy, SQLAlchemy 2, SQLite, pytest and existing Make targets. No new dependencies or schema migration.

## Approved sources, baseline and integration boundaries

Read `AGENTS.md`, `CONTEXT.md`, this item's `meta.json`, both locales and latest handoff, accepted `docs/adr/0016-replayable-particle-selections.md`, and design §6.3 before execution. The user has approved this contract; do not request repeat architecture approval for these specified relations and location rules.

At preparation time models contain seven metadata tables and `begin_run` accepts only a location owned by `Run.input`. FEAT-0009 intentionally adds two tables first; FEAT-0010 adds Selection as the tenth. Re-read current owning files at execution rather than transplanting prepared line numbers over predecessor work. FEAT-0006 is concurrently adding invocation JSON and public metadata plumbing; integrate its completed API rather than recreate its collector or schema.

| Owning file | Planned responsibility |
| --- | --- |
| `pynbodyext/orm/selections.py` (new) | Strict versioned SelectionSpec serialization, canonical descriptor generation, reconstructible filter validation and replay of one definition against a validated LoadedInput root. No SQL transaction, loading dispatch, persistent filter cache or second DAG executor. |
| `pynbodyext/orm/models.py` | Selection metadata with explicit root/view relations and nullable Run.selection_id / Run.selection. Keep scientific arrays in ResultStore. |
| `pynbodyext/orm/repository.py` | Register/get/load/run selection APIs; re-read/validate root metadata, descriptor, view identity and selected root location; eagerly load detached relations; extend existing lifecycle validation. |
| `pynbodyext/orm/loading.py` | Clarify LoadedInput can contain an explicitly replayed view with its root location/provider. Keep load_snapshot's root and transformation validation intact. Place a small shared loaded-root validator here only if needed by registration; do not weaken root-only loader validation. |
| `pynbodyext/orm/__init__.py` | Export Selection and SelectionSpec alongside existing ORM public types. |
| `tests/test_orm_selections.py` (new) | Real tiny pynbody views, strict JSON/indices/filter behavior, registration/reopen/replay, runs, corruption and receipt recovery. |
| `tests/test_orm_loading.py` | Retain arbitrary-view load_input rejection and direct location ownership rules; regression coverage for the new explicitly associated exception. |
| `tests/test_orm_associations.py` | Change only approved metadata-table expectations from nine to ten after FEAT-0009; preserve no-payload assertions and existing lifecycle tests. |
| `tests/test_orm_environment.py` | Verify selection invocation context records the selected root URI and existing logical loader/revision without changing the context schema. |
| `CONTEXT.md` and `docs/work/FEAT-0010/{en.md,zh-CN.md,meta.json,handoff.md}` | Implementation-phase contract documentation, paired actual outcomes and honest validation/status. |

Run existing `tests/test_logical_snapshot_identity.py`, `tests/test_package_layers_contract.py` and `tests/test_layering_contract.py` unchanged. `orm` sits below concrete `filters`/`properties`/`transforms`: use core FilterBase, CalculatorSignature and its existing decoder in ORM code; do not directly import concrete filter modules from ORM. Tests/project code explicitly import the filter classes they use. No provider format/digest algorithm or core execution change is planned.

## Approved amendment before implementation changes — 2026-11-09

The user approved proposal A: “首版遵循根的 family 块顺序，跨块重排明确拒绝（推荐）”. This supersedes unrestricted cross-family ordering wherever the preparation below described it. Exact order and duplicates remain supported within the root's family-block order. With dm=2/star=2, `[1,0,1,3,2,3]` stays unchanged; `[3,1,3,0]` and `[3,1]` raise ValueError without sorting or partial metadata. Convert the retained universal-order regression to explicit rejection only after this recorded amendment; add replay family-layout-change failure coverage through an already-committed Run. Do not build an adapter or mislabel families. See family-order-decision.md and amended ADR-016/bilingual spec.

## Concrete representation

Use one schema identifier, `pynbodyext.orm.selection/v1`, for the persisted selection definition. SelectionSpec is an immutable/copying value object with the approved factories:

```python
SelectionSpec.indices([3, 1, 3, 0])
SelectionSpec.family("star")
SelectionSpec.filter(Sphere("30 kpc"))
```

Provide `to_json()` / `from_json(text)` for the versioned value, implemented in selections.py. JSON has exactly `schema`, `kind`, and the one kind-specific field:

```json
{"schema":"pynbodyext.orm.selection/v1","kind":"indices","indices":[3,1,3,0]}
{"schema":"pynbodyext.orm.selection/v1","kind":"family","family":"star"}
{"schema":"pynbodyext.orm.selection/v1","kind":"filter","calculator_signature":{}}
```

The last field contains the complete existing signature dictionary from `signature.as_dict(full=True)`, not the empty illustrative object. Reject unknown/missing fields, unknown schema/kind, malformed JSON, non-object envelopes and invalid nested signatures. Parse without silently accepting duplicate JSON keys; do not coerce tuple/array/bool-mask inputs into the indices factory. Canonical JSON uses `sort_keys=True, separators=(",", ":"), allow_nan=False`; public reads return detached values so caller mutation cannot rewrite a registered definition.

Indices are an exact built-in list of exact built-in ints. Reject bool, floats, NumPy integers, nested lists, strings, negatives and out-of-range indices. Keep original ordering and duplicates; empty list is valid. Bounds are checked against `len(loaded_root.sim)` at registration and replay. Convert to `np.asarray(indices, dtype=np.int64)` only after validation when creating the pynbody view, including the empty case. A descriptor with different order or multiplicity is different.

Family requires a nonempty built-in string. Resolve through pynbody's family API without creating a new family and require membership in `root.families()`; a globally known but absent family fails explicitly instead of becoming an empty subset. Preserve the literal name in the definition. Replay yields `root[family]`, checked with the existing configured provider.

Filter requires a FilterBase calculator with both stable and constructible CalculatorSignature. Serialize the signature, reconstruct with existing `calculator_from_signature`, and re-encode the rebuilt node to check stable/constructible and matching signature identity; do not trust serialized flags alone. Reconstructing is not evaluating. Evaluate the reconstructed filter once against the untransformed root through its existing `.run(...)` without a ResultStore; obtain its public boolean mask. Reject errors, mask dimensions other than one, length different from `len(root)`, and nonboolean public mask dtype explicitly before `root[mask]`. Keep existing FilterBase normalization semantics; do not broaden the accepted definition to a bare callable or a property calculator. Unavailable/unreconstructible classes fail through the existing decoder; do not introduce plugin installation or a new signature registry.

Use the following metadata fields:

```python
class Selection(Base):
    # Integer primary key; all foreign keys use RESTRICT.
    id: Mapped[int]
    root_input_id: Mapped[int]
    view_input_id: Mapped[int]
    schema: Mapped[str]
    definition_json: Mapped[str]
    expected_count: Mapped[int]
    descriptor_key: Mapped[str]  # unique full sha256, not a truncated display hash
    root_input: Mapped[DataObject]
    view_input: Mapped[DataObject]

# On Run, independent of input_location_id:
selection_id: Mapped[int | None]
selection: Mapped[Selection | None]
```

`expected_count` is the ROOT particle count at registration, not selected view length. Derive it from the validated LoadedInput; callers cannot supply it. Add a nonnegative count CHECK and explicit foreign_keys on the two DataObject relationships. The canonical descriptor comprises `root_input_id`, `expected_count`, `schema` and the parsed canonical definition; hash the complete canonical JSON with sha256. On an existing key compare the descriptor fields and registered view identity rather than treating hash equality as permission to overwrite. Different definitions can have the same view_input_id; do not place a unique constraint on view_input_id or save one mutable definition on DataObject. An identical canonical descriptor returns the same Selection idempotently.

## Root, view and Run validation rules

1. **Registration:** require LoadedInput; re-read its registered input and selected location from this repository. Run `_validate_input_metadata` on the fresh root, require snapshot kind/no parent, require location belongs to that root, and match the supplied detached metadata/provider to those registered facts. Require SimSnap, `sim.ancestor is sim`, valid local filename and a provider identity exactly equal by `sim_identity_key` to the registered root identity. A translated/rotated root produces session identity and is rejected, even if a supplied LoadedInput field claims otherwise. A subview in LoadedInput, view DataObject, stale revision, forged location or mismatched provider fails before metadata writes or filter evaluation.
2. **Materialize once:** validate/freeze the definition, take `len(root)`, evaluate its selector once, obtain the view's actual configured provider identity, and require the registered logical root prefix plus valid existing subset suffix. Use `_input_row` and one metadata transaction to register/reuse the materialized DataObject and Selection atomically. Recheck fresh root metadata/location in that transaction without doing filter work there. Do not call `register_input` in a separate transaction that can leave a new view behind when Selection insertion fails.
3. **Descriptor validation:** one shared repository helper checks schema, strict parsed definition, expected_count type/nonnegativity, canonical descriptor hash, root hierarchy, view hierarchy/parent/root logical prefix, and consistency of root_input_id/view_input_id relations. Return the validated root/view/spec for callers. Eager-load root/view timestep→simulation, view parent, root locations, Run.selection, and the selected DataLocation before session close.
4. **Replay:** `_prepare_load(selection.root_input_id, location_id)` still selects and validates the root location and loader without IO. Load with the unchanged `load_snapshot`, require exact typed root signature and count, reconstruct the selection, then require its actual provider identity equals the registered view identity by `sim_identity_key`. Return `LoadedInput(sim=view, input=selection.view_input, location=root_location, sim_identity=root_provider)`. Compare revision and loader as part of the full logical prefix, not by a hand-picked label. A changed filter/family membership producing another digest fails before the requested analysis calculator runs.
5. **Managed runs:** `run_selection(calculator, selection_id, *, store_name, location_id=None, reuse=True, options=None)` preflights backend, descriptor, recipe, loader binding and root location. Commit `Run(input_id=view.id, selection_id=selection.id, input_location_id=root_location.id)` before loader IO, then call `_execute_run(run, execute)` where execute loads/replays and calls `calculator.run(view, store=backend, sim_identity=provider, reuse=reuse, options=options)`. Never call `load_selection` before beginning the Run. Even a result cache hit must load/replay and validate first.
6. **Only explicit selection permits a root location:** extend begin_run with nullable keyword `selection_id=None`. When absent, keep current `location.data_object_id == run.input_id` ownership rule, including when a view has a parent. When present, require Selection.view_input_id equals input_id, validate its entire descriptor/root/view chain, and require selected location belongs to Selection.root_input_id. Reject an unrelated root, unrelated selection and location-only parent inference. The location may be null for a manually begun selection Run; managed run_selection always supplies the selected root location.
7. **Shared lifecycle validation:** `_validate_run_metadata` repeats the exact selection/ownership rules when selection_id is present and still returns the VIEW signature. `_validate_ref` compares the receipt to that view and recipe, not to the root. This makes finish_run, fetch_result, unlinked_versions, audit and explicit finish_run recovery share one validation seam; never add a permissive recovery shortcut. Validate a selection Run's descriptor and root even if input_location_id is null. Audit uses metadata and pinned-reference existence without loading snapshots or decoding scientific payloads.
8. **Invocation JSON:** integrate FEAT-0006's finished begin_run/context flow. The existing `execution_context.input` records the selected ROOT URI and logical loader_id/revision from the selected view's root prefix. The VIEW identity remains on Run.input and ResultLink's RecordRef. Do not add an identity field, change scope/schema, replace producer environment or recompute context during recovery. Add an execution_metadata keyword to run_selection only if the completed FEAT-0006 API consistently exposes it for managed entry points; forward and validate it through the same collector before Run/IO.

There is no new guarantee against same-count particle reordering or in-place edits when revision is omitted. Count checks detect changed count; the provider digest detects changed selected index sequence, not changed values at those positions. Document this explicitly and test the actual behavior without promising automatic data fingerprints.

## Task 1 — Strict SelectionSpec and replay tests

- [ ] Create `tests/test_orm_selections.py` with a file-backed SQLite metadata database, separately bound SQLAlchemyResultStore, and a fixture constructing a real four-particle pynbody root. Give it an absolute `_filename`, `mass=[10.,20.,30.,40.]`, meaningful positions and explicit star/dm families. Reopen repositories against the same files and dispose all engines on teardown. Project/test imports supply concrete filter classes.
- [ ] Add parameterized public factory/JSON tests: ordered duplicates and empty indices round-trip; input-list mutation has no effect; invalid indices `True`, `1.0`, NumPy int, negative, nested list, string, tuple, NumPy array and bool mask are rejected. Test unknown/extra/missing schema/kind/definition keys and duplicate JSON keys. Bounds are a registration/replay check, not an arbitrary maximum at construction.
- [ ] Add filter cases: ordinary Sphere and a composed stable reconstructible filter round-trip; a callable-dependent unstable filter and a stable but nonconstructible filter fail. Use a stable test FilterBase whose signature reconstructs but returns a wrong-length/2-D mask to verify explicit mask errors through the real filter runtime. Passing ParamSum or a lambda to SelectionSpec.filter fails before selector execution.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_selections.py -k 'spec or mask' -q` and record a red failure caused by missing SelectionSpec/API, not a bad import or fixture.
- [ ] Implement selections.py's strict value object and single replay operation. Keep the public mask contract check at this boundary rather than changing FilterBase's normalization behavior. Run the focused cases to green, then the existing signature and layer tests.

Concrete public test shape after the API exists:

```python
indices = [3, 1, 3, 0]
spec = orm.SelectionSpec.indices(indices)
text = spec.to_json()
indices[:] = [0]
assert orm.SelectionSpec.from_json(text).to_json() == text
assert json.loads(text)["indices"] == [3, 1, 3, 0]
with pytest.raises((TypeError, ValueError)):
    orm.SelectionSpec.indices([True])
```

## Task 2 — Atomic registration and many definitions per view

- [ ] Add `test_registration_materializes_once_and_reuses_descriptor`: load a root once, register a counting stable reconstructible FilterBase, assert exactly one evaluation per registration call, correct expected ROOT count and recorded view identity. Repeating registration returns the same Selection and does not overwrite data. Reject out-of-range indices before evaluating a filter or writing metadata.
- [ ] Add `test_distinct_definitions_can_share_view`: register indices matching the star-family index sequence and a family definition (and a filter selecting the same sequence); require different Selection IDs/descriptor keys, one view DataObject ID and intact independent definitions after reopen. Register reversed or duplicate-containing indices and require distinct provider identities. Do not equate family and indices definitions just because they happen to select the same particles.
- [ ] Add registration rejection cases for non-LoadedInput, a LoadedInput holding a subview, view metadata, transformed root, stale revision/provider mismatch, invalid root hierarchy, foreign location, missing registered input and invalid family. No Selection, view or Run is created on failure. Use `dataclasses.replace` to construct invalid bundles without faking SimSnap behavior.
- [ ] Add SQLAlchemy event failure injection on Selection INSERT after a new view would be inserted. Assert rollback leaves neither the new view nor Selection while previously registered root metadata survives. Use ordinary independent transactions/queries to observe persisted rows, not the repository's implementation helper.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_selections.py -k 'registration or definitions' -q`; record red before adding Selection/model exports and register_selection/get_selection. Implement using one SQL transaction after out-of-transaction materialization, then require the tests green.

Concrete identity assertions:

```python
selection = repo.register_selection(loaded_root, orm.SelectionSpec.indices([3, 1, 3, 0]))
assert selection.expected_count == len(loaded_root.sim)
assert selection.root_input_id == loaded_root.input.id
assert selection.view_input.signature == loaded_root.sim_identity(loaded_root.sim[[3, 1, 3, 0]])
assert selection.view_input.parent_id == selection.root_input_id
assert repo.runs() == []
```

## Task 3 — Reopen, replay and data-change boundaries

- [ ] Add `test_ordered_duplicate_indices_replay_after_reopen`: register `[3,1,3,0]`, close/reopen metadata and payload engines with equivalent loader binding, call get_selection and load_selection, and assert `view.get_index_list(view.ancestor).tolist() == [3,1,3,0]`, selected masses and exact typed provider identity. Read root/view hierarchy and chosen location after sessions are closed. Include empty indices, all particles and repeated single particle.
- [ ] Add family and Sphere replay against real SimSnap data; assert selected root-relative indices and identity, then test missing family and changed filter membership explicitly. A changed selected-index digest raises rather than silently re-registering the view or attaching a new definition.
- [ ] Parameterize root revision/loader/hierarchy corruption and loaded-root particle-count changes; assert failure before the requested analysis calculator or payload cache lookup/write. Distinguish malformed stored metadata failures at preflight from valid metadata whose loaded content fails after IO.
- [ ] Add two root locations and choose one explicitly; assert wrong-root/unknown locations fail without IO, omitted choice fails for aliases, normalized returned filename is acceptable, and failed selection/loading never falls back to the other alias.
- [ ] Retain/add `test_load_input_still_rejects_view_with_selection`: even a view with a valid Selection remains rejected by `load_input(view_id)`; callers must name the Selection through load_selection. Existing digest-only halo/subset inputs remain rejected. Descriptor registration does not make catalogue objects loadable.
- [ ] Add `test_omitted_revision_does_not_detect_same_count_edits`: indices select unchanged positions after the loader changes masses at those positions; replay succeeds with the same provider identity. With a saved analysis result reuse returns the same pinned version; reuse=False recomputes a new version reflecting edits. A separately registered new revision is a separate root/selection/cache identity. This explicitly tests the limit instead of asserting that reordering must be detected.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_selections.py tests/test_orm_loading.py tests/test_logical_snapshot_identity.py -q`, capture new load_selection failures first, implement prepare/replay/load and shared validation, then repeat to green.
- [ ] Add a `@pytest.mark.data` case using `_test_data / 'gadget3/data/snapshot_103/snap_103.hdf5'`: load with real pynbody.load, register a small ordered duplicate subset, reopen SQLite, replay and compare indices/provider identity and a mass calculation with the direct view. Keep this small in memory; do not download or write testdata through new code.

## Task 4 — Selection Run creation, invocation facts and errors

- [ ] Add `test_selection_running_is_committed_before_loader`: register the selection first, reset observed calls, and use an execution loader that queries through a separate repository. Before returning the root it must observe a running row with view input, exact selection_id, root location, and FEAT-0006 context containing the selected root URI/loader/revision. Assert no original metadata connection remains checked out during IO. Inject Run INSERT/commit failure separately and require zero execution loader calls.
- [ ] Add successful run, cache-hit and refresh cases. Require distinct Run IDs with the chosen definition's selection_id, view RecordRef cache key, root location relation, unchanged initial receipt on hit, and new version on reuse=False. Run two definitions selecting the same view and assert they retain different selection_id while legitimately sharing a payload key/version.
- [ ] Add metadata-preflight cases: unknown Selection/store, unstable recipe, malformed descriptor, wrong root/view hierarchy, missing loader, no/ambiguous/foreign location and invalid execution_metadata when exposed. Assert no new Run or loader/analysis/payload work. Keep existing begin_run without selection ownership rules intact.
- [ ] Add after-begin loader OSError, invalid loaded root, changed count, missing family, invalid mask and changed digest cases. Each raises RunExecutionError with run_id and chained original cause, leaves failed with the stored selection/location/context and no ResultLink or requested analysis payload. `load_selection` alone propagates the original error and creates no Run.
- [ ] Add a stable analysis calculator that raises, a collected-errors calculation with `RunOptions(errors="collect")`, and a result-store write failure. Assert the same existing failed/RecordedRun behavior as run_input, with selection/location retained. Selection replay uses no persistent filter-result cache.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_selections.py -k 'run or preflight or failure' -q`; record expected absent run_selection/selection relation failures. Implement begin_run selection keyword, eager loading and run_selection through `_execute_run`; do not duplicate its exception handling.
- [ ] Run `tests/test_orm_loading.py`, `tests/test_orm_environment.py` and `tests/test_orm_associations.py` to ensure direct run/run_input contexts, root-only ownership and existing failures are preserved.

Run invariants tested through public results:

```python
recorded = repo.run_selection(ParamSum("mass"), selection.id, store_name="results")
assert recorded.run.selection_id == selection.id
assert recorded.run.input_id == selection.view_input_id
assert recorded.run.input_location.data_object_id == selection.root_input_id
assert recorded.result.store_ref.sim_signature == recorded.run.input.signature
assert recorded.run.execution_context["input"]["uri"] == loaded_root.location.uri
```

## Task 5 — Shared finalization, audit and explicit recovery

- [ ] Add `test_selection_metadata_link_failure_preserves_receipt`: inject failure on INSERT into result_links after a valid selection calculation. Catch RunPersistenceError and assert non-null pinned result_ref, result and run_id; independently reopen the running Run and inspect selection, root location, view input and invocation JSON. Remove injection, explicitly finish_run with that exact receipt, fetch the original result, repeat identical finish idempotently, and assert clean audit. No automatic attachment is added.
- [ ] Add failure-recording UPDATE injection for loader/selection/calculation errors. Require RunPersistenceError with committed running row and inspectable selection/location/context, no forged receipt; remove injection, call fail_run explicitly, and verify failed. For collected errors preserve result on the persistence exception just as `_execute_run` currently does.
- [ ] Add a corruption matrix applied through explicit session writes after a valid Run/receipt exists: unknown/extra schema fields, changed definition with stale key, changed expected_count with stale key, changed root_input relation/revision, view relation/identity/parent mismatch, wrong Run.selection_id, and unrelated Run input_location. On each, finish_run recovery and fetch_result reject, unlinked_versions rejects, and succeeded audit reports invalid_link. For a running run, audit may retain its existing unfinished classification, but finish recovery must validate before transition. Assert no repair of descriptor/view/receipt occurs.
- [ ] Add receipt negatives: root identity receipt instead of view identity, another selection's different materialized view, another recipe, unpinned reference, missing version and conflicting terminal completion. Require rejection leaving the original running/terminal state unchanged. A receipt from another definition with the SAME materialized view and recipe is legitimate: retain Run.selection_id rather than claiming receipt alone proves which definition produced it.
- [ ] Add direct begin_run location regression: view+parent root location with selection_id omitted fails; mismatched selection_id fails; matching selection plus matching root location succeeds. Corrupt location ownership after begin and assert recovery rejects. A merely matching parent is never sufficient authorization.
- [ ] Keep audit read-only: block loaders, monkeypatch payload codec decode to fail if called, and run successful selection audit. Root/view/descriptor validation must inspect metadata without replaying filters, reloading data or decoding values. `unlinked_versions` reports candidates only and does not infer definition ownership.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_selections.py tests/test_orm_associations.py tests/test_orm_loading.py -q`; record the specific pre-extension root-location/recovery failures and turn them green by extending the common `_validate_run_metadata` seam, keeping `_validate_ref` pinned view/recipe validation.

## Required failure semantics

| Failure point | Required effect |
| --- | --- |
| Invalid SelectionSpec, loaded registration root, location or registration bounds | Original explicit error; no partial selection/view metadata, Run, requested calculation or payload write. |
| Invalid persisted descriptor/root/view, store/recipe/binding/location preflight | Original explicit error before Run creation or loader IO. |
| Run insertion/commit failure | Original metadata error; no loader/selector/analysis execution. |
| Loader/root/count/family/mask/digest failure after Run begins | Failed Run plus RunExecutionError with original cause; selection/location/context remain recorded. |
| Requested analysis raises or payload write fails | Existing RunExecutionError and failed row; retain selected definition and root location. |
| Requested analysis collects errors | Existing RecordedRun with failed status and error-bearing result; no successful ResultLink. |
| Failure while recording failure | Existing RunPersistenceError; inspectable running row; explicit fail_run recovery. Preserve collected result when available. |
| Final metadata/link commit fails | RunPersistenceError with exact pinned receipt/result; pending row preserved; explicit finish_run revalidates descriptor, root/view and root location. |
| Stale/invalid descriptor or location during explicit recovery | Reject without terminal transition or payload mutation; never use an alternate root/location or latest version. |
| Same-count edited/reordered particles, unchanged manual identity/revision | No guaranteed detection; unchanged selected indices can reuse the same cache key. Explicit refresh or new revision remains caller responsibility. |

## Task 6 — Schema assertions, docs and complete validation

- [ ] After FEAT-0009, assert the metadata table set includes its approved two additions plus `pynbodyext_orm_selections` and totals ten. Keep every no-payload-column assertion in `test_metadata_tables_have_no_payload_columns`. Selection definition/index lists are replay metadata; do not add scientific result arrays or result payload columns to SQL models.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_selections.py tests/test_orm_loading.py tests/test_orm_environment.py tests/test_logical_snapshot_identity.py tests/test_orm_associations.py tests/test_orm_catalogues.py tests/test_package_layers_contract.py tests/test_layering_contract.py -q` and record exact results. Use `-m data` on the data selection case if a focused data rerun is needed; no silent skipping or fixture replacement.
- [ ] Update CONTEXT.md's ORM/loading vocabulary with Selection versus DataObject identity, expected ROOT count, explicit root-owned location exception, Run.selection_id, managed run timing, typed identity/digest/revision limits and original-coordinate filter evaluation. Update the existing root-only description to state load_input remains root-only while load_selection names a replay definition.
- [ ] Update both work-item locales with implemented behavior and actual tests, set meta status honestly, append handoff with red→green evidence, dependency state and any limitations. Run `make docs-check`.
- [ ] Run `make check` and `make test-all` because this changes public metadata and loading behavior. Report exact counts and Linux/SQLite scope; no certification of other SQL dialects/platforms and no invented migration coverage. New module import-layer checks and pure-calculator import isolation must remain green.
- [ ] Prepare the scoped PR description using the repository template and actual validation evidence. Follow active parent/user authorization for any commit/push/PR operation; this plan grants no Git actions. Do not claim predecessor merge, publication or completion based solely on this plan.

## Preparation self-review

The planned tasks cover all five acceptance criteria: reopen/order/typed identity (Task 3), family/filter failures (Tasks 1 and 3), root revision/count and selection digest rejection (Task 3), failed runs plus receipt recovery and root-location validation (Tasks 4 and 5), and the documented/manual revision guarantee (Task 3 and documentation). Many definitions sharing one DataObject and explicit Run.selection_id are covered in Tasks 2 and 4. The schema change is approved in both locales; it changes the expected table set/count while retaining no-payload assertions.

This is preparation only. No source, test, CONTEXT, work-item status or Git change is made, and no red/green or suite-pass result is claimed here. Execute only after the parent resumes FEAT-0010 following verified local prerequisites.
