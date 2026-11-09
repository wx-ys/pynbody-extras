# FEAT-0006 Execution Context Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. This preparation changes only this file. Execute after FEAT-0005 has passed its local checks; verified local prerequisites do not imply a remote merge.

**Goal:** Persist one immutable, versioned invocation snapshot for every ORM Run, including runs which reuse a payload or fail during loading/calculation.

**Architecture:** Keep collection and caller-metadata validation in the new ORM-local `environment.py`. Store canonical JSON text on `Run` and expose a detached decoded snapshot; commit it with the initial running row, never during finalization. Reuse the current Calculator DAG, pinned ResultLink and failure/recovery lifecycle.

**Tech stack:** Python 3.11+, standard-library JSON/importlib.metadata/platform/sys, SQLAlchemy 2, SQLite, existing pytest and Make targets. No dependency additions or migrations.

## Approved sources and ownership

Read `AGENTS.md`, `CONTEXT.md`, `docs/work/FEAT-0006/{meta.json,en.md,zh-CN.md,handoff.md}`, accepted `docs/adr/0012-run-execution-context.md`, and design §5.2 before execution. The approval already covers this increment; the choices below specify the approved contract rather than change architecture.

| Owning file | Responsibility |
| --- | --- |
| `pynbodyext/orm/environment.py` (new) | Validate/copy `execution_metadata`, collect known versions without importing native, build versioned JSON snapshot. No SQL, environment scanning, filesystem inspection, Git, seeding or cache logic. |
| `pynbodyext/orm/models.py` | Add non-null execution-context Text storage on Run and a read-only `execution_context` property returning a fresh JSON object. |
| `pynbodyext/orm/repository.py` | Add the keyword to `run`, `run_input`, `begin_run`; validate before execution or Run creation; capture selected facts once; insert snapshot with running row. Keep finalization/recovery from updating it. |
| `tests/test_orm_environment.py` (new) | Snapshot shape, validation, copying, read-only access, invocation/reuse semantics, version lookup and native import boundary. Use temporary file-backed SQLite and existing real tiny pynbody snapshots. |
| `tests/test_orm_loading.py` | Selected-location facts before IO, loader/returned-root failure retention, preflight/commit failures, explicit recovery. Keep existing tests intact. |
| `CONTEXT.md` (implementation phase only) | Document invocation-only meaning, null/unknown values, selected-location URI semantics, JSON mutation boundary and no automatic random seeding. |
| `docs/work/FEAT-0006/{en.md,zh-CN.md,meta.json,handoff.md}` (implementation phase only) | Paired outcomes, actual red/green evidence, validation counts and honest local/review status. |

Existing regression owners to run without rewriting: `tests/test_orm_associations.py` and `tests/test_package_layers_contract.py`. No core, payload schema, native implementation, loader implementation or shared plan changes are needed.

## Concrete representation and APIs

Use this initial schema and exact meaning consistently in code/tests/docs:

```json
{
  "schema": "pynbodyext.orm.execution-context/v1",
  "scope": "invocation",
  "python": "3.11.0",
  "packages": {"pynbodyext": "0.0.1", "numpy": "2.0.0", "pynbody": null},
  "native": null,
  "input": {"uri": "/data/snap", "loader_id": "default", "revision": null},
  "tool_versions": {"analysis-script": "r7"},
  "random_seed": 42
}
```

Version values above are illustrative. `python` uses `platform.python_version()`; package versions use installed distribution metadata for the three named packages. Catch `importlib.metadata.PackageNotFoundError` and record null; do not manufacture a package version from a guessed release. Native is null when `sys.modules.get("pynbodyext._native")` is absent/None. If already loaded, use `{"module": "pynbodyext._native", "version": <string-or-null>}` from its ordinary module namespace; missing/non-string `__version__` is null. Null native means no loaded-module information was observed, not that the extension cannot work. Do not call `find_spec`, `import_module`, `util.deps.native_import_error`, or import `util.deps` to collect it: that module currently attempts native loading on import.

Suggested internal signatures (not new package exports):

```python
def validate_execution_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]: ...
def execution_context_json(
    metadata: Mapping[str, Any], *, uri: str | None,
    loader_id: str, revision: str | None,
) -> str: ...
```

Add keyword-only `execution_metadata: Mapping[str, Any] | None = None` to each existing public entry point, preserving all current positional and keyword parameters. `None` and `{}` normalize to `{"tool_versions": {}, "random_seed": None}`. Accept only the two top-level keys. `tool_versions`, if supplied, must be a mapping with exact built-in string keys and string values; `random_seed` must be exact built-in int or None (reject bool, float and NumPy integer). A supplied null `tool_versions` is invalid. Reject unknown keys and non-mapping input with `ValueError` naming `execution_metadata` and the invalid field. Validate before JSON encoding to avoid silent key coercion; encode/decode with `allow_nan=False` to detach caller-owned data. Do not require seed positivity or non-empty version strings: neither restriction exists in the approved contract.

Persist canonical text with `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)`. Use `Run._execution_context_json: Mapped[str] = mapped_column("execution_context", Text)` and a getter-only `Run.execution_context` returning `json.loads(self._execution_context_json)`. There is no repository setter or replacement method. Mutating a returned dict cannot change either the same detached object on its next read or the database. Direct session access to the private backing field remains an intentional caller-owned write, as existing direct ORM writes do; do not add triggers or an event-based immutability system.

## Input facts and transaction timing

1. `run_input`: validate/copy execution_metadata at entry, retain existing `_store`/`_prepare_load`/stable-recipe preflight, then snapshot the exact `DataLocation.uri` handed to `load_snapshot`, with loader_id/revision from its registered logical input. Commit context with the running row before calling the loader. The returned snapshot filename may differ from the selected multipart prefix; record the selected prefix, not that later filename. Retain the existing selected location FK.
2. `begin_run`: validate/copy metadata before any Run insert. Load/validate its registered input and recipe in the existing transaction, extract loader_id/revision from the logical identity, and copy URI when `input_location_id` is explicit. With no location selected, URI is null even if the input happens to have one or several locations; manual begin_run must not imply it loaded anything.
3. `run`: validate/copy metadata before registering input/recipe or calculating. Snapshot URI as null because the caller supplied an already loaded snapshot and no managed location was selected. Its existing normalized ancestor filename may still be registered as location metadata; that does not make it a selected input fact. Extract known loader_id/revision from the logical input. Preserve `input_location_id=None`: `test_load_reopen_detached_metadata_and_existing_run` explicitly expects this. Do not select an arbitrary alias or add observed_uri plumbing.
4. Keep `begin_run` as the context-construction/insert point. `run` and `run_input` pass copied validated execution_metadata through the existing call. The sequential loading path must record the same selected URI used by its detached request; do not add observed-URI arguments or change location-selection rules. Concurrent caller-owned direct session edits during an invocation are outside the current concurrency contract; do not add silent reconciliation, retries or long-held transactions to address them. No database transaction remains open during IO.
5. Collect versions once for each new invocation before execution, never inside `finish_run`, `fail_run`, `get_run`, `runs` or recovery. Since native may first load during execution, the snapshot truthfully describes the pre-execution state.

`scope="invocation"` describes the current consumer even on cache hits. It does not supply a producer environment for the pinned payload. Do not add guessed producer linkage, read another Run as proof of provenance, or modify ResultRecord/ResultLink. Documentation explicitly says payload producer environment is unknown in this increment, including when the cached payload predates ORM tracking.

## Task 1 — Red tests for JSON validation and copied snapshots

- [ ] Create `tests/test_orm_environment.py` with its own small SQLite fixture modeled on `tests/test_orm_associations.py`: named SQLAlchemyResultStore, LogicalSnapshotIdentity, ConfiguredSnapshotIdentity and a two-particle mass snapshot. Dispose engines on teardown.
- [ ] Add `test_begin_run_context_round_trips_and_copies_metadata`: register input and ParamSum recipe, call begin_run with `{"tool_versions": {"script": "r1"}, "random_seed": 42}`; assert schema/scope and all normalized keys, mutate original nested mapping and decoded context, reopen SQLite, and assert the first context remains unchanged. Assigning `run.execution_context = {}` raises AttributeError. With no selected location URI is null; loader_id and revision remain the registered values.
- [ ] Parameterize `test_invalid_execution_metadata_is_preflight` over the three public entry points and the cases `[]`, `{"unknown": 1}`, `{"tool_versions": None}`, `{"tool_versions": []}`, `{"tool_versions": {1: "v"}}`, `{"tool_versions": {"a": 1}}`, `{"random_seed": True}`, `{"random_seed": 1.0}`, `{"random_seed": "1"}`, `{"random_seed": float("nan")}`. Require `ValueError`, no Run, no payload version, no loader call and no calculator invocation. For direct run additionally assert no new input/recipe registration was performed.
- [ ] Add accepted-value cases for omitted metadata, empty dict, negative/zero integer seed, null seed and empty string tool version. Do not compare full locally installed version literals.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_environment.py -q`; record the expected initial TypeError for unsupported execution_metadata / AttributeError for absent execution_context, not an environment/import failure.
- [ ] Implement validators/collector, model backing/property and public keyword plumbing to turn these tests green. Test the production API rather than duplicating validator internals in tests.

## Task 2 — Red tests for known versions and the native import boundary

- [ ] Add `test_missing_package_versions_are_null`: monkeypatch the environment module's distribution lookup to return fixed strings for two packages and raise PackageNotFoundError for one; call begin_run and assert precisely the stored versions and Python version. Repeat after changing the lookup and ensure the first Run stays unchanged.
- [ ] Add `test_context_reads_only_already_loaded_native`: after importing the ORM test prerequisites, temporarily remove native from sys.modules, install a meta-path blocker/counter for `pynbodyext._native`, and call begin_run. Assert no native import attempt and `native is None`. Insert a fake `types.ModuleType` with a string `__version__`; begin a separate Run and assert its native info. Repeat without `__version__` and expect null version. Restore sys.modules with monkeypatch cleanup. This targets collector side effects without incorrectly blaming imports performed earlier by existing calculator modules.
- [ ] Add `test_pure_calculation_still_excludes_orm` using a subprocess based on the existing association test: block sqlalchemy, run ParamSum on a tiny snapshot, and assert neither sqlalchemy nor any `pynbodyext.orm` module is in sys.modules. The existing layering checks remain unchanged.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_environment.py -k 'versions or native or excludes_orm' -q`; record the specific missing-context failure before implementing the corresponding collector behavior.
- [ ] Implement native inspection directly through sys.modules and standard-library metadata lookup. Re-run the full new module and existing package-layer contract.

## Task 3 — Red tests for invocation lifetime, reuse and recovery

- [ ] Add `test_success_failure_hit_contexts_survive_reopen`: make a successful run with environment A and explicit metadata, a cache hit with changed package lookup/metadata B, and a raised execution failure with metadata C using a stable PropertyBase test calculator. Dispose/reopen SQLite and compare each context to its original copied dict; success/hit/failure keep distinct run IDs, and hit uses exactly the first pinned store_ref. Add a collected-errors case using `RunOptions(errors="collect")` and assert failed context survives as well.
- [ ] Add `test_cache_hit_without_known_producer_context`: save a result directly through calculator.run into the bound backend before any ORM Run; then execute through repository with metadata B and assert reused=True, the original pinned ref unchanged, and the stored scope/versions/metadata describe B only. No producer environment is inferred or attached to the payload.
- [ ] Add `test_finalization_and_recovery_do_not_replace_context`: manually begin_run, produce a pinned receipt, change mocked version data, finish and idempotently finish again; compare exact context. Begin another run and fail it after changing mocked versions; context stays exact. Exercise existing INSERT-result-link failure injection, reopen its still-running Run, recover explicitly with finish_run and assert the original context survived all stages.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_environment.py -k 'reopen or producer or recovery' -q`, capture red failures for absent/incomplete context, then implement the remaining lifecycle plumbing. Keep existing UPDATE statement field lists limited to status/time/reuse/error and ResultLink insertion; no context replacement during recovery.

## Task 4 — Red tests for selected facts, preflight and retained loading failure

- [ ] In `tests/test_orm_loading.py`, add `test_context_snapshots_selected_prefix_before_loader_io`. Register an identity with non-default loader_id and explicit revision, two aliases, and explicitly choose the second location. Inside loader create a second repository observer; assert its committed running context already has the exact selected URI/loader_id/revision and supplied metadata, with zero checked-out original-repo connections. Return a snapshot with a normalized filename different from selected prefix and assert context still names the selected prefix.
- [ ] In that test commit an ordinary session edit to DataLocation.uri after completion, mutate the caller's tool_versions dict, dispose/reopen metadata SQLite, and assert the original context is unchanged although the relationship now exposes edited metadata. A later invocation may snapshot the new URI; never retroactively rewrite the first context.
- [ ] Add `test_loader_failure_retains_context_after_reopen` parameterized over loader OSError and invalid returned object. Record execution_metadata, assert RunExecutionError cause/run_id and failed row, reopen and compare the selected input facts and metadata. Also inject failure in status UPDATE: RunPersistenceError retains run_id and the committed running snapshot; remove injection, explicitly fail_run, and assert exact snapshot remains.
- [ ] Add `test_invalid_metadata_never_starts_loader_or_run` for run_input and direct begin_run with an otherwise valid input/recipe. Check ValueError, no loader calls, repo.runs()==[], no payload writes. Add non-validation collector failure injection and begin INSERT failure injection: both must propagate original preflight/metadata exception, create no Run, and perform no loader IO. Only exceptions after a successful begin go through RunExecutionError/RunPersistenceError.
- [ ] Add `test_direct_run_context_has_no_selected_uri`: direct run on a named root and on its logical view, checking URI is null, loader_id/revision are known and existing input_location_id=None. Repeat with two registered aliases and assert no selection is inferred. Manual begin_run without selected location also remains null rather than choosing an alias.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_loading.py -k 'context or metadata_never' -q` and record red before implementing the selected facts flow in repository.py. Re-run the entire loading module, including existing alias, ownership and preflight tests.

## Failure-semantics checklist

| Trigger | Outcome required |
| --- | --- |
| Invalid caller metadata | ValueError before Run creation, loader/calculation/payload work; do not wrap as RunExecutionError. |
| Unknown package distribution | Record null, continue; not an execution failure. |
| Unexpected context collector/JSON or begin INSERT failure | Propagate original exception before execution; no loader/calculation and no new Run. Metadata collection is not silently replaced with fabricated data. |
| Loader/returned-root/calculator/payload-write failure after begin commit | Existing failed Run and RunExecutionError with cause; initial context retained. |
| Failure while recording execution failure | Existing RunPersistenceError; committed running context remains inspectable after reopen. |
| Final result-link commit failure | Existing RunPersistenceError with receipt/result; committed context retained and explicit finish_run recovery preserves it. |
| Cache hit | Separate invocation snapshot and Run; same pinned payload ref; payload producer environment remains unknown. |

## Task 5 — Integration verification and paired handoff

- [ ] Run `.venv/bin/python -m pytest tests/test_orm_environment.py tests/test_orm_associations.py tests/test_orm_loading.py tests/test_package_layers_contract.py -q` and record exact results. Do not introduce SQLAlchemy/native imports into lower layers to satisfy collector tests.
- [ ] Update only the relevant ORM glossary/loading paragraphs in CONTEXT.md during implementation. Explain null package/native state, scope=invocation, immutable JSON reads, selected URI or null for direct run, direct session responsibility, and that random_seed is recorded without changing RNG state or cache keys.
- [ ] Run `make check`, `make test-all`, and `make docs-check`; the full suite is required because public ORM behavior/loading paths change. Record exact counts, data-download/environment limits, and distinguish Linux/SQLite evidence from untested platforms/SQL dialects.
- [ ] Update both work-item locales, metadata status and append handoff with actual red and green evidence, suggested scoped PR summary and any limitations. Do not mark a remote merge or release. Follow active parent/user publishing authorization; this plan grants no Git actions.

## Resolved implementation details within the approved contract

- Read-only means repository APIs do not replace stored JSON and reads return copies; it does not require a general SQLAlchemy immutability enforcement framework.
- No URI is guessed for manual begin_run or direct run. Both record null without a selected location, even when a root filename is known; direct run preserves its existing null location FK. run_input records the selected URI even if loading fails or returns a normalized filename.
- bool is rejected as an integer seed; missing seed and explicit null both mean unspecified. Tool strings are not interpreted or validated as semantic versions.
- Native absence records unknown loaded state rather than availability=False. Capturing pre-execution state must not itself load native.
- Invocation metadata does not participate in CalculatorSignature, logical identity or result-store keys, so changing tool versions/seed alone still permits the same pinned cache hit.
- No payload producer record is introduced. A pre-existing cached payload can be consumed by a new Run with known consumer context while its producer environment remains unknown.

Preparation validation: this plan is based on the current owning files and approved documents. No runtime/test file was changed and no red/green or passing-suite claim is made by this preparation.
