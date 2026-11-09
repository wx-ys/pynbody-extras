# Serial Recorded Studies Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` on the parent's implementation followup. Do not dispatch agents or implement runtime behavior from this preparation task. Steps use checkboxes for tracking.

**Goal:** Execute an explicitly requested root-input × labelled-calculator study serially through tracked Runs, retaining actual values, errors, Run IDs and pinned result references in a portable report.

**Architecture:** Put orchestration and report serialization in the ORM layer above `AnalysisRepository.run_input`; keep Calculator DAG, ResultStore and SampleTable contracts intact. Complete read-only validation of the whole request before the first Run or loader call. Treat metadata persistence failure as a stop with the original recovery receipt and a partial report.

**Tech Stack:** Python 3.11+, dataclasses/JSON, existing ValueCodec/RecordRef, SQLAlchemy metadata and payload stores, SQLite/pynbody/pytest. No new dependency or schema migration.

---

## Preparation boundary and evidence

This plan is the sole preparation deliverable. Do not change runtime, tests, status, CONTEXT or Git state during preparation. ADR-018 and both FEAT-0012 specifications are approved; implementation starts only on the parent's followup after verified local FEAT-0011. Verified local predecessors do not imply remote merge.

Read `AGENTS.md`, `CONTEXT.md`, `docs/work/INDEX.md`, FEAT-0012 metadata/both locales/handoff, ADR-018, design §7, ORM records/repository/loading, ValueCodec, SampleTable serialization, `tests/test_orm_loading.py`, `tests/test_orm_associations.py` and `tests/test_sample_refs.py`. Re-read changed predecessor source before implementation.

Existing seams:

- `repo._store(name)` checks bindings without touching payloads; `repo._prepare_load(input_id, location_id)` validates detached hierarchy, location and loader binding without calling the loader. `run_input` repeats these checks before starting its Run. They are available to this adjacent ORM module without changing a core API.
- `run_input` registers a stable recipe, commits a running Run, loads even on a cache hit, calls the calculator, and returns `RecordedRun(run, result)`. New invocation means new Run, even when reusing exactly the same payload version.
- `RunExecutionError.run_id` refers to a successfully recorded failed Run and chains the execution exception. A returned `RecordedRun` can also have failed status when Result contains collected errors.
- `RunPersistenceError` preserves `run_id`, `result_ref`, `result` and its metadata exception cause. It must remain the exact recovery receipt; a pinned saved payload does not establish a completed metadata association.
- SampleTable already captures actual pinned `result.store_ref` and serializes values with ValueCodec. Reuse those leaf value/ref codecs, not its orchestration or row sentinel semantics. A successful study value of None is distinguished by cell status.
- FEAT-0010 will expand loading to some views. Studies remain **root-only**, so explicit kind/parent checks must survive that expansion rather than inheriting `_prepare_load`'s current root restriction.

## Proposed minimal public API

Create `pynbodyext/orm/study.py`; export `StudyCell`, `StudyReport`, `StudyPersistenceError`, `run_study` in `pynbodyext/orm/__init__.py`. Create `tests/test_orm_study.py`. Only implementation followup updates CONTEXT and bilingual work-item completion evidence.

```python
def run_study(repo, input_ids, calculators, store_name, location_ids=None,
              reuse=True, errors="raise") -> StudyReport: ...

@dataclass(frozen=True, slots=True)
class StudyCell:
    input_id: int
    label: str
    run_id: str
    status: str                    # "succeeded" or "failed"
    value: Any = None
    ref: RecordRef | None = None
    errors: tuple[ErrorInfo, ...] = ()

@dataclass(frozen=True, slots=True)
class StudyReport:
    store_name: str
    input_ids: tuple[int, ...]
    identities: tuple[tuple[Any, ...], ...]   # same order as input_ids
    columns: tuple[tuple[str, str], ...]      # ordered label/full signature hash
    cells: tuple[StudyCell, ...]             # completed attempts in execution order
    def cell(self, input_id: int, label: str) -> StudyCell: ...
    def to_dict(self, *, codec=None) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data, *, codec=None) -> StudyReport: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, text) -> StudyReport: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> StudyReport: ...

class StudyPersistenceError(RuntimeError):
    # report: completed attempts before the interrupted cell
    # receipt: the exact original RunPersistenceError object
    # input_id / label: interrupted coordinate
    # run_id / result_ref / result: unchanged aliases from receipt
    ...
```

Use actual CalculatorBase, AnalysisRepository, ErrorInfo, RecordRef and path types in implementation annotations with repository TYPE_CHECKING conventions. Preserve ErrorInfo's error_type/message/phase/traceback_text explicitly in JSON; never serialize a Python exception, ORM model, live Result or repository. Loading report metadata must not reconstruct calculators, load snapshots, fetch results, mutate Runs or locate a backend. The store name labels ownership but is not a resolver.

The report includes the full requested axes, and cells for completed attempts only. An interrupted metadata cell is absent from `report.cells`; its coordinate, Run ID, Result and candidate pinned ref are on StudyPersistenceError. This avoids presenting an unfinalized association as an ordinary report success. `cell` raises KeyError for a missing/unattempted coordinate. Build tuples from local lists when returning/checkpointing so a saved partial report cannot grow while later work proceeds.

Use schema `pynbodyext.orm.study/v1`. JSON contains exactly schema/store_name/input_ids/identities/columns/cells; columns and cells are ordered lists rather than JSON objects keyed by user labels. Encode identities and values via ValueCodec; encode refs with existing `RecordRef.to_json`. A report is a checkpoint of completed attempts, with no automatic executor/resume method, no repository URL and no binding credentials. Live calculators need stable signatures as run_input already requires; **do not impose JobSpec's constructibility restriction on live study calculators**, because reports do not rebuild them.

## Task 1: Full-list preflight before scientific work

**Files:** new study module, ORM exports, new study tests.

- [ ] Add tests importing `run_study` inside test bodies so first red evidence is a missing API, rather than an environment/collection failure. Run `.venv/bin/python -m pytest tests/test_orm_study.py -q` and retain expected failure output.
- [ ] Snapshot iterable input IDs once into a tuple and mapping items once into an ordered tuple. Require calculators and optional location_ids to implement Mapping; do not accept a list of pairs. Require input and location IDs to be exact built-in positive ints, excluding bool, floats, NumPy integers, zero and negatives. Reject repeated input IDs before querying metadata.
- [ ] Require exact nonempty, non-whitespace string labels and reject duplicate labels observed from mapping `.items()` before converting to a dict. An ordinary dict cannot retain duplicate entries; test a custom Mapping whose items repeats a key as well as duplicate JSON report labels. Distinct labels may use the same calculator/signature and still create distinct Runs.
- [ ] Validate exact bool reuse, errors in the two string policies, and a valid bound store name before any `run_input`. Reject unknown location mapping keys, including positive IDs absent from input_ids; values are positive exact ints. None means no explicit selection, not a permissive mapping value.
- [ ] For every calculator, require CalculatorBase, compute its signature and require stability. Validate any already-registered Recipe with the same hash using the same payload identity comparison as register_recipe, in a read-only repository session; do not call register_recipe during validation. Signature-construction and malformed recipe errors propagate before work.
- [ ] For every input, get metadata, require root snapshot kind and no parent, then call `_prepare_load` for hierarchy/selection/URI/loader validation. Save validated typed signature and actual selected location ID. Explicitly pass that selected ID during execution so an ambiguous-location change cannot silently select another source.
- [ ] Build the entire input-major/calculator-minor job list only after all axes and bindings pass. Empty axes may return a valid zero-cell report after validating the supplied axes and bindings; no loader, Run or payload access occurs. Do not pre-load snapshots or query result-store hits.
- [ ] Parameterize late invalid second input/calculator/location/binding cases and assert no loader/calculator calls, no Runs, no newly registered Recipes, and no payload versions. Include a valid FEAT-0010 selection that direct run_input can load but study must reject as a non-root.

Representative late-list contract (fixture is implemented in Task 2):

```python
def test_late_bad_input_prevents_all_science(study_env):
    import pytest
    from pynbodyext.orm import run_study
    repo, inputs, calls, store = study_env
    with pytest.raises(KeyError):
        run_study(repo, [inputs[0].id, 999999],
                  {"mass": ParamSum("mass")}, "results")
    assert calls == []
    assert repo.runs() == []
    assert store.list_refs() == []
```

## Task 2: Serial tracked cells and historical cache references

**Files:** study module and tests.

- [ ] Add a fixture with separate metadata/payload SQLite files, two LogicalSnapshotIdentity roots and absolute selected locations. Trusted loader produces real two-particle pynbody roots, populates mass/pos, sets `_filename`, and records calls. Use masses summing to 6 and 12 and position x summing to 1 and 2 so cells have distinguishable scientific values. Dispose all engines in fixture teardown.
- [ ] Run two inputs × ordered `{"mass": ParamSum("mass"), "x": ParamSum("x")}`; assert exactly four Runs, four cells and loader calls in input-major order. Inspect input/location/recipe/store/result-link mapping for each Run and compare every ref directly to its actual Result/store metadata receipt. Never synthesize refs from calculator/input axes or refetch current cache to obtain report values.
- [ ] Implement the loop through `repo.run_input(calc, input_id, store_name=..., location_id=selected, reuse=reuse)` without supplying an options override. Preserve each live calculator's configured options; there is no JobSpec-style default-options-only restriction. Successful cell value is `recorded.result.value`, ref is its actual pinned store_ref, Run ID is recorded.run.id. Require succeeded status and pinned ref for success; unexpected return states are errors that stop.
- [ ] Repeat same study: four new distinct Run IDs, same four pinned refs/values and all new Runs reused. Same-signature calculators under two distinct labels still produce separate Runs, even when their refs agree. No shared DAG optimization or snapshot cache is added.
- [ ] Change loader values and invoke reuse=False. Assert changed actual numerical values, new pinned versions and unchanged prior report refs. Save the earlier report, dispose/reopen both databases and load it without a repository. Explicitly fetch each saved ref from the rebound payload backend and assert original values; verify old Runs still fetch their original versions and repository audit is clean. Invalidating current cache must not erase those historical versions.

## Task 3: Scientific failure policies and exact persistence receipts

**Files:** study module and tests.

- [ ] Cause an actual scientific failure using ParamSum on an absent field, in an ordered column between valid columns. In record mode catch only RunExecutionError; obtain its already-failed Run metadata, preserve its ID and error type/message in an ErrorInfo, append a failed cell with no ref and continue subsequent jobs. Preserve the exception's original diagnostics; do not invent a successful value or fetch a current cached result for that coordinate.
- [ ] For returned failed RecordedRun, record its actual `result.errors`, Run ID and failed status; keep failure ref None. Exercise this branch with a live calculator configured with `.options(errors="collect")`, so the test uses actual calculator execution and failed SQL metadata, also proving configured options remain supported. Check the currently available `.options` spelling after predecessor integration; use an equivalent configured RunOptions on the calculator if needed, without broadening the study signature.
- [ ] With errors="raise", propagate RunExecutionError unchanged and stop before later jobs. For a returned failed RecordedRun raise RunExecutionError(run.id), chained from an explanatory RuntimeError carrying its collected diagnostics; do not retry or mark the Run a second time.
- [ ] Catch RunPersistenceError before RunExecutionError regardless of errors policy. Construct StudyPersistenceError with a snapshot report of prior completed attempts and interrupted coordinate. Its receipt `is` the caught error; run_id/result_ref/result are unchanged aliases. `raise wrapped from receipt` retains the full original chain through receipt.__cause__. Never call finish_run/fail_run/delete/unlinked_versions automatically on this path.
- [ ] Force a real metadata finalization failure on the second attempted cell with a SQLAlchemy before_cursor_execute listener rejecting INSERT into result links. Assert one completed report cell, second pending Run and exact original receipt/Result/pinned ref, both policies halting all later cells, and payload version still fetchable. Capture the original RunPersistenceError at the delegate boundary only to assert object identity; do not replace it with a fake exception.
- [ ] Remove the listener and explicitly call `repo.finish_run(error.run_id, error.result_ref)` in test/application code, then inspect succeeded Run, exact fetched value and clean audit. The study function must not perform recovery. Also cover fail_run metadata failure after a loader/science exception: no saved ref, exact cause chain and no later jobs.
- [ ] Unexpected exceptions from preflight/Run creation/registration, including raw metadata errors before a receipt exists, propagate and stop even with errors="record". Do not catch broad Exception as a continuable cell error. Existing run_input owns classification into RunExecutionError versus RunPersistenceError; this item does not rewrite that lower contract.

## Task 4: Report JSON validation and explicit reinvocation

**Files:** study module and tests.

- [ ] Implement explicit dict/JSON/save/load methods with ValueCodec and RecordRef. Serialize all ErrorInfo fields explicitly; preserve successful None via status. Include float/NumPy values, SimArray with real units, nested values and typed identities in roundtrip checks using numpy assertions and unit equality.
- [ ] Validate exact schema/field sets, top-level/collection shapes, exact positive input IDs and nonempty unique labels/full hashes, matching identities length, unique coordinates and Run IDs, known axis membership and allowed statuses. Reject refs without version IDs, refs whose typed `sim_identity_key` differs from their row, and refs with calculator hash differing from their column. Success requires pinned ref and empty errors; failure requires errors and no ref. Validate before returning a restored report.
- [ ] Reject duplicate JSON object keys and non-finite JSON constants explicitly, then use ValueCodec's own representation for legitimate non-finite values. Reject duplicate cells/columns/inputs, unknown fields/schema, foreign coordinates, invalid error field types and malformed references. Do not silently coerce bool/float IDs or overwrite duplicate labels.
- [ ] Test typed identity comparison by constructing otherwise matching refs for identities containing 1/True/1.0/"1"; they must remain distinct, including rejection of equal-by-Python-but-wrong-type ref identity. ORM root identities currently contain strings; test codec/ref validation independently rather than falsifying ORM identity metadata.
- [ ] Assert load performs no ORM/payload/loader access by restoring report JSON with relevant entry points patched to fail. No calculator reconstruction occurs; report contains signature hashes sufficient to validate refs without importing plugins.
- [ ] Save/load a partial report after the real metadata failure. Show omitted coordinates remain missing. Explicitly invoke run_study again for a chosen subset: new Runs are created for every requested cell, cached payloads may be reused, and no run/ref in the loaded checkpoint is rewritten or automatically attached. No exactly-once claim or resume token is introduced.

## Task 5: Verify and hand off the scoped increment

- [ ] Focused: `.venv/bin/python -m pytest tests/test_orm_study.py tests/test_orm_associations.py tests/test_orm_loading.py tests/test_sample_refs.py -q`.
- [ ] Boundary: `.venv/bin/python -m pytest tests/test_layering_contract.py tests/test_package_layers_contract.py tests/test_calculate_contract.py -q`; confirm pure calculators and SampleTable stay ORM-free, and ORM does not import higher built-in property/filter layers.
- [ ] Run `make check`, `make test-all`, `make docs-check` once the increment is green. Retain exact results and limits; this plan records future checks, not an execution claim. SQLite/Linux coverage does not certify other SQL dialects/platforms.
- [ ] Update CONTEXT during implementation with root-only study API, deterministic serial order, preflight boundary, actual pinned refs, failure/receipt semantics, read-only JSON, new-invocation resume and store-name ownership. Sync both FEAT-0012 locales, update honest metadata status and append handoff with red/green evidence and local-prerequisite caveat.
- [ ] Review only this item's source/test/docs diff, preserve all unrelated dirty files, and deliver the scoped increment to parent. Git/PR actions follow active parent/user authorization; this preparation does not itself authorize them.

## Self-review checklist

- [ ] Every approved acceptance criterion has a behavioral task: actual 2×2 science, new Runs on reuse, old refs after reopen, recorded failure continuation, metadata stop/recovery, values/units/typed refs/errors, malformed preflight inputs.
- [ ] Entire request validates before the first load, calculator evaluation, Run or Recipe mutation; later invalid jobs cannot cause earlier science.
- [ ] FEAT-0010 view loading and FEAT-0011 manifest constructibility do not accidentally broaden/narrow the study's approved live-root contract.
- [ ] Only RunExecutionError/returned failed RecordedRun allow record-mode continuation. Persistence receipts remain exact and are never auto-attached or deleted.
- [ ] Report API/field names, serialization and test assertions match the proposed minimal API; no ORM import is added below ORM.
