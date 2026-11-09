# Root-input JobSpec Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` task by task when the parent resumes this item. Do not dispatch agents or start runtime implementation from this preparation task. Steps use checkboxes for tracking.

**Goal:** Replay a strictly validated root-input job in the same explicit metadata repository through `AnalysisRepository.run_input`, preserving scientific signatures and pinned result versions.

**Architecture:** Add an immutable description in the ORM layer, above the existing Calculator DAG and ResultStore. Explicit application imports register trusted calculator types; manifests cannot import a type or supply loader/store bindings. Reconstruction delegates to existing signature decoding only after a complete recursive trust preflight; execution delegates to `run_input` after a separate root-only check.

**Tech Stack:** Python 3.11+, dataclasses/JSON, existing calculator signatures and RunOptions, SQLAlchemy ORM, SQLite and pytest/subprocess integration tests. No dependencies.

---

## Preparation boundary and inspected evidence

This plan alone is the preparation deliverable. Do not change runtime, tests, status, CONTEXT, Git state or prerequisite files during preparation. ADR-017 and both FEAT-0011 specifications are accepted. Execute only on the parent's followup after the verified local FEAT-0010 increment; local availability does not imply remote merge.

Read: `AGENTS.md`, `CONTEXT.md`, `docs/work/INDEX.md`, this item's `meta.json`, both specifications and handoff, ADR-017, design §7, `interfaces.py`, `result/signature.py`, `nodes/serde.py`, `runtime/options.py`, ORM loading/repository/models, `tests/test_orm_loading.py`, `tests/test_signature_roundtrip.py`, `tests/test_signature_type_identity.py`, and package layering tests.

Important existing behavior:

- `interfaces.resolve_signature_type(name)` checks `_SIGNATURE_TYPES`, then **imports** an arbitrary path. Registry entries exist only for explicitly declared names/aliases; a class-path-only calculator is not registered merely because its class exists.
- `_Decoder.enum_` separately calls `import_object(payload["class"])`. Guarding only calculator nodes leaves a second manifest-controlled import path.
- Array signature decoding already uses `np.load(..., allow_pickle=False)` and refuses hash-only arrays. JobSpec must never invoke ValueCodec or add pickle fallback.
- `CalculatorSignature.from_dict` coerces envelope flags with `bool(...)`; JobSpec must check supplied flag types and reject false flags before trusting this parser.
- `default_options` are excluded from calculator signatures. Checking signature stability alone would silently discard caller options.
- `run_input` currently checks store, input/location/loader and recipe before starting a Run; loader IO begins after committing running. FEAT-0010 will expand loading to reconstructible views, so JobSpec must retain its own **root-only** guard after that change.
- ORM is below the built-in `filters`, `properties` and `transforms` layers. `jobs.py` must not import these packages to populate its registry; application code explicitly imports required types.

## Frozen public contract

Create `pynbodyext/orm/jobs.py`, export `JobSpec` from `pynbodyext/orm/__init__.py`, create `tests/test_orm_jobs.py`, and update CONTEXT only during implementation. Work-item bilingual completion/status/handoff updates follow verification. Do not edit the signature core, introduce an executor, or broaden registry behavior.

Use schema `pynbodyext.orm.job/v1` with **exactly** these six JSON fields. Emit all six, including null `location_id`:

```json
{
  "schema": "pynbodyext.orm.job/v1",
  "calculator_signature_text": "<existing calculator signature JSON text>",
  "input_id": 1,
  "location_id": null,
  "store_name": "results",
  "reuse": true
}
```

The schema spelling is the implementation's new versioned namespace; do not accept aliases or infer a missing schema. `schema`, signature text and store name have `type(value) is str`. IDs have `type(value) is int and value > 0`; `location_id` alone also accepts None. `reuse` has `type(value) is bool`. Store name must contain a non-whitespace character; retain its exact spelling as a backend label. Unknown/missing fields, unsupported schema, malformed JSON, duplicate object keys and non-finite JSON constants fail explicitly. Reject subclasses/coercions, bool IDs, float IDs, zero and negative IDs. The parser must not accept arbitrary dictionary keys through dataclass defaults.

Public shape:

```python
@dataclass(frozen=True, slots=True)
class JobSpec:
    calculator_signature_text: str
    input_id: int
    store_name: str
    location_id: int | None = None
    reuse: bool = True
    schema: str = "pynbodyext.orm.job/v1"

    @classmethod
    def from_calculator(cls, calculator, input_id, *, store_name,
                        location_id=None, reuse=True): ...
    def to_json(self): ...
    @classmethod
    def from_json(cls, text): ...
    def execute(self, repo): ...
```

Implementation annotations use real CalculatorBase, AnalysisRepository and RecordedRun types with existing TYPE_CHECKING conventions. No `options`, credentials, URL, loader, callable, snapshot, pickle, `_EvaluationPlan` or repository object is a field. Local database IDs refer to the explicitly supplied metadata repository; this is not database migration or global identity portability. Arbitrary scientific parameter strings are not scrubbed for secrets; the format serializes recipes rather than serializing repository bindings.

## Task 1: Strict immutable JSON boundary

**Files:** new `tests/test_orm_jobs.py`, new `pynbodyext/orm/jobs.py`, ORM export.

- [ ] Add focused tests that import the module inside the test body so the first failure is the missing JobSpec API, not collection/environment breakage.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_jobs.py -q` and record the expected missing-API failure.
- [ ] Add strict JSON helpers: an `object_pairs_hook` raises ValueError for duplicate keys; a `parse_constant` callback rejects NaN/Infinity; top-level decoded value must have exact dict type. Validate exact field set before constructing JobSpec.
- [ ] Implement frozen fields and shared validation used by `__post_init__`, `from_json` and `to_json`; serialize explicitly named fields with `json.dumps(..., sort_keys=True, allow_nan=False)`. Do not serialize arbitrary instance state.
- [ ] Parameterize each bad field with wrong built-in types and invalid values. Assert a dataclasses FrozenInstanceError on assignment, exact JSON field set, roundtrip equality, and independence from mutable caller state.

Representative contract test (imports are intentionally local for red evidence):

```python
def test_json_roundtrip_is_frozen_and_has_only_job_fields():
    import json
    from dataclasses import FrozenInstanceError
    import pytest
    from pynbodyext.orm import JobSpec
    from pynbodyext.properties import ParamSum

    job = JobSpec.from_calculator(ParamSum("mass"), 1, store_name="results")
    encoded = json.loads(job.to_json())
    assert set(encoded) == {"schema", "calculator_signature_text", "input_id",
                            "location_id", "store_name", "reuse"}
    assert encoded["schema"] == "pynbodyext.orm.job/v1"
    assert encoded["location_id"] is None
    assert JobSpec.from_json(job.to_json()) == job
    with pytest.raises(FrozenInstanceError):
        job.reuse = False
```

## Task 2: Signature validation without expanding trusted types

**Files:** jobs module and jobs tests only. Keep generic signature APIs unchanged.

- [ ] Add a recursive walk of parsed signature JSON before any reconstruction. Inspect **every** nested list/dict, including pipeline outputs, scopes, transform state, dataclass init values, generic identity values, encoded dictionaries and `type=calculator` wrappers.
- [ ] For class-bearing calculator nodes (`dataclass`, `generic`, `pipeline`, `transform_chain`), require exact nonempty string `class` and look it up directly in `interfaces._SIGNATURE_TYPES`; missing membership raises ValueError naming the unavailable registered type. Do not call `resolve_signature_type` for membership. Intrinsic expression/filter/scoped/combined node kinds have fixed decoders, not arbitrary class lookup.
- [ ] For each `type=enum`, split the path exactly as `import_object` does; require its module already in `sys.modules` and its named object in `vars(module)` with `isinstance(candidate, type)` and `issubclass(candidate, Enum)`. Use dictionary access, not module `getattr`/`__getattr__`, and never import the named module. Reject unloaded/unavailable enum classes before reconstruction. Do not globally monkeypatch a resolver or a decoder.
- [ ] Validate all optional signature envelope flags with exact bool type; reject `stable=False` or `constructible=False`, including nested signature wrappers. Parse using CalculatorSignature, require its stable/constructible properties, and perform the guarded `calculator_from_signature` call. Decoder errors for unsupported/lambda/hash-only-array payloads remain preflight failures. Re-encode the rebuilt graph and require stable/constructible again, so forged optimistic flags cannot make a session-only graph eligible.
- [ ] Check default RunOptions on the live graph **before serialization** in `from_calculator`, and on the rebuilt graph before accepting/executing a job. Use an identity-visited walk over `node.dependencies()` (includes dynamic parameters and scope) and dataclasses fields of `RunOptions()`; require exact option-value types and equality to defaults. Reject `.options(cache=False)`, callbacks/progress sinks, and nested non-default options before recipe conversion. No option fields enter JSON.
- [ ] Require the rebuilt root to be CalculatorBase. Revalidate during `execute` because application registration/code may have changed since reading JSON. Do not cache a live calculator inside frozen JobSpec.
- [ ] Add a temporary importable module whose top-level code writes a marker. An unknown calculator class from it must fail before the marker is created, both at root and deeply nested. Repeat through `type=enum`; instrument imports of the target only, not all normal decoder imports.
- [ ] Test already-registered aliases and an explicitly registered custom rebuildable class; test an available class-path-only custom class is rejected. Test stable-but-nonconstructible and unstable recipes separately, plus forged flags on lambda/hash-only payloads.
- [ ] Preserve existing general decoder version-warning/alias semantics; this item does not silently introduce a new global version policy. Check current/recorded signatures in ordinary replay tests. If exact-hash rejection across aliases/version changes becomes necessary, record that separate contract decision before implementing it.

The private registry read is an intentional **narrowing** at the ORM boundary. Exposing a new core registry API or changing import fallback would exceed this item's file scope. The complete recursive walk must finish before the first constructor runs. Constructors/from_signature_payload of already-registered application types remain trusted application code; JobSpec is not a sandbox for hostile trusted plugins.

## Task 3: Root-only execution delegates to run_input

**Files:** jobs module and jobs tests; re-read FEAT-0010 repository/loading code after integration.

- [ ] Add tests with spies for both loader and calculator execution plus `repo.runs()` assertions. Unknown schema/type fails while loading the manifest; missing store/loader, unknown input, foreign/missing location, ambiguous locations and selected nonlocal URI fail during execute preflight. None of those failures may execute a calculation, invoke the loader, or create a Run.
- [ ] Before delegating, inspect `repo.get_input(input_id)` and require the registered root snapshot: `kind == "snapshot"` and `parent_id is None`; reject catalogue objects and every selection/view even if FEAT-0010 can now reload them. Delegate hierarchy/location/loader validation to current `run_input`; do not implement a parallel loading stack.
- [ ] Execute exactly `repo.run_input(calculator, input_id, store_name=store_name, location_id=location_id, reuse=reuse)` with default options. Return RecordedRun unchanged; preserve RunExecutionError/RunPersistenceError and receipts rather than wrapping/retrying.
- [ ] Assert selected location/reuse forwarding and inspect the actual recorded input/location/recipe/result link. Add one loader failure and one scientific missing-field failure to confirm existing failed-Run semantics survive delegation. Existing loading tests own exhaustive metadata recovery checks.
- [ ] Register a valid FEAT-0010 selection and assert JobSpec rejects it before loader/Run, while direct `run_input` remains able to process it. This catches the integration trap that the current root restriction will no longer be inherited.

## Task 4: New-process scientific replay and pinned cache history

**Files:** jobs tests only. Use real SQLite ResultStore/AnalysisRepository and real pynbody arrays, not mocked Result or calculator reconstruction.

- [ ] In a parent pytest process create metadata and payload SQLite files under tmp_path, register a logical root and local prefix, explicitly import ParamSum, create its JobSpec, and write only `job.to_json()` to a job file. Build a two-particle root with masses `[2.0, 4.0]`, positions, and valid `_filename`; first value is 6.0. Record first signature/hash, run ID and pinned RecordRef.
- [ ] Launch `.venv/bin/python` through `sys.executable` and `subprocess.run(..., check=True, capture_output=True, text=True, cwd=repository_root)`. Pass local file paths as ordinary subprocess arguments; do not use a shell. Child code explicitly imports ParamSum to register it, reopens both databases, binds trusted loader and named store, and reads JobSpec JSON.
- [ ] First child execution uses a loader with masses `[10.0, 20.0]` and reuse=True. Assert returned scientific value **6.0**, same full calculator signature/hash and same pinned ref, new distinct Run ID, `reused is True`; loading must still occur on a cache hit.
- [ ] Child creates `dataclasses.replace(job, reuse=False)` and executes again. Assert value **30.0**, same recipe signature/cache key, distinct version ID/ref and distinct Run ID, `reused is False`. Fetch first Run through metadata and confirm its pinned historical value is still 6.0; fetch refreshed Run and confirm 30.0. `repo.audit()` returns no issues. Emit only small JSON summaries to stdout; parent parses and asserts actual numbers/signatures/refs/run IDs.
- [ ] Dispose repository/store engines in finally blocks. Assert no shared Python objects are used across parent/child. A reopened object in the same process alone does not satisfy this criterion.
- [ ] Add manifest shape assertions showing that backend URLs and loader/callable names used by the test do not enter serialized fields. Reject extra keys such as credentials/options/loader/store_url/pickle rather than serializing them. Ordinary result value decoding stays inside the pre-existing trusted ResultStore API.

## Task 5: Verification and integration handoff

- [ ] Focused: `.venv/bin/python -m pytest tests/test_orm_jobs.py tests/test_orm_loading.py tests/test_signature_roundtrip.py tests/test_signature_type_identity.py -q`.
- [ ] Boundary: `.venv/bin/python -m pytest tests/test_layering_contract.py tests/test_package_layers_contract.py tests/test_calculate_contract.py -q`. ORM must not import higher built-in plugin layers, and pure calculator imports must remain ORM-free.
- [ ] Run `make check`, `make test-all`, and `make docs-check`; retain exact outputs and expected red-to-green evidence. These are future validation steps, not claims of execution during this plan.
- [ ] During implementation only, update CONTEXT with schema/API, same-repository IDs, required explicit type/loader/store bindings, strict reconstruction preflight, default-options limitation, root-only rule and inherited lifecycle/errors. Sync both FEAT-0011 locales, set an honest status, append handoff with counts and local-prerequisite caveat.
- [ ] Review the final diff against the approved file scope, retain all pre-existing dirty files, and prepare only the scoped increment for parent review. No commit/push/PR or migration is authorized by this preparation plan itself; follow the parent's active authorization at execution time.

## Self-review checklist

- [ ] Every acceptance criterion maps to Tasks 1–4, including separate process/value/hash checks, new Run on hit, pinned refresh history and no calculation on preflight failures.
- [ ] No global import hook, monkeypatch, automatic plugin import, second execution engine, selection schema or implicit binding has been introduced.
- [ ] Recursive preflight covers enum as well as calculator class paths; default options are checked before signature conversion can discard them.
- [ ] FEAT-0010 root-loading expansion cannot expand JobSpec v1's input boundary.
- [ ] No preparation action changes status or advertises tests that have not run.
