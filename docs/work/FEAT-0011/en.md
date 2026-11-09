# FEAT-0011: Replay root-input jobs from a versioned JSON manifest

## Context and goal

Provide a small portable job description within one explicit metadata repository using existing signatures and run_input.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: FEAT-0006

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-017](../../adr/0017-root-input-job-manifests.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §7 using frozen JobSpec with schema, calculator_signature_text, input_id, location_id, store_name and reuse. Public APIs: JobSpec.from_calculator(...), to_json(), from_json(), execute(repo). Strict built-in field types; positive integer IDs, no bool IDs, nonempty store_name. Only default RunOptions. Refuse unstable or nonconstructible signatures, unknown fields/schema; rebuild only already available registered calculator types. execute calls repo.run_input; bindings supplied at execution, never serialized. Root inputs only; selection jobs are a later separate schema. Do not expose _EvaluationPlan or call this a new execution engine.

## File scope

- `pynbodyext/orm/jobs.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_jobs.py`
- `CONTEXT.md`

## Acceptance criteria

- [x] A new Python process reopens SQLite, binds loader/store and obtains the same scientific value and signature.
- [x] Second execution records a new cache-hit Run with same pinned ref; reuse=False saves a new version.
- [x] Missing loader/store/type and invalid schema fail at the documented preflight stage without executing user calculations.
- [x] Manifest contains no credentials, callable, live snapshot, executable pickle or absolute backend URL.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Update both locales, meta.json and handoff with red/green evidence.
- [ ] Parent integration workflow: prepare one scoped PR using the template when publishing is authorized.

```bash
.venv/bin/python -m pytest tests/test_orm_jobs.py tests/test_orm_loading.py tests/test_signature_roundtrip.py -q
```

## Local implementation and validation

Implemented frozen `JobSpec` with schema `pynbodyext.orm.job/v1` and exactly the six approved fields. `from_calculator(calculator, input_id, *, store_name, location_id=None, reuse=True)` copies signature text; strict `from_json` rejects missing/unknown/duplicate fields, non-finite constants, unsupported schema and invalid built-in types. Default RunOptions are checked throughout the live graph before signature conversion and throughout its reconstruction.

Before any signature constructor, the entire nested payload is checked for explicitly registered calculator names/aliases, matching registered signature versions and already-loaded Enum classes. Project code must explicitly import/register required types before reading/executing a job. Importable class-path-only custom types are deliberately rejected. After reconstruction, recipe identity must match with registered aliases normalized recursively and numeric equivalence preserved; constructor/rebuilder parameter drift is rejected. This strict job recipe/version check leaves general registry/decoder behavior unchanged; no automatic module import, plugin installation or global monkeypatch. Registered constructors/rebuilders remain trusted application code.

`execute(repo)` rebuilds/rechecks the signature, rejects every non-root input (including valid FEAT-0010 selections), then delegates to `run_input` unchanged. Store/loader/location preflight creates no Run and performs no scientific work. Later loading/calculation errors retain tracked failure semantics. No bindings, backend URLs, callables, live snapshots or pickle are serialized; recipe parameter strings are preserved without automatic secret redaction. IDs remain local to the explicit metadata repository.

TDD red evidence: expected missing `orm.JobSpec` API, **1 failed**, `/tmp/feat0011-red.log`. Initial new-module green: **55 passed**, `/tmp/feat0011-jobs.log`; two additional security regressions cover nested signature-envelope flags and a loaded non-Enum callable escape. The subprocess test reopens real SQLite databases, explicitly imports ParamSum and binds loader/store; it checks original value **6**, a new cache-hit Run returning **6** with the same signature/pinned ref despite loaded masses totaling **30**, and `reuse=False` value **30** with a distinct version while old Run fetch remains **6**.

Review identified recipe drift: stale versions were only warned by the general decoder, and reconstructed parameters could differ. Four root/nested version and dataclass/generic constructor regressions went **4 red → green** (`/tmp/feat0011-drift-red.log`, `/tmp/feat0011-drift-green.log`); the updated new jobs module is **62 passed**. Final-source focused/check/all runs are repeated after this correction; earlier successful counts/logs are explicitly retained as `*-pre-drift.log`, not final evidence. Source/tests are frozen for independent re-review. Verified local FEAT-0006/0009/0010 precede this increment; no remote merge or publication is claimed. SQLite/Linux verification does not certify other SQL dialects/platforms.

Pre-alias corrected-source verification: focused **226 passed** in 22.94s (`/tmp/feat0011-focused-pre-alias.log`); `make check` passed lint/mypy/docs and **1572 passed, 2 skipped, 5 deselected** in 62.95s (`/tmp/feat0011-check-pre-alias.log`); full suite **1577 passed, 2 skipped** in 64.29s (`/tmp/feat0011-all-pre-alias.log`). These successful logs predate the alias-only correction below and are not final-source evidence. Both skips require optional pandas.

The parent final probe found a valid explicit-alias-only class (no `signature_name`) was rejected when the rebuilt signature used its canonical class path. Two root/nested alias regressions reproduced this (**2 red**, `/tmp/feat0011-alias-red.log`). The job now carries a local mapping of already-vetted registered types and their canonical names into rebuilt validation and identity normalization. Initial payload names still require explicit registry membership; no canonical name is globally registered and the class-path-only/from_calculator policy is unchanged. JSON roundtrip and real tracked execution return **18** for both root/nested alias-only jobs; unknown returned types and parameter drift remain rejected. New-module green: **65 passed** (`/tmp/feat0011-alias-green.log`). Source/tests remain frozen for renewed independent review. Status remains review; no Git/remote action.

Final alias-corrected verification: focused **229 passed** in 24.15s (`/tmp/feat0011-focused.log`); `make check` lint/mypy (123 source files)/docs passed with **1575 passed, 2 skipped, 5 deselected** in 67.25s (`/tmp/feat0011-check.log`); `make test-all` **1580 passed, 2 skipped** in 67.09s (`/tmp/feat0011-all.log`). Both skips require optional pandas. Final documentation validation passed (`/tmp/feat0011-docs.log`); independent review/publication is the parent workflow.

Parent confirmed renewed alias-corrected specification and quality reviews both passed; targeted reviewer verification was **10 passed, 55 deselected**. The parent independently checked the final focused/check/all evidence above and is continuing the sequential work. Status remains review, with no commit/PR/remote merge claimed. Runtime/tests stayed frozen throughout final verification and re-review.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify local predecessors without claiming remote merge. Commit/push permissions follow the active user authorization, not this template.
