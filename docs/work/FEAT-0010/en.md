# FEAT-0010: Reload explicit root-relative particle selections

## Context and goal

Reload registered derived views from a persisted selection definition, never from an index digest alone.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: FEAT-0006

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-016](../../adr/0016-replayable-particle-selections.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §6.3. Add SelectionSpec schema for indices/family/filter, tied to a registered root and expected particle count. register_selection requires a LoadedInput to materialize and validate the selection once; persist the resulting provider identity plus definition. Indices are a JSON list of built-in integers, ordered and duplicate-preserving subject to root family-block order; reject bool, negatives and out-of-range values. Filter signature must be stable and constructible. Register only an untransformed root. Add load_selection(selection_id, location_id=None) returning LoadedInput for the view with root provider; run_selection delegates the existing run flow and records failures. Location ownership checks refer to the root; this is a deliberate extension of root-only Run location rules requiring an ADR and regression tests. Existing arbitrary view load_input rejection remains.

Persist Selection rows with root_input_id, view_input_id, schema, definition_json, expected_count and a unique canonical descriptor key. Different definitions may share the same materialized view DataObject; do not overwrite either definition. Add nullable Run.selection_id to retain which definition was used. Implemented API: SelectionSpec.indices(list[int]), SelectionSpec.family(str), SelectionSpec.filter(calculator); repo.register_selection(loaded_root, spec) -> Selection; repo.get_selection(selection_id); repo.load_selection(selection_id, *, location_id=None); repo.run_selection(calculator, selection_id, *, store_name, location_id=None, reuse=True, options=None, execution_metadata=None). Registration derives expected_count from loaded_root rather than trusting caller input. run_selection commits running after metadata preflight, before loading; the selected view and definition are both validated during finalization and recovery.

## Approved family-order revision — 2026-11-09

The user explicitly chose proposal A: “首版遵循根的 family 块顺序，跨块重排明确拒绝（推荐）”. This replaces the original unrestricted-order requirement before changing its acceptance test. Preserve exact indices and duplicates while requiring selected family blocks to follow the root's family order. For a dm-then-star root with two particles in each family, `[1,0,1,3,2,3]` is valid and remains unchanged; `[3,1,3,0]` or `[3,1]` raises explicit ValueError without sorting or metadata writes. If replayed root family layout makes a saved sequence unsupported, run_selection records failed after Run creation and before analysis. The upstream contiguous-family view constraint is detailed in [the approved decision](family-order-decision.md). A custom view adapter is outside this increment.

## File scope

- `pynbodyext/orm/selections.py`
- `pynbodyext/orm/models.py`
- `pynbodyext/orm/repository.py`
- `pynbodyext/orm/loading.py`
- `tests/test_orm_selections.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_associations.py` (approved metadata-table expectation)
- `tests/test_orm_loading.py` (existing regression coverage)
- `CONTEXT.md`

The approved Selection model intentionally adds one metadata table after FEAT-0009 (nine to ten). Update the schema-table expectation in test_orm_associations.py while retaining all no-payload assertions; Run.selection_id is an explicit metadata relation.

## Acceptance criteria

- [x] Reopen root/selection metadata and reproduce identical ordered indices and typed provider identity.
- [x] Family and reconstructible filter select expected particles; missing family or wrong-length mask fails explicitly.
- [x] Changed root revision/particle count or changed materialized selection digest rejects before calculation.
- [x] Load/selection failures after Run creation record failed; final metadata failure preserves receipt and explicit recovery validates the root location.
- [x] No guarantee against same-count reordered/edited particles without revision; document and test only the stated guarantee.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Update both locales, meta.json and handoff; retain red/green evidence.
- [x] Parent workflow: prepare one scoped PR using the template.

```bash
.venv/bin/python -m pytest tests/test_orm_selections.py tests/test_orm_loading.py tests/test_logical_snapshot_identity.py tests/test_orm_associations.py -q
```

## Local implementation evidence

SelectionSpec is immutable canonical JSON with strict schema/kind/fields and a full stable/constructible FilterBase signature. Indices reject booleans, NumPy integers, negatives and out-of-range values; family-block order follows the approved amendment. Family absence, malformed/incorrect-length masks and changed materialized digest fail explicitly. Filters use the existing DAG in original coordinates, with scoped transforms rejected and root identity revalidated after materialization. No persistent mask cache is introduced.

Selection rows hold separate root/view relations, derived expected root count, definition and full canonical descriptor hash. Different definitions may share one view without overwriting, and identical definitions register idempotently. View and Selection insertion share one transaction; injected Selection insertion failure rolls both back. get_selection/load_selection return detached metadata with the immediate root/view hierarchy loaded. load_input remains root-only even when a view has a Selection.

run_selection commits the view Run, exact selection_id, selected ROOT location and copied invocation context before loader/selector IO. It delegates _execute_run, preserves cache-hit/refresh and collected-errors behavior, and accepts the same execution_metadata as run_input. Only a matching validated Selection authorizes a root-owned Run location; ordinary parent inference remains forbidden. finish/fetch/audit/unlinked_versions/recovery validate descriptor, root/view hierarchy, exact view receipt and root-location ownership. Receipt failure recovery is explicit, idempotent for the same pinned reference, and never chooses latest. Audit does not load data or decode values.

Initial red: **58 cases failed** for missing SelectionSpec/selection APIs (/tmp/feat0010-red.log). The approved family-order amendment retained explicit rejecting regressions: **3 red** message/behavior checks before the explicit guard (/tmp/feat0010-family-red.log). Strict signature-envelope checks had **3 red** cases (/tmp/feat0010-red-validation.log). A later malformed scope=null regression reproduced AttributeError and went red→green as explicit ValueError (/tmp/feat0010-scope-red.log, /tmp/feat0010-scope-green.log). The module reached **86 passed** before that final added scope case (/tmp/feat0010-green.log); final validation includes **87 new cases**, including real-file replay via pynbody.load, native views, raw SQLite constraints, post-registration data/family-layout changes and injected metadata/payload failures.

Final-source full suite: **1515 passed, 2 skipped** in 60.06s (/tmp/feat0010-all.log); both skips require optional pandas. Final focused suite: **326 passed** in 33.55s (/tmp/feat0010-focused.log). `make check` passed lint, mypy (**122 source files**), **1510 passed, 2 skipped, 5 deselected** in 57.94s and docs validation (27 work items), recorded in /tmp/feat0010-check.log. Both independent specification and quality reviews reported no findings on the final frozen source. Status is review; no Git action, remote merge, PR or release is claimed. SQLite/Linux is verified; other SQL dialects/platforms and a custom arbitrary-family-order view are outside this increment. No migration, dependency, core/provider rewrite or automatic data fingerprint was added. Same-count edits/reordering without a new revision can keep the same key; tests explicitly require reuse=False or a new revision for changed computations. Metadata table count changes from nine to ten while all no-payload assertions remain intact.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify local predecessors before continuing without claiming remote merge. Commit/push permissions follow the active user authorization, not this template.

## Integration outcome — 2026-11-09

Merged via [FEAT-0010 PR](https://github.com/wx-ys/pynbody-extras/pull/50), merge commit `448fe1156f19d32aed42118cd108c1cfe0cb6c74`. Current status: done; this supersedes the historical local-only/review notes above. Exact scoped integration checks are recorded in handoff.md and the [integration record](../../design/integration-2026-11-09.md). No release was published.
