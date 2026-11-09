# FEAT-0010 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-016 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.

## 2026-11-09 — approved family-order requirement revision

- Real four-particle dm/star testing reproduced pynbody 2.8 IndexingViewMixin's exact failure: `ValueError: Families must retain the same ordering in the SubSnap` for `[3,1,3,0]` and `[3,1]`. Same-family arbitrary order/duplicates and root-family-block order such as `[1,0,1,3,2,3]` are supported. `allow_family_sort=True` changes input order and is excluded.
- Parent obtained explicit user approval for proposal A: “首版遵循根的 family 块顺序，跨块重排明确拒绝（推荐）”. This replaces the original universal-order requirement, not an excuse to suppress a failing test. English/Chinese contracts, ADR-016, family-order-decision.md and plan were amended before corresponding source/test edits.
- Next: convert dedicated cross-family acceptance to explicit ValueError/no partial writes; add unsupported replay family-layout failure after Run creation; finish full focused/check/all and reviews. Other lifecycle/metadata tests continue unchanged. No custom view adapter, sorting, family relabeling or source fixture substitution.

## 2026-11-09 — local implementation, verified source freeze and reviews

- Done: immutable SelectionSpec JSON (indices/family/stable constructible filter), strict field/schema/signature validation, separate Selection descriptors and view identities, Selection model and nullable Run.selection_id, atomic registration, detached get/load, and run_selection through the existing _execute_run lifecycle. No second DAG/store executor, dependency, provider format or migration.
- Registration validates a registered LoadedInput untransformed root/provider/location, derives ROOT expected_count, evaluates once outside SQL, revalidates root afterward, and atomically inserts/reuses view plus definition. Different definitions can share a view/cache identity without overwrite. Scoped transforms are rejected before execution; a custom filter leaving an external transform is rejected after materialization without metadata writes.
- User-approved family amendment: exact index order/duplicates remain within root family-block order. `[1,0,1,3,2,3]` on dm=2/star=2 replays unchanged; `[3,1,3,0]` and `[3,1]` raise explicit root-family-block ValueError, never sort. A changed family layout on replay records failed after the committed Run and before requested analysis. See the preceding approval entry and family-order-decision.md.
- Loading verifies root logical identity/revision, root count and materialized view digest. Existing arbitrary/digest-only view load_input rejection remains. Same-count reordered/edited values can retain index identity without a new revision: tests demonstrate cached value reuse, explicit reuse=False refresh and separate revision identity, without claiming automatic detection.
- Runs commit before root loader/selector IO, retain exact definition and root source location, and snapshot root URI/loader/revision with the unchanged invocation-context schema; Run.input and pinned receipts identify the view. Matching validated Selection is the only exception to direct input location ownership. finish/fetch/audit/unlinked_versions/explicit recovery share descriptor, root/view, root-location and pinned view/recipe checks. Audit performs no loading/value decoding. Loader/selector/calculation/store errors and collected errors retain the existing failed/exception semantics; failure-write and final-link failures preserve running metadata and explicit receipts for recovery.
- TDD: initial 58 cases failed for missing SelectionSpec/APIs (/tmp/feat0010-red.log). Strict signature-envelope cases: 3 red (/tmp/feat0010-red-validation.log). Approved family guard cases: 3 red (/tmp/feat0010-family-red.log). Final malformed scope=null regression reproduced AttributeError then passed with explicit ValueError (/tmp/feat0010-scope-red.log, /tmp/feat0010-scope-green.log). Earlier module green: 86 passed (/tmp/feat0010-green.log); final source contains 87 new cases. The approved schema expectation change 9 -> 10 first failed independently (/tmp/feat0010-schema-red.log); all no-payload assertions remain intact.
- Final-source focused: 326 passed in 33.55s (/tmp/feat0010-focused.log), covering selections/loading/context/logical identity/associations/catalogues/package/core layering, including real pynbody.load file replay. Pre-scope focused/check logs are explicitly retained as *-pre-scope.log and are not final verification evidence.
- Final-source make check: lint pass, mypy 122 source files pass, 1510 passed / 2 skipped / 5 deselected in 57.94s; docs27 pass (/tmp/feat0010-check.log). make test-all: 1515 passed / 2 skipped in 60.06s (/tmp/feat0010-all.log). Both skips require optional pandas. Parent reported independent specification and quality reviews both passed with no findings on the renewed final freeze.
- Files: pynbodyext/orm/{selections.py,models.py,repository.py,loading.py,__init__.py}; tests/test_orm_selections.py and test_orm_associations.py (approved table count/addition); owning CONTEXT.md section; this item's bilingual specs/meta/handoff/plan/family-order-decision, ADR-016 amendment and generated docs/work/INDEX.md. Existing loading/context/catalogue/core tests remain intact.
- Env/limits: repository .venv, Linux/SQLite only. Other SQL dialects/platforms, arbitrary cross-family-order view adapters, catalogue loading, content fingerprinting and garbage collection are not certified or implemented. Direct session/SQL metadata edits remain caller responsibility; corruption is rejected by repository validation, never silently repaired.
- Next: parent integration/publication workflow and sequential FEAT-0011. Local prerequisites FEAT-0006/0009 were verified, not claimed remotely merged. Status review; no Git commands, commit/push/PR/release or subagents used. Runtime source frozen for review; subsequent edits in this entry are documentation/status only.

```bash
.venv/bin/python -m pytest tests/test_orm_selections.py tests/test_orm_loading.py tests/test_orm_environment.py tests/test_logical_snapshot_identity.py tests/test_orm_associations.py tests/test_orm_catalogues.py tests/test_package_layers_contract.py tests/test_layering_contract.py -q
make check
make test-all
make docs-index
make docs-check
```

## 2026-11-09 — scoped integration verification

- User authorized commit, PR and merge. This increment was reconstructed in isolation after verified predecessors.
- `make check`: 1512 passed, 2 skipped, 5 deselected in 49.53s; `make test-all`: 1517 passed, 2 skipped in 52.14s. Log: `/tmp/pynbody-publish-test-9.log`.
- Independent complete-source review found no actionable findings; slice tests protect extraction boundaries. Publication/merge URL and final status are recorded after the remote operation is confirmed.
