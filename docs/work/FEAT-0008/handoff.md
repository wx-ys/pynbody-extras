# FEAT-0008 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-014 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.

## 2026-11-09 — local implementation and validation

- Done: added `PostprocessorBase` as a property role over the existing DAG. The
  inherited required keyword-only Param source is the only dynamic field;
  process(value) receives its public value and units unchanged. Static dataclass
  fields, nested source recipes, signature_version and separate stable/
  constructible flags use the existing serializer; no new node kind or registry.
- Validation guards: Calculator-only source (explicit Result/RecordRef refusal),
  no additional dynamic fields/specs or unsigned/static/converting source,
  calculate-override rejection before signature registration. Exported from node
  package, core facade and actual short `pynbodyext.calculate` facade.
- Exception ownership decision recorded in ADR-014 before implementation: only
  invoke this processor's own applied scope cleanup callback, respect scope/
  transform NEVER and nonrevertible handles, preserve/log original exception if
  cleanup fails, and leave generic engine/PropertyBase success behavior unchanged.
  Native compound handles can cascade through earlier native transforms; a
  retained native NEVER step can block an older handle's revert. Adapter callback
  ownership does not promise stronger effect isolation than the existing handle.
  Pre-adapter transform/filter failures are outside this local exception path.
  Existing last-recorded-phase behavior may report revert after cleanup while
  preserving the original error type/message/traceback.
- Red: `/tmp/feat0008-red.log` records expected missing public API ImportError.
  `/tmp/feat0008-cleanup-red.log`: 6 failed, 30 passed for missing cleanup/callback
  logs. Initial version-test setup was corrected to sign a fresh instance after
  a version change, consistent with existing recipe-signature memoization. New
  exploratory callback-order/effect-isolation expectations were corrected after
  inspecting pynbody native cascading revert; no existing tests were modified.
- Green: focused acceptance/signature/Param/runtime/error/store/version/layering
  suite below passed 164 tests in 2.77s, including 39 new postprocessor cases.
  Single/all-ALWAYS chain cleanup, no-owned-scope ancestor callback ownership,
  explicit NEVER/nonrevertible cases, native semantics, cleanup failure and
  collect_partial failure followed by correct scoped sibling science are covered.
- Full: `make check` passed lint, mypy (120 source files), 1355 passed/2 skipped/
  4 deselected in 40.00s, docs check (27 items). `make test-all`: 1359 passed/
  2 skipped in 41.00s. Skips require optional pandas. Logs:
  `/tmp/feat0008-focused.log`, `/tmp/feat0008-check.log`,
  `/tmp/feat0008-test-all.log`. `make docs-index` and `make docs-check` passed.
- Files: nodes/postprocessors.py (new), nodes/__init__.py, core/calculate/__init__.py,
  pynbodyext/calculate.py, tests/test_postprocessors.py (new), CONTEXT.md, ADR-014,
  this item's plan/en.md/zh-CN.md/meta.json/handoff.md, generated work INDEX.md.
- Next: parent independent review and integration. Status review; no Git,
  commit/push/PR/release publication or unrelated feature source changes. Local
  prerequisites verified (including sequential FEAT-0007), not claimed merged.
- Limits: Linux/SQLite verified, no claim about other dialects/platforms, generic
  success cleanup, filter/apply failures before adapter execution, or isolation
  beyond transform callback semantics. No dependencies or migrations added.

```bash
.venv/bin/python -m pytest tests/test_postprocessors.py tests/test_calculate_contract.py tests/test_signature_roundtrip.py tests/test_layering_contract.py tests/test_signature_stability.py tests/test_signature_type_identity.py tests/test_calculate_params.py tests/test_calculate_runtime.py tests/test_calculate_errors.py tests/test_store_reuse.py tests/test_result_versions.py -q
make check
make test-all
make docs-index
make docs-check
```
