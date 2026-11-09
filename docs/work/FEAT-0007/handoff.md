# FEAT-0007 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-013 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.

## 2026-11-09 — Codex / local implementation and verification

- Done: implemented ADR-013 in the existing store seam. Frozen ExternalFileRef
  and ExternalFileInspection preserve caller descriptors, expose strict versioned
  JSON and explicit status/reason/checksum_verified observations, and reject
  invalid local paths/URI, field values, schemas and fields. ValueCodec has an
  explicit external_file_ref_v1 tag; construction/JSON/codec do no file IO.
- Ownership: callers own external files. Default inspection only stats; checksum
  reads require explicit verification plus a supplied digest. Access errors
  propagate. Old pinned results preserve descriptors but cannot restore bytes
  after external overwrite. Bilingual specs and CONTEXT document this limit.
- Tests: added 93 cases covering strict types/JSON/URI, URI parser control
  normalization, filesystem-free codec operations, nested object arrays, named
  Pipeline outputs, absent-file persistence, PermissionError/OSError propagation,
  checksum/size changes, both memory and reopened SQLite, and same-key refresh
  preserving old descriptors while leaving overwritten bytes unchanged.
- Red: `/tmp/feat0007-red.log` records 1 assertion failure for missing public API;
  `/tmp/feat0007-inspect-red.log` records 13 missing-method failures;
  `/tmp/feat0007-codec-red.log` records 5 failures/12 passes exposing opaque fallback
  instead of the required explicit codec tag; `/tmp/feat0007-uri-red.log` records
  4 failures/21 passes for URI spellings urlsplit silently normalizes. The initial
  named-output test setup was corrected to use existing `.keep("files")` before
  codec implementation; no runtime behavior was changed to accommodate that setup.
- Green: descriptor stage 59 passed; inspection stage 13 passed; codec integration
  stage 185 passed. Final focused command below: 260 passed in 3.94s.
- Full: `make check` passed lint, mypy (119 source files), 1316 passed/2 skipped/
  4 deselected in 43.55s, and docs check (27 work items). `make test-all` passed
  1320 tests/2 skipped in 44.33s. Logs: `/tmp/feat0007-focused.log`,
  `/tmp/feat0007-check.log`, `/tmp/feat0007-test-all.log`.
- Files: store/external.py (new), store/codec.py, store/__init__.py,
  tests/test_external_file_refs.py (new), CONTEXT.md, this item's en.md/zh-CN.md/
  meta.json/handoff.md and generated docs/work/INDEX.md. Preparation plan remains
  historical; no other feature source, tests or plans changed in this increment.
- Next: independent review and parent integration. Status review; no Git commands,
  commit, push, PR or release publication performed. Predecessors were verified
  locally, not claimed merged. No database/schema migration or dependency added.
- Limits: Linux/SQLite verified only; no claim about other SQL dialects/platforms.
  Inspection does not lock concurrent external writes or prove default readability.

```bash
.venv/bin/python -m pytest tests/test_external_file_refs.py tests/test_value_codec.py tests/test_result_store.py tests/test_sql_store.py tests/test_result_versions.py tests/test_layering_contract.py tests/test_package_layers_contract.py -q
make check
make test-all
```

## 2026-11-09 — scoped integration verification

- User authorized commit, PR and merge. This increment was reconstructed in isolation after verified predecessors.
- `make check`: 1318 passed, 2 skipped, 4 deselected in 34.35s; `make test-all`: 1322 passed, 2 skipped in 34.97s. Log: `/tmp/pynbody-publish-test-6.log`.
- Independent complete-source review found no actionable findings; slice tests protect extraction boundaries. Publication/merge URL and final status are recorded after the remote operation is confirmed.
