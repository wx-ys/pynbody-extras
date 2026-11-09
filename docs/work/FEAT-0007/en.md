# FEAT-0007: Encode explicit local external-file result references

## Context and goal

Allow a saved result to describe an external file without pretending a path string pins file content.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: BUG-0010, BUG-0011

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-013](../../adr/0013-local-external-file-references.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §5.3 with a frozen ExternalFileRef and versioned codec tag. Require a local absolute URI/path, nonempty media_type, optional nonnegative integer size_bytes, optional lowercase 64-hex sha256. Reject unknown fields/schema, booleans as sizes, remote URLs and relative paths. Explicit inspect(verify_checksum=False) reports available/missing/changed with reason; access errors propagate. Encoding/decoding performs no IO. All files remain caller-owned; no copy/delete or automatic externalization.

## File scope

- `pynbodyext/core/calculate/store/external.py`
- `pynbodyext/core/calculate/store/codec.py`
- `pynbodyext/core/calculate/store/__init__.py`
- `tests/test_external_file_refs.py`
- `CONTEXT.md`

## Acceptance criteria

- [x] Nested reference round-trips through JSON, memory and reopened SQL store with exact descriptor equality.
- [x] Construction/decode work when file is absent; explicit inspection reports missing.
- [x] Modified size/checksum reports changed; no checksum means no claim of equality.
- [x] Refreshing a result preserves old descriptor but cannot restore an overwritten external file; this limitation appears in docs.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Update both locales, meta.json and handoff; prepare one scoped PR using the template and red/green evidence.

```bash
.venv/bin/python -m pytest tests/test_external_file_refs.py tests/test_value_codec.py tests/test_result_versions.py -q
```

## Implemented API and ownership

Import `ExternalFileRef` and `ExternalFileInspection` from
`pynbodyext.core.calculate.store`. The frozen descriptor preserves caller text;
the explicit codec tag is `external_file_ref_v1`, with strict descriptor schema
`pynbodyext.result.external-file-ref/v1`. JSON requires exactly schema/uri/media_type/
size_bytes/sha256, including null optionals; unknown/duplicate fields are rejected.
Only absolute local paths and absolute file URIs with empty/localhost authority
are accepted. URI query/fragment, malformed escapes and nonlocal authorities fail.

```python
from pynbodyext.core.calculate.store import ExternalFileRef, ValueCodec

ref = ExternalFileRef("/data/results/map-v1.png", "image/png")
restored = ValueCodec().decode(ValueCodec().encode(ref))  # no file access
assert restored == ref
observation = restored.inspect(verify_checksum=True)
# If absent: missing/not_found. Without sha256: available/checksum_not_supplied.
# checksum_verified is true only when an explicitly requested digest matches.
```

Inspection returns status/reason/checksum_verified. Size mismatch is changed;
requested SHA256 mismatch is changed, including an equal-size overwrite. Default
availability is a stat observation, not proof of readability or content equality.
PermissionError/other access errors propagate. Missing files can be constructed,
serialized and stored; only explicit inspection observes them. Symbolic links
follow ordinary stat/open semantics. Concurrent external writes are not locked.

ResultStore versions pin only the descriptor. An old pinned result retains its
original size/hash after refresh, but fetching it cannot restore overwritten
external bytes. Callers retain ownership and maintain distinct/versioned paths
for reproducible historical files. No copies, deletion or automatic externalization.

Red evidence: missing public API (1 assertion failure), missing inspection (13
failures), missing explicit codec tag (5 failures, 12 passing), and URI parser
whitespace normalization (4 failures, 21 passing). Green focused suite: 260 passed;
logs are `/tmp/feat0007-{red,inspect-red,codec-red,uri-red,focused}.log`.

Full validation: `make check` passed lint, mypy (119 source files), 1316 tests
(2 skipped, 4 deselected), and docs validation (27 items). `make test-all` passed
1320 tests (2 skipped). Logs: `/tmp/feat0007-check.log` and
`/tmp/feat0007-test-all.log`. Verified on Linux/SQLite only. Status is review;
local prerequisites were verified, not asserted merged. Both locales, metadata
and handoff are updated. PR publication is deferred to the parent workflow.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify predecessors are merged before continuing. Commit/push permissions follow the active user authorization, not this template.

## Integration outcome — 2026-11-09

Merged via [FEAT-0007 PR](https://github.com/wx-ys/pynbody-extras/pull/47), merge commit `5103136ec2837ec428ea722743c24637be067616`. Current status: done; this supersedes the historical local-only/review notes above. Exact scoped integration checks are recorded in handoff.md and the [integration record](../../design/integration-2026-11-09.md). No release was published.
