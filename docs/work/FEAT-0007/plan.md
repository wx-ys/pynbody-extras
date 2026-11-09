# FEAT-0007 ExternalFileRef Implementation Plan

> **For agentic workers:** Execute inline, task by task, after the parent confirms FEAT-0006 verification. Use the executing-plans workflow for implementation; this preparation task authorizes this plan file only. Checkboxes below describe future work, not completed work.

**Goal:** Persist an explicit, immutable descriptor of a caller-owned local file while keeping all filesystem inspection explicit.

**Architecture:** Add a stdlib-only value type in the existing store layer and teach ValueCodec its explicit versioned representation. Existing ResultRecord, memory/SQL adapters and pinned RecordRef semantics carry the descriptor without new tables, executors or file management. ADR-013 already accepts this core contract; the details below fill in API choices within that decision.

**Tech Stack:** Python 3.11+, frozen slotted dataclasses, pathlib/os/stat/urllib.parse, json, hashlib, existing ValueCodec and ResultStore, pytest and SQLite through the existing SQLAlchemy extra. No new dependencies.

## Preparation and scope

Read for this plan: AGENTS.md, CONTEXT.md, docs/work/INDEX.md, this item's meta.json, en.md, zh-CN.md and handoff.md; ADR-013; design §5.3; store/codec.py, store/base.py and store/__init__.py; tests/test_value_codec.py, tests/test_result_store.py, tests/test_result_versions.py, tests/test_sql_store.py and the two import-layer contract suites.

The prerequisite object-array and structured-array codec branches are present in the inspected local source. This is not a statement about remote merge status. Parent sequencing controls when implementation starts. Do not edit unrelated local changes.

| File | Future responsibility |
| --- | --- |
| `pynbodyext/core/calculate/store/external.py` (new) | ExternalFileRef, ExternalFileInspection, lexical descriptor validation, strict JSON and explicit inspection |
| `pynbodyext/core/calculate/store/codec.py` | One explicit encode branch and one explicit decode branch; existing recursive handling supplies nested support |
| `pynbodyext/core/calculate/store/__init__.py` | Export ExternalFileRef and ExternalFileInspection |
| `tests/test_external_file_refs.py` (new) | Descriptor/JSON/codec/inspection/store/version contract tests |
| `CONTEXT.md` | Implemented glossary entry and descriptor-versus-file-history limitation |
| `docs/work/FEAT-0007/{en.md,zh-CN.md,meta.json,handoff.md}` | Synchronized implementation evidence and review status after validation |

No ResultStore schema bump or edits to base.py/SQLAlchemy tables are needed. No dependency, DAG, ORM, cache-key or existing test edits are planned.

## Concrete API choices

### Frozen descriptor and JSON

Use `@dataclass(frozen=True, slots=True)` with exactly these constructor fields:

```python
uri: str
media_type: str
size_bytes: int | None = None
sha256: str | None = None
```

Expose `to_json(self) -> str`, `from_json(cls, text: str) -> ExternalFileRef`, and `inspect(self, verify_checksum: bool = False) -> ExternalFileInspection`. The frozen fields are the entire equality/hash identity. Do not add filesystem facts at construction time or mutate fields during inspection. Do not coerce Path, bytes, NumPy integers or string numbers into valid fields.

Choose descriptor schema `pynbodyext.result.external-file-ref/v1`. `to_json()` always emits exactly `schema`, `uri`, `media_type`, `size_bytes`, `sha256`, with compact separators, sorted keys and explicit null optionals. `from_json()` requires the exact same field set and schema; optional constructor fields are nullable in this wire format, rather than omitted. Reject non-object JSON, absent required wire fields, unknown fields and unsupported schemas with ValueError. Reject duplicate object keys with a small object_pairs_hook rather than accepting the last value silently. Invalid field types/values raise ValueError from both construction and JSON loading; unknown constructor keywords keep the normal dataclass TypeError.

Validate `uri` and `media_type` as strings whose stripped form is nonempty, while preserving their original text. No MIME registry lookup or content sniffing: nonempty media_type is the approved requirement. `size_bytes` is None or `type(value) is int and value >= 0`; bool is rejected. `sha256` is None or a string matching `[0-9a-f]{64}` exactly. No normalization to lowercase or stripping of supplied descriptors.

### Local location validation without IO

Use a private `_local_path(uri: str) -> Path` for lexical validation and inspection path conversion. Store the original uri; never resolve symlinks, call absolute()/resolve(), expand `~`/environment variables, inspect existence, or normalize away text in the descriptor.

- A bare absolute platform-local path is accepted as literal path text. Test the bare absolute path case before URL parsing, so `/tmp/a#b?c` remains a valid filename. Reject a NUL character.
- Otherwise accept only a `file:` URI with an absolute decoded local path, empty authority or `localhost` authority (case-insensitive), and no query/fragment. Reject credentials, ports, nonlocal authorities, every non-file scheme and relative/empty URI paths.
- Decode URI percent escapes strictly as UTF-8; reject malformed `%` escapes and decoded NUL. Spaces and Unicode survive round-trip; percent decoding is used solely to locate the file during inspect. Bare path percent characters stay literal.
- This implementation is verified on Linux. Use the running platform's absolute-path rule and do not claim Windows or network-filesystem certification. Symlinks follow normal local stat/open semantics during explicit inspection.

Examples to accept: `/tmp/map.png`, `/tmp/a#b?c`, `file:///tmp/map%20one.png`, `file:/tmp/map.png`, `file://localhost/tmp/map.png`. Reject `map.png`, `~/map.png`, `file:map.png`, `https://host/map.png`, `s3://bucket/map.png`, `file://other-host/tmp/map.png`, `file:///tmp/map.png?x=1`, `file:///tmp/%00map.png`, and malformed escapes.

### Inspection result and checksum semantics

Choose a second frozen slotted dataclass `ExternalFileInspection` with three fields:

```python
status: Literal["available", "missing", "changed"]
reason: str
checksum_verified: bool = False
```

This is a returned value, not a persistable file snapshot; no additional JSON or codec branch is required for it. Reasons below are stable machine-readable strings documented alongside status. `available` reports the observation made by this call, not a promise of immutability or content equality.

| Observation (in evaluation order) | status | reason | checksum_verified |
| --- | --- | --- | --- |
| stat/open raises FileNotFoundError, including disappearance between those operations | missing | `not_found` | false |
| stat target is not a regular file | changed | `not_regular_file` | false |
| supplied size differs from stat size | changed | `size_mismatch` | false |
| verify_checksum is false | available | `checksum_not_checked` | false |
| verification requested but descriptor has no sha256 | available | `checksum_not_supplied` | false |
| computed SHA256 differs | changed | `checksum_mismatch` | false |
| computed SHA256 matches | available | `checksum_match` | true |

Only `verify_checksum=True` with a supplied sha256 opens the file for reading. Read binary blocks (1 MiB is sufficient), update hashlib.sha256 and compare hexdigest; avoid loading an entire scientific file. Checking size first can report a change without hashing. Validate the flag as a bool to avoid accidental truthy strings.

Catch FileNotFoundError only for missing. PermissionError and other OSError subclasses propagate, including from stat or checksum reads; do not relabel them missing/changed. A successful default stat can report available even if the file is not readable; this is a metadata observation and the reason makes checksum nonverification explicit. Do not use exists()/os.access(), which can hide errors or give an unreliable authorization precheck.

Do not claim atomicity under concurrent external writes. This increment supplies no locks, copies or transactional file snapshots. A directory becomes changed rather than being opened; a broken symlink becomes missing. Document these semantics in method docstrings.

### Codec representation

Choose an explicit versioned tag with a structured descriptor payload:

```json
{"__kind":"external_file_ref_v1","descriptor":{"schema":"pynbodyext.result.external-file-ref/v1","uri":"/tmp/map.png","media_type":"image/png","size_bytes":null,"sha256":null}}
```

Keep `_to_payload()` / `_from_payload()` private helpers on ExternalFileRef (or equivalent private module functions) as the shared strict validation seam for JSON and codec. The public API remains small. Encode before opaque fallback so descriptors never become pickle. Decode must require the exact envelope fields `__kind` and `descriptor`, validate the descriptor mapping, then reconstruct the frozen type. Unknown versions continue to reach the existing unknown-kind error.

No encoding/decoding branch may call inspect/stat/open. This includes `ValueCodec.kind()`, `size_bytes()` and `content_hash()`, which internally encode values. A content hash describes the descriptor bytes, not the external file. Existing scalar/array/object/structured tags and ordinary path strings stay unchanged. Keep external.py independent of codec.py/base.py to avoid circular imports; codec.py may import its sibling external module.

## Task 1: Descriptor and strict serialization

- [ ] Add the first public API regression in tests/test_external_file_refs.py using a test-local import from the existing store package; assert the exported type exists. Run `.venv/bin/python -m pytest tests/test_external_file_refs.py -q` before implementation and retain the failure that names the missing ExternalFileRef API. A broken dependency import is not valid red evidence.
- [ ] Add parameterized descriptor/JSON tests covering frozen assignment failure, exact equality/hash, all nullable combinations, absent-file construction, local path/file-URI examples above, media_type types/blank values, size values `True`, `False`, `-1`, `1.0`, `"1"`, valid zero, and malformed/uppercase/short/long SHA256.
- [ ] Test schema/field rejection through both from_json and the future codec decoder: missing schema, unknown schema, extra field, absent wire field, nonmapping payload, malformed JSON and duplicate keys. Preserve a caller's exact uri/media_type strings after valid JSON round-trip.
- [ ] Implement the frozen value, shared lexical validators and strict payload/JSON helpers in external.py, then export both new types from store/__init__.py. Reuse stdlib only.
- [ ] Run the descriptor-focused selection and record its green result: `.venv/bin/python -m pytest tests/test_external_file_refs.py -q -k 'descriptor or json or local_path'`.

Representative descriptor behavior to turn into a test:

```python
def test_descriptor_json_without_existing_file(tmp_path):
    from dataclasses import FrozenInstanceError
    from pynbodyext.core.calculate.store import ExternalFileRef

    ref = ExternalFileRef(str(tmp_path / "absent.bin"), "application/octet-stream")
    assert ExternalFileRef.from_json(ref.to_json()) == ref
    assert hash(ExternalFileRef.from_json(ref.to_json())) == hash(ref)
    with pytest.raises(FrozenInstanceError):
        ref.uri = "/tmp/replaced.bin"
```

## Task 2: Explicit inspection

- [ ] Before adding inspect, write behavioral tests for the complete status/reason matrix and run `.venv/bin/python -m pytest tests/test_external_file_refs.py -q -k inspect`; retain the missing-method/behavior red evidence.
- [ ] Use real tmp_path files to test missing, regular file, directory, broken symlink, matching/mismatching size, matching SHA256 and equal-size overwrite with different SHA256. Assert default inspection never claims checksum verification, including when the descriptor contains a correct digest. Assert verify=True without a digest does not read the file.
- [ ] Inject PermissionError into Path.stat and Path.open in separate tests; require propagation. Patch within a context after fixture setup and do not rely on chmod, because tests may run as root. For default inspection patch open to fail the test if called; for checksum inspection observe that binary read is used.
- [ ] Implement stat-first inspection with the matrix above. Use stat.S_ISREG; stream the hash only on explicit verification with a supplied digest. Catch only FileNotFoundError. Test disappearance after stat by making the checksum open raise FileNotFoundError.
- [ ] Run the inspect selection again and record green. Document that successful stat is metadata availability, matching digest is an observation, and the type performs no file mutation.

Representative equal-size overwrite test:

```python
def test_inspect_equal_size_overwrite(tmp_path):
    import hashlib
    from pynbodyext.core.calculate.store import ExternalFileRef

    path = tmp_path / "map.bin"
    path.write_bytes(b"first")
    ref = ExternalFileRef(str(path), "application/octet-stream", 5, hashlib.sha256(b"first").hexdigest())
    assert ref.inspect(verify_checksum=True).checksum_verified
    path.write_bytes(b"other")
    assert ref.inspect().reason == "checksum_not_checked"
    observed = ref.inspect(verify_checksum=True)
    assert (observed.status, observed.reason, observed.checksum_verified) == ("changed", "checksum_mismatch", False)
```

## Task 3: IO-free explicit codec and nested values

- [ ] Write a test requiring the explicit external_file_ref_v1 tag, then a nested JSON round-trip test containing dict/list/tuple descriptors and a shaped object array of descriptors. The current opaque fallback can already round-trip a dataclass, so equality alone does not provide valid red evidence; assert the exact explicit wire envelope and that pickle is never used.
- [ ] Assert a plain absolute filename string still encodes as `str`; normal numeric and existing object/structured-array regressions remain intact. Test malformed descriptor/schema/envelope data and an unsupported `external_file_ref_v2` tag.
- [ ] Under a scoped monkeypatch, make Path.stat, Path.open and builtins.open fail if touched, and make inspect fail if called. Build an absent-file descriptor, run to_json/from_json, ValueCodec encode/decode/kind/content_hash/size_bytes, then assert exact equality. Patch after importing modules to avoid confusing module-import IO with descriptor operations. Patch pickle.dumps/loads in the explicit codec test to guard against fallback.
- [ ] Run `.venv/bin/python -m pytest tests/test_external_file_refs.py -q -k codec` before the branches and save the expected explicit-tag failure.
- [ ] Add the explicit branches with exact envelope validation and the shared descriptor payload validator. Update codec documentation to name the new supported type and descriptor hash semantics.
- [ ] Run `.venv/bin/python -m pytest tests/test_external_file_refs.py tests/test_value_codec.py tests/test_result_store.py -q` and record green.

## Task 4: Stored nested descriptors and external overwrite history

- [ ] Add tests parameterized over memory and SQLite using the existing tests/test_result_versions.py fixture pattern (including engine.dispose in finally). Use a module-level PropertyBase dataclass with a stable signature that returns a nested dict/list/tuple descriptor and a named Pipeline output; let normal run/store handling encode both value and named entries.
- [ ] Keep the external file absent during at least one memory/SQL save and fetch; require success and equal descriptors. Inspect afterwards and require missing. Persistence must not validate external existence.
- [ ] Add a SQLite-only reopen regression: save a result, serialize its pinned RecordRef, dispose the engine, construct a new store over the same file and fetch through RecordRef.from_json. Assert exact descriptor type/fields/equality and the saved named output, not merely a matching URI.
- [ ] For the history regression, define a stable calculator whose fixed path parameter is in its signature and whose calculate method explicitly observes size/hash to build each new descriptor. Run on a caller-created file containing `first`, overwrite that same path with equal-size `other`, then refresh using `reuse=False` and the same simulation/calculator key. This makes generation of fresh facts the calculator's explicit responsibility, not a side effect of constructing ExternalFileRef.
- [ ] Require two different pinned refs; fetch the first and require its original size/hash/URI unchanged; fetch the second and require the new descriptor. Inspect the old descriptor with verification and require changed/checksum_mismatch. Assert actual file bytes remain `other`, fetch creates no restored file and the directory has no extra payload copies. Run this on memory and on SQLite after reopen; pair the saved receipt with its owning backend.
- [ ] Run the new store/history selections first, preserving expected missing tag/API behavior, then implement only fixes in external.py/codec.py if necessary. Store adapters and version tables should need no changes.
- [ ] Run the complete focused command: `.venv/bin/python -m pytest tests/test_external_file_refs.py tests/test_value_codec.py tests/test_result_store.py tests/test_sql_store.py tests/test_result_versions.py tests/test_layering_contract.py tests/test_package_layers_contract.py -q`.

## Task 5: Documentation and final verification

- [ ] Extend CONTEXT.md with ExternalFileRef, its public import path, supported fields, strict schema and inspection semantics. State explicitly: ResultStore versions pin descriptors only; fetching an old version cannot restore a caller-overwritten file. Reproducible historical contents require caller-maintained distinct/versioned paths and optional checksums.
- [ ] Add one concise usage example to both work-item locales showing absent-file construction/codec persistence, explicit checksum inspection, and the overwrite limitation. Keep wording synchronized. A checksum is caller-supplied evidence unless explicit inspection verifies it; no checksum means no content-equality claim.
- [ ] Run `make check` and `make test-all` because this touches public persistence behavior; record exact counts, skips and limitations. Check import-layer tests without exceptions or new upward imports. No new global packages or testdata changes.
- [ ] Update bilingual acceptance checkboxes, meta.json to review (not merged/done), and append a handoff entry with red/green commands, exact outcomes, local prerequisite basis and Linux/SQLite verification scope. Regenerate the index with `make docs-index`, then `make docs-check`.
- [ ] Supply the parent a single-item review summary. Git/PR operations follow the active user authorization and parent coordination; this plan is not authorization to commit or publish.

## Acceptance coverage and execution boundary

| Approved requirement | Planned coverage |
| --- | --- |
| Frozen strict descriptor, local absolute URI/path, nonempty media_type, validated nullable size/SHA256 | Task 1 constructor/path/JSON matrix |
| Unknown fields/schema, bool sizes, remote and relative locations rejected | Tasks 1 and 3 strict payload/envelope tests |
| Construction/decode work without a file; codecs perform no IO | Tasks 1 and 3 monkeypatched filesystem boundary; Task 4 absent-file persistence |
| Explicit available/missing/changed plus reason; access errors propagate | Task 2 full matrix and PermissionError injection |
| Changed size/checksum and no unchecked equality claim | Task 2 equal-size overwrite, size mismatch and checksum flags |
| Nested JSON, memory, reopened SQL exact descriptor equality | Tasks 3 and 4, including object-array and named Pipeline output |
| Refresh pins old descriptor but cannot restore overwritten external contents | Task 4 same-key refresh on both backends, SQLite reopen; Task 5 bilingual docs |
| Caller ownership; no copying/deletion/automatic externalization | Tasks 3/4 IO boundary and directory assertions; Task 5 docs |

Preparation completion means only that this plan exists. No source/test/CONTEXT/status changes or verification runs are claimed by this file. Do not start implementation until the parent sends its follow-up after FEAT-0006 verification.
