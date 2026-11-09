# FEAT-0005: Pin SampleTable cell provenance to saved result versions

## Context and goal

Preserve exact saved-result provenance when a sample is refreshed or exported.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: BUG-0010, BUG-0011, BUG-0012

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-011](../../adr/0011-sample-cell-references.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §5.1: add refs as the final defaulted SampleRow field; add optional keyword-only store_name to SampleTable. Capture result.store_ref only after success, clear stale refs on failure/non-persisted success, preserve refs during fill/where/save/load. New sample JSON schema includes refs and store_name; restore with RecordRef.from_json, validate pinned ref against row typed identity and column signature, reject wrong-key/unpinned refs. Load never dereferences a backend. Keep legacy schema readable with empty refs, without inventing historical versions. Export dataframe references indexed by typed identity, not duplicate display names.

## Implemented behavior

`SampleRow.refs` is the final defaulted field. `add` and store fills copy the
actual `Result.store_ref`; a failed retry (including `errors="raise"`) or a
successful run without persistence removes the prior cell reference. Failed
store reads also clear that cell's stale reference while preserving its existing
value/error diagnostics. Nonempty cells and cache misses remain untouched by fill.
`from_store` and `where` retain the explicit optional `store_name`.

JSON schema `pynbodyext.sample.table/v2` stores reference JSON strings and the
backend label. Import rejects unpinned, wrong typed identity, wrong signature and
unknown-column references before use. Legacy v1 imports keep values with empty
references. Loading binds only an explicitly supplied store and never queries it.

DataFrame metadata includes `store_name`, ordered `identity_keys`, and
`identities`/`refs` maps keyed by `sim_identity_key`. Identity values use
`ValueCodec`; reference values use `RecordRef.to_json()`. Duplicate display names
remain valid. Successful `None` retains the existing missing-cell ambiguity; this
increment does not introduce a new cell state machine.

## File scope

- `pynbodyext/core/calculate/sample.py`
- `tests/test_sample_table.py`
- `tests/test_sample_refs.py`
- `CONTEXT.md`

## Acceptance criteria

- [x] Value 6 then refresh 12: exported first table ref still fetches 6 after DB reopen and cache invalidation.
- [x] Cache hit preserves exact ref; no-store/session-only failure paths leave no stale reference.
- [x] JSON/where/DataFrame preserve typed identities and backend label, including duplicate row display names.
- [x] Wrong identity/hash/unpinned reference is rejected on import; missing backend is not automatically rebound.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Update both locales, meta.json and handoff; keep the verified local increment in review without committing, pushing or opening a PR in this task.

```bash
.venv/bin/python -m pytest tests/test_sample_refs.py tests/test_sample_table.py tests/test_result_versions.py -q
```

New regression/API tests were added before source edits. See the handoff for exact red/green commands, counts and optional pandas verification.

## First regression scenario

```python
import pynbody
from pynbodyext.core.calculate import SampleTable, InMemoryResultStore
from pynbodyext.properties import ParamSum


def test_sample_keeps_original_receipt_after_refresh(tmp_path):
    sim = pynbody.new(2)
    sim["mass"] = [6.0, 0.0]
    identity = lambda sim: ("sim", "sample-reference-fixture")
    calc = ParamSum("mass")
    store = InMemoryResultStore()
    first = SampleTable({"mass": calc}, store=store, store_name="results", sim_identity=identity)
    row = first.add(sim)
    pinned = row.refs["mass"]
    sim["mass"] = [12.0, 0.0]
    calc.run(sim, store=store, sim_identity=identity, reuse=False)
    path = tmp_path / "sample.json"
    first.save(path)
    loaded = SampleTable.load(path, store=store)
    assert loaded.rows[0].refs["mass"] == pinned
    assert float(store.fetch(pinned).value) == 6
```

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; use the approved, verified local predecessors sequentially without claiming they are merged. Commit/push permissions follow the active user authorization, not this template.

## Integration outcome — 2026-11-09

Merged via [FEAT-0005 PR](https://github.com/wx-ys/pynbody-extras/pull/45), merge commit `1fbde1c4aeca11844cb7ede0339731756a33832f`. Current status: done; this supersedes the historical local-only/review notes above. Exact scoped integration checks are recorded in handoff.md and the [integration record](../../design/integration-2026-11-09.md). No release was published.
