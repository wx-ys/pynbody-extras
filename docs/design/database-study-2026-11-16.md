# Database studies and detached result tables

[中文](database-study-2026-11-16.zh-CN.md) · [ADR-021](../adr/0021-database-studies-and-detached-tables.md) · [SPIKE-0002](../work/SPIKE-0002/en.md)

Date: 2026-11-16. **Accepted**. See [FEAT-0018](../work/FEAT-0018/en.md) for implementation and verification.
Design review baseline: `d862742`. See the [Database guide](../database.md) for use of the implemented APIs.

This design covers two workflows: one object calculated/saved/reopened, and a fixed sample calculated/retried/exported. Both use the same result-store address and execution owner. A Study saves a request; a StudyTable saves an observation or execution outcome. Reading either never starts a calculation.

## 1. Review findings and decisions

These findings review the preceding usage proposal against current source and tests; they are not claims that all target behavior already exists.

| ID | Inconsistency or unnecessary complexity | Decision |
| --- | --- | --- |
| R01 | A thin wrapper around ORM `run_study` would still load particles on hits and reject selections. `run_input` explicitly loads before execution. | Share the Database lookup-before-load coordinator with single-object calls; keep EvalEngine and ResultStore as owners. Do not wrap the old root-only runner unchanged. |
| R02 | A Study-level cache key, or an implicit Pipeline containing all columns, would break single/batch reuse when adding or renaming columns. | Persist each cell under the existing exact input × full CalculatorSignature key. Study names and layout are not computation keys. |
| R03 | `SampleTable.compute/add/fill_from_store` mutate values; its missing sentinel also overlaps successful None. Calling it an immutable report would change an existing contract. | Add one detached, read-only **StudyTable** for both high-level `run` and `collect`. Keep legacy SampleTable and ORM StudyReport interfaces unchanged. Reuse codecs and shared presentation helpers where appropriate. |
| R04 | Re-collecting after failed refresh can show old successful values and conceal the failed attempt. | Run tables record this invocation; collect tables record selected current cache versions. Never repair a failed cell with an old value. |
| R05 | `not_run` cannot describe an invocation that started but whose completion could not be confirmed. | Add `incomplete` to the table outcome model; keep any recovery receipt separate from its successful-result reference. |
| R06 | A saved signature is not necessarily constructible; live calculator options and loader functions do not survive reopening. | Save signatures as data. Reconstruct only when computation is needed; allow explicit exact-signature live bindings. Loader functions remain process-local. |
| R07 | A path or `store_name="default"` cannot prove the ownership of exported object/result references. | Add a persistent Database UUID namespace; retain typed exact input keys and existing pinned RecordRefs. Paths are location hints, not ownership. |
| R08 | In the proposed examples, `study.calculations` would need to rebuild classes merely to extend a saved study. | Expose an immutable label → CalculatorSignature mapping; the factory also accepts signatures. No reconstruction during inspection. |
| R09 | A frozen dataclass can still expose mutable arrays/dicts, letting stored table values diverge from pinned provenance. | Detached tables protect their values with read-only storage or defensive copies. No implicit refresh or in-place editing interface. |
| R10 | `errors="record"`, `errors="collect"`, and `on_error` give the same high-level intent different spellings. | New Study accepts only `errors="raise"` or `"collect"`; preserve existing lower-level spellings without a new alias there. |
| R11 | `not_applicable` has no agreed scientific outcome protocol. Path-only registration, automatic load sharing and cache submission from reviewed tables enlarge this change substantially. | Defer those additions. A successful None is valid; a scientific exception is failed. Do not infer applicability. First version uses registered exact inputs and serial isolated execution. |
| R12 | Detached loading through unrestricted ValueCodec may deserialize opaque pickles, contradicting the promise that project code is unnecessary. | New table files accept the portable, non-opaque codec subset; reject opaque/pickle tags before decoding. Legacy formats are unchanged. |

Source evidence: [Database execution](../../pynbodyext/database/database.py), [object collections](../../pynbodyext/database/objects.py), [ORM execution](../../pynbodyext/orm/repository.py), [ORM studies](../../pynbodyext/orm/study.py), [SampleTable](../../pynbodyext/core/calculate/sample.py), [ValueCodec](../../pynbodyext/core/calculate/store/codec.py). Existing tests include [database](../../tests/test_database.py), [catalogues](../../tests/test_database_catalogues.py), [studies](../../tests/test_orm_study.py), and [sample references](../../tests/test_sample_refs.py).

## 2. Public vocabulary and scope

| Object | Owns | Does not own |
| --- | --- | --- |
| Database | Registration, durable Study definitions, input resolution, execution coordination and result access | Scientific calculation or a second Calculator DAG |
| Input handle | A registered exact root/view, possibly with a catalogue origin | Its particle arrays or an automatically discovered history |
| Calculator / Result | Scientific definition / one calculation's value and provenance | Sample membership |
| Study | Ordered fixed objects × ordered labelled signatures | Latest values, mutable query membership, or a background job |
| StudyTable | Detached values, outcomes, actual version references and invocation context | Database connections, executable calculators or a resume operation |

`db.study(...)` constructs a bound request in memory from owned registered handles or detached references; it does not accept raw particle snapshots or register new data implicitly. `study.save(name=...)` persists its definition, and `db.studies[name]` reopens it. `run()` and `collect()` each return a new StudyTable; saving a table never inserts its values into the result cache. No separate Sample, Workspace or job scheduler is introduced.

Target API:

```python
db = pe.Database("analysis.sqlite", read_only=False)
optional = obj.results.get(cc)               # None only for an absent result
study = db.study(objects=objects, calculations={"mass": cc})
definition_id = study.save(name="sample-v1")
study = db.studies["sample-v1"]
plan = study.plan(mode="auto")
table = study.collect(missing="raise")       # or "keep"
table = study.run(mode="auto", errors="raise", cells=None)
study = study.bind_calculators({"mass": live_cc})  # optional exact bindings
table.save("sample-v1.json")
table = pe.StudyTable.load("sample-v1.json")
```

`pe.StudyTable` is a lazy root export, backed by dependency-free calculation-layer data types. A name such as `sample-v1` identifies a human-chosen definition, not a moving latest-version alias. Re-saving identical content under the same name is idempotent; different content under that name fails. Different names may point to the same definition. Study definition IDs are content hashes, not new cache keys.

## 3. Identity and single/batch result sharing

The authoritative computation address remains:

```text
K = (sim_identity_key(exact_input_signature), full_calculator_signature_hash)
saved value = (owning_database_id, pinned_RecordRef(K, version_id))
```

Full signature payload validation remains mandatory on hits. Display names, column labels, Study definition IDs and catalogue lookup paths do not replace K. Logical data revision, loader identity and selection distinctions retain their existing identity semantics. File contents and in-place particle edits are not automatically fingerprinted.

A detached object reference contains Database UUID, exact typed input signature, canonical object locator and frozen display label. The locator preserves a catalogue origin where present, but the computation key remains the exact input identity. SQL primary-key integers and basenames are insufficient portable references. The factory accepts owned handles or these detached references and validates ownership; it does not bind foreign references by matching a name.

First-version rows must have distinct exact input keys. Selecting the same input twice, including two catalogue handles resolving to that same input, is an explicit duplicate-input error; do not silently merge them. Repeated-row statistics and cross-catalogue comparison tables are outside this version. Columns have unique nonempty labels; multiple labels may intentionally use one signature. In serial auto mode the later identical column can reuse the earlier saved value; refresh creates a new version for each requested cell in row-major order. No exactly-once computation guarantee is implied.

The factory freezes copied signature JSON, input references and axis order. Its digest includes a schema identifier, owning Database UUID, frozen object descriptors, ordered column labels and full signature payloads, but not save name, timestamps or live file-location hints. `study.objects` is an immutable tuple of references; `study.calculations` is an immutable mapping to CalculatorSignature values. A factory argument may be a Calculator or a stable CalculatorSignature. No live reference is the saved definition.

A live Calculator binding must still match its frozen signature immediately before a request starts and before its cell executes. Mutation is an error, not an unnoticed change to the definition. A new scientific definition requires a new Study. Runtime diagnostic options are not serialized as scientific identity; scientifically consequential settings, including randomness when relevant, must be represented by the Calculator signature or input identity. Mode/errors are resolved for each invocation and recorded separately.

For a quiescent store, single-object save → batch auto, batch save → single-object auto, and two differently named Studies must return the same pinned reference for the same K. Under concurrent refresh, each returned value must match its own pinned ref; two separate current-cache reads need not select the same version.

## 4. Read paths, planning and execution

### 4.1 Read-only behavior

`results[cc]`, `results.get(cc)`, `fetch(ref)`, Study inspection, `plan` and `collect` perform no writes. Metadata-only catalogue resolution must not invoke a helper that materializes and registers selections. A supported saved membership description may derive the exact key without loading particles. An object that cannot be resolved to an exact input from registered metadata is rejected before Study construction; a future adapter may extend coverage.

`get()` returns None only when that exact result is absent. Ambiguity, invalid signatures, corrupt payloads and decoding failures propagate. A successful Result whose value is None is still a Result.

`plan(mode=...)` reads bounded metadata, not scientific payloads. Per-cell actions are `reuse`, `compute`, or `blocked`; reasons distinguish absent calculator/loader bindings from known metadata errors. File availability, memory cost and scientific success remain unchecked without IO. Plans are advisory: `run` revalidates, and a plan reserves neither cache versions nor execution resources.

`collect` first captures current pinned refs for its requested grid within one local database read snapshot, then reads those exact versions. It creates no Runs and never reconstructs a calculator. `missing="raise"` reports absent coordinates; `"keep"` retains them with status missing. Other errors always propagate; missing is not a policy for accepting corruption. Concurrent later refresh cannot replace the captured refs. The snapshot guarantee covers this database's metadata, not external files or particle data.

### 4.2 Execution and policies

For `run`, explicit mode wins, then an applicable same-database `db.use()` mode, otherwise auto. A foreign ambient database cannot redirect a bound Study. `errors` defaults to raise. Existing individual Calculator policies remain; new Study does not accept `collect_partial` as a whole-cell success policy.

Request validation is read-only and checks axes, ownership, selectors, signature integrity and permissions before any registration or Run creation. Necessary materialization metadata is written only during actual cell execution after request validation. Auto determines hits before requiring executable calculators, loaders or available locations. Missing computational bindings needed by selected cache misses, or by manual/refresh, are preflight errors before any cell runs; cached-only studies work without them. Once execution starts, actual file IO and scientific failures are cell failures. If cache state changes after preflight, revalidate the affected cell and expose partial progress rather than silently changing policy.

The same Database coordination seam serves `cc.run(obj)` and each Study cell: exact lookup → optional load → EvalEngine → ResultStore → Run association. ORM `run_study` remains a lower-level compatibility interface; its root-only, load-first preflight cannot be reused unchanged for this contract. Initial high-level execution is serial and row-major. There is no automatic Pipeline fusion, per-snapshot shared live particle cache, parallel backend or promise of cross-column DAG reuse. Stateful or nonrevertible transformations must not leak into later cells.

| Mode | Read result cache | Compute | Save successful cells | Persist Runs |
| --- | --- | --- | --- | --- |
| manual | No | Every selected cell | No | No |
| auto | Yes | Misses | Yes | Yes, including hits |
| refresh | No | Every selected cell | Yes, new versions | Yes |

Read-only databases allow reads and manual computation over already registered inputs, but reject auto/refresh before any selected-cell work, even if all would hit. `bind_loader` and `bind_calculators` only change process-local bindings and are allowed. Manual computation may read particles; its table has no fabricated Run IDs or saved-result refs.

## 5. Outcomes, old values and recovery

StudyTable has `origin="run"` or `"collect"`, effective invocation policies (for run), capture timestamps, full frozen source definition, displayed axes and per-cell outcomes. A filtered table retains the source definition ID and original coordinates while narrowing displayed rows. It cannot silently become the authoritative report for a new invocation.

| Status | Value present | Successful ref | Error/reason | Meaning |
| --- | --- | --- | --- | --- |
| succeeded | Yes, even if None, NaN or empty | Required for auto/refresh/collect; absent for manual | No execution error | A confirmed value |
| missing | No | None | Absent cache entry | collect only |
| failed | No public successful value | None | Required structured diagnostic | Attempt failed; an old cache version may still exist |
| not_run | No | None | Optional selection/stop reason | This invocation did not attempt this coordinate |
| incomplete | No public successful value | None | Required interruption/persistence reason | Started but completion is not confirmed |

Serialization uses an explicit value-presence flag; `value=None` alone cannot encode absence. `from_store` is True for collect and reused successful cells, False for freshly computed successful cells, and None otherwise; `run_id` identifies only an actual persisted invocation. A failed calculation's partial diagnostics may be retained separately, never as a successful cell value. Dedicated `not_applicable` is deferred until an explicit scientific outcome contract is approved; neither empty values nor exceptions imply that status.

Consider an old version V1 and a refresh that fails. The run table reports failed with no ref; V1 remains current; a later collect may return succeeded/V1. Neither operation claims the other occurred. Recovery that subsequently confirms V2 does not mutate a previously returned incomplete table; a new observation is necessary. A failed cell's `ref` never points to V1 or to an unconfirmed save receipt.

`errors="collect"` continues only after a cell's ordinary execution failure has been recorded reliably (or retained in memory in manual mode). Incomplete persistence, invalid configuration or inability to record failure stops the study. `StudyExecutionError` on raise carries `.report` and the failed coordinate, with the original exception as cause. `StudyPersistenceError` carries `.report`, interrupted coordinate and the existing exact receipt/result_ref when available. Metadata failure before a cell starts leaves it not_run; a started unconfirmed cell is incomplete. No whole-study rollback removes previously completed versions.

Catchable user interruption propagates as an interruption, with best-effort partial report attached; it is never converted into an ordinary failure and continued. Forced process termination cannot promise a final report. An existing Run may remain running. Restart is a new explicit invocation that reuses confirmed cache entries; no exactly-once/resume claim is made. Incomplete persistence receipts must be inspected/recovered explicitly, not automatically resolved by retry.

`table.cells(status="failed")` returns exact cell selectors containing source definition ID, database namespace, input key, column label and signature hash. A run validates the full selector set before effects; changed definitions and foreign tables are rejected. Selectors are never expanded into a row-set × column-set product. Empty selection does no work and returns not_run cells. Repeated auto retries may reuse a valid cache produced elsewhere; retrying a failed refresh requires explicit refresh. Untargeted cells in a retry table stay not_run. Do not merge an older run report into a new one implicitly.

## 6. Independent reopening and ownership

| Operation after restarting Python | Owning database | Particle files | Calculator implementation | Loader binding |
| --- | --- | --- | --- | --- |
| `db.studies[name]`, inspect definition | Required | No | No | No |
| `study.plan()` | Required | No | No reconstruction; binding availability may be inspected | No call |
| `study.collect()` | Required, with selected payloads | No | No | No |
| `study.run(auto)` with all hits | Required, writable | No | No | No |
| `study.run` with computation | Required | Required as specified by input | Required for those cells | Required for those cells |
| `StudyTable.load(path)`, inspect/filter/export | No | No | No project calculator code | No |

Study is independent of the old Python session and any result-table file, but still requires its owning database. StudyTable is independent of Study storage and of the database itself for embedded portable values. Calculator-free cached reading assumes supported portable payloads; legacy opaque results may still require their value type implementation and are not granted a new portability guarantee. Neither is a particle archive. ExternalFileRef values remain descriptors; inspecting or resolving external bytes is explicit.

A stored definition keeps full signatures, not just hashes or Python objects. `study.bind_calculators({label: cc})` returns a new live binding of the same definition, validates exact signatures and leaves the saved definition unchanged. Bindings are optional if supported classes can be reconstructed when needed. `collect` and all-hit auto remain usable even for a stable but nonconstructible calculator with no live binding. No arbitrary import is attempted merely to open a definition or table.

Database UUID is initialized in additive metadata on writable initialization of this feature. Read-only opening never creates it, upgrades schemas or creates a missing file. Older databases may retain existing read capabilities; requesting Study features without the necessary schema gives an explicit compatibility error. No existing cache address or payload is rewritten. A byte-for-byte backup preserves its namespace; copied databases may diverge, so a matching UUID alone does not establish that a requested pinned ref is present. Verify the exact ref when accessing that backend. Automatic merging/forking of databases is outside scope.

Exported table references contain the database namespace and optional non-authoritative location hint. Loading never follows hints or searches for databases. Passing `table.objects` to `db.study` is an explicit reattachment request; namespace, exact identities and locators must agree. Renaming a file cannot turn a foreign database into the owner. Missing referenced versions fail explicitly rather than falling back to current cache.

## 7. StudyTable persistence and data access

Proposed schema identifiers: `pynbodyext.database.study/v1` for definitions and `pynbodyext.study.table/v1` for detached tables. This is not SampleTable v3 or an implicit upgrade of ORM StudyReport JSON.

Definition fields are schema, database UUID, ordered object descriptors and ordered labelled full signatures. Stored metadata adds content digest, save-name bindings and creation time. Table fields are schema, full source definition/digest, origin/capture metadata, effective policies, displayed axes, and cells. Each cell has its coordinate, status, value-presence flag, portable encoded value when present, pinned ref when confirmed, run ID when real, from_store and diagnostics. Recovery information is separate from confirmed cell refs. A table file contains enough data to validate coordinates and signatures without the database. Every displayed row/column coordinate has exactly one explicit cell, including missing/not_run/incomplete states; omission is invalid. Filtered tables preserve the complete source definition separately from their displayed subset.

Validate schemas, duplicate JSON fields, typed identities, full signature hashes against payloads, definition digest, duplicate/unknown coordinates, status/value/ref consistency, pinned versions and owning namespace before accepting a file. Unsupported versions fail explicitly. Do not accept opaque/pickle codec tags, including nested ones, in the new format. Use existing portable ValueCodec tags for supported NumPy/SimArray/scalar/container values and ExternalFileRef descriptors. Ordinary successful None is supported. No changes to legacy codec policy are implied.

`save(path, overwrite=False)` first validates and encodes completely, then writes through a temporary file and atomically publishes on the same filesystem. Existing files are refused unless overwrite is explicit; failed validation/encoding leaves them untouched. This does not promise durability against arbitrary hardware failure. A nonportable value causes an explicit export error; the original database Result remains available and a later portable projection is a separate operation.

The table owns detached value snapshots. Returned arrays are protected or copied, and mutable nested/custom containers are defensively copied where supported. Inspection, unit conversion, filtering and export cannot mutate the stored values or refs. No `compute`, `fill_from_store`, automatic database binding, or generic in-place editing is exposed on StudyTable.

`cell(obj_or_ref, label)` and `row.cell(label)` return cell views; `row[label]` requires succeeded and returns a protected value, otherwise raises a cell-value error. `where(predicate)` explicitly filters this snapshot and keeps order/provenance; it never loads particles. `column(label, units=...)` provides numeric scalar extraction with a mask for non-success states, never treating successful zero/NaN as an absent cell. Successful None or nonnumeric/nonscalar values require per-cell access instead of implicit numeric coercion. Incompatible dimensions fail. Different compatible units require an explicit target unit; no first-row-unit guessing.

`to_dataframe(units=None, include_status=True, include_refs=True)` is a derived view with explicit unit-conversion choices and status/ref columns by default; it does not mutate the table. CSV is an interchange projection, not the lossless archive. Full-value portable JSON is the first-version detached artifact; ref-only/lazy huge-payload tables are deferred.

## 8. Two target stories

### 8.1 One galaxy, reuse and history

```python
import pynbody
import pynbodyext as pe
from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamSum

stellar_mass = ParamSum("mass").filter(FamilyFilter("star"))
sim = pynbody.load("/data/galaxy_099")
sim.physical_units()
result = stellar_mass.run(sim)
with pe.Database("analysis.sqlite") as db:
    first_ref = result.save(db, name="stellar_mass")
    obj = db.snapshots["galaxy_099"]
    key = obj.key
    assert stellar_mass.run(obj).store_ref == first_ref
    fresh = stellar_mass.run(obj, mode="refresh")
    assert fresh.store_ref != first_ref
    old = db.fetch(first_ref)

with pe.Database("analysis.sqlite", read_only=True) as db:
    obj = db.snapshots[key]
    saved = obj.results[stellar_mass]  # works without the particle file
    optional = obj.results.get(ParamSum("mass"))
```

Variables above span a notebook close/reopen. In a new process, retain the exact object key and serialize pinned refs with the existing RecordRef JSON together with their database namespace. Duplicate short names fail rather than select the first row. For logical identity use the current loaded-root `db.register(sim, ...)` before computation; path-only registration from the preceding sketch is deferred.

### 8.2 Fixed sample, retries, reopen and publication

Assume registered, replayable catalogue objects 7, 12 and 23 already exist; creating their catalogue follows the current [database guide](../database.md). Creating a Study does not scan or load a catalogue.

```python
with pe.Database("sample.sqlite") as db:
    cat = db.snapshots["snapshot_099"].catalogues["groups"]
    objects = [cat.halos[i] for i in (7, 12, 23)]
    study = db.study(objects=objects, calculations={
        "mass": ParamSum("mass"), "stellar_mass": stellar_mass,
    })
    study.save(name="galaxies-v1")
    print(study.plan())
    before = study.collect(missing="keep")
    table = study.run(errors="collect")
    table.save("attempt-1.json")

# A new process needs no previous table to continue the saved definition.
with pe.Database("sample.sqlite") as db:
    study = db.studies["galaxies-v1"]
    current_attempt = study.run(mode="auto", errors="collect")
    current_attempt.save("attempt-2.json")

# A different process needs no database or calculator implementation to inspect.
frozen = pe.StudyTable.load("attempt-1.json")
for cell in frozen.cells(status="failed"):
    print(cell)  # exact selectors, not particle objects

# After fixing a temporary execution problem, retry only those exact cells.
with pe.Database("sample.sqlite") as db:
    study = db.studies["galaxies-v1"]
    retried = study.run(cells=frozen.cells(status="failed"), errors="collect")
    retried.save("retry-1.json")
    publication = study.collect(missing="keep")
    print(publication)  # review missing states before choosing the paper sample
    publication.save("paper-sample-v1.json")
```

For an old failed refresh use `mode="refresh"` on retry. Saving a report does not mark incomplete persistence as recovered. A strict publication gate may call `collect(missing="raise")`; this guarantees current cache coverage, not that the most recent refresh succeeded.

```python
with pe.Database("sample.sqlite") as db:
    study = db.studies["galaxies-v1"]
    publication = pe.StudyTable.load("paper-sample-v1.json")
    # Extend definitions without reconstructing existing calculators.
    expanded = db.study(objects=study.objects, calculations={
        **study.calculations,
        "gas_mass": ParamSum("mass").filter(FamilyFilter("gas")),
    })
    expanded.save(name="galaxies-v2")
    expanded_table = expanded.run()

    # Explicitly select only successful sufficiently massive objects.
    selected = publication.where(
        lambda row: row.cell("stellar_mass").status == "succeeded"
        and row["stellar_mass"].in_units("Msol") > 1e10
    )
    next_study = db.study(objects=selected.objects, calculations={"mass": ParamSum("mass")})
```

Table column labels are not object aliases: use `objects[0].results[stellar_mass]`, not an unregistered string alias. In a quiescent store that ref equals the corresponding successful auto/collect cell's ref.

## 9. Layering, alternatives and compatibility

The recommended Module seam is Database coordination above ORM, delegating science to EvalEngine and persistence to ResultStore. Study definitions/resolution live at that upper layer. Detached StudyTable/Cell/ObjectRef data and codec validation live in the calculation layer without SQLAlchemy or Database imports. Root StudyTable export must be lazy and covered by the package import contract. ORM report compatibility can use explicit internal conversion; lower-layer tables must not discover upper-layer repositories.

Alternatives rejected: making `cc.run(list)` return a different type based on input cardinality; changing mutable SampleTable methods into immutable operations; a second batch executor with independent cache policy; implicit Pipeline fusion; one new framework for all time-series/multi-input workflows. A dedicated detached table adds one public type but avoids changing existing builder semantics and keeps both new result paths identical.

ADR-021 adds read-only Database opening, results.get, Study definitions/bindings/coordination, detached reports and additive namespace metadata. It preserves existing ResultStore keys, immutable versions, Result.save semantics, accepted lower-level APIs and tests. Existing SampleTable v1/v2 and ORM study/v1 files continue through their current loaders. There is no silent conversion or fabricated run/ref history.

## 10. Acceptance scenarios

Scenario IDs link the design to the test coverage recorded in FEAT-0018.

| ID | Scenario | Required observation |
| --- | --- | --- |
| A01 | Single save then batch auto | Identical pinned ref; zero particle loads on hit |
| A02 | Batch auto then single auto | Identical K/ref, no duplicate payload on hit |
| A03 | Rename Study/column or add another column | Existing computation reused; only missing calculations execute |
| A04 | Change parameter, signature version, input revision or selection | Distinct K; no false hit |
| A05 | Duplicate exact input under two catalogue origins | Explicit rejection before writes, not silent row merging |
| A06 | Two columns with one signature | Ordered auto reuse; ordered refresh versions, exact per-cell refs |
| A07 | Read-only plan/collect for unmaterialized supported catalogue handles | No selection registration, payload decode during plan, particle IO or Runs |
| A08 | Offline particles/unbound loader and all cache hits | Collect and writable auto succeed; no loading |
| A09 | Missing executable class with all hits versus one miss | Inspection/collect/hits succeed; required missing binding fails preflight |
| A10 | Bind wrong signature or mutate live calculator | Reject without executing the mismatched definition |
| A11 | Failed refresh with old successful cache | Run cell failed/ref=None; collect reads old ref; old table unchanged |
| A12 | Successful None, zero, NaN and empty array | Success preserved across file roundtrip, distinct from missing |
| A13 | Ordinary execution failure, collect policy | Later cells continue, failed cell has no stale ref |
| A14 | Raise policy | Partial report contains failure and later not_run cells |
| A15 | Persistence failure before/after beginning a cell | not_run/incomplete distinguished; receipt separate; later cells stop |
| A16 | Exact recovery after a partial report was saved | No duplicate payload, no mutation of old report |
| A17 | Retry two diagonal failed cells | Exactly two cells targeted; no Cartesian expansion |
| A18 | Retry selectors from changed/foreign definition | Full request rejected before work |
| A19 | Close/reopen saved Study without its old table | Same typed ordered axes/signatures; no calculator reconstruction on open |
| A20 | Remove database and project calculator package; load table | Embedded values/units/statuses/refs remain usable without backend access |
| A21 | Foreign database with same names and SQL IDs | Reject reattachment; no name-based substitution |
| A22 | Relocated backup with same UUID but missing pinned version | Explicit missing-version error; no latest fallback |
| A23 | Concurrent refresh while collecting | Values match initially captured pinned refs |
| A24 | Mutate a value obtained from a table | Original table/provenance unaffected or write rejected |
| A25 | Unsupported schema, duplicate field, corrupt ref or nested opaque tag | Reject before unsafe decoding; no IO/execution side effects |
| A26 | Export encoding failure or existing file | Existing artifact unchanged; overwrite must be explicit |
| A27 | Read-only open, missing file, old schema; auto even on all hits | No creation/migration; explicit permission/capability errors |
| A28 | Manual run and empty axes/empty selectors | No Runs/payload writes; honest unpersisted success/not_run states |
| A29 | Filter/convert/export columns with units and failed states | Order/refs kept, masks explicit, incompatible or ambiguous units rejected |
| A30 | Layer/import compatibility | No SQLAlchemy/Database dependency in detached table; legacy tests intact |
| A31 | Save same definition/name twice, then conflicting content | Idempotent same binding; conflict does not overwrite |
| A32 | Keyboard interrupt versus forced process exit | No continue-on-interrupt; no promised final report/exactly-once execution |

## 11. Deliberately deferred

Path-only registration, dynamic query samples, repeated-row/bootstrap studies, multiple databases in one Study, distributed scheduling, automatic shared particle loading, bulk submission of manually reviewed tables to the cache, multi-input/time-tree calculations, scientific not-applicable outcomes, arbitrary opaque portable objects, and ref-only lazy table artifacts are separate designs. They are not prerequisites disguised as unfinished parts of this proposal.

The three review questions are resolved at contract level: single and batch share K and coordination; invocation outcomes never inherit old success; Study reopens from its database and StudyTable reopens from its own file.
