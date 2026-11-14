# Database user experience design

Date: 2026-11-14. Source baseline: `41dc264`.

Status: implementation authorized by the user, including commit/pull/merge.
Implementation and validation are tracked in FEAT-0013–0017. The current user guide
is [docs/database.md](../database.md); this document records the accepted design.
[中文](database-ux-2026-11-14.zh-CN.md) · [Architecture proposal](../adr/0020-database-user-interface.md)

## 1. The user model

Calculators compute; Results carry outputs; Database saves and finds them.
Attributes expose fixed capabilities, brackets select collection members, and
database objects load particles through `.load()`.

```python
import pynbody
import pynbodyext as pe
from pynbodyext.properties import ParamSum

sim = pynbody.load('/data/snapshot_099')
cc = ParamSum('mass')
res = cc.run(sim)
db = pe.Database('analysis.sqlite')
ref = res.save(db, name='total_mass')
snap = db.snapshots['snapshot_099']
saved = snap.results['total_mass']
saved.value
saved.calculator
particles = snap.load()
db.close()
```

These are proposed examples, not executable current tutorials. `save()` returns
a pinned RecordRef and updates the same Result's store_ref. Database accepts a
local SQLite filename. Ordinary calculations require neither Database nor
SQLAlchemy. SQLite is the only initially certified database dialect.

## 2. Execution and persistence modes

```python
with pe.Database('analysis.sqlite') as db:
    with db.use(mode='auto'):
        first = cc.run(sim)
        second = cc.run(sim)
        fresh = cc.run(sim, mode='refresh')
    assert second.from_store
    cached = cc.run(db.snapshots['snapshot_099'])
```

| Mode | Persistent lookup | Save after computation |
| --- | --- | --- |
| manual | No | Only explicit save |
| auto | Yes | Yes on miss |
| refresh | No | Yes, a new version |

A raw input without a database or context defaults to manual. Explicit
`database=db`, a database object input, and `db.use()` default to auto. An explicit
mode overrides the applicable context. Invalid modes fail before execution.
Database resolution is explicit argument, input's owning database, then current
context. An input owned by A rejects an explicit B. An ambient B neither rebinds
an A input nor supplies its mode: the input uses A's default instead.
ContextVar scopes nest, restore after exceptions, and do not hold transactions
or write locks while calculating. Exiting use does not close the database;
exiting Database does. Decoded Result values remain usable after closure.

Save destination resolution is explicit argument, current context, then Result's
bound database. Absence raises an error suggesting `save(db)`. A different target
is an explicit copy, never a reinterpretation of input identity. This differs
intentionally from execution ownership: a save context selects a destination.
A selected closed database fails rather than reopening or redirecting writes.

Existing explicit store/reuse calls remain supported and bypass ambient Database
configuration. Combining them with explicit database/mode or database object
inputs fails before execution. The old reuse=False meaning remains; new calls
use mode so that two persistence policies cannot coexist in one invocation.

## 3. Collections and names

```python
db.snapshots
snap = db.snapshots['snapshot_099']
snap.results
snap.results[cc]
snap.results['total_mass']
snap.load()
db.simulations['TNG50-1'].snapshots['099']
```

Collections contain registered objects, without filesystem or catalogue scans.
Representations show bounded metadata tables and counts without particle loads,
large-value decoding or calculator reconstruction. Result listings include name,
calculator summary, short hash, value type/shape/units, version and save time.
Snapshot means one time output; Simulation means the whole simulation. Path-only
snapshots can lack Simulation/Timestep membership. Directory names do not infer
scientific identities. Snapshot basename is a display name, never an identity.
Each snapshot has an unambiguous key. Duplicate-name lookup raises ambiguity and
shows keys/locations; key lookup resolves it. Missing entries raise KeyError.

Explicit result aliases are unique within an input and bind full calculator
hashes. Rebinding the same hash is allowed; another hash raises a conflict.
Pretty names may repeat and do not automatically become string lookup aliases.
`results[cc]` always selects by full signature. Aliases do not affect caching;
dependencies, parameters, signature versions and Pipeline output names do.
Lookup returns the current saved Result, never computes, and raises KeyError on
absence. Previously obtained Results/receipts stay pinned after refresh.
Each successful save appends a version, consistent with ResultStore; an auto hit
does not. Pipeline named outputs stay in the root result and are not invented
independent reconstructible calculators.

Result.calculator retains its current contract: an available calculator or None.
Its signature/expression remains inspectable without successful reconstruction.
This refines the earlier suggestion to raise from the property and avoids
breaking the existing field contract.

## 4. Identity and deferred save

Plain run followed by save must derive the same key as auto execution. High-level
run defaults to snapshot_identity for ordinary file inputs. Direct EvalEngine
keeps its session default; in-memory and externally transformed inputs remain
session-only. This high-level default change requires an accepted ADR and tests.
Save must not infer past input/calculator identity from their current live state.

Capture an immutable input description and CalculatorSignature before execution:
root location, identity, loader ID, known unit-loading convention, root count and
necessary root-relative selection. Results do not strongly retain the full sim.
DAG transforms remain computations, not rewritten root locations. Subsequent
calculator edits, labels or cwd changes cannot change this result's key.
Save serializes the Result's current values, as existing stores do; editing its
value does not recompute and must not be represented as untouched original output.

| Input | Automatic behavior |
| --- | --- |
| Local root | Normalized absolute path identity, built-in pynbody loader, multipart prefixes allowed |
| Registered logical root | Explicit logical identity/revision/loader, no path fallback |
| Particle view | Root identity plus existing digest, accompanied by replayable selection |
| Memory-only or externally transformed input | Manual calculation works; persistent auto/save fails explicitly |
| Unknown loading convention | Do not claim reloadability; identify missing loader information |

Advanced registration is `db.register(sim, simulation='TNG50-1', snapshot='099',
revision='r1', loader='default') -> Snapshot`. The project namespace comes from
Database configuration, defaulting to `default`. Use the returned Snapshot or
db.use context for subsequent calculations. Registration declares equivalence;
it does not rewrite or merge earlier path-keyed results. Multiple revisions at
one path require a selected Snapshot; raw sim resolution cannot guess a revision.

There is no full particle fingerprint. In-place array edits require refresh;
distinct data versions require revision. Refresh does not archive particle files.
Moving a default path produces a new identity; explicit logical registration
supports location-independent reuse. Missing captured persistent identity cannot
be repaired by promoting a session ID at save time.

## 5. Halos, subhalos and loading

```python
snap = db.snapshots['snapshot_099']
halo = snap.halos[0]
halo.results[cc]
particles = halo.load()
res = cc.run(halo, mode='auto')
halo = snap.catalogues['subfind'].halos[0]
subhalo = snap.catalogues['subfind'].subhalos[12]
```

Brackets preserve original typed catalogue IDs, not row positions. snap.halos
requires a unique/default catalogue; ambiguity raises. The explicit persistent
setter is `db.set_default_catalogue(snap, 'subfind')`. Physical host relationships
remain separate from root/view input ownership.

Saving a real Halo registers its root and object. Record catalogue namespace,
finder/revision/loader when reliably available; `_halo_number` alone is not enough.
An otherwise replayable view with unknown catalogue belongs in
`snap.selections[key]`, with a visible registration notice, not an invented halo
catalogue. `db.register_catalogue(snap, catalogue, name='subfind', revision='v1')`
imports metadata from a supported pynbody catalogue without reading all particle
memberships. Unsupported adapters fail explicitly; universal finder support is
not promised.

Load requires a supported catalogue locator or copied root-relative indices;
a digest is not a replay definition. Preserve Selection order, duplicates and
family-block restrictions without sorting. First successful materialization fixes
the catalogue object's exact input identity/selection. A changed catalogue
revision creates another identity. A metadata-only object with no exact input
identity must load on first run; it cannot claim a load-free hit. Validate root
identity, expected count and selection digest on loading. Same-count content
changes still require user-managed revision.

The built-in root loader is pynbody.load. Bind custom project code with
`db.bind_loader('custom', callable)`; it accepts a location and returns a root,
with options pre-bound in project code. Rebind after reopening. Never serialize
callables or credentials. Arbitrary earlier pynbody.load kwargs cannot be inferred
reliably; incompatible loading conventions require explicit loader registration.
Multiple locations require `.load(location_id=...)`; failure does not try another
copy automatically.

## 6. Load-free cache and failures

A database object resolves exact identity/signature before lookup. Hits invoke
neither snapshot loader, halo finder nor selector, and work while files are
offline. They assert a saved result for the registered identity, not unchanged
disk contents. Only misses/refresh load and validate, then delegate to EvalEngine.
Raw SimSnap inputs have already incurred the caller's loading cost.

Values, units, named outputs and pinned receipts use the existing codec/store.
Corruption, decoding failure and signature mismatch are not misses. Loading,
calculation or encoding failure cannot replace current successful results or
create successful associations. Every auto/refresh invocation has a Run, including
hits; manual computation creates none. Later manual save records a save operation
and captured provenance, not a fabricated past Run. Hit invocation environment
must not be substituted for historical producer environment.

One SQLite file does not make current independent transactions atomic. Initially
retain payload-then-metadata recovery: association failures expose the pinned
receipt; `db.recover(ref)` validates input, recipe and save operation before
finishing missing links. Recovery never recomputes, appends another payload, or
guesses a latest receipt. Unverifiable provenance rejects recovery. Register the
save-operation description before writing payloads so failures remain recoverable.

## 7. Ownership and exclusions

New pynbodyext/database owns the user interface, browsing, names, loader bindings
and execution coordination. Extend existing ORM input/browsing relationships to
accept paths; reuse result tables. Do not create per-snapshot SQL tables, another
result store, another DAG or another executor. Core owns SQLAlchemy-free execution
and save protocols/context; Database implements them. Core never imports ORM or
Database; pe.Database is lazy-exported.

Keep one owner for logical identities, immutable versions, codec, selection replay
and run recovery. Old explicit store/reuse behavior stays independent of ambient
configuration. Existing tests remain intact; tests of the changed high-level
default may change only for the accepted requirement recorded in the ADR.

Exclude remote downloads, particle archival, automatic content fingerprints,
inferred scientific identities, distributed scheduling, arbitrary SQL certification,
legacy database migration and historical garbage collection. Concurrent misses may
compute twice; keys/databases/versions must remain isolated. No global exactly-once
execution guarantee.

## 8. Delivery stages and acceptance

| Stage | Deliverable | Required evidence |
| --- | --- | --- |
| A | Root run/save/reopen/results/load | Real files and units round-trip; automatic schema; frozen save identity; duplicate basenames isolated |
| B | Modes and context | Hit skips computation; refresh keeps historical receipts; nested/exception restoration; cross-database rejection |
| C | Database objects as inputs | Zero loader/selector calls on hit; offline hit works; miss/refresh loads |
| D | Halo/subhalo/selection browsing/loading | Catalogue isolation, original IDs, reopened ordering/duplicates/units, unavailable selection is explicit |
| E | Logical registration, custom loaders, recovery, tutorials | No revision guessing; missing loader still permits reads; payload/link failure recovers; runnable onboarding |

Create bilingual work items per stage; demonstrate expected failing public-interface
tests before implementing each change. Write the detailed implementation plan after
design review; this table is sequencing, not a completed execution plan.
Run make check, and make test-all for these public/data-path changes. Preserve
low-level identity/version/catalogue/selection/import tests. Prove no-I/O behavior
with call counts, not just from_store. Final examples must use real temporary
pynbody files, not merely spoof a filename on in-memory snapshots.

## 9. Handoff

This increment adds paired design documents, proposed ADR-020 and index links.
No package/test edits, work-item status changes, example replacements, commits,
pushes or PR creation. Earlier uncommitted example documentation stays intact.
