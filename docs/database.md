# Save and reuse calculations

[中文](database.zh-CN.md) · [Runnable example](database_quickstart.py)

Use a calculator to compute, a Result to inspect the output, and a Database to
save and find it. Database stores results and input descriptions in one local
SQLite file; particle files stay at their original locations.

A full `pynbodyext` installation includes SQLAlchemy. With the lightweight
`pynbodyext-core` distribution, enable its `store` extra to use Database. Pure
calculations do not require SQLAlchemy. In a repository checkout, `make setup`
prepares the project virtual environment.

The following notebook-style blocks reuse the same open `db` in order and close
it at the end. Replace the snapshot path with yours. The script at the end creates
its own data for a complete runnable example.

## Calculate, then save

```python
import pynbody
import pynbodyext as pe
from pynbodyext.properties import ParamSum

sim = pynbody.load('/data/snapshot_099')
sim.physical_units()
cc = ParamSum('mass')
res = cc.run(sim)

db = pe.Database('analysis.sqlite')
ref = res.save(db, name='total_mass')
snap = db.snapshots['snapshot_099']
saved = snap.results['total_mass']
print(saved.value)
print(saved.calculator)
particles = snap.load()
```

`save()` creates the tables and registers the input, calculator signature and
encoded result. Its return value is a pinned `RecordRef`; the same reference is
available as `res.store_ref`. You can save after the original particle object is
gone because `run()` captures the input identity and calculator signature.

`name` is optional. Use the calculator itself for an exact lookup:

```python
saved = snap.results[cc]
```

A name binds one calculator signature within one input. Reusing that name for a
different calculation raises a conflict. Display names and short hashes do not
control cache identity. `saved.calculator` is a runnable calculator when its
registered type can be reconstructed; otherwise it is `None`, while the value
and stored signature remain available.

## Let repeated calculations use the database

```python
with db.use(mode='auto'):
    first = cc.run(sim)
    again = cc.run(sim)
    assert again.from_store

    fresh = cc.run(sim, mode='refresh')
    assert fresh.store_ref != first.store_ref

old = db.fetch(first.store_ref)
versions = db.versions(first.store_ref)
```

| Mode | Read saved result | Save computed result |
| --- | --- | --- |
| `manual` | No | Only when you call `res.save()` |
| `auto` | Yes, if the exact input and calculation match | Yes, on a miss |
| `refresh` | No, recompute | Yes, as a new version |

Without a database or context, `cc.run(sim)` uses manual mode. `db.use()` defaults
to auto. An explicit `database=db` or a database object input defaults to auto
unless an applicable context supplies a mode. An explicit `mode` overrides it.

```python
res = cc.run(sim, database=db, mode='auto')

with db.use(mode='manual'):
    res = cc.run(sim)
    ref = res.save(name='total_mass')
```

A bare `save()` uses the current context or the Result's bound database. Without
either, pass `save(db)`. Closing a database leaves decoded values usable, but
further database operations require an open Database. Contexts restore outer
settings on exit, including after exceptions.

Every successful explicit save creates a new version. Auto hits reuse their
existing version; earlier receipts remain readable after refresh. A lookup such
as `snap.results[cc]` raises `KeyError` if absent and does not compute.

## Browse objects and reload particles

```python
db.snapshots                         # registered snapshots
snap = db.snapshots['snapshot_099']
snap.results                         # saved result summaries
snap.load()                          # pynbody particles
```

Collection displays show metadata without loading particles or decoding result
arrays. Results written through a low-level store for a registered input still
appear; missing summaries show `summary unavailable` without reading payloads.
Collections contain registered objects, not a scan of your data directory.
Two paths can have the same basename. If a name is ambiguous, inspect
`db.snapshots` and choose `db.snapshots[snap.key]` using the full key.

A database handle also works as calculator input:

```python
res = cc.run(snap)                    # lookup first; load only on a miss
```

Once the object's exact input is registered, a cache hit does not call its
particle loader. You can read saved results while the particle files are offline.
Raw `cc.run(sim, database=db)` cannot avoid loading you already performed yourself.

## Preserve an exact particle selection

```python
with db.use():
    selected_result = cc.run(sim[[2, 0, 2]])

snap = db.snapshots['snapshot_099']
selected = next(iter(snap.selections))  # this example has one registered selection
particles = selected.load()
assert len(particles) == 3
saved = selected.results[cc]
```

The stored selection preserves order and duplicates. Pynbody requires selected
particles to follow the root's family-block ordering; a cross-family interleaving
raises an error. Reloading checks the root count and selection identity. The
checks do not detect arbitrary same-count edits to particle data.

## Browse halos

The qualified adapter in this release supports `HaloNumberCatalogue`, whose
integer array assigns a halo number to each loaded particle. The following assumes
your `sim` provides a `grp` array and its catalogue includes halo ID 7:

```python
from pynbody.halo.number_array import HaloNumberCatalogue

catalogue = HaloNumberCatalogue(sim, array='grp', ignore=-1)
res = cc.run(catalogue[7])
res.save(db, name='halo_mass')

snap = db.snapshots['snapshot_099']
halo = snap.halos[7]
halo.results['halo_mass']
halo.load()
cc.run(halo)                          # a saved hit avoids particle loading
```

Keep the catalogue alive until saving so the library can capture its metadata.
For this adapter, saving a live Halo registers the catalogue with a
membership-derived revision. An unavailable or unsupported catalogue produces a
notice and leaves the saved object accessible through `snap.selections`.

You can instead name and register the catalogue yourself:

```python
cat = db.register_catalogue(snap, catalogue, name='groups', revision='v1')
halo = cat.halos[7]
```

`7` is the catalogue's original ID, not its row position. Registration copies the
already loaded membership array and lists IDs without materializing each halo.
Multiple catalogues or revisions require a full catalogue key or an explicit
default: `db.set_default_catalogue(snap, cat.key)`. An exact particle selection
can share cached results across catalogue namespaces when its identity matches.

`cat.subhalos` and `halo.subhalos` browse registered subhalo metadata. Reloading a
subhalo requires an explicit saved selection binding. The number-array adapter
does not infer hosts or subhalo memberships. Other halo finders need qualified
adapters; the library does not guess their constructor options.

## Name simulations or bind a custom loader when needed

Default snapshot identity uses the absolute file path. For explicit scientific
identity, register a simulation, snapshot and optional data revision:

```python
def load_custom(path):
    loaded = pynbody.load(path)       # bind format-specific options here
    loaded.physical_units()
    return loaded

snap = db.register(sim, simulation='simulation-A', snapshot='099',
                   revision='r1', loader='custom')
db.bind_loader('custom', load_custom)
res = cc.run(snap)
db.simulations['simulation-A'].snapshots['099']
```

Use the returned handle or `db.use()` for subsequent calculations. Registration
does not rewrite results saved earlier under a path identity. Multiple revisions
at one path require a selected handle; raw input lookup reports ambiguity.
Rebind custom loaders after reopening. Cached results remain readable without a
loader binding. Database does not recover arbitrary kwargs from an earlier
`pynbody.load()` call or serialize your loader function.

Array edits do not change the default cache key. Use `refresh` to replace the
current result, or a new data `revision` to keep input versions distinct. Moving
a default path changes identity; explicit logical identities support location
aliases. A Calculator's declared signature version must change when its scientific
behavior changes for the same parameters. Particle files are not archived.

Persistent saving requires stable file-backed input and calculator identities.
In-memory `pynbody.new()` objects, partially loaded file roots and active external
transformations cannot silently become full-file cache entries. A partial default
file load needs a distinct explicit logical identity and a custom loader that
defines how to reload it; do not reuse the full-file default identity. Put
supported transforms inside the Calculator DAG. SQLAlchemy storage is optional; SQLite is
the validated backend, and no legacy-database migration is provided.

For a metadata-finalization failure with a saved receipt, inspect
`DatabasePersistenceError.result_ref` and call `db.recover(ref)` after fixing the
cause. Recovery validates that exact version and does not recompute or append
another payload. Existing low-level `store=`/`reuse=` calls remain available;
keep them separate from the Database/mode arguments.

Close the connection when you finish the notebook's database operations:

```python
db.close()
```

In scripts, use `with pe.Database(...) as db:` for lifecycle management, keeping
operations that need the database inside that context.

## Run the complete example

From the repository root:

```bash
PYTHONPATH=. .venv/bin/python docs/database_quickstart.py
```

The script writes and reloads a four-particle Tipsy file, checks units, saves and
refreshes results, reloads an exact selection and a halo, then reopens SQLite
with a loader that raises. Its final cache-hit assertions prove that root,
selection and halo results can be reused without loading particles. It prints
the temporary output directory and keeps the files for inspection.
