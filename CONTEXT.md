# Context

Single-context repo. This file defines the vocabulary used across the calculator
and result-store subsystems, plus the architecture invariants the tests enforce.

## Glossary

Use these terms exactly. Don't drift to synonyms.

- **Calculator DAG** — the composed execution graph (`CalculatorBase` +
  mixins, `PropertyBase`, `PostprocessorBase`, `FilterBase`, `TransformBase`, `Pipeline`). Each node
  produces a `ResultNode`; the graph is resolved and executed by `EvalEngine`.
- **`CalculatorSignature`** (`result/signature.py`) — a frozen, structured
  description of a calculator (class + `deps` + parameters), with a content
  hash (sha256 of schema + payload) and a JSON text. *Not* a display name.
  Two independent flags: **constructible** (the payload can rebuild the
  calculator) and **stable** (the hash is the same in every session, so it can
  key data that outlives the process; false when a part is identified by `id()`
  or a function repr — `LambdaProperty`, a node without `signature_payload`).
  A `Pipeline` is identified by its ordered output names and children; its
  `name`, like every node's, is a label.
- **Signature name** — how a payload names a calculator class: the class's own
  `signature_name` (conventionally its public import path, e.g.
  `pynbodyext.filters.Sphere`), else its class path. Not inherited: a subclass
  is another calculator. Registered at class creation
  (`interfaces.register_signature_type`), unique, and resolved back by
  `interfaces.resolve_signature_type`, which also knows `signature_aliases`, so
  moving or renaming a class keeps stored keys readable.
- **Signature version** — a class's own `signature_version` (default 0, omitted
  from the payload). Bump it when what the calculator computes changes for the
  same parameters: the key changes, so stale stored results stop matching.
- **`from_signature_payload`** — optional classmethod, the inverse of
  `signature_payload()`; it makes a hand-written (non-dataclass) node
  rebuildable from its signature.
- **`sim_signature`** — a pluggable address tuple describing *which* simulation
  object a result belongs to. Its tag says how far it can be trusted: `("sim",
  ...)` is stable across sessions, `("session", id(obj))` is valid in this
  process only (`interfaces.is_stable_sim_identity`). Produced by
  `EvalEngine.make_sim_signature(sim)`.
- **Typed simulation key** (`store/identity.py::sim_identity_key`) — versioned
  canonical JSON used by `RecordRef`, result-store lookup/listing and
  `SampleTable` row indexing. Built-in strings, booleans, integers, finite floats,
  None and recursive tuples are supported. Types distinguish identities:
  `1`, `"1"`, `True` and `1.0` are different; floating `-0.0` and `0.0` agree.
  Providers convert NumPy scalar IDs to built-ins explicitly; mutable values,
  arbitrary objects and non-finite floats are rejected. This is independent of
  calculator-parameter numerical equivalence. See ADR-006: there are no existing
  databases or caches to migrate; no legacy compatibility is required.
- **`engine.make_sim_signature`** — the stable-identity seam. Default is
  id-based (`("session", id(sim))`). `snapshot_identity` is the stable provider
  for pynbody snapshots: `("sim", <abs path>)`, plus `("halo", <number>,
  <digest>)` or `("subset", <digest>)` for a sub-snapshot, the digest hashing the
  particle indices it selects. It falls back to a session identity for a
  snapshot without a file or with a transformation applied outside the
  calculator. `calc.run(sim, store=<persistent store>)` uses it by default.
- **`LogicalSnapshotIdentity`** (`runtime/logical_identity.py`) — explicit
  project/simulation/timestep string IDs, optional data `revision` (default None),
  and user-managed `loader_id` (default "default"). None revision means data edits
  do not change the key; users refresh affected computations with `reuse=False`.
  A supplied revision or changed loader ID creates a separate cache identity.
  This data revision is distinct from `RecordRef.version_id`, which identifies
  a result save. Labels assert equivalence; they do not prove file contents equal.
- **`ConfiguredSnapshotIdentity`** — callable `sim_identity` provider mapping
  explicit local paths/prefixes to logical identities. Paths are normalized at
  configuration time and are not part of configured cache keys. Aliases may share
  an identity; conflicting normalized locations raise ValueError. The copied
  configuration has versioned JSON round-trip with absolute locations. Unknown
  snapshots retain `snapshot_identity` fallback, and externally transformed roots
  or views stay session-only. No file scanning, fingerprints, automatic loading or
  global registry. See ADR-008; revision is optional by explicit user choice.
- **Read-through store** — `calc.run(sim, store=store)` returns the stored
  result when the key is present (`result.from_store`), and computes and stores
  it otherwise; `reuse=False` recomputes, saves a new version and updates the
  current cache. `result.store_ref` identifies the saved/reused version, or is None
  when persistence did not occur. A persistent store is
  neither read nor written when either half of the key is session-only.
  In-place array edits are not part of the key.
- **Content-addressed key** — the composite store identity:
  `(sim_signature, calculator_signature.hash)`, the full 64-character hash. Two
  identical computations derive the same key; a changed calculator or object
  derives a different one. Numbers compare by value (`CalculatorSignature.identity()`:
  `30` and `30.0` share a key). A hit is checked against the requested payload
  (`SignatureMismatchError`), and a persistent store refuses an unstable
  signature (`UnstableSignatureError`). `short_hash()` is for display only.
- **`ResultStore`** (`store/base.py`) — the persistence seam. `store`/`fetch`/
  `get`/`has`/`delete`/`list_refs`/`list_versions` over six abstract primitives
  (`_put`, `_get_by_key`, `_delete`, `_list`, `_get_version`, `_list_versions`).
  Every save atomically appends a version and updates the current cache.
- **`RecordRef`** — a typed simulation + full calculator hash address, optionally
  pinned by `version_id`. `cache_key` identifies the computation; `key` also
  includes its version when pinned. `store()` and `Result.store_ref` return pinned
  references, whose `fetch` remains unchanged by refresh or cache invalidation.
  `get`/`has`/`load` and unpinned refs use the current cache; `list_refs()` lists
  current unpinned addresses; `list_versions(ref)` lists saved versions in insertion
  order. `delete` only invalidates current cache (a pinned ref must still be current),
  retaining all historical versions without promoting an older one. Reference JSON
  preserves identity types and version, and must be paired with its owning backend.
  No automatic historical GC or legacy migration; see ADR-007.
- **`SignatureCatalog`** (`store/catalog.py`) — keys (full signature hashes)
  mapped to their signatures and optional labels: turns a key back into the
  calculator (`catalog.calculator(key)`) and its expression. Looked up by full
  key, unique prefix of >= 8 characters, or label; a label names one key.
  Saved as JSON; `ResultStore.catalog()` collects the calculators a store holds.
- **`SampleTable`** (`sample.py`, the `sample` layer above `bins`) — objects
  (rows, keyed by simulation identity) × calculators (columns, keyed by
  signature), so each cell is a result-store key and a study computes only what
  the store lacks. `compute` / `add` / `fill_from_store` / `from_store`;
  `column`, `where`, `missing`; `to_dataframe` (column metadata in
  `df.attrs["pynbodyext"]`), lossless `save` / `load` (JSON, values via
  `ValueCodec`, columns via the catalog). Cells are separate runs: shared
  dependencies are not shared between columns. Default labels for configured
  logical snapshots include their logical fields plus any halo/subset suffix;
  callers can still supply an explicit row name. `SampleRow.refs` holds each
  successful cell's actual pinned `Result.store_ref`; retries clear stale refs
  on failure or success without persistence. Optional `store_name` labels the
  owning backend; filtering and exports retain it without automatically locating
  or querying a backend. JSON schema `pynbodyext.sample.table/v2` preserves and
  validates pinned refs against typed row identity and column signature; v1
  remains readable with empty refs. DataFrame metadata keys identities and refs
  by `sim_identity_key`, with ordered `identity_keys` alongside possibly duplicate
  display names. Loading only restores values/references; explicitly call
  `store.fetch(row.refs[label])` to read a historical version. Successful `None`
  still shares the existing missing-cell sentinel. See ADR-011.
- **Record** (`store/base.py::ResultRecord`) — one self-describing, codec-encoded
  row (value + named + provenance), serializable via `to_json`/`from_json`.
  The current schema is `pynbodyext.result.store/v4`; JSON loading refuses missing
  or unsupported schema versions and missing/empty `version_id`.
- **`ValueCodec`** (`store/codec.py`) — portable value serialization (numpy,
  pynbody `SimArray` with units, scalars, nested structures, pickle fallback).
  Object arrays retain shape, C-order element boundaries and units via a
  versioned tag. Structured arrays use NPY with pickle disabled; object fields,
  nested dtype metadata and unreadable/non-lossless dtype headers are rejected
  before saving. Ordinary numerical-array tags remain unchanged. Legacy object
  tags remain readable with their original shape limitations.
- **tangos** — the Python package this store is modeled on; a relational DB that
  interns name strings. We deliberately diverge.
- **Repr style** — how a calculator object renders in a notebook, set by
  `display.set_repr_style` (session-wide) or `display.repr_style` (a `with`
  block): `rich` (cards, `<details>`, badges, inlined CSS) or `simple`
  (GitHub-sanitizer-safe: `<h4>`/`<h5>`/`<p>`/`<table>`/`<pre>`/`<code>`/
  `<strong>`/`<br>` only, no attributes — so a notebook pushed to GitHub renders
  cleanly). `github` and `plain` are retired names for `simple`.
  *Not* a data property: every style shows the same fields, only the markup
  differs.
- **`Card`** (`display.py`) — the style-neutral description of what an object
  shows: a title, `label -> value` rows and blocks (`Fields`, `Table`, `Metrics`,
  `Pre`, `Section`). Values are fragments (plain values, `Code`, `Badge`), never
  pre-rendered HTML. Objects build a card in `_repr_card()`; one renderer per repr
  style turns it into markup (`register_repr_style` adds a dialect).
- **`InfoView`** (`display.py`) — mixin for a value object that renders itself.
  The subclass defines `_display_rows()` once; the mixin derives the compact
  `__repr__` and the per-style HTML. Adopted by `ProvenanceInfo`, `PerfSummary`,
  `ValueSummary`, `ErrorInfo`, `PhaseRecord`, `AccessObservation` and the result
  dict views.
- **`TextReport`** (`display.py`) — a `str` subclass returned by the view
  attributes (`Result.execution_tree` / `.performance` / `.cache`,
  `CalculatorBase.dependency_tree`, the `Result.report_*()` builders). It behaves
  like `str` everywhere but also renders as HTML, so one name serves both the
  text and the notebook view.
- **Section attribute** — the rule tying the two display layers together: a
  *drill-down* section (`Section(attr=...)`) is reachable as an attribute whose
  name is the section title in snake_case (``Execution tree`` ->
  ``.execution_tree``). The `simple` card does not inline drill-downs; it lists
  exactly those names under *Details*, so a reader drills down by typing an
  attribute, never by calling a method. A section without an `attr` is inlined in
  every style, so nothing is hidden without a way to reach it. Card sections
  render from the attribute's own rows, so the two cannot drift.
- **View-only attribute** — a public attribute that exists only to render
  something already reachable elsewhere. Deliberately avoided: the detail lives
  on the object itself (via `InfoView`) or on the report method (via
  `TextReport`).

## Persistence entry points

The store seam is reachable from the live execution paths:

```python
engine = EvalEngine(sim_identity=lambda sim: ("sim", "snapshot_103", "halo_0"))
engine.run(calc, sim, store=store)                    # direct engine
calc.run(sim, store=store, sim_identity=path_identity)  # calculator-level
calc(sim, store=store, sim_identity=path_identity)    # __call__ alias
```

The result is persisted automatically on a clean run keyed by
`(sim_identity(sim), root CalculatorSignature.hash)`. `store` and
`sim_identity` are both optional, so existing callers are unaffected.

To give a snapshot a location-independent identity, pass an explicit provider:

```python
from pynbodyext.calculate import ConfiguredSnapshotIdentity, LogicalSnapshotIdentity

logical = LogicalSnapshotIdentity("my-project", "simulation-A", "099")
identity = ConfiguredSnapshotIdentity({
    "/data/old/snapshot_099": logical,
    "/data/new/snapshot_099": logical,
})
result = calc.run(sim, store=store, sim_identity=identity)
# After changing data, explicitly refresh each affected calculation/subset:
refreshed = calc.run(sim, store=store, sim_identity=identity, reuse=False)
```

For separate data versions use `revision="r2"`; for interpretation changes use a
new `loader_id`. Construct a new provider when changing bindings or IDs. Without
revision there is no automatic invalidation of other cached computations/subsets.
The provider serializes with `to_json()` / `from_json()`; paths are local location
configuration, so moving to another machine requires rebinding its locations.
These APIs also work with `EvalEngine(sim_identity=identity)` and
`SampleTable(..., sim_identity=identity)`.

Loading back is the mirror image — `store.load(sim_signature=...,
calculator_signature_text=...)` deserializes the stored signature text into a
runnable calculator and returns a `Result` with `.calculator` attached (or
`None` on a miss):

```python
result = store.load(
    sim_signature=("sim", "snapshot_103", "halo_0"),
    calculator_signature_text=record.calculator_signature_text,
)
if result is not None:
    rerun = result.calculator.run(sim)   # the recipe is rebuilt from the text
```

A saved reference can be serialized and fetched after recomputation or reopening
its database. Keep the backend location alongside it:

```python
from pynbodyext.core.calculate.store import RecordRef

first = calc.run(sim, store=store, sim_identity=path_identity)
saved_reference = first.store_ref.to_json()
refreshed = calc.run(sim, store=store, sim_identity=path_identity, reuse=False)
original = store.fetch(RecordRef.from_json(saved_reference))  # first saved result
assert original.store_ref == first.store_ref
assert refreshed.store_ref != first.store_ref
```

## ORM associations (`pynbodyext.orm`)

An explicitly imported layer above `calculate`, with SQLAlchemy 2 models and an
`AnalysisRepository`. Importing/running pure calculators does not import this
package or require SQLAlchemy. For now use the existing `store` extra; the planned
full/lightweight distribution split is a separate packaging decision (ADR-005).

- **Simulation / Timestep** — logical project/simulation and timestep hierarchy.
  `Simulation.logical_id` is the provider's simulation_id;
  `Timestep.logical_id` is its timestep_id. Database PKs do not replace these IDs.
- **DataObject** — exact configured logical root/view identity, including optional
  revision, loader and any halo/subset suffix. Views link to their root input.
  Only logical identities are accepted; path/session fallback is not promoted.
- **DataLocation** — root input URI metadata. Aliases and historical reuse of a
  location by different revisions are allowed. Explicit loader bindings can load
  local absolute paths/prefixes for registered roots (ADR-010); no fingerprinting.
- **Recipe** — the existing stable CalculatorSignature hash and JSON DAG. ORM
  records the recipe; it does not implement a second graph or execution engine.
- **Run** — one invocation, with input/recipe FKs, named result backend, wall-clock
  start/end, status (`running`, `succeeded`, `failed`), error details and nullable
  `reused`. Cache hits also get separate Run records. Failed is terminal; retry
  means a new Run. Manual recovery may leave reuse unknown. `execution_context`
  is a versioned JSON snapshot committed with the initial running row, before
  loading/calculation: Python and installed pynbodyext/NumPy/pynbody versions,
  already-loaded native module information, selected URI and logical
  loader_id/revision, plus explicit `tool_versions` and `random_seed`. Missing
  package/native version information is null; collection never imports native.
  Native null means no loaded-module information was observed, not unavailable.
  This is the invocation environment, including on cache hits; the pinned
  payload's producer environment remains unknown. `execution_context` returns a
  fresh decoded copy with no setter, and repository finalization/recovery never
  rewrites it. Caller-owned direct session writes remain caller responsibility.
- **ResultLink** — one root output's pinned RecordRef JSON and result-store schema,
  associated with a Run. The backend name lives on Run. Arrays/named pipeline
  outputs remain in ResultStore; no payload copies in ORM.

```python
from pynbodyext.orm import AnalysisRepository
from pynbodyext.core.calculate.store import SQLAlchemyResultStore

payloads = SQLAlchemyResultStore("sqlite:///payloads.sqlite")
repo = AnalysisRepository("sqlite:///metadata.sqlite", stores={"results": payloads})
recorded = repo.run(calc, sim, sim_identity=identity, store_name="results")
run_id = recorded.run.id
stored = repo.fetch_result(run_id)  # exact saved version, even after refresh
assert stored.store_ref == recorded.result.store_ref
```

`sim_identity` is the configured logical provider from the example above. Metadata
and payload commits are separate, even with a shared database URL: commit running,
execute/save/reuse, then commit status and result link together. Raised execution
or store errors are wrapped in `RunExecutionError` with run_id and original cause;
collected calculator errors return a failed RecordedRun and its Result. Metadata
finalization errors raise `RunPersistenceError` with run_id, result_ref and result
when available. A running record remains inspectable; a commit whose outcome is
uncertain must be checked before recovery.

After fixing the metadata failure, explicitly validate/attach the known receipt:

```python
issues = repo.audit()  # does not decode scientific payload values
candidates = repo.unlinked_versions(run_id)  # candidates, never automatic choices
recovered = repo.finish_run(run_id, known_ref)  # caller explicitly selects a receipt
```

`finish_run` checks input hierarchy, recipe, backend and pinned version; its status
transition and link insertion are atomic. Same-reference completion is idempotent;
conflicting terminal transitions fail. Unlinked means unreferenced by this metadata
repository under that store name, not globally orphaned or safe to delete. Audit
reports unfinished/missing/invalid associations; neither operation mutates payloads.
No automatic retry, attachment, GC or distributed exactly-once promise (ADR-009).

`get_run` / `runs(input_id=..., recipe_hash=..., status=...)` load the immediate
input→timestep→simulation, recipe and result-link chain before closing their session.
For collection navigation/joins use `with repo.session() as session:` with the
exported mapped classes and ordinary SQLAlchemy `select`. Use an explicit transaction
for intentional ORM writes. SQLite foreign keys/unique constraints are enforced.
The repository does not scan all external store data or promise concurrency retries;
concurrent registration may raise a uniqueness conflict. SQLite is tested; other
SQL dialects need validation; conditional UPDATE RETURNING needs SQLite 3.35+.
Complete particle-container hierarchy, RunStep and MergerTree models remain later increments.

### Loading registered inputs

`AnalysisRepository(..., loaders={loader_id: callable})` copies explicit, trusted
project-code bindings. Each callable accepts a local absolute path/prefix and
returns a root `pynbody.snapshot.SimSnap`. Pre-bind parameters in project code;
reopen with equivalent bindings. There is no implicit default loader, persisted
callable, automatic configuration fingerprint or directory discovery. Changing
interpretation requires a new loader_id. Revision remains optional/manual.

```python
import pynbody
from pynbodyext.calculate import LogicalSnapshotIdentity
from pynbodyext.orm import AnalysisRepository

repo = AnalysisRepository(
    "sqlite:///metadata.sqlite",
    stores={"results": payloads},
    loaders={"default": pynbody.load},
)
data = repo.register_input(
    LogicalSnapshotIdentity("my-project", "simulation-A", "099"),
    location="/data/simulation-A/snapshot_099",
)
recorded = repo.run_input(calc, data.id, store_name="results")
assert recorded.run.input_location.uri == "/data/simulation-A/snapshot_099"
original = repo.fetch_result(recorded.run.id)

# Inspect or load independently, without creating a Run:
registered = repo.get_input(data.id)  # hierarchy, parent and locations are loaded
loaded = repo.load_input(data.id)
result = calc.run(loaded.sim, sim_identity=loaded.sim_identity, store=payloads)
```

`get_input` returns detached metadata with timestep/simulation, parent and locations
loaded. **LoadedInput** bundles `sim`, `input`, `location` and `sim_identity`; that
provider retains the existing view/transformation identity rules. A normalized
filename returned by the loader is bound to the registered logical identity;
this declares equivalence, not proof of file contents. Invalid roots, subviews,
missing/local-relative filenames and active external transformations are rejected.

With multiple locations pass `location_id=registered.locations[...].id` to either
loading entry point; no location is guessed and no failure switches to another
copy. Local absolute file paths and multipart prefixes are supported, without an
`isfile` precheck. Remote/relative locations can remain metadata but cannot load.
Registered halo/subset objects cannot be reconstructed from their index digests;
root-only loading is the deliberate first increment.

`run_input` validates input hierarchy, selected location ownership, bindings and
stable recipe before creating a Run. It then commits running **before loader IO**,
without holding a metadata transaction during loading/calculation. Loading or
returned-object validation errors are recorded as failed and raise
`RunExecutionError` with run_id and the original cause. Metadata failure follows
`RunPersistenceError` and explicit recovery, as above. `load_input` alone propagates
errors and creates no Run. Even cache hits load in this increment.

**Run.input_location_id / input_location** records the selected source relation;
existing `run(calc, sim, ...)` leaves it null because it did not perform a managed
load. This relation is not a historical copy of mutable DataLocation metadata.
Run queries eagerly load it; reference validation/audit check ownership. Binding
loaders never imports ORM into the calculator layer. No new dependency or legacy
schema migration is provided (ADR-010).

## Architecture invariants (each one is a test)

These are the rules a change has to keep, and the test that says so.  When a rule
gets in the way, change the rule deliberately and record the decision rather than
working around it silently.

| Invariant | Test |
|-----------|------|
| A `core/calculate` module imports only from the layers below it | `tests/test_layering_contract.py` |
| Any package module imports only from its own layer or a lower one, and every module has a declared layer | `tests/test_package_layers_contract.py` |
| The image layer knows the calculator layer only in `plot/image/data.py`, and only for typing | `tests/test_image_boundary_contract.py` |
| `bins/` reaches `pynbodyext.plot` only inside functions, so the calculator import stays matplotlib-free | `tests/test_bins_import_weight.py` |
| Only `bins/_pynbody_render.py` calls `pynbody.sph._render`, against a pinned upstream signature | `tests/test_pynbody_render_contract.py` |
| Every node kind encodes, decodes and prints — and the encoder's kinds are exactly the printers' handlers | `tests/test_signature_roundtrip.py`, `tests/test_calculate_contract.py` |
| Run options are declared once, and a misspelt override fails by name | `tests/test_calculate_contract.py` |
| The three tree renderers (dependency, node, execution) share one walker and one truncation wording | `tests/test_calculate_render.py` |

## Image layer (`pynbodyext/plot/image/`)

The vocabulary for turning a 2-D map into a figure. Code lives in
`pynbodyext/plot/image/`. Every operation is a free function over a plain `numpy`
2-D array, and the same operations are methods of `ImageData`, which is the
object a caller normally holds.

- **`ImageData`** (`plot/image/data.py`) — a 2-D array plus the metadata needed
  to display and measure it. Geometry is per axis: `x_edges`/`y_edges` are the
  bin edges (unevenly spaced allowed — logarithmic, quantile, or explicit) and
  `extent` (`(xmin, xmax, ymin, ymax)`) is shorthand for evenly spaced bins.
  Metadata is per axis too: `x_units`/`y_units` may differ and `x_label`/`y_label`
  name the axes, while `label`/`units` describe the values (and go on the colour
  bar). `ImageData.from_bins(bins2d, query)` reads one query of a 2-D
  `BinNDResult` together with its grid, units and axis names; `.with_data(...)`
  carries the metadata through a processing chain; `.pixel_size` is the
  `(dy, dx)` needed to express a kernel width in physical units, and refuses when
  a direction has bins of unequal width. Display accordingly:
  `.display.imshow()` for evenly spaced bins, `.display.pcolormesh()` for arbitrary
  edges, `.display.draw()` to let it pick.
  `ImageData` is a single class (no capability base classes). Each family is a
  **view** — `SmoothOps`, `PsfOps`, `ComposeOps`, `AdaptiveOps` — reached as a
  property of `image.process`: `image.process.smooth.gaussian(fwhm=2)`,
  `image.process.psf.convolve(fwhm=3)`, `image.process.noise.poisson(...)`,
  `image.process.compose(other)`, `image.process.adaptive.bin(signal)`.
  Display is the other entry point: `image.display.normalize/to_rgba/draw/imshow/
  pcolormesh/contour/add_colorbar`. Every operation has exactly one spelling.
- **`ImageDataView`** (`plot/image/ops.py`) — "an `ImageData` seen from one angle":
  it holds the image and forwards its data and metadata (`data`, `shape`, `ndim`,
  `extent`, edges, centers, uniformity, `pixel_size`, units, labels) plus
  `add_colorbar`. Every view of an image derives from it, so nothing forwards the
  same property twice. **`ImageOps`** extends it with the two things a capability
  family needs: `derive()` (return a new image with the operation recorded) and
  `kernel_scale()` (pixel size, or `None` on uneven bins). `AdaptiveMap` derives
  from it directly — it *is* a view of its painted map, not a capability.
- **`register_ops`** / `ImageData.register_ops("tessellation", TessellationOps)`
  adds a capability family through the registry, so a new processing stage is a new
  module and a registration — no base-class list to edit
  (`ImageData.views()` lists what is registered; an instance's `.ops` is a different
  thing — the chain of operations already applied to *that* image).
- **`ImageOp`** (`plot/image/ops.py`, next to the views and the registry) — one
  recorded step of a chain (`name`, `params`), appended to `ImageData.ops` by every
  processing method and summarised by `repr(image)`.
- Methods that produce an image return a new
  `ImageData` — geometry and units intact — with the call appended to
  **`ops`** (`tuple[ImageOp, ...]`, `ImageOp(name, params)`), which is what
  `repr(image)` summarises.
- **Two entry points: `display` and `process`.** `ProcessOps` (in
  `ops.py`, the module that also holds the views and the registry) exposes the five
  families that change or measure the values; `DisplayOps` exposes the seven
  operations that show them. So `dir(image)` is short and unambiguous —
  `image.process.smooth.gaussian(...)` versus `image.display.imshow(...)` — and
  `ImageData` itself carries only what it *is*: values, geometry, units, labels,
  `ops`, `with_data`, `from_bins`/`from_bins_array`/`as_image`, the registry, and
  those two properties.
- **Colour bars** are docked to their panel, not floated: `add_colorbar(artist |
  image, loc=…)` (also `image.display.add_colorbar(…)`, and `colorbar=True|loc` +
  `colorbar_kwargs` on `display.draw`/`imshow`/`pcolormesh`/`contour`) uses
  `make_axes_locatable` with a shared divider per panel, so `size="5%"`,
  `pad=0.05`, `label_pad` and `tick_label_size` are under the caller's control and
  several bars (e.g. the two sides of `compose.imshow`) coexist.
- **Scales are explicit and shared.** `display.draw`/`imshow`/`pcolormesh`/
  `contour` and `add_colorbar` take `log=True` (or a `norm=`), so a map that spans
  decades — a density, say — is drawn and labelled on one scale: `log=True` builds a
  `LogNorm` over the positive values (or explicit `vmin`/`vmax`), contour levels
  become geometric, the colour bar comes out logarithmic on every side, and asking
  for a scale a drawn artist does not have is an error rather than a mislabelled
  bar. `symmetric=True` (zero-centred) and `log=True` are mutually exclusive.
  Matplotlib's *string* spellings are understood too — `norm="log"` behaves exactly
  like `log=True` (it is resolved to a norm instance by `cmaps.as_norm`, so the
  levels become geometric as well), and `norm="symlog"`, `"logit"`, `"asinh"` and
  `"linear"` are accepted where this matplotlib has them; an unknown name is an
  error rather than a silently linear scale.
- **Every user-facing method documents its own arguments.** The free functions stay
  the detailed reference, but a method may not answer a parameter with "as in
  :func:`the_function`" — it lists its own `Parameters`, `Returns` and an `Examples`
  block, and names the function that implements it. `tests/test_plot_image_docs.py`
  enforces that, rejecting pointer-style parameter entries, because the methods are
  what users read.
  A `MapStyle` may carry a `norm` too (or a `stretch`, not both), and
  `MapStyle.norm_for(data)` — used by `compose.imshow` — returns a norm matching the
  stretch exactly (`FuncNorm` over the data range, per `stretch_functions`), so a
  stretched map's colour bar no longer shows a linear scale.
- **Annotations name real types.** The vocabulary is `plot/image/_types.py` —
  `MapLike` (an image, a binned array or a plain array), `MaskLike`, `KernelWidth`
  (a number, a pynbody unit, or a `(y, x)` pair) — plus `numpy.typing.ArrayLike`,
  the `matplotlib` artist classes and `UnitLike`; a signature never answers with a
  bare `Any` (it stays only inside a container, `dict[str, Any]`, and on forwarded
  `**kwargs`, where it is honest). The bridge is typed on both sides:
  `BinsArray.image` is an `ImageData`, and `ImageData.from_bins_array` takes a
  `BinsArray`. `tests/test_plot_image_typehints.py` keeps both from drifting.
  An `Artist` is anything a drawing method returns — an image, a mesh, a contour
  set, a scatter `PathCollection`, or a bar — so a scatter profile can go straight
  to `add_colorbar` without a type error. The annotations are deferred (PEP 563)
  and the names they use are imported under `TYPE_CHECKING`, which the lint config
  requires (`ruff` `TC`); resolving them therefore takes
  `plot/image/_types.resolve_type_hints`, which imports what they name — matplotlib's
  drawing stack, the calculator — only when asked. That is deliberate: importing
  `pynbodyext.plot.image` stays free of `matplotlib.pyplot` and of the calculator
  (about 350 ms and 270 ms), and the tests pin it.
- **Contours** are a display verb alongside `imshow`: `image.display.contour(ax=ax,
  levels=…, filled=…)` (and `AdaptiveMap.contour`) draws on the bin centres, so it
  is correct for uneven bins and overlays an existing image when passed the same
  axes. The artist is a `ContourSet`, which `add_colorbar` accepts directly.
- **Orientation is type-driven.** A binned array is `(x, y)`; an image is
  `(row=y, column=x)`. `as_image(bins2d["mass.sum"])` — and every function that
  takes a second map (`adaptive.bin(signal)`, `compose(other)`, `adaptive_bin_map`)
  — recognises a binned array by its `bins`/`grid` attributes and transposes it, so
  `bins.s["count"]` means what it looks like. A *raw* array cannot be told apart
  from an image, so when its shape is the transpose of the image's, the error says
  exactly that instead of failing obscurely (silent when the grid is square).
  The rule holds for the **methods** (signals, noise maps, masks, the other side of
  a composite): they accept an image-like input. The free functions underneath take
  arrays you have already oriented, which is the array-level contract.
- **`BinNDResult.imshow`** (`core/calculate/bins/plot.py`) — a one-line bridge:
  it builds `ImageData.from_bins(self, query)` and calls `.display.draw()`. A bin grid is
  `(x, y)` and an image is `(row=y, column=x)`, so `from_bins` transposes; uneven
  bins therefore come out as a `pcolormesh` rather than being forced onto a
  regular pixel grid. No image drawing is implemented in the calculator layer.
  The same bridge is reachable from the array itself: `bins2d["mass.sum"].image`
  (a `BinsArray` property) hands back the `ImageData`, so
  `bins2d["mass.sum"].image.display.draw()` or `.display.contour(ax=ax)` need no
  further import.
- **Adaptive bin map** (`AdaptiveMap`, `plot/image/adaptive.py`) — a partition of
  a map into regions of comparable *capacity* made with PowerBin (centroidal
  power diagrams, the successor of Voronoi binning). The capacity is the
  `signal` map, or `(S/N)²` when a `noise` map is given. `.bin_num` assigns every
  binned pixel to a region, `.bin_value`/`.bin_capacity`/`.bin_count` describe the
  regions, and `.value` paints the region values back at full resolution with
  unbinned pixels left non-finite. *Not* a smoothing operation: it never mixes
  pixels into an average they do not belong to.
  It carries the geometry of the grid it was binned *from* (`x_edges`/`y_edges`,
  per-axis units and labels), and `.to_image_data()` hands that on; `.display.draw()`
  draws even grids with `imshow` and uneven ones with `pcolormesh`.
- **Map mask** (`plot/image/compose.py`) — a pair of complementary soft masks in
  `[0, 1]` split by a line at `line_angle`, with a transition ramp whose width is
  a fraction of the image diagonal (`width`, so the look is resolution
  independent; `width <= 0` gives a hard split). Drives `blend_images`,
  `blend_stack` and `compose_maps`, which stitch two maps drawn with different
  colour maps into one RGBA image whose alpha is the coverage.
- **`sauron_cmap`** (registered as `"sauron"`/`"sauron_r"`, `plot/image/cmaps.py`)
  — **Michele Cappellari & Eric Emsellem's SAURON colormap** (Leiden, 2001): black
  → blue → cyan → **green on zero (0.50)** → yellow → red → light grey (0.9, not
  white). `SAURON_POSITIONS` (11 control points, symmetric about 0.5) and
  `SAURON_RGB` are the published table, and the map is built from it exactly as the
  reference implementation (`plotbin/sauron_colormap.py`, Cappellari 2014-2024,
  https://purl.org/cappellari) does — a test asserts the two look-up tables are
  identical whenever `plotbin` is installed, so `cmap="sauron"` means the same
  colours whoever registered it first. This is the map used for the stellar
  velocity fields of the SAURON and ATLAS³D surveys; cite that lineage, not this
  package, for the colours. `vel_cmap`/`vel_cmap_r` are aliases. `to_rgba` maps a
  2-D array plus a stretch to RGBA.
- **PSF** (`plot/image/psf.py`) — the observational point-spread function:
  `gaussian_psf(fwhm=..., e=..., theta=...)` builds a kernel, `convolve_psf`
  applies it (forward direction, for comparing a model to data), and
  `wiener_deconvolve` / `richardson_lucy` invert it (visualisation only — they
  amplify noise).
- **Noise** (`plot/image/noise.py`, `image.noise.*`) — what the map looks like once
  it is *counted*. `noise.gaussian(sigma=… | snr=…)` adds Gaussian white noise (a
  scalar or per-pixel σ map, or σ = |data| / snr), and
  `noise.poisson(exposure=…, background=…)` samples counting noise at a given
  exposure (bigger = deeper; variance `(data + background) / exposure`). Both are
  seeded (`rng=`), leave non-finite pixels alone, honour a mask, and record what
  they did in `.ops`.

## Newly implemented contracts

- **`ExternalFileRef`** (`store/external.py`, exported by `core.calculate.store`)
  — frozen descriptor of a caller-owned absolute local path or file URI (empty or
  localhost authority), nonempty `media_type`, nullable nonnegative integer
  `size_bytes` and lowercase 64-hex `sha256`. Strict descriptor JSON uses
  `pynbodyext.result.external-file-ref/v1`; ValueCodec uses `external_file_ref_v1`.
  Construction/JSON/codec operations never access the file, and ordinary strings
  remain strings. Explicit `inspect(verify_checksum=False)` returns a frozen
  `ExternalFileInspection(status, reason, checksum_verified)`; statuses are
  available/missing/changed, access errors propagate, and SHA256 reads occur only
  when requested with a supplied digest. Available without a verified checksum
  does not establish content equality or readability. ResultStore versions pin
  descriptors only: fetching an old version cannot restore overwritten external
  bytes. Callers maintain distinct/versioned paths for reproducible history; no
  automatic copy/delete/externalization is provided. See ADR-013.
- **`PostprocessorBase`** (`nodes/postprocessors.py`) — a value-consuming property
  role exported by `pynbodyext.calculate`. Decorated scientific subclasses
  implement `process(value)`; the adapter owns `calculate(sim, params)` and
  delivers `params.source` unchanged, including units. Its required keyword-only
  `source` is one Calculator dependency declared through `Param`; raw `Result`,
  `RecordRef`, and extra dynamic parameters are rejected. Other dataclass fields
  are static configuration read through `self`. The existing Calculator DAG,
  property/runtime lifecycle, signatures, and result-store receipts remain the
  owning seams; there is no second executor or signature node kind. See ADR-014.
  On an exception the adapter invokes cleanup for its own applied bound scope,
  respecting revert policies and nonrevertible handles; it does not invoke an
  inherited ancestor's cleanup callback. Native compound handles retain their
  existing cascading revert semantics, and cleanup failures are logged while
  preserving the original exception. Transform/filter application failures before
  entering the adapter and generic property success cleanup are outside this path.
- **Catalogue** — scientific catalogue metadata for a validated snapshot root.
  Its canonical typed identity includes that root's logical signature plus
  `catalogue_id`, `finder`, optional `revision` and `loader_id`; None revision is
  a real identity component, independent of the root data revision. Registration
  requires no loader binding or location and performs no snapshot/payload IO.
- **CatalogueObject** — halo/subhalo metadata unique by catalogue, kind and
  `typed_object_key`. Built-in string/int/bool/finite float keys use
  `sim_identity_key`: `1`, `"1"`, `True` and `1.0` differ; signed floating zeros
  agree. Only subhalos may have a host, which must be a halo in the same catalogue;
  an unknown host can remain None. Physical hosts do not change `DataObject.parent`
  or existing snapshot-view/provider signatures and do not imply loadable objects.
- **Selection / SelectionSpec** — a versioned root-relative replay definition
  (`indices`, named `family`, or stable constructible FilterBase signature), kept
  separately from its materialized view DataObject. Registration requires a
  validated untransformed LoadedInput root and materializes once; expected_count
  records the ROOT particle count. Canonical descriptors distinguish definitions
  even when they select the same view. Indices are strict built-in integer lists,
  ordered and duplicate-preserving within root family-block ordering; cross-block
  reordering/interleaving raises ValueError without sorting. For a dm=2/star=2
  root, `[1,0,1,3,2,3]` is preserved, while `[3,1,3,0]` is rejected. This approved
  limit follows pynbody's contiguous-family views. Bounds and family presence are explicit.
  Filters run through the existing DAG in original coordinates without persistent
  mask caching; scoped transforms are rejected and root identity is checked again
  after evaluation. See ADR-016.

### Root-input job manifests

**JobSpec** (`orm/jobs.py`) is a frozen root recipe/address with strict schema
`pynbodyext.orm.job/v1` and exactly six JSON fields: `schema`,
`calculator_signature_text`, `input_id`, nullable `location_id`, `store_name`,
and `reuse`. IDs are positive built-in integers (never bool); store_name is a
nonempty built-in string and reuse is a built-in bool. IDs belong to the explicitly
supplied metadata repository; this format does not migrate IDs between databases.

```python
from pynbodyext.orm import JobSpec
from pynbodyext.properties import ParamSum  # explicit import registers the type

job = JobSpec.from_calculator(ParamSum("mass"), data.id, store_name="results")
manifest = job.to_json()
# In another process, reopen the same metadata repository and bind loaders/stores:
replayed = JobSpec.from_json(manifest).execute(repo)
```

Construction/`from_json` rejects unknown/missing/duplicate fields, unsupported
schemas, invalid built-in types and unstable/nonconstructible recipes. Before any
calculator constructor, the whole nested signature is checked against explicit
registered names/aliases; enum parameters require already-loaded Enum classes.
Recorded calculator versions must match their registered classes before any
constructor runs. Rebuilt recipe identity must match after normalizing registered
aliases and numerical equivalence; parameter drift fails before loading a snapshot.
An explicit alias is sufficient even when its class has no `signature_name`:
rebuilt canonical names are recognized only through a job-local mapping of the
already-vetted types, without registering those names globally. Incoming canonical
class paths still require explicit registration.
Project code must import/register required calculators first. A class-path-only
custom calculator is excluded even when importable. General signature decoding's
import fallback is unchanged; the job boundary never imports a manifest-selected
application module. General signature version warnings remain unchanged outside
JobSpec. Registered constructors and `from_signature_payload` are
trusted project code, so this is not a sandbox for hostile registered plugins.

Only default RunOptions are accepted across the graph, checked before signature
conversion can discard live options and again after reconstruction. The manifest
contains no loader/store objects, credentials, backend URL, callable, live snapshot
or executable pickle; scientific parameter strings are retained without secret
redaction. `execute(repo)` rechecks type availability and root-only input, then
delegates to `repo.run_input`. Missing bindings or invalid input/location fail
before loader IO, calculation or Run creation. Selections/views remain excluded
from v1, including those supported by the separate `run_selection` API.

The existing runner owns subsequent lifecycle/errors. Each execution creates a
new Run, even on a cache hit with the same pinned ref; `reuse=False` saves a new
version while prior references retain historical values. Loader/calculation/store
and metadata failures retain RunExecutionError/RunPersistenceError receipts and
explicit recovery. No new executor, automatic binding or migration (ADR-017).
