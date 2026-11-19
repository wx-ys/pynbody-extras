# Design review of particle membership and result queries

Draft, 2026-11-19. Source baseline `c56120f` includes the ADR-021 implementation. [中文](database-member-queries-2026-11-19.zh-CN.md)

Implementation completed on 2026-11-20. The [Database guide](../database.md) documents the shipped API. [ADR-023](../adr/0023-selection-observations.md) and [ADR-024](../adr/0024-saved-members-and-output-discovery.md) record the runtime and storage contracts.

An analyst might start at a galaxy and look for its stars inside an aperture, or start at a published value and ask which particles, coordinate frame and calculation produced it. Use `obj.members[selector]` to browse registered members, `obj.results.where(...)` to discover saved outputs, and a pinned result reference to trace a particular measurement. Catalogues, single runs and Study already have public interfaces. The member collections, reverse uses, result query predicates and selection-frame reload below are proposals.

## From a subhalo to stars inside a disk

Assume the catalogue records the host relationship and can supply exact membership for this subhalo. The example inspects the sole origin after the first saved disk measurement.

```python
from pynbodyext.filters import Disc
from pynbodyext.properties import AngMomVec, KappaRot
from pynbodyext.transforms import AlignVec

sub = db.snapshots["snapshot_099"].catalogues["groups"].halos[42].subhalos[7]
stars = sub.members.register("star")
face_on = AlignVec(AngMomVec())
disc = Disc("10 kpc", "1 kpc")
krot = KappaRot().transform(face_on).filter(disc)

result = krot.run(stars)
disk_stars = stars.members[disc]
disk_stars.origin.context
disk_stars.used_by
```

`register` explicitly establishes a membership in the source object's snapshot frame by default. It can reuse an identical confirmed observation. A successful database calculation automatically links the filtered views it actually used. The example evaluates the disk filter once. After reopening the database, continue from `sub.members["star"]`. The string `"star"` and `FamilyFilter("star")` address one selector definition, as do their dark-matter equivalents. Custom filters use their full persistent signatures.

Brackets match registered origins directly under the source and group matches by exact particle input. No match reports absence and available registration information. Distinct particle inputs produce an ambiguity with candidates. **Several origins for one set of particles still allow a direct `.load()`.** For example, a user can recover the same stars selected in several galaxy orientations without first choosing a frame. `.origins` lists the matching origins and `.origin` is available when unique. A verified empty selection is also a valid member handle.

`members.where(...)` yields one item per distinct origin. `.one()` requires a unique match. Select an origin to inspect its context or replay its coordinates. Without a pinned origin, `.used_by` combines uses of the matching origins and identifies the origin of each use. Further member navigation searches only the origin branches represented by the handle. Explicit registration defaults to the exact input in snapshot coordinates. A nested origin path is recorded only when its parent origin is established.

```python
stars.members.where(filter=Disc)
chosen = stars.members.where(filter=disc, transform=face_on).one()
native_particles = chosen.load()
disk_particles = chosen.load(frame="selection")
```

`.load()` reads the corresponding snapshot revision at the saved, ordered root-relative indices, preserving duplicates and the captured unit convention. With `frame="selection"`, an independently loaded view receives the recorded operations that were active at selection time. Rotations, position shifts and velocity shifts with known targets, units and parameters are suitable first cases. Report the unavailable step when replay is incomplete. Native-frame membership can still be loaded. Both forms require the loader to open the original input revision. Overwriting the source file in place can destroy particle values that indices and matrices alone cannot recover.

## From a saved output to its members

```python
hits = db.results.where(objects=sub.host.subhalos, transform=face_on)
all_discs = stars.results.where(filter=Disc)
same_aperture = stars.results.where(filter=disc)
same_frame = stars.results.where(context=chosen.origin.context)
```

- `objects` restricts the direct inputs. The first line searches the host's subhalos. The example saved `krot` on `stars` and it appears in `stars.results`. Use `chosen.results` for calculations run directly on the selected disk stars. Use `chosen.used_by` for pinned outputs that consumed that origin, with output path and role.
- `transform=` matches the output recipe's entire ordered transform chain. One instance denotes one step. Use `(center, face_on)` for centering followed by alignment. `filter=` matches the output's direct filter expression. A class selects its root type and an instance selects the full expression. All predicates match the same output. `scope=` matches the full scope, retaining nested scope boundaries, order and revert policy.
- `uses=` searches any recipe occurrence and returns its path and role. For example, `uses=Sphere` can find a negated sphere, a sphere inside a compound aperture, or a sphere used to determine an axis. The hit identifies the operator and use. `where(filter=Sphere)` selects outputs directly filtered by an expression rooted at Sphere.
- `context=` matches the observed coordinate context. For member queries this is the context at selection time. For result queries it is the context of the final measurement. Comparison includes root input revision, affected targets, units and the ordered operations actually applied. Galaxies aligned by the same method can have different rotation matrices. Outputs with unknown frames remain discoverable by recipe with their recording status visible.

Result discovery defaults to metadata for current confirmed versions, including the scientific object locator, full calculator signature, output path, pinned reference and context availability. Listing reads metadata only. A standalone result yields one row. A named pipeline leaf has a key path such as `("kinematics", "krot")`. Different outputs of one saved pipeline share its version reference. An explicit `output=()` selects root results, including pipeline roots. Use `calculator=` with a complete calculator or stored signature to select an exact recipe.

To redraw an old figure, start at the original Study cell. Here `profiles` is a successfully saved pipeline with a `krot` output that consumed `disc`.

```python
from pynbodyext import Database, StudyTable

frozen = StudyTable.load("paper-sample.json")
cell = frozen.cell(frozen.objects[0], "profiles")
with Database("analysis.sqlite", read_only=True) as db:
    hit = db.results.where(ref=cell.ref, output=("krot",)).one()
    prior = hit.members.where(role="output_filter", filter=disc).one()
    value = db.fetch(hit.ref).value["krot"]
    particles = prior.load(frame="selection")
```

`ref=` pins the owning database and historical version. A later refresh still leaves the original origins reachable. A legacy result with a recipe alone exposes its value and signature and marks membership as unrecorded. If an output used the same filter more than once, the occurrence's DAG node path distinguishes its uses. The detached StudyTable file carries values readable on its own. Looking up the origins requires its database.

## Comparing subhalos with their hosts

For the question “how does subhalo stellar mass vary with host total mass?”, use subhalos as sample rows. The number of stars is a sample property. Even a starless subhalo contributes to the population denominator. Here `cat` is the selected catalogue and `targets` is a Study subsample with distinct, addressable inputs.

```python
from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamSum

total_mass = ParamSum("mass")
stellar_mass = total_mass.filter(FamilyFilter("star"))
sample = db.study(objects=targets, calculations={"Mstar": stellar_mass})
coverage = sample.collect(missing="keep")

roster = cat.subhalos.to_dataframe()
satellites = db.results.where(objects=cat.subhalos, calculator=stellar_mass, output=()).to_dataframe()
hosts = db.results.where(objects=cat.halos, calculator=total_mass, output=()).to_dataframe()
sat_refs = satellites[["object_locator", "ref"]].rename(columns={"ref": "stellar_mass_ref"})
host_refs = hosts[["object_locator", "ref"]].rename(
    columns={"object_locator": "host_locator", "ref": "host_mass_ref"}
)
pairs = roster.merge(sat_refs, on="object_locator", how="left", validate="one_to_one")
pairs = pairs.merge(host_refs, on="host_locator", how="left", validate="many_to_one")
```

`coverage` retains values and gaps in the fixed sample. Both left joins start from the full catalogue roster, keeping a row even if the subhalo measurement, host mass or particle binding is missing. The proposed catalogue `to_dataframe()` returns the scientific locator, host locator and membership availability. The same method on a result query extracts metadata and pinned references. Read the values with the corresponding `db.fetch(ref).value`. When given a collection of scientific locators, return one row per requested locator, recipe and output path, even where two rows reuse one value. Stellar mass summation and κrot on an empty population each follow their calculator's success or failure contract.

The separate satellite and host queries each pin the versions they observe. A refresh can occur between them. If a figure needs both populations observed at one cache instant and the inputs are distinct, include both populations and both calculators in one Study. Call `collect(missing="keep")` once and project the required rows and columns.

## Applicability across analysis workflows

| Scenario | Entry point | What affects the interpretation |
|---|---|---|
| 01 Isolated galaxy with one file and a root snapshot | `snap.members["star"]`, `snap.results` | Path inputs work directly and members are addressed relative to the root. |
| 02 Two catalogues each contain halo 42, with hosts and subhalos in one catalogue | `snap.catalogues[key].halos[42].subhalos[7]`, `sub.host` | The catalogue supplies the host. Table joins use a locator qualified by database, root, catalogue revision, kind and typed object number. |
| 03 Catalogue metadata without particles, or a subhalo known to have zero stars | Browse the catalogue, or open a verified empty member | A missing binding leaves membership unknown. A count of zero requires established family and membership information. The existing root-level `SelectionSpec.family` raises for an absent family. A verified empty set can be stored as empty indices. |
| 04 Two selections choose exactly the same particles, or a manual selection includes duplicate indices | `member.origin`, `member.load()` | An identical exact input can share indices and results. Each origin, order and multiplicity matters. Bare indices establish an exact input. A catalogue association needs supporting evidence. |
| 05 Measure κrot and Re in the same disk | `stars.results.where(transform=face_on)`, `disk_stars.used_by` | The recipes have separate addresses. Both outputs can use the same membership. The Re calculator defines the projection, weight or bandpass. |
| 06 Measure stellar mass on a subhalo, or total mass directly on its stars | `sub.results[stellar_mass]`, `stars.results[total_mass]` | Matching values can have distinct input and recipe keys. `stars.used_by` leads to measurements that consumed that stellar selection. |
| 07 Use stars for the orientation and all particles for the disk measurement | `AlignVec(AngMomVec()).measure_with(FamilyFilter("star"))` | Stars carry the incoming frame and the `transform_measure` role. The disk carries the resulting frame and the `output_filter` role. |
| 08 Use twice a measured radius as an adaptive aperture | `Sphere(2 * rhalf)`, or first select the stars | Both operands of one compound filter evaluate on one input. Selecting stars before measuring the radius changes its input. Store the radius recipe, resolved value and units. |
| 09 Compound, negated and exclusion masks | `members[a & b]`, `results.where(uses=Sphere)` | Record the intersection actually formed. A two-level member path requires an intermediate view that was consumed. |
| 10 Center, align and change apertures, or use different frames in sibling pipeline outputs | Whole-chain `transform=` or a fixed version plus `output=` | Link each filter to its consuming outputs. The restored parent frame, root versus subset target, and transforms still active all affect the observed context. |
| 11 A refresh changes the membership or the orientation for the same aperture | `db.results.where(ref=..., output=...)` | A historical plot refers to its original particles, matrix and version. Recipe and transform-method queries may still yield more than one origin. |
| 12 Measure Mstar for subhalos and join the host mass | `db.study(objects=halo.subhalos, calculations=...)` | Take one row per subhalo and apply the stellar filter in the column. Multiple starless subhalos keep distinct rows. Select one mass per host for a many-to-one left join on locators. |
| 13 Discover which sample members have saved results and fill gaps later | `db.results.where(objects=..., ...)`, `study.collect(missing="keep")` | Discovery returns saved hits. Collect preserves missing cells in a fixed sample of addressable inputs. A catalogue roster can also represent objects whose particle membership is unknown. |
| 14 Deferred manual save, automatic hit, failed refresh or interrupted save | `res.save(db)`, `db.fetch(ref)`, `db.recover(ref)` | Deferred saves use membership facts frozen during computation. A cache hit uses evidence from the producer. Failed attempts have status. Confirmed saved outputs acquire successful use links. |
| 15 A collaborator has only the database or exported table, without particle files or custom calculator code | Inspect metadata in a read-only database, read values with `StudyTable.load` offline | Browsing and identifying versions need no calculator reconstruction or particle reads. Particle reload still requires its loader. |
| 16 Periodic wrapping, slicing or rotation inside custom code | Inspect completeness in `member.origin.context` | Exact root indices can be reloaded independently. Wrapping may move different particles by different offsets. Custom operations need explicit reporting for selection-frame replay. |
| 17 Many apertures in a large catalogue or repeated catalogue IDs across timesteps | Restrict by snapshot and object collection, page indexed recipe matches | Each timestep has its own address. Share indices for consumed views to avoid copying masks into every result. Cross-snapshot particle tracking would require different records. |

## Two measurements that change the query semantics

**A family filter and an adaptive aperture.** Put six particles on the x axis. Dark matter occupies positions 10, 20 and 30, each with mass 10. Stars occupy positions 1, 2 and 6, each with mass 1. For `rhalf = ParamContain(frac=0.5, cal_key="rxy", parameter="mass")`, the projected half-mass radii on all particles and on stars are 14 and 2 in example units. These two calculations return stellar masses of **3 and 2**, respectively.

```python
from pynbodyext.calculate import Pipeline
from pynbodyext.filters import Sphere
from pynbodyext.properties import ParamContain

rhalf = ParamContain(frac=0.5, cal_key="rxy", parameter="mass")
star = FamilyFilter("star")
ap = Sphere(2 * rhalf)
mass = ParamSum("mass")
joint = mass.filter(star & ap).run(sim)
sequential = Pipeline({"mass": mass.filter(ap)}).filter(star).run(sim)
```

At this baseline `mass.filter(star).filter(ap)` has the same signature as `mass.filter(star & ap)`. To evaluate the second condition on the first selected population, start from its member handle or use a genuinely nested scope. Registration must follow the input actually supplied to each filter.

**Two sibling outputs of a pipeline.** A second six-particle snapshot starts with two dark matter particles followed by four stars at the x, y and z unit vectors and the origin in that order. Apply `Disc(3.0, 0.1)` to `sim.star` on separate fresh snapshots. The native disk selected root indices `[2, 3, 5]`. Aligning the x direction to the z axis yielded `[3, 4, 5]`. Both counts were three. Running the rotated output first on one live snapshot yielded `[3, 4, 5]` for both outputs. The root coordinates changed and a `Rotation` remained active. Reversing the order yielded `[2, 3, 5]` for the native output. This was observed on the ordinary Property/Pipeline path in the stated version.

Recipe discovery can therefore use signatures. Matching an actual frame requires observing active transforms at selection and output consumption, with untracked changes marked explicitly. For scopes requesting automatic restoration, acceptance requires each output order to agree with independent runs and includes a check of the caller's root coordinates. A calculation that explicitly retains a transform should record its lifetime and target. [Scope execution](../../pynbodyext/core/calculate/runtime/engine.py) · [Pipeline outputs](../../pynbodyext/core/calculate/nodes/pipeline.py) · [Transform measurement input](../../pynbodyext/core/calculate/nodes/transforms.py)

## Persistence and acceptance

| Data | Identity and responsibility |
|---|---|
| `CatalogueObject`, `DataObject`, `Selection` | Catalogue host, exact input and ordered particle selection for a root revision. The catalogue membership binding links the scientific object to an input. |
| Proposed `SelectionOrigin` | One immutable observation of a selection. It records the direct source locator (including its origin when nested), full filter, observed selection context, resolved parameters and exact membership reference. Multiple observations can share one definition. |
| Proposed `SelectionUse` | One use links a pinned result version, output path, DAG occurrence path and role to an origin. Multiple uses can share an origin. A cache reuse invocation may have its own Run. |
| Output recipe index and output observation | Index ordered scopes and dependency roles per named output of a Recipe. Record the observed measurement context and its completeness per confirmed version and output path, including outputs with no filter. |
| `ResultStore`, `Study`, `StudyTable` | The cache key is exact input × full calculator signature. A Study freezes ordered inputs and columns. A registered member can bind the existing exact-view locator for Study input. The v1 table rejects duplicate exact input rows. When several scientific parents must stay distinct, use the parents as rows or keep their locators in the sample roster. |

Capture root-relative indices when the filter is consumed, respecting existing family-block order and duplicates. Deferred saves can hold immutable descriptions and shared index blocks while releasing the live snapshot. Record transform order, affected targets, measurement sources, realized matrices or unit-bearing offsets, and replay capability. A missing step makes the corresponding context incomplete. Contiguous ranges, sparse indices or shared index blocks are internal storage choices. Reload must reproduce the sequence element for element.

In the baseline implementation, `cat.halos[number]` enumerates catalogue rows and matches the number in Python. A single lookup in a large catalogue therefore reads the collection's metadata. Query on catalogue, kind and typed number for direct access, and on `host_id` to enumerate hosted objects. Bulk browsing should query a collection as a set and bound previews to a page. [Catalogue collections](../../pynbodyext/database/catalogues.py)

Core produces database-independent observations. Database links them through the existing save confirmation and recovery path. Recovering the same receipt is idempotent, and concurrent registration of an identical origin resolves to one row. Many versions can share a registered origin. A metadata query pins its matching versions in one read snapshot so a concurrent refresh cannot replace an observed value. Existing signatures suffice to rebuild the recipe index in an older database. Historical members require actual recorded observations.

### Priority acceptance scenarios

1. After either `calc.run(obj)` or `run(sim).save(db)`, retrieve the member with a newly constructed equivalent filter. Disable the particle loader and filter execution for metadata browsing. On particle reload, compare with the original `get_index_list(root)` element for element.
2. Repeat both numerical examples above. Cover negation and combined filters, inherited scopes, roles before and after alignment, and retained versus automatically restored transforms. On an independent copy, verify replayed positions, affected velocities, units and target sets for supported rotations and shifts.
3. Distinguish two starless subhalos, an unbound catalogue row, a missing result and a failed refresh in the sample. A historical result keeps its original members. A hit adds no new producer observation.
4. Join subhalo and host results across two catalogue revisions and verify keys and cardinality. Measure SQL calls, payload decodes, particle loads and duplicate index bytes for read-only, legacy and large-sample cases. Query catalogue collections in bulk. Metadata discovery should perform zero particle loads and zero scientific payload decodes.

The proposed interfaces above are acceptance targets. Existing contracts are described in [ADR-021 at the reference revision](https://github.com/wx-ys/pynbody-extras/blob/c56120ffb31f0461450393eaa4039dfa25369da7/docs/adr/0021-database-studies-and-detached-tables.md), the [particle selection decision](../adr/0016-replayable-particle-selections.md) and [catalogue object tests](../../tests/test_database_catalogues.py).

The baseline validation covered fast tests for catalogues, Database Study, StudyTable, signature round trips and ORM selections. There were 139 passes, one skip for optional pandas in the project environment and one data test deselected by marker. In a separate environment with pandas, the example's two joins preserved all five synthetic roster rows, distinguished equal numbers across catalogues and rejected duplicate host keys.
