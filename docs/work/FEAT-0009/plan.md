# Catalogue-scoped object metadata implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox syntax for tracking. This document is preparation only; execute after the FEAT-0008 handoff and explicit follow-up from the coordinating agent. Do not infer a merge or permission to publish from this plan.

**Goal:** Register and query catalogue scientific identities and halo/subhalo host metadata without loading snapshots or changing existing calculation identities.

**Architecture:** Keep the SQLAlchemy models and `AnalysisRepository` as the owning ORM seam. A small `orm/catalogues.py` module owns typed catalogue/object identity and label/kind validation; repository methods own session transactions, hierarchy validation and detached results. Reuse `sim_identity_key` and `_validate_input_metadata`; no core module imports ORM.

**Tech stack:** Python 3.11+, SQLAlchemy 2, existing uv-managed `.venv`, SQLite behavioral tests and pytest.

## Approved contract and observed baseline

- Read `AGENTS.md`, `CONTEXT.md`, `docs/work/INDEX.md`, this item's `meta.json`, both locale specs and the latest handoff before implementation. ADR-015 records explicit approval; the design's §6.2 specifies object uniqueness as catalogue + kind + typed object key.
- Existing `AnalysisRepository.session()` returns sessions with `expire_on_commit=False`; write methods use `with self.session() as session, session.begin():`. Exceptions roll back the whole operation. Return through an eager query after the transaction, rather than relying on whichever relationships a newly constructed object happens to have populated.
- `_validate_input_metadata` already checks logical identity, canonical key, kind, project/simulation/timestep hierarchy and root/view parent invariants. `get_input` eagerly loads the relations it needs. Root validation must use both this validator and a requirement that the validated kind is `snapshot`.
- `sim_identity_key` supports more than catalogue object keys do: first require `type(object_key) in (str, int, bool, float)`, then canonicalize. This rejects None, tuples, NumPy scalars and subclasses; the helper rejects non-finite floats. Signed floating zeros intentionally share one identity.
- `tests/test_orm_associations.py::test_metadata_tables_have_no_payload_columns` asserts seven tables. The authorized two new tables change that contract to nine. Before editing that test, update the bilingual file scope/spec to record this necessary schema-expectation change; preserve its existing no-payload-column assertion. Do not remove, skip or weaken it.
- No migration, snapshot/catalogue loader, provider signature change, `DataObject.parent` change, `run_input` change or automatic catalogue population belongs here. The null catalogue revision is independent of the root input's revision.

## File map

| Path | Responsibility |
|---|---|
| `pynbodyext/orm/models.py` | Add mapped `Catalogue` and `CatalogueObject`, FKs, uniqueness/check constraints and named forward relationships. |
| `pynbodyext/orm/catalogues.py` | Pure catalogue identity, object-key and kind/label validation, using the existing typed identity helper. |
| `pynbodyext/orm/repository.py` | Six public register/get/list methods, metadata validation, eager query builders and transaction boundaries. |
| `pynbodyext/orm/__init__.py` | Export the two models; keep ORM opt-in. |
| `tests/test_orm_catalogues.py` | Typed identity, idempotence, hierarchy, detached return, SQLite reopen/join and atomic-failure behavior. |
| `tests/test_orm_associations.py` | Update only the seven-to-nine metadata schema expectation after documenting the requirement change. |
| `CONTEXT.md` | Explain catalogue identity, distinct host hierarchy, eager relation boundary and metadata-only limit. |
| `docs/work/FEAT-0009/{en.md,zh-CN.md,meta.json,handoff.md}` | Document the schema-expectation adjustment and implementation evidence in sync. |

## Task 1: Add the smallest failing catalogue registration slice

- [ ] Create `tests/test_orm_catalogues.py` using `pytest.importorskip("sqlalchemy")` before importing ORM. Use a temporary file-backed SQLite repository and a `LogicalSnapshotIdentity("project", "simulation", "099")`; dispose engines in fixture teardown. No snapshot fixture or downloaded testdata is needed.
- [ ] Add this initial behavioral test, keeping imports of new models inside test functions until public exports exist:

```python
def test_catalogue_registration_is_idempotent_with_null_revision(repo):
    root = repo.register_input(LogicalSnapshotIdentity("project", "simulation", "099"))
    first = repo.register_catalogue(root.id, "main", finder="finder")
    second = repo.register_catalogue(root.id, "main", finder="finder")
    assert first.id == second.id
    assert first.revision is None and first.loader_id == "default"
    assert first.root_input.id == root.id
    assert repo.get_catalogue(first.id).root_input.kind == "snapshot"
    assert [row.id for row in repo.catalogues(root.id)] == [first.id]
```

- [ ] Run `.venv/bin/python -m pytest tests/test_orm_catalogues.py::test_catalogue_registration_is_idempotent_with_null_revision -q`. Record the expected missing `register_catalogue` failure; an environment/import failure is not acceptable red evidence.

## Task 2: Implement catalogue identity and minimal repository methods

- [ ] In `catalogues.py`, define a `CatalogueObjectKey = str | int | bool | float` alias, `_nonempty_label`, `validate_kind`, `catalogue_identity_key` and `typed_object_key`. Keep this helper module independent of repository/models to avoid an import cycle. The identity-building code is:

```python
def catalogue_identity_key(root_signature, catalogue_id, finder, revision, loader_id):
    for name, value in (("catalogue_id", catalogue_id), ("finder", finder), ("loader_id", loader_id)):
        _nonempty_label(value, name)
    if revision is not None:
        _nonempty_label(revision, "revision")
    return sim_identity_key((
        "pynbodyext.orm.catalogue/v1", root_signature,
        catalogue_id, finder, revision, loader_id,
    ))

def typed_object_key(value):
    if type(value) not in (str, int, bool, float):
        raise TypeError("catalogue object keys require built-in str, int, bool or finite float")
    return sim_identity_key((value,))

def validate_kind(kind):
    if type(kind) is not str or kind not in ("halo", "subhalo"):
        raise ValueError("catalogue object kind must be halo or subhalo")
```

  Give these functions concrete annotations when implementing. `_nonempty_label` follows repository `_nonempty`: exact built-in string and nonblank `strip()`, without trimming or otherwise changing the original label. Do not duplicate the typed JSON codec or concatenate ambiguous strings. Empty strings are permitted as object keys: object keys are identifiers, not the nonempty catalogue labels.
- [ ] Add `Catalogue` with integer PK, `root_input_id` FK to `DataObject` (`RESTRICT`, indexed), non-null unique `identity_key` text, `catalogue_id`, `finder`, nullable `revision`, and `loader_id`. Declare `root_input` and an `objects` relationship. The canonical key includes the validated logical root signature and all four labels, including None revision; a nullable multi-column SQL unique constraint alone is insufficient.
- [ ] Add a private eager catalogue query builder selecting `Catalogue` with `root_input -> timestep -> simulation` and `root_input.parent` loaded. Do not eager-load unbounded object collections. Loading root locations is unnecessary for this metadata-only contract.
- [ ] Add `_validated_catalogue_root(session, root_input_id)` using an eager `DataObject` query within the registration transaction. Unknown IDs raise `KeyError`. Call the existing `_validate_input_metadata` and require snapshot kind; views and malformed roots raise `ValueError`. Do not call `_prepare_load`: no location or loader binding is required.
- [ ] Implement `register_catalogue(root_input_id, catalogue_id, *, finder, revision=None, loader_id="default")`. Validate labels, open one transaction, validate root, compute canonical key, select by that key, insert/flush if missing. Existing row metadata must agree with its canonical identity and root FK; reject conflicts rather than silently repairing. Commit then return `get_catalogue(row_id)`.
- [ ] Implement `get_catalogue(catalogue_pk)` and `catalogues(root_input_id)` via the eager query. Unknown get raises `KeyError`; lists filter by root FK and order by `Catalogue.id`. Follow existing list behavior: no matches returns `[]`. `get_catalogue` validates its root/labels/canonical identity; registration must also reject a corrupt existing row.
- [ ] Export `Catalogue` and rerun the initial test to green. Add parametrized tests showing separate rows for changes to root project/simulation/timestep/revision/loader and catalogue ID/finder/revision/loader, including None versus literal `"None"` revision. Repeating all inputs yields the same PK before and after SQLite reopen.

## Task 3: Add typed object and physical host behavior test first

- [ ] Add tests against the public methods below before implementing them:

```python
def test_typed_keys_and_host_hierarchy(repo):
    root = repo.register_input(LogicalSnapshotIdentity("project", "simulation", "099"))
    catalogue = repo.register_catalogue(root.id, "main", finder="finder")
    objects = [repo.register_catalogue_object(catalogue.id, key) for key in (1, "1", True, 1.0)]
    assert len({row.id for row in objects}) == 4
    assert len({row.typed_object_key for row in objects}) == 4
    assert [repo.register_catalogue_object(catalogue.id, key).id for key in (1, "1", True, 1.0)] == [
        row.id for row in objects
    ]
    child = repo.register_catalogue_object(catalogue.id, 1, kind="subhalo", host_id=objects[0].id)
    assert child.id != objects[0].id
    assert child.catalogue.id == catalogue.id and child.host.id == objects[0].id
    assert repo.get_catalogue_object(child.id).host.kind == "halo"
    assert [row.id for row in repo.catalogue_objects(catalogue.id, kind="subhalo", host_id=objects[0].id)] == [child.id]
```

- [ ] Run `.venv/bin/python -m pytest tests/test_orm_catalogues.py::test_typed_keys_and_host_hierarchy -q`; record missing object API failure.
- [ ] Add `CatalogueObject` with integer PK, indexed catalogue FK (`catalogue_fk`), `kind`, non-null `typed_object_key` text and nullable indexed self-FK `host_id`, all FKs `RESTRICT`. Use `UniqueConstraint("catalogue_fk", "kind", "typed_object_key")`, a halo/subhalo kind check, and a check requiring `host_id IS NULL OR kind = 'subhalo'`. Add `catalogue`, scalar self `host` (`remote_side`) and `subhalos` reverse collection; use explicit relationship FKs to avoid mapper ambiguity. No payload columns, external numbering-as-PK or `DataObject.parent` changes.
- [ ] Implement `register_catalogue_object(catalogue_pk, object_key, *, kind="halo", host_id=None)`. Validate kind/key, then within one transaction load and validate catalogue/root, resolve host if supplied, require registering a subhalo and host kind halo in the same catalogue. Unknown catalogue/host raises `KeyError`; invalid kind/ownership/halo-with-host raises `ValueError`.
- [ ] Select existing row by `(catalogue_fk, kind, typed_object_key)`. Identical host metadata returns the same PK; a different host, including None versus a supplied host, is a conflict and raises `ValueError`. Never silently attach, detach or reparent during idempotent registration. An unhosted subhalo is valid.
- [ ] Insert/flush only after all validation and return through `get_catalogue_object` after transaction exit. An existing row's host/canonical-key metadata is also validated. No automatic upsert/retry is needed; concurrent uniqueness violations may propagate as in existing registration methods.
- [ ] Implement `get_catalogue_object(object_pk)` with `catalogue` and scalar `host` eagerly loaded. Eager-load catalogue's `root_input -> timestep -> simulation` chain when that chain is promised in the new docs/tests; do not imply arbitrary recursive/collection traversal. Unknown PK raises `KeyError`.
- [ ] Implement `catalogue_objects(catalogue_pk, *, kind=None, host_id=None)`, sharing the eager query, filtering by catalogue and optional kind/host and ordering by object PK. `None` is an omitted filter, consistent with existing repository lists; filtering for unhosted-only objects remains a caller-owned SQLAlchemy query. Validate a supplied kind; no matching catalogue/object returns `[]`.
- [ ] Export `CatalogueObject`; rerun the object test to green. Add tests for same numbers in two catalogue namespaces; halo/subhalo same key coexistence; unhosted subhalos; exact metadata idempotence; host conflict preserving original; `-0.0` versus `0.0`; rejected None/list/tuple/bytes/NumPy/custom keys and NaN/infinities. Do not compare external key values with Python `==` to establish identity.

## Task 4: Exercise atomic validation failures and detached results

- [ ] Add parametrized root rejection tests adapting `tests/test_orm_loading.py::test_invalid_inputs_fail_before_loading`: unknown ID, registered subset and halo views, corrupt root hierarchy, kind, canonical identity key, malformed identity JSON and parent. Perform intentional corruption with `repo.session()` and a transaction. Each `register_catalogue` call raises before any catalogue write; the preexisting input metadata is not repaired or otherwise changed.
- [ ] Add atomic object rejection cases: unknown catalogue, unknown host, host in a different catalogue under the same root, host under another root, subhalo as host, halo-with-host and invalid kinds. Count `Catalogue`/`CatalogueObject` rows before/after each call and assert unchanged IDs/host assignments. Follow the failed call with a valid registration to show the repository is usable after rollback.
- [ ] Add label cases for empty/blank/nonstring catalogue ID/finder/loader, and revision values other than None or a nonblank built-in string. Verify no partial catalogue row exists. Root revision and catalogue revision are separate labels; a catalogue revision does not mutate the root.
- [ ] Assert `sqlalchemy.inspect(returned).detached` for register/get/list results. Access `catalogue.root_input` and its documented root hierarchy, and `object.catalogue` / `object.host`, after the internal session closes. Cover a fetched unhosted object whose `host` is None so unloaded null relations do not accidentally pass only constructor paths.
- [ ] Inject a `RuntimeError` on an `INSERT INTO pynbodyext_orm_catalogues` and separately on `INSERT INTO pynbodyext_orm_catalogue_objects` using the existing `before_cursor_execute` pattern. Always remove the listener in `finally`. Verify the failed transaction leaves no partial rows; retry succeeds with the same intended metadata. Exception propagation is sufficient; no new persistence exception family is required.
- [ ] Verify DB unique/FK/check guarantees using caller-owned transactions: duplicate canonical catalogue key with null revision, duplicate `(catalogue_fk, kind, typed_object_key)`, unknown FK, invalid kind and halo-with-host raise `IntegrityError`. Keep same-catalogue/halo-host validation at repository boundary unless an equally small portable DB constraint is demonstrably necessary; do not claim a plain self-FK enforces these scientific restrictions for arbitrary raw SQL writes.

## Task 5: SQLite reopen, joins and metadata-only boundary

- [ ] Register two catalogues, hosts and children; dispose the metadata engine, reopen the file with no stores or loaders bound, and assert all register/get/list results retain identity, host FKs and deterministic PK ordering. Repeat registration to prove idempotence survives process-local session state.
- [ ] Test ordinary mapped joins through the full hierarchy:

```python
with repo.session() as session:
    found = session.scalars(
        sa.select(orm.CatalogueObject)
        .join(orm.CatalogueObject.catalogue)
        .join(orm.Catalogue.root_input)
        .join(orm.DataObject.timestep)
        .join(orm.Timestep.simulation)
        .where(orm.Simulation.project_id == "project", orm.CatalogueObject.kind == "subhalo")
        .order_by(orm.CatalogueObject.id)
    ).all()
    assert [row.host.kind for row in found] == ["halo"]
```

  Add a self-join with `sa.orm.aliased(orm.CatalogueObject)` if SQL host filtering is needed; an ORM relationship access inside a live session alone is not a join test for host recovery.
- [ ] Bind a loader that fails if called and a separate empty result store. Run catalogue registration/get/list/reopen operations and assert loader-call list, runs, recipes and result-store refs remain empty. No root location is required. Inspect the metadata schema for the absence of payload/value columns.
- [ ] Update the bilingual spec's file scope and record the seven-to-nine schema expectation change before changing the old test. Update only its expected table count; retain all payload-column protections.
- [ ] Run `.venv/bin/python -m pytest tests/test_orm_catalogues.py tests/test_orm_associations.py tests/test_orm_loading.py tests/test_sim_identity_store.py tests/test_package_layers_contract.py -q`. The existing subprocess test also proves calculators remain usable without SQLAlchemy; do not introduce lower-layer ORM imports or new optional dependency rules.

## Task 6: Verify and hand off one increment

- [ ] Run `make check`, `make test-all` (public metadata API/schema behavior changes), and `make docs-check`. Report actual counts and failures; SQLite/Linux results do not certify other SQL dialects/platforms.
- [ ] Update `CONTEXT.md`: add catalogue/object vocabulary and exact eager relation promises; replace the statement that catalogues are a future increment while retaining future complete hierarchy/RunStep/MergerTree scope. State that catalogue scientific identity and physical hosts do not alter snapshot-view signatures or imply reconstructible selection/loading.
- [ ] Synchronize both locale acceptance checklists and execution evidence; set `meta.json.status` according to the verified local state; append red/green commands and results, files, limits and next step to `handoff.md`. Reflect any required index update through the existing work-item workflow.
- [ ] Request review as required by the repository architecture rule and current parent workflow. Do not claim the item merged or approved merely because ADR-015 approved the contract. Follow active authorization for committing/pushing/PR creation; this preparation task authorizes none of them.

## Self-review of this plan

All four acceptance criteria map to Tasks 2–5: namespace/revision/loader separation, typed keys/null uniqueness, atomic root/host rejection, and SQLite reopen/joins without IO. Immediate relation loading and deterministic lists have explicit tests. No task changes loader/provider signatures, core ownership or `DataObject.parent`. The only additional test-file change is the documented new-table schema expectation. This file records future work and does not claim red/green execution or runtime completion.
