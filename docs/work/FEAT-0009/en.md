# FEAT-0009: Register catalogue-scoped halo and subhalo metadata

## Context and goal

Distinguish catalogue scientific identity from snapshot-view identity and physical host hierarchy.

Sources: [revised design](../../design/v1-2026-11-09.md), [review](../../design/review-2026-11-09.md).
Prerequisites: No new work-item dependency; start from current main.

This contract was explicitly approved by the user on 2026-11-09 and recorded in [ADR-015](../../adr/0015-catalogue-object-metadata.md). Implement after its verified local prerequisites; do not claim they have merged.

## Scope and contract

Implement design §6.2 as metadata only. Add Catalogue(root_input_id,catalogue_id,finder,revision,loader_id) with a typed canonical identity key for uniqueness including null revision. Add CatalogueObject(catalogue_fk,kind,typed_object_key,host_id optional). Repository register/get/list methods are idempotent and return detached immediate relations. Object keys accept built-in string/int/bool/finite float using sim_identity_key; cross-type keys differ. A host must be a halo in the same catalogue; only subhalos may have a host. Catalogue root must be a validated snapshot DataObject. Do not alter existing provider signatures, DataObject.parent semantics or run_input.

Repository methods: register_catalogue(root_input_id, catalogue_id, *, finder, revision=None, loader_id="default") -> Catalogue; get_catalogue(catalogue_pk); catalogues(root_input_id); register_catalogue_object(catalogue_pk, object_key, *, kind="halo", host_id=None) -> CatalogueObject; get_catalogue_object(object_pk); catalogue_objects(catalogue_pk, *, kind=None, host_id=None). Validate nonempty string labels; catalogue identity includes root typed identity and all four labels. Changing finder/revision/loader creates another catalogue. The list methods order by primary key for deterministic results.

## File scope

- `pynbodyext/orm/models.py`
- `pynbodyext/orm/catalogues.py`
- `pynbodyext/orm/repository.py`
- `pynbodyext/orm/__init__.py`
- `tests/test_orm_catalogues.py`
- `tests/test_orm_associations.py` (approved metadata schema count only)
- `CONTEXT.md`

The approved two new metadata models intentionally increase the schema table count from seven to nine. Before adjusting test_orm_associations.py, retain its no-payload-column assertion and update only the expected approved metadata-table set/count. This is the specified schema addition, not suppression of a failure.

## Acceptance criteria

- [x] Two catalogues with identical halo numbers coexist; different revisions/loaders isolate identity.
- [x] Object keys 1, "1", True and 1.0 remain distinct; null revision duplicates are rejected/idempotent.
- [x] Unknown hosts, cross-catalogue hosts, halo-with-host and non-root catalogue parents fail atomically.
- [x] SQLite reopen and select joins recover hierarchy; no snapshot loading or payload writes occur.

## Execution and validation

- [x] Read AGENTS.md, CONTEXT.md, this bilingual spec/metadata/handoff and the listed tests.
- [x] Turn every criterion into behavioral tests; record red failures for missing behavior/API, not a broken environment.
- [x] Implement the smallest scoped change; record any core contract decision before changing it.
- [x] Run focused tests, then make check and make test-all; report exact results and limits.
- [x] Update both locales, meta.json and handoff; retain red/green evidence.
- [ ] Parent workflow: prepare one scoped PR using the template.

```bash
.venv/bin/python -m pytest tests/test_orm_catalogues.py tests/test_orm_associations.py tests/test_package_layers_contract.py -q
```

## Local implementation evidence

Catalogue identity uses the root logical signature and all catalogue labels through
sim_identity_key, including None revision. Objects are unique by catalogue, kind
and typed key; exact built-in strings, ints, bools and finite floats are accepted,
with signed floating zero equivalence. Catalogue loader/revision labels remain
metadata and do not change or bind snapshot providers.

Registration is idempotent for identical metadata. A different requested host,
including None versus a host, raises ValueError without reparenting. Unknown
catalogue/host/get IDs raise KeyError; root/host/kind/label conflicts raise
ValueError, unsupported key types TypeError and non-finite floats ValueError.
Unhosted subhalos are valid. List methods order by primary key, with None filters
omitted and no matching rows returning an empty list.

Detached catalogue results load root, root parent and timestep/simulation; object
results also load their catalogue/root hierarchy and scalar host. Collections and
further host relations require a caller-owned session. DB unique/FK/check
constraints protect keys and basic kind/host rules; same-catalogue halo hosts and
logical hierarchy validation are enforced by repository methods. Direct SQL
writes are caller responsibility.

Red: all **69 new cases failed** because register_catalogue did not exist
(/tmp/feat0009-red.log). Green: **69 passed** in 4.43s
(/tmp/feat0009-green.log). Focused catalogue, association, loading, typed identity
and package layer regressions: **163 passed** in 14.10s
(/tmp/feat0009-focused.log). Metadata-only tests bind a loader that must never be
called and verify empty result-store refs, recipes and runs; SQLite reopen and
mapped root/host joins recover hierarchy.


Full validation at the frozen FEAT-0009 source checkpoint: `make check` passed
lint, mypy (**121 source files**), **1424 passed, 2 skipped, 4 deselected** in
44.40s and docs validation (27 work items). `make test-all`: **1428 passed,
2 skipped** in 45.16s. The two skips require optional pandas. Logs:
`/tmp/feat0009-check.log`, `/tmp/feat0009-all.log`. Independent specification and
quality reviews reported no findings. Local status is review; no merge, commit,
push, PR or release publication is claimed. SQLite/Linux is verified; other SQL
dialects/platforms and catalogue loading are outside this increment. No migration
or dependency was added. The no-payload assertion remains intact while the
approved metadata schema count changed from seven to nine.

## Handoff boundary

No incidental DAG rewrite, database migration, automatic GC or release publication. Keep one reviewable increment per item; verify local predecessors before continuing without claiming remote merge. Commit/push permissions follow the active user authorization, not this template.
