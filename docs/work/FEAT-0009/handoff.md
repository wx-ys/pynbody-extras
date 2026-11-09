# FEAT-0009 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-015 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.


## 2026-11-09 — catalogue metadata implemented, locally verified and reviewed

- Done: added Catalogue/CatalogueObject metadata and six register/get/list methods.
  Canonical identity includes the validated root logical signature and every label,
  including None revision. Object keys use exact built-in scalar types and existing
  sim_identity_key; catalogue + kind + typed key namespaces halo/subhalo IDs.
- Detached relations: catalogue root/parent/timestep/simulation; object catalogue/
  root hierarchy and scalar host. Lists order by PK. Unknown IDs fail explicitly;
  identical registration is idempotent, conflicting hosts never reparent. Only a
  subhalo can have a halo host from the same catalogue; missing hosts are permitted.
- TDD: all 69 new cases failed first for missing register_catalogue (not environment
  breakage), then 69 passed in 4.43s. Logs: /tmp/feat0009-red.log,
  /tmp/feat0009-green.log. The bilingual spec approved the necessary old metadata
  schema expectation change 7 -> 9 before its test edit; no-payload assertion kept.
- Focused: catalogue/association/loading/typed identity/package layering suite,
  163 passed in 14.10s (/tmp/feat0009-focused.log). Real SQLite joins/reopen,
  null-revision uniqueness, raw FK/check constraints, corrupted metadata,
  unsupported keys, atomic validation and injected insertion rollback covered.
- Full at source freeze: make check passed lint, mypy (121 source files),
  1424 passed/2 skipped/4 deselected in 44.40s, docs (27 items); make test-all:
  1428 passed/2 skipped in 45.16s. Optional pandas explains both skips. Logs:
  /tmp/feat0009-check.log and /tmp/feat0009-all.log. Parent reported independent
  specification and quality reviews both passed without findings.
- Files: pynbodyext/orm/{catalogues.py,models.py,repository.py,__init__.py},
  tests/test_orm_catalogues.py (new), tests/test_orm_associations.py (schema count),
  owning CONTEXT.md catalogue section, this item's en.md/zh-CN.md/meta.json/
  handoff.md and generated docs/work/INDEX.md. plan.md remains preparation history.
- Limits: SQLite/Linux only; metadata-only, no snapshot/catalogue loading, payload
  writes, provider/run_input changes, dependency or migration. Scientific host/root
  restrictions are repository checks; arbitrary raw SQL writes remain caller-owned.
- Next: parent integration/publication workflow. Status review; no Git commands,
  commit/push/PR/release performed. Local FEAT-0008 predecessor verified, not merged.

```bash
.venv/bin/python -m pytest tests/test_orm_catalogues.py tests/test_orm_associations.py tests/test_orm_loading.py tests/test_sim_identity_store.py tests/test_package_layers_contract.py -q
make check
make test-all
make docs-index
make docs-check
```
