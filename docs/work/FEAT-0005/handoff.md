# FEAT-0005 handoff

## 2026-11-09 — Codex / review and task preparation

- Done: inspected baseline 5df15dd and wrote bilingual acceptance contract; no runtime code changed.
- Next: read meta.json and prerequisites; implement only after applicable architecture review.
- Commands: baseline make check passed (1082 passed, 1 skipped, 4 deselected).
- Failing test: not added/run; this is a planned increment, not an implemented feature.
- Blockers: see contract review and prerequisites in the specification.
- Files: en.md, zh-CN.md, meta.json; source/test paths are listed in the spec.
- Env: use repository .venv; no global dependency installation.

## 2026-11-09 — explicit contract approval

- User approved all FEAT-0005–0012 contracts in this chat; decision recorded in ADR-011 before source edits.
- Local sequential implementation authorized; no repeat approval needed. Prerequisites verified locally, not claimed merged.

## 2026-11-09 — local implementation ready for review

- Decision: implemented accepted ADR-011 on locally verified BUG-0010/11/12;
  no claim of merge and no repeat approval. Work plan: `plan.md`.
- Changes: final defaulted `SampleRow.refs`; keyword-only optional `store_name`;
  actual pinned receipts captured after compute/cache hit/store fill; stale refs
  cleared on failed retries (including `errors="raise"`), failed fill reads, and
  successful runs without persistence. Filled nonempty cells/misses keep their
  existing state, and BUG-0012 recovery/error behavior remains covered.
- Exports: v2 JSON persists label and receipt JSON; v1 reads values with empty
  refs. Import validates pinned version, typed row identity, column hash and
  known column. Load never queries a store or resolves a backend label.
  `where`/`from_store` retain store_name. DataFrame identities and refs use typed
  identity keys, with ordered identity_keys, so duplicate display names survive.
- RED: `.venv/bin/python -m pytest tests/test_sample_refs.py -q` -> **14 failed,
  1 skipped** (missing refs/store_name/v2 schema; optional pandas absent).
  Log: `/tmp/feat0005-red.log`.
- Additional RED: `.venv/bin/python -m pytest
  tests/test_sample_refs.py::test_failed_fill_clears_stale_receipt_and_preserves_existing_diagnostics -q`
  -> **1 failed**, detecting a stale receipt after a backend exception.
  Log: `/tmp/feat0005-red-fill.log`.
- GREEN: `PYTHONPATH=/tmp/pynbody-test-libs .venv/bin/python -m pytest
  tests/test_sample_refs.py tests/test_sample_table.py tests/test_sample_recovery.py
  tests/test_result_versions.py -q` -> **81 passed**.
  Log: `/tmp/feat0005-focused-pandas.log`.
- Validation: `make check` -> lint/typecheck/docs passed, **1170 passed,
  2 skipped, 4 deselected**; `make test-all` -> **1174 passed, 2 skipped**.
  Logs: `/tmp/feat0005-check.log`, `/tmp/feat0005-test-all.log`.
  Both skips are optional pandas cases in the ordinary environment; both ran in
  the successful focused run with pandas 2.3.3 and its dependencies installed
  under `/tmp/pynbody-test-libs`. Project dependency files were unchanged.
- Files: `pynbodyext/core/calculate/sample.py`, `tests/test_sample_refs.py`,
  `CONTEXT.md` (SampleTable section only), this item's plan/locales/meta/handoff;
  regenerated `docs/work/INDEX.md` for review status. Existing tests and unrelated
  source/docs/design changes were preserved.
- Status: **review**, not done/merged. No commits, pushes or PR created.
- Limits: successful None still shares the existing missing-cell sentinel;
  historical refs require their explicit owning backend. SQLite/Linux exercised;
  other SQL dialects/platforms unverified. No migration, implicit dereference,
  automatic history GC or external publication added.
