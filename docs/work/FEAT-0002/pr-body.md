## What changed

Implement FEAT-0002: explicitly configured snapshots keep the same cache identity after relocation. `LogicalSnapshotIdentity` supplies project/simulation/timestep IDs, optional `revision`, and a loading-configuration label; `ConfiguredSnapshotIdentity` maps local paths to those identities through the existing `sim_identity` parameter. Calculator.run, EvalEngine and SampleTable use the provider without an execution or persistence rewrite.

Per the user's decision, `revision` defaults to None. Data edits do not automatically invalidate that identity; callers use `reuse=False` to refresh each affected computation, or supply a revision to separate data versions. `loader_id` defaults to "default" and is user-maintained. Locations are excluded from logical cache keys; aliases assert equivalence without proving file contents equal. No snapshot scans, fingerprinting or database migration are added.

Unknown locations retain the existing path/session fallback, and externally transformed roots/views remain session-only. Versioned configuration JSON validates fields, absolute locations and conflicting bindings. Existing halo/subset addressing is shared. SampleTable default labels now include the logical fields instead of giving every logical snapshot the name `v1`; explicit names and row keys are unchanged.

## Evidence

- [x] Wrote failing tests first: 45 failures from missing logical identity APIs, then all 45 passed.
- [x] Added two failing SampleTable label regressions (`v1` / `v1:halo s` for every logical row), then fixed the naming helper; logical identity + SampleTable suites: 70 passed, 1 skipped.
- [x] Identity/store/sample/layering focused suite before label follow-up: 150 passed, 1 skipped.
- [x] Independent code review: no remaining actionable findings; public str/Path/mixed-mapping mypy probe passed.
- [x] `make check`: ruff/mypy/docs pass; 1020 passed, 1 skipped, 3 deselected.
- [x] `make test-all`: 1023 passed, 1 skipped.

## Architecture

- [x] ADR added: `docs/adr/0008-configured-logical-snapshot-identity.md`, following accepted V1 ADR-003/004 and the user's explicit optional-revision decision.
- [x] `CONTEXT.md`, both FEAT-0002 locales, metadata and handoff describe behavior and limits.
- [x] Existing import layers, default providers, execution and store schema remain intact; no new dependencies.

## Checklist

- [x] Existing tests were not weakened to get a green run
- [x] Public behavior and user-facing docs are updated
- [x] No testdata, virtual environments, build output, secrets or unrelated local files are included
