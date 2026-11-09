# FEAT-0002: Configure logical snapshot identities independently of location

## Context

PR #37 provides immutable result references. The default snapshot identity still
uses an absolute path. Moving a snapshot therefore loses cache reuse, and neither
the path nor a saved result version identifies a scientific data revision.
The supplied V1 ADR-003/004 require user/project logical IDs, separate locations,
and no default full-content hashing. The user selected optional revision on
2026-11-09: omit it and refresh with reuse=False when the data changes.

## Goal

Add an opt-in configured identity provider through the existing sim_identity seam.
Moving an explicitly registered snapshot keeps its identity; unknown snapshots
keep the existing path/session fallback. No execution or persistence rewrite.

## Design

See [ADR-008](../../adr/0008-configured-logical-snapshot-identity.md).
LogicalSnapshotIdentity is frozen, with required non-empty string project_id,
simulation_id and timestep_id; revision is an optional non-empty string (None by
default); loader_id is a non-empty string defaulting to "default". IDs are literal:
no string coercion, case folding, or stripping. Whitespace-only fields are invalid.
The loader label is a user-maintained identifier for interpretation options, not
an automatic inventory of pynbody settings/plugins.

ConfiguredSnapshotIdentity accepts a mapping of local paths/prefixes to these
identities. It snapshots the configuration and normalizes paths to absolute paths
without resolving symlinks or touching files. Aliases may point to the same identity;
conflicting identities at the same normalized path raise ValueError. Reconfiguration
means constructing another provider. No global registration or implicit mutation.

The provider selects by ancestor.filename. It returns a versioned stable tuple
containing all logical fields, plus existing halo/subset addressing where applicable.
An externally transformed root or view remains session-only. Unknown paths retain
snapshot_identity behavior, and unconfigured in-memory snapshots remain session-only.

Configuration JSON has a strict versioned schema, absolute paths and all logical
fields; loading validates fields, duplicates and schema before use. Relative JSON
paths are rejected to avoid reinterpreting config under another working directory.
No remote loading or file access occurs during configuration or identity lookup.

## Acceptance criteria

- [x] Explicit logical identity stays stable after moving/aliasing the path.
- [x] Changing project, simulation, timestep, revision or loader changes the key.
- [x] Missing revision allows reuse until explicit reuse=False; refresh preserves
  old immutable result references.
- [x] Same subset/order on relocated replicas matches; different indices/order and
  halo number separate identities. External transforms remain session-only.
- [x] Existing unconfigured path/session behavior is unchanged.
- [x] Conflicting normalized paths and malformed configuration are rejected.
- [x] Configuration JSON round-trip works across fresh processes and SQLite reopen.
- [x] Calculator.run, EvalEngine and SampleTable use the provider without changes
  to their control flow; memory and SQLite read-through both work. Default
  SampleTable labels include logical fields and retain halo/subset suffixes.
- [x] Existing tests remain intact; make check and make test-all pass.

## Non-goals and limits

No ORM tables, catalog/halo logical registry, automatic discovery, fingerprinting,
file IO, data loading, migration, packaging changes or historical GC. A configured
identity asserts equivalence: identical labels do not prove identical contents.
Without revision, updates require refreshing each affected computation (including
any subsets); there is no blanket invalidation. With revision, update the provider
configuration so all subsequent keys use the new revision. Loading options need a
new loader_id when they affect interpretation. Index-based subset identity assumes
unchanged parent particle order, managed by revision or explicit refresh.

## Test plan

See [implementation plan](plan.md). Add tests/test_logical_snapshot_identity.py;
run it red before adding the implementation. Exercise read-through on both stores,
SQLite reopen, invalid configuration, process stability, subsets and transformations.

## Outcome

Implemented on codex/logical-snapshot-identity, based on main aaf9b92.
Initial API suite: 45 failures before implementation, then 45 passed. Added two
failing logical row-name regressions and fixed only the naming helper; identity
and SampleTable suites: 70 passed, 1 skipped. Earlier focused integration/layering
suite: 150 passed, 1 skipped. All 47 new cases are included in final validation.

Final make check: ruff/mypy/docs passed; 1020 passed, 1 skipped, 3 deselected.
Final make test-all: 1023 passed, 1 skipped. Existing tests were unchanged.
Independent review found a public factory typing issue (fixed with concrete static
JSON construction) and confirmed the final implementation and naming correction.
Publication/merge are authorized; no remaining implementation questions.

PR: [#38](https://github.com/wx-ys/pynbody-extras/pull/38).
