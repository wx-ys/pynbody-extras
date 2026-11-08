# FEAT-0001: Pin stored result references to immutable versions

## Context

The current store overwrites a result under its calculation key, so an old
store() reference reads a new value after recomputation. User approved continuing
the V1 plan after PR #36, with no legacy database/cache compatibility required.

## Goal

A saved reference keeps its original result while read-through uses the latest
successful save. Surface the actual version through Result.store_ref.

## Non-goals

ORM Run tables, logical snapshot registry, automatic GC, payload deduplication,
external files, migration and distributed scheduling.

## Acceptance criteria

- [x] Recompute under the same cache key preserves the first saved reference.
- [x] Current lookup and list_refs retain their existing latest-result semantics.
- [x] Version listing, reference JSON and SQLite reopen preserve history.
- [x] Cache invalidation never deletes history or invalidates a newer version
  when invoked with an older pinned reference.
- [x] Returned mutable values cannot modify saved versions.
- [x] A write failure rolls back history and cache changes together.
- [x] Concurrent successful SQLite saves preserve all returned versions.
- [x] Engine write/hit/refresh returns the exact store_ref; no-store and failed
  runs do not invent saved versions.
- [x] Identity/signature/version mismatches fail before payload decoding.
- [x] make check and make test-all pass with red-to-green evidence.

## Design notes

See [ADR-007](../../adr/0007-immutable-result-versions.md). RecordRef has optional
version_id and a separate cache_key. store() returns pinned references;
list_refs() remains a view of current cache addresses. delete() invalidates only
current cache and is version-conditional for pinned references. Versioned
fetch/contains remain valid after cache deletion. Record schema v4; no migration.

## Test plan

Write tests/test_result_versions.py, confirm failures, implement shared semantics
and both adapters, then run focused, layering, fast and full suites. Preserve all
existing tests. Inject a cache write failure after version insertion, and verify
both old cache and old version remain while no orphan version is added.

## Outcome

Implemented on base main 8f191b2. Initial contract suite: 25 failures before
implementation; expanded version suite: 37 passed. Full verification: make check
973 passed, 1 skipped, 3 deselected, with ruff/mypy/docs checks passing;
make test-all 976 passed, 1 skipped. The existing corruption test exposed a current
record alias: memory reads now resolve the immutable version selected by cache,
matching the SQL adapter. Related regression suite: 77 passed, existing tests unchanged.

Publication and merge are authorized. Historical GC and ORM integration remain
separate milestones; schema v4 intentionally has no legacy migration.

## Open questions

None for this scoped storage milestone. Historical versions are retained without
automatic GC; backend identity and reference lifecycle will be managed by ORM later.
