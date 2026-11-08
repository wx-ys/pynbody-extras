# BUG-0009: Prevent typed simulation identities from sharing result-store keys

## Context

Continue the incremental V1 plan after codec/SPH/median fixes merged on main
3a30cff. _sig_text stringifies identity components; Python tuple equality also
collapses bool/int/float SampleTable rows.

## Goal

One typed contract for lookup, replacement, deletion, lists and SampleTable rows.
CalculatorSignature numerical equivalence is unchanged.

## Actual behavior

RecordRef(("sim", 1), "a" * 64).key equals the key for ("sim", "1").
Saving both under one calculator overwrites one result.

## Expected behavior

Distinct typed inputs coexist; repeated identities reuse only their own results.

## Reproduction

Run `.venv/bin/python -m pytest tests/test_sim_identity_store.py -q`.

## Non-goals

No legacy compatibility or migration (explicit user decision: no old databases
or caches exist). No logical registry, historical result versions, ORM, new
dependencies, pickle-policy or calculator-hash changes.

## Acceptance criteria

- [x] Distinct str/int/bool/float/None/nested tuple identities have distinct keys.
- [x] Memory and SQLite retain independent values, filtered lists, lookup,
  deletion and refresh; reopen preserves identity types.
- [x] SampleTable add/from_store/save/load/where retain distinct typed rows.
- [x] Unsupported values and non-finite floats fail explicitly.
- [x] Current record schema loads; unknown/missing schemas fail explicitly.
- [x] Focused tests, make check and make test-all pass with red-to-green evidence.

## Design notes

[ADR-006](../../adr/0006-typed-simulation-store-identity.md) records the decision.
Use built-in scalar tags and recursive tuples; normalize signed floating zero.
Providers explicitly convert NumPy IDs. ResultRecord becomes v3. No migration.

## Test plan

Add public-behavior regressions and confirm collisions/row collapse. Implement
one leaf encoder used by store/table boundaries, then run focused and full tests.

## Outcome

Baseline fast suite: 916 passed, 1 skipped, 3 deselected on pynbody 2.8.0.
Red: new regression suite produced 19 failed / 1 passed before implementation.
Green: focused identity/store/reuse/sample suite: 80 passed / 1 skipped.
Final make check: lint and mypy (111 files) passed; fast tests 936 passed /
1 skipped / 3 deselected; docs validated 10 items. make test-all: 939 passed /
1 skipped. git diff --check passed. No existing assertion was weakened.

The initial baseline docs check raced with creation of this work item and reported
an out-of-date index; regenerating the index resolved it. The final check above
includes the finished work-item layout. No production baseline failure remained.

Self-review covered type-tag separation, nested identities, signed zero,
unsupported IDs, SQLite reopen, filtering/deletion, SampleTable round trips,
record-schema rejection and import layering. The maintainer explicitly authorized publication and merge in this chat.
PR #36 contains the completed repair; merge is subject to remote checks. Historical result versioning remains the next milestone.

## Open questions

None for this repair; user resolved database compatibility.
