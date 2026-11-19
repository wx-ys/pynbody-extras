# BUG-0013 · Restore Pipeline branch transform scopes

## Reproduction

Two named outputs select stars with the same `Disc(3.0, 0.1)` filter.
One first rotates the snapshot with `AlignVec`. On a fresh copy, the rotated
output selects root indices `[3, 4, 5]` and the native output selects
`[2, 3, 5]`. On `c56120f`, putting the rotated output first gave
`[3, 4, 5]` for both. Putting the native output first gave the expected
indices but left 6 of the caller's 18 position components changed.

```bash
.venv/bin/python -m pytest -q tests/test_pipeline_branch_scopes.py::test_named_output_order_restores_entry_frame
```

## Phase 1 acceptance criteria

- **Frames.** In either output order the Pipeline agrees with both standalone
  measurements on fresh snapshots. An output owns its default reversible
  transforms through filtering, input validation, calculation and
  materialization. On success or failure it restores the entry positions and
  velocities before the next output runs. A Pipeline-level transform spans
  all of its outputs. Nested Pipelines restore their parent's frame. A
  subset transform applies to its requested target, and a native handle
  already active on entry remains under the caller's control.
- **Values.** Returned numerical views of the particles retain the values
  and units measured in their branch, including after a caller subsequently
  reverts its own handle. This includes ragged arrays of particle measurements.
  Eager Bin queries measure in the transformed frame. Each per-bin Calculator
  and Filter selector releases its default transform before the next subset.
  A later lazy Bin query uses the live particle arrays.
- **Transform lifetime.** Standalone Property and Transform calls keep their
  existing lifetimes. An explicit `NEVER` on a bound scope, transform or chain
  retains the requested mutation for later outputs and the caller. Standalone
  Postprocessor exception cleanup follows [ADR-014](../../adr/0014-dag-value-postprocessors.md).
- **Failures.** Applied reversible steps unwind in reverse order, including
  completed steps in a chain that fails during application. Under
  `collect_partial` a sibling can run after a science failure once cleanup
  succeeds. A handle that cannot restore its frame stops subsequent outputs.
  If the science step also failed, its exception remains primary. A failed
  Pipeline produces no successful stored result.
- **Persistence.** A recipe containing a Pipeline gets a new
  `CalculatorSignature` hash, including nested and custom Pipelines. An
  upgraded Database in `auto` mode computes the corrected result on its first
  request and reuses that version on the next. Pinned references and frozen
  Study columns still read the historical value. A standalone Property keeps
  its existing key. [ADR-022](../../adr/0022-pipeline-transform-scopes.md)

## Verification

Both output orders failed against `c56120f` while the 52 existing baseline
tests passed. A SQLite regression seeds the old recipe and result, closes
the Database and opens it again to exercise cache reuse, refresh history and
frozen Study columns. The tests and follow-up verification are recorded in
the [handoff](handoff.md).

```bash
.venv/bin/python -m pytest -q tests/test_pipeline_branch_scopes.py tests/test_pipeline_scope_versions.py tests/test_postprocessors.py
```

Indexed particle subsets restore their exact coordinates after all three periodic wrapping conventions. An invalid alignment direction or parallel up vector is reported before mutation, so independent outputs can still run under `collect_partial`. [PR #57](https://github.com/wx-ys/pynbody-extras/pull/57)
