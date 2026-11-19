# ADR-023 · Observe consumed particle selections and measurement frames

2026-11-20 · Phase 2 contract · [FEAT-0019](../work/FEAT-0019/en.md)

A full Calculator run on an addressable particle input returns detached
`Result.selection_observations`. The same execution records these facts for
manual runs, subsequent saves, and Database-coordinated computations. The
field-access observer, tracing policies and runtime cache are independent.
Value-only batch execution allocates no selection recorder. A pure calculation
may accept inputs without root-relative particle addresses.

## Evidence at execution

A Filter computes a mask. Reading the resulting filtered view records a
Selection origin on first consumption. This includes scoped inputs, a
Transform’s measurement Filter, and a DAG node reading the raw
`FilterResult.filtered_sim`. A standalone mask, Boolean operand mask or
custom calculator’s private particle slice produces zero inferred origins.
A shared filtered view supplies its ordered root indices once. Duplicate
particle indices keep their positions in the sequence.

Each origin carries its direct input indices, its optional upstream origin,
the root snapshot identity and family layout, the complete Filter signature,
and the resolved values and units passed to the Filter hook. A root input
uses a null parent index sequence. Boolean operands retain their resolved
parameters at their paths in the canonical Filter expression. The selection
keeps its Filter’s measured frame even when a Filter-local rotation has
already unwound before the selected particles reach the leaf. Boolean
operands measured in different frames yield a mixed, incomplete selection
context with their exact membership. For example,
a family and adaptive sphere combined by intersection resolve the aperture
on their common input. A Pipeline filtered to that family gives its child
sphere a different input and links the two origins.

Each measured output records its physical context. The executor follows the
actual dependency edges, including cache hits, to associate each consumed
origin with its leaf output, DAG occurrence and role. Roles are
`output_filter` and `transform_measure`. A Pipeline-level Filter belongs to
each leaf inheriting its selection. An unfiltered leaf still has a measured
context. Output paths are tuples of user keys, with the empty tuple for a
standalone result.

## Frames and detachment

An explicitly qualified Transform records the applied native rotation matrix
or signed translation, its units, the actual root or ordered subset target,
and its complete method signature. Reversion removes the step from later
frames. An independently applied transform, an opaque custom step and a
periodic wrap carry an incomplete-frame reason. A run-scoped native
registration hook also detects temporary transforms applied and reverted
inside a science hook. Concurrent runs route notifications by execution
context and root. pynbody exposes a pre-existing native handle on its
target. An earlier mutation on an unrelated view that pynbody does not
expose on the input or its ancestors cannot be recovered from particle
arrays. Actual-frame comparison uses
the typed root identity, layout, unit conventions, targets and ordered
numerical operations. Complete contexts support exact matching. The
declaration stays available for method-based navigation.

The observer reads available array-unit metadata and the physical unit basis.
A field first loaded by the science hook can supply its units at
materialization. Observing a family-only result requires no position or
velocity IO. Arrays, parameter values and operation receipts become immutable
data. The observations contain no snapshot, FilterResult, Calculator or
native handle. An explicit request to retain raw runtime values continues to
use the existing recording policy.

The versioned JSON representation supports detached Database staging and
recovery. A missing observation on a historical value differs from a
recorded output with an empty list of uses. Exact membership size determines
the storage cost. An observation records root indices and operations, without
copying the particle fields. Replay requires access to the captured input
version and unit convention. [ADR-016](0016-replayable-particle-selections.md)
and [ADR-022](0022-pipeline-transform-scopes.md)
