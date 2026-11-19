# ADR-022 · Pipeline transform scopes

2026-11-19 · Phase 1 contract approved for implementation · [BUG-0013](../work/BUG-0013/en.md)

A Pipeline evaluates named outputs on a shared particle snapshot. An output
with a default reversible transform owns that transform through filtering,
input validation, computation and value materialization. It restores the entry
frame before the next output starts, on success and on failure. A transform on
the Pipeline itself spans all of its outputs. A child applies its own steps on
the requested snapshot or subset in that frame, allowing the caller's native
transform handle to remain active. Borrowed numerical outputs retain the
values and units seen when measured, including arrays grouped into ragged
object arrays. Each eager per-bin Calculator or Filter runs in the Bin's
current frame and releases its default transforms before the next query or
subset. A Bin's later lazy queries keep their documented dependence on its
live input.

An explicit NEVER policy retains the requested mutation. A bare Transform
returned by a Pipeline output also belongs to that branch and its default
handle has already been reverted on return. When a default scope
cannot restore its frame, execution reports the failure and stops evaluating
siblings, including under collect_partial. If the science step also failed,
that exception remains primary. Standalone Property and Transform calls retain
their existing lifetime, and ADR-014 still governs standalone Postprocessor
exception cleanup.

pynbody requires native handles on the same target to unwind in reverse
application order. If a child explicitly retains a newer handle, an enclosing
default scope may be unable to undo its older handle. This produces a cleanup
error and stops subsequent branches. The still-active native handles are
available to the caller in stack order. Ordinary fully reversible nested
scopes restore each layer exactly once.

Each invocation owns a mutable particle tree for its duration. Concurrent analyses
can use independent roots. Callers that share one live root across threads must
coordinate all readers, writers and native transforms on that root. A custom
transform can reject invalid inputs in `prepare_params`, before any mutation.
If an apply hook raises without returning an undo handle, the runtime cannot
establish which arrays that hook changed and stops dependent measurements.

The Pipeline recipe records the execution semantics in its signature. A new
auto run computes under the new key. Fixed historical receipts, immutable
Study columns and independent Property keys keep their meaning.
