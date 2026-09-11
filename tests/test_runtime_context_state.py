"""ExecutionContext groups its mutable per-run state into cohesive sub-objects."""

from __future__ import annotations

import dataclasses

from pynbodyext.core.calculate.runtime.context import ExecutionContext
from pynbodyext.core.calculate.runtime.options import RunOptions


def test_execution_context_state_is_grouped() -> None:
    names = {f.name for f in dataclasses.fields(ExecutionContext)}
    assert {"nodes", "mutation", "records"} <= names
    # the previous flat mutables must no longer be top-level fields
    flat = {
        "node_registry",
        "runtime_store",
        "named_registry",
        "warnings",
        "errors",
        "log_events",
        "access_observations",
        "mutation_generation",
        "unknown_mutation_generation",
        "field_generations",
        "last_error_node_id",
        "root_signature",
        "_node_counter",
    }
    assert names & flat == set(), f"ungrouped fields remain: {names & flat}"


def test_execution_context_substate_defaults() -> None:
    ctx = ExecutionContext(sim=None, sim_signature=(), run_id="r", options=RunOptions(), engine=None)
    assert ctx.nodes.registry == {}
    assert ctx.nodes.named == {}
    assert ctx.mutation.generation == 0
    assert ctx.mutation.field_generations == {}
    assert ctx.records.errors == []
    assert ctx.records.warnings == []
