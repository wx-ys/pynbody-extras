"""Multi-output calculator pipelines.

:class:`Pipeline` groups several calculators into one shared execution run.
This is the standard way to compute several related outputs while reusing
shared dependencies, cache state, trace collection, and performance reporting.

Each pipeline output is registered under its dictionary key in
:attr:`Result.named`, so outputs can be retrieved after one run with
:meth:`Result.get`.

Use :class:`Pipeline` when you want to:

- evaluate several outputs in one run
- reuse overlapping dependencies automatically
- expose outputs by stable names
- apply one filter or transform scope to several outputs at once

Basic Example
-------------
Group several outputs under stable names::

    from pynbodyext.core.calculate import Pipeline, PropertyBase


    @PropertyBase.dataclass
    class StellarMass(PropertyBase[float]):
        def calculate(self, sim, params=None):
            return float(sim["mass"].sum())


    @PropertyBase.dataclass
    class MeanTemperature(PropertyBase[float]):
        def calculate(self, sim, params=None):
            return float(sim["temp"].mean())


    pipe = Pipeline({"mass": StellarMass(), "temp_mean": MeanTemperature()}, name="basic_summary")

    result = pipe.run(sim)
    print(result.value["mass"])
    print(result.get("temp_mean"))

Why Pipeline Helps
------------------
The main benefit of :class:`Pipeline` is shared execution, not just returning
a dictionary. If two outputs depend on the same child calculator, that child is
evaluated once inside the same run context.

This is especially useful for analysis summaries built from overlapping
building blocks.

Scoped Pipeline
---------------
A pipeline composes with filters and transforms like any other calculator::

    hot_summary = Pipeline({"mass": StellarMass(), "temp_mean": MeanTemperature()}, name="hot_summary").filter(
        TemperatureAbove(1.0e5)
    )

    result = hot_summary.run(sim, progress="phase")
    print(result.value["mass"])
    print(result.reports["trace_tree"])

Inspecting Results
------------------
A pipeline result is still a normal :class:`Result`::

    result = pipe.run(sim, progress="phase")
    print(result.value)
    print(result.get("mass"))
    print(result.reports["perf"])
    print(result.reports["cache"])

Subclassing
-----------
Most users should compose :class:`Pipeline` directly rather than subclassing
it.

Subclass :class:`Pipeline` only when you need custom multi-output
orchestration, such as non-standard failure handling or output registration
rules.

Notes
-----
Keep output keys stable and unique. If a pipeline output name collides with
another named node, execution will fail.

If one child seems to recompute unexpectedly, inspect the shared dependency
graph through the trace and cache reports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.result.enums import BuiltinKinds, ErrorPolicy

from .base import CalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput


class Pipeline(CalculatorBase[dict[str, Any], dict[str, Any]]):
    """Calculator that returns a dictionary of named outputs.

    Parameters
    ----------
    outputs : mapping of str to CalculatorBase
        Output names and calculators to evaluate.
    name : str, optional
        Optional name for the pipeline root node.
    """

    node_kind = BuiltinKinds.COMBINED

    def __init__(self, outputs: Mapping[str, CalculatorBase[Any, Any]], *, name: str | None = None) -> None:
        super().__init__(name=name)
        if not outputs:
            raise ValueError("Pipeline requires at least one output.")
        self.outputs = dict(outputs)
        self._validate_output_names()

    def _validate_output_names(self) -> None:
        output_keys = set(self.outputs)
        for key, child in self.outputs.items():
            if not key:
                raise ValueError("Pipeline output names must be non-empty.")
            if child.name is not None and child.name != key and child.name in output_keys:
                raise ValueError(
                    f"Pipeline output {key!r} uses child name {child.name!r}, which conflicts with another output key."
                )

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return list(self.outputs.values())

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("outputs", tuple(self.outputs.keys()))]
        if self.name is not None:
            fields.append(("name", self.name))
        if not self.scope.is_empty:
            fields.append(("scope", self.scope.short_label()))
        return fields

    def _repr_summary_rows(self) -> list[tuple[str, Any]]:
        rows = super()._repr_summary_rows()
        rows.append(("outputs", ", ".join(self.outputs)))
        return rows

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> dict[str, Any]:
        values: dict[str, Any] = {}
        with ctx.phase(self, "calculate"):
            for key, child in self.outputs.items():
                try:
                    node_result = ctx.evaluate(child, input)
                except Exception:
                    if ctx.options.errors != ErrorPolicy.COLLECT_PARTIAL:
                        raise
                    if ctx.nodes.last_error_id is None:
                        raise
                    node_result = ctx.nodes.registry[ctx.nodes.last_error_id]
                    values[key] = None
                else:
                    values[key] = ctx.nodes.runtime[node_result.node_id].public_value
                existing = ctx.nodes.named.get(key)
                if existing is not None and existing != node_result.node_id:
                    raise ValueError(f"Duplicate named pipeline output {key!r}.")
                ctx.nodes.named[key] = node_result.node_id
        return values
