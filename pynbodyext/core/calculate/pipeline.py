"""Multi-output calculator pipelines.

A :class:`Pipeline` groups several calculators into one shared evaluation run.
This is useful when multiple outputs depend on overlapping subgraphs and should
share dependency resolution, runtime cache, provenance, progress reporting, and
diagnostics.

Each output is registered under its dictionary key in :attr:`Result.named`, so
pipeline outputs can be retrieved with :meth:`Result.get` after one run.

When To Use Pipeline
--------------------
Use :class:`Pipeline` when you want to:

- evaluate several outputs in one shared run
- reuse overlapping dependencies automatically
- expose outputs by stable names
- apply one filter/transform scope to several outputs at once

Most users compose :class:`Pipeline` directly rather than subclassing it.

Basic Usage
-----------
A basic pipeline groups named outputs::

    from pynbodyext.core.calculate import Pipeline, PropertyBase

    class StellarMass(PropertyBase[float]):
        def instance_signature(self):
            return ("stellar_mass",)

        def calculate(self, sim):
            return float(sim["mass"].sum())

    class MeanTemperature(PropertyBase[float]):
        def instance_signature(self):
            return ("mean_temperature",)

        def calculate(self, sim):
            return float(sim["temp"].mean())

    pipe = Pipeline(
        {
            "mass": StellarMass(),
            "temp_mean": MeanTemperature(),
        },
        name="basic_summary",
    )

    result = pipe.run(sim)
    print(result.value["mass"])
    print(result.get("temp_mean"))

Why Pipeline Helps
------------------
The main advantage of :class:`Pipeline` is not just returning a dictionary.  It
ensures all outputs participate in one shared execution graph.  If two outputs
depend on the same child calculator, that dependency is only evaluated once.

For example, a reusable property can be shared across outputs::

    class CountParticles(PropertyBase[int]):
        def instance_signature(self):
            return ("count_particles",)

        def calculate(self, sim):
            return int(len(sim))

    count = CountParticles().keep("count")

    pipe = Pipeline(
        {
            "count": count,
            "mass": StellarMass(),
        }
    )

    result = pipe.run(sim)
    print(result.get("count"))

Scoped Pipeline
---------------
Pipelines compose with filters and transforms like any other calculator::

    import numpy as np

    from pynbodyext.core.calculate import FilterBase

    class TemperatureAbove(FilterBase):
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold

        def instance_signature(self):
            return ("temperature_above", self.threshold)

        def build_mask(self, sim, params):
            return sim["temp"] > self.threshold

    hot_summary = Pipeline(
        {
            "mass": StellarMass(),
            "temp_mean": MeanTemperature(),
        },
        name="hot_summary",
    ).filter(TemperatureAbove(1.0e5))

    result = hot_summary.run(sim, progress="phase")
    print(result.value["mass"])

Custom Pipeline Subclass
------------------------
Subclass :class:`Pipeline` only when you need custom multi-output orchestration,
such as special failure handling or output registration rules.  In most cases,
plain composition is simpler and keeps signatures easier to reason about.

Inspection Example
------------------
Pipelines work especially well with the result reporting tools::

    result = pipe.run(sim, progress="phase")
    print(result.pipeline_report())
    print(result.reports["trace_tree"])
    print(result.perf_summary)

Notes
-----
Pipeline output keys should be stable and unique.  Prefer short semantic names
such as ``mass``, ``re``, ``temp_mean``, or ``hot_fraction`` over positional or
temporary labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import CalculatorBase
from .enums import BuiltinKinds, ErrorPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .context import ExecutionContext, NodeInput


class Pipeline(CalculatorBase[dict[str, Any],dict[str, Any]]):
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
                    f"Pipeline output {key!r} uses child name {child.name!r}, "
                    "which conflicts with another output key."
                )

    def instance_signature(self) -> tuple[Any, ...]:
        return ("pipeline", tuple((key, child.signature()) for key, child in self.outputs.items()))

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
                    if ctx.last_error_node_id is None:
                        raise
                    node_result = ctx.node_registry[ctx.last_error_node_id]
                    values[key] = None
                else:
                    values[key] = ctx.runtime_store[node_result.node_id].public_value
                existing = ctx.named_registry.get(key)
                if existing is not None and existing != node_result.node_id:
                    raise ValueError(f"Duplicate named pipeline output {key!r}.")
                ctx.named_registry[key] = node_result.node_id
        return values
