"""Core calculator framework for composable pynbody analyses.

The :mod:`pynbodyext.core.calculate` package implements the execution model
behind calculator graphs. A calculator graph is a directed acyclic graph of
nodes that read values, build masks, apply temporary mutations, or evaluate
several outputs in one shared run.

The framework is organized around four common node roles:

- :class:`PropertyBase` for read-only derived values
- :class:`FilterBase` for boolean masks and scoped selection
- :class:`TransformBase` for temporary mutations
- :class:`Pipeline` for grouped multi-output evaluation

A single run evaluates shared dependencies once, records provenance, trace,
cache, and performance diagnostics, and returns a :class:`Result`.

Where Most Users Should Start
-----------------------------
Most users only need:

- :class:`PropertyBase`
- :class:`FilterBase`
- :class:`TransformBase`
- :class:`Pipeline`
- :class:`RunOptions`
- :class:`Result`

If you only want the short public import path for the common role classes, use
:mod:`pynbodyext.calculate`.

Choosing A Base Class
---------------------
Choose the narrowest base class that matches the node.

1. Use :class:`PropertyBase` for a read-only derived value.
2. Use :class:`FilterBase` for a boolean mask.
3. Use :class:`TransformBase` for a temporary mutation.
4. If none of those fit, but the node still follows the standard runtime
   lifecycle, subclass :class:`RuntimeCalculatorBase`.
5. Only subclass :class:`CalculatorBase` directly when you need to implement
   :meth:`CalculatorBase.execute` yourself.

This matches the framework structure itself:
:class:`PropertyBase`, :class:`FilterBase`, and :class:`TransformBase` all
build on top of :class:`RuntimeCalculatorBase`, while
:class:`CalculatorBase` is the lowest-level escape hatch.

Package Layout
--------------
User-facing authoring modules:

- :mod:`.base` for :class:`CalculatorBase`
- :mod:`.template` for :class:`RuntimeCalculatorBase`
- :mod:`.properties` for property nodes
- :mod:`.filters` for filter nodes
- :mod:`.transforms` for transform nodes
- :mod:`.pipeline` for grouped outputs
- :mod:`.fields` and :mod:`.declarative` for dataclass-style definitions

Runtime and debugging modules:

- :mod:`.context` for run options and per-run execution state
- :mod:`.result` for finished run outputs and reports
- :mod:`.params` for dynamic parameter resolution
- :mod:`.scopes` for filter and transform composition
- :mod:`.engine` for graph evaluation
- :mod:`.cache`, :mod:`.trace`, and :mod:`.perf` for diagnostics

Quick Start
-----------
Define a simple property calculator in the same dataclass style used by the
user-facing property modules::

    from pynbodyext.core.calculate import PropertyBase

    @PropertyBase.dataclass
    class StellarMass(PropertyBase[float]):

        def calculate(self, sim, params=None):
            return float(sim["mass"].sum())

    result = StellarMass().run(sim)
    print(result.value)

Scope one calculator with a filter in the same style used by
:mod:`pynbodyext.filters`::

    import numpy as np

    from pynbodyext.core.calculate import FilterBase, PropertyBase

    @FilterBase.dataclass
    class TemperatureAbove(FilterBase):
        threshold: float

        def calculate(self, sim, params=None):
            return sim["temp"] > self.threshold

    @PropertyBase.dataclass
    class MeanTemperature(PropertyBase[float]):

        def calculate(self, sim, params=None):
            return float(np.asarray(sim["temp"]).mean())

    result = MeanTemperature().filter(TemperatureAbove(1.0e5)).run(sim)
    print(result.value)

Apply a temporary transform in the same style used by
:mod:`pynbodyext.transforms`::

    from pynbodyext.core.calculate import PropertyBase, TransformBase

    @TransformBase.dataclass
    class XShift(TransformBase[dict[str, object]]):
        dx: float

        def build_handle(self, sim, target, params=None):
            original = target["x"].copy()
            target["x"] = target["x"] + self.dx
            return {"target": target, "original_x": original}

        def cleanup(self, ctx, handle):
            handle["target"]["x"] = handle["original_x"]

        def is_revertible(self, handle):
            return True

    @PropertyBase.dataclass
    class XMean(PropertyBase[float]):

        def calculate(self, sim, params=None):
            return float(sim["x"].mean())

    result = XMean().transform(XShift(1.0)).run(sim)
    print(result.value)

Evaluate several outputs in one run::

    from pynbodyext.core.calculate import Pipeline

    pipe = Pipeline(
        {
            "mass": StellarMass(),
            "hot_temp": MeanTemperature().filter(TemperatureAbove(1.0e5)),
        },
        name="basic_summary",
    )

    result = pipe.run(sim, progress="phase")
    print(result.value["mass"])
    print(result.get("hot_temp"))
    print(result.reports["trace_tree"])

Result Surface
--------------
Every run returns a :class:`Result`. The most commonly used parts are:

- ``result.value`` for the public root value
- ``result.get(name)`` for named outputs
- ``result.perf_summary`` for aggregate performance counters
- ``result.reports`` for trace, cache, and performance reports
- ``result.diagnostics`` for raw debugging payloads

Notes
-----
This package root is an index and import surface. Detailed authoring examples
live in the role-specific submodules, while execution and debugging details
live in the runtime-oriented submodules.

When writing new calculators, prefer the style already used in
:mod:`pynbodyext.properties`, :mod:`pynbodyext.filters`, and
:mod:`pynbodyext.transforms`: dataclass-based definitions, :class:`Param`
fields where needed, and the narrowest role-specific hook.
"""

from .base import BoundCalculator, CalculatorBase, CombinedCalculator
from .cache import CacheEvent, ExecutionValue, RuntimeCache
from .context import (
    ExecutionContext,
    FilterResult,
    LogEvent,
    NodeInput,
    NullProgressSink,
    ProgressSink,
    RunOptions,
    TransformResult,
    resolve_value,
)
from .declarative import dataclass_calc
from .engine import EvalEngine
from .enums import (
    BuiltinKinds,
    CachePolicy,
    EffectPolicy,
    ErrorPolicy,
    NodeKind,
    NodeStatus,
    RecordPolicy,
    RevertPolicy,
    normalize_error_policy,
    normalize_kind,
    normalize_revert_policy,
)
from .exceptions import CalculatorError, CycleError
from .expr import ConstantProperty, LambdaProperty, OpProperty
from .fields import Param, ParamSpec, ParamView, collect_param_specs
from .filters import AndFilter, FilterBase, NotFilter, OrFilter
from .observer import AccessEvent, AccessObservation
from .params import DynamicParamSpec, dynamic_value_dependencies, dynamic_value_signature, resolve_dynamic_value
from .perf import PerfCollector
from .pipeline import Pipeline
from .properties import PropertyBase
from .result import (
    ErrorInfo,
    PerfSummary,
    PhaseRecord,
    ProvenanceInfo,
    Result,
    ResultNode,
    ValueSummary,
)
from .runtime import CalcRuntime, TransformRuntime
from .scopes import Scope, ScopeSpec, TransformScope
from .signature import CalculatorSignature, calculator_from_signature, calculator_to_signature
from .template import RuntimeCalculatorBase
from .trace import TraceCollector, TraceEvent
from .transforms import TransformBase, TransformChain, TransformPlan, TransformStep, chain_transforms

__all__ = [
    "CalculatorBase",
    "RuntimeCalculatorBase",
    "BoundCalculator",
    "CombinedCalculator",
    "EvalEngine",
    "CalculatorError",
    "CycleError",
    "ExecutionContext",
    "RunOptions",
    "NodeInput",
    "FilterResult",
    "TransformResult",
    "ProgressSink",
    "NullProgressSink",
    "LogEvent",
    "resolve_value",
    "FilterBase",
    "AndFilter",
    "OrFilter",
    "NotFilter",
    "TransformBase",
    "TransformChain",
    "TransformPlan",
    "TransformStep",
    "chain_transforms",
    "CalcRuntime",
    "TransformRuntime",
    "ScopeSpec",
    "Scope",
    "TransformScope",
    "Pipeline",
    "PropertyBase",
    "dataclass_calc",
    "ParamSpec",
    "ParamView",
    "Param",
    "collect_param_specs",
    "ConstantProperty",
    "LambdaProperty",
    "OpProperty",
    "DynamicParamSpec",
    "dynamic_value_dependencies",
    "dynamic_value_signature",
    "resolve_dynamic_value",
    "Result",
    "ResultNode",
    "AccessEvent",
    "AccessObservation",
    "ValueSummary",
    "PhaseRecord",
    "ErrorInfo",
    "ProvenanceInfo",
    "PerfSummary",
    "RuntimeCache",
    "ExecutionValue",
    "CacheEvent",
    "CalculatorSignature",
    "calculator_from_signature",
    "calculator_to_signature",
    "TraceCollector",
    "TraceEvent",
    "PerfCollector",
    "NodeKind",
    "BuiltinKinds",
    "normalize_kind",
    "NodeStatus",
    "RecordPolicy",
    "RevertPolicy",
    "normalize_revert_policy",
    "EffectPolicy",
    "ErrorPolicy",
    "normalize_error_policy",
    "CachePolicy",
]
