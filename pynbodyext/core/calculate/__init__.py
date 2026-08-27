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
The implementation is organized into six layers:

- :mod:`.nodes` contains calculator node authoring primitives and roles:
    :mod:`.nodes.base`, :mod:`.nodes.runtime_base`, :mod:`.nodes.properties`,
    :mod:`.nodes.filters`, :mod:`.nodes.transforms`, :mod:`.nodes.pipeline`,
    and :mod:`.nodes.expr`.
- :mod:`.params` contains declarative parameter support:
    :mod:`.params.fields`, :mod:`.params.declarative`, and
    :mod:`.params.resolution`.
- :mod:`.runtime` contains execution mechanics:
    :mod:`.runtime.engine`, :mod:`.runtime.context`, :mod:`.runtime.input`,
    :mod:`.runtime.options`, :mod:`.runtime.runtime`, :mod:`.runtime.cache`,
    :mod:`.runtime.scopes`, and :mod:`.runtime.progress`.
- :mod:`.result` contains finished run models, signatures, enums, and
    exceptions: :mod:`.result.result`, :mod:`.result.signature`,
    :mod:`.result.enums`, and :mod:`.result.exceptions`.
- :mod:`.diagnostics` contains :mod:`.diagnostics.trace`,
    :mod:`.diagnostics.perf`, and :mod:`.diagnostics.observer`.
- :mod:`.display` stays independent for shared text, HTML, and mime rendering.

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
        {"mass": StellarMass(), "hot_temp": MeanTemperature().filter(TemperatureAbove(1.0e5))}, name="basic_summary"
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

from .bins import (
    Bin1D,
    BinAxis,
    BinND,
    BinNDResult,
    BinParticlesAccessor,
    BinsArray,
    SubBinNDResult,
    has_axes,
    has_axis,
    register_bin_algorithm,
)
from .diagnostics.observer import AccessEvent, AccessObservation
from .diagnostics.perf import PerfCollector
from .diagnostics.trace import TraceCollector, TraceEvent
from .nodes.base import CalculatorBase, CombinedCalculator
from .nodes.expr import ConstantProperty, LambdaProperty, OpProperty
from .nodes.filters import AndFilter, FilterBase, NotFilter, OrFilter
from .nodes.pipeline import Pipeline
from .nodes.properties import PropertyBase
from .nodes.runtime_base import RuntimeCalculatorBase
from .nodes.transforms import TransformBase, TransformChain, TransformPlan, TransformStep, chain_transforms
from .params import DynamicParamSpec, dynamic_value_dependencies, dynamic_value_signature, resolve_dynamic_value
from .params.declarative import dataclass_calc
from .params.fields import Param, ParamSpec, ParamView, collect_param_specs
from .result.enums import (
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
from .result.exceptions import CalculatorError, CycleError
from .result.result import ErrorInfo, PerfSummary, PhaseRecord, ProvenanceInfo, Result, ResultNode, ValueSummary
from .result.signature import CalculatorSignature, calculator_from_signature, calculator_to_signature
from .runtime import CalcRuntime, TransformRuntime
from .runtime.cache import CacheEvent, ExecutionValue, RuntimeCache
from .runtime.context import ExecutionContext, LogEvent, resolve_value
from .runtime.engine import EvalEngine
from .runtime.input import FilterResult, NodeInput, TransformResult
from .runtime.options import RunOptions
from .runtime.progress import NullProgressSink, ProgressSink
from .runtime.scopes import Scope, ScopeSpec, TransformScope

__all__ = [
    "CalculatorBase",
    "Bin1D",
    "BinND",
    "BinAxis",
    "BinNDResult",
    "SubBinNDResult",
    "BinsArray",
    "BinParticlesAccessor",
    "register_bin_algorithm",
    "has_axis",
    "has_axes",
    "RuntimeCalculatorBase",
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
