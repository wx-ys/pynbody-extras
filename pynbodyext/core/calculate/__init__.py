"""Composable calculator framework for pynbody snapshots.

The :mod:`pynbodyext.core.calculate` package provides a small but expressive
execution framework for building reusable analysis graphs on top of
pynbody-like simulation snapshots.

A calculator graph is a directed acyclic graph of nodes.  Each node belongs to
one of four common roles:

- property nodes read values from the active snapshot view
- filter nodes build boolean masks and select subsets
- transform nodes mutate the active frame temporarily
- pipeline nodes evaluate several outputs in one shared run

The engine evaluates shared dependencies once, records provenance, trace, cache,
and performance diagnostics, and returns a :class:`Result` object whose content
is convenient for scripts, notebooks, and debugging.

What This Package Exports
-------------------------
The package root exposes the main public abstractions used to define and run
calculator graphs:

- :class:`CalculatorBase` for custom execution nodes
- :class:`PropertyBase` for read-only derived values
- :class:`FilterBase` for boolean selection masks
- :class:`TransformBase` for temporary state mutations
- :class:`Pipeline` for named multi-output evaluation
- :class:`RunOptions`, :class:`Result`, and supporting runtime/result types

Most end users compose existing calculators.  Most library authors subclass one
of the four base classes above.

Quick Start
-----------
Define and run a simple property calculator::

    from pynbodyext.core.calculate import PropertyBase

    class StellarMass(PropertyBase[float]):
        def instance_signature(self):
            return ("stellar_mass",)

        def calculate(self, sim):
            return float(sim["mass"].sum())

    result = StellarMass().run(sim)
    print(result.value)

Combine a property with a custom filter::

    import numpy as np

    from pynbodyext.core.calculate import FilterBase, PropertyBase

    class TemperatureAbove(FilterBase):
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold

        def instance_signature(self):
            return ("temperature_above", self.threshold)

        def build_mask(self, sim, params):
            return sim["temp"] > self.threshold

    class MeanTemperature(PropertyBase[float]):
        def instance_signature(self):
            return ("mean_temperature",)

        def calculate(self, sim):
            return float(np.asarray(sim["temp"]).mean())

    calc = MeanTemperature().filter(TemperatureAbove(1.0e5))
    result = calc.run(sim, progress="phase")
    print(result.value)

Apply a temporary transform before another calculator::

    from pynbodyext.core.calculate import TransformBase, PropertyBase

    class XShift(TransformBase[dict[str, object]]):
        def __init__(self, dx):
            super().__init__()
            self.dx = dx

        def instance_signature(self):
            return ("x_shift", self.dx)

        def build_handle(self, sim, target, params=None):
            original = target["x"].copy()
            target["x"] = target["x"] + self.dx
            return {"target": target, "original_x": original}

        def cleanup(self, ctx, handle):
            handle["target"]["x"] = handle["original_x"]

        def is_revertible(self, handle):
            return True

    class XMean(PropertyBase[float]):
        def instance_signature(self):
            return ("x_mean",)

        def calculate(self, sim):
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
    result = pipe.run(sim)
    print(result.value["mass"])
    print(result.value["hot_temp"])

Inspect reports and diagnostics after a run::

    result = pipe.run(sim, progress="phase")
    print(result.perf_summary)
    print(result.reports["trace_tree"])
    print(result.reports["cache"])

How To Choose A Base Class
--------------------------
Choose the narrowest base class that matches your use case:

- subclass :class:`PropertyBase` when the node only reads from the active
  snapshot and returns a value
- subclass :class:`FilterBase` when the node computes a boolean mask
- subclass :class:`TransformBase` when the node mutates the active frame and
  returns a handle that may later be cleaned up
- subclass :class:`CalculatorBase` directly only when you need custom execution
  semantics that do not fit the three specialized bases

In most custom nodes you should implement:

- :meth:`instance_signature` so structurally equivalent calculators share a
  stable signature
- :meth:`declared_dependencies` when the node depends on other calculators
- the execution hook required by the chosen base class:
  :meth:`PropertyBase.calculate`,
  :meth:`PropertyBase.calculate_with_params`,
  :meth:`FilterBase.build_mask`,
  :meth:`FilterBase._build_mask_runtime`,
  :meth:`TransformBase.build_handle`,
  :meth:`TransformBase._build_handle_runtime`,
  or :meth:`CalculatorBase.execute`

Results And Diagnostics
-----------------------
Every run returns a :class:`Result`.  It contains:

- ``value``: the public root value
- ``root`` and ``nodes``: execution records for evaluated nodes
- ``named``: nodes registered with ``named()`` or pipeline outputs
- ``perf_summary``: aggregate timing and cache counters
- ``reports``: human-readable trace, cache, and performance reports
- ``diagnostics``: raw trace/cache/log events and other debug-oriented data

For a deeper overview of the execution model, see :mod:`.base`,
:mod:`.properties`, :mod:`.filters`, :mod:`.transforms`, :mod:`.pipeline`,
:mod:`.params`, :mod:`.context`, and :mod:`.result`.
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
    "ValueSummary",
    "PhaseRecord",
    "ErrorInfo",
    "ProvenanceInfo",
    "PerfSummary",
    "RuntimeCache",
    "ExecutionValue",
    "CacheEvent",
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
