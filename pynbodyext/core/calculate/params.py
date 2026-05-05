"""Dynamic parameter support for calculator-valued arguments.

Dynamic parameters let calculators, callables, constants, arrays, and nested
containers be accepted by the same constructor.  Values are resolved at run
time with access to the active :class:`ExecutionContext` and
:class:`NodeInput`.

This mechanism is what allows a calculator constructor to accept values such as:

- a plain number like ``10.0``
- a callable like ``lambda sim: sim["r"].max() / 2``
- another calculator whose value is computed first
- a nested mapping or sequence containing any of the above

Why Dynamic Parameters Exist
----------------------------
Without dynamic parameters, every calculator constructor would need to decide
up front whether an argument is a literal value or a runtime-dependent value.

With this module, calculator authors can write a single class that supports
both static and runtime-resolved arguments in a uniform way.

Basic Example
-------------
A property can declare that one constructor argument should be resolved in the
units of a simulation field::

    from pynbodyext.core.calculate import PropertyBase

    class MassInsideRadius(PropertyBase[float]):
        dynamic_param_specs = {"radius": "r"}

        def __init__(self, radius):
            super().__init__()
            self.radius = radius

        def instance_signature(self):
            return ("mass_inside_radius", self.radius)

        def calculate_with_params(self, sim, params=None):
            radius = params["radius"]
            mask = sim["r"] < radius
            return float(sim["mass"][mask].sum())

This class can then be used with either a constant radius or a runtime value.

Constant, Callable, And Calculator Inputs
-----------------------------------------
Dynamic parameters can come from several sources::

    fixed = MassInsideRadius(10.0)
    from_callable = MassInsideRadius(lambda sim: sim["r"].max() / 2)

A calculator-valued argument works too, as long as the argument is itself a
calculator whose public value can be resolved before use.

Declaring Dynamic Parameters
----------------------------
Calculator authors opt in by defining ``dynamic_param_specs`` on the class::

    class RadiusBetween(FilterBase):
        dynamic_param_specs = {"rmin": "r", "rmax": "r"}

        def __init__(self, rmin, rmax):
            super().__init__()
            self.rmin = rmin
            self.rmax = rmax

        def instance_signature(self):
            return ("radius_between", self.rmin, self.rmax)

        def build_mask(self, sim, params):
            return (sim["r"] >= params["rmin"]) & (sim["r"] < params["rmax"])

The values of ``rmin`` and ``rmax`` may now be constants, callables, or
calculators, and they will be normalized before ``build_mask`` runs.

Nested Structures
-----------------
Dynamic values can also appear inside mappings and sequences.  This is useful
when a calculator accepts structured parameter bundles::

    {
        "inner": 5.0,
        "outer": lambda sim: sim["r"].max() / 2,
    }

The helper functions in this module preserve signatures and dependencies across
such nested structures.

Key Helpers
-----------
The most commonly important helpers for framework authors are:

- :func:`dynamic_value_signature` to make runtime-valued arguments contribute a
  stable signature fragment
- :func:`dynamic_value_dependencies` to extract nested calculator dependencies
- :func:`resolve_dynamic_value` to resolve one dynamic value in a runtime or
  standalone context

When You Usually Do Not Need This Module
----------------------------------------
Most end users do not need to import anything from this module directly.
Instead, they benefit from it indirectly when using calculators that accept
runtime-valued arguments.

This module is most relevant when writing new calculator classes or debugging
why a parameter is not being resolved as expected.

Notes
-----
If a calculator accepts constructor arguments that may be runtime-dependent,
its :meth:`instance_signature` should usually include the unresolved value
object, while dynamic resolution itself should happen through
``dynamic_param_specs`` and the base-class hooks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import Parameter, signature as inspect_signature
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.util._type import get_signature_safe

from .context import ExecutionContext, NodeInput

if TYPE_CHECKING:
    from .base import CalculatorBase
    from .context import ExecutionContext, NodeInput, RunOptions

TPublic = TypeVar("TPublic")


@dataclass(frozen=True, slots=True)
class DynamicParamSpec:
    """Unit metadata for a dynamic parameter.

    Parameters
    ----------
    field_name : str, optional
        Simulation field whose units should be used when converting unit-like
        values.
    target_units : object, optional
        Explicit target units.  When omitted, ``field_name`` units are used if
        available.
    optional_units : bool, default: True
        If true, unit conversion is skipped when the field has no units.
    """

    field_name: str | None = None
    target_units: Any | None = None
    optional_units: bool = True


def dynamic_value_signature(value: Any) -> Any:
    """Return a stable signature fragment for a dynamic parameter value."""
    from .base import CalculatorBase

    signature: Any

    if isinstance(value, CalculatorBase):
        signature = value.signature()
    elif isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        signature = value.item() if isinstance(value, np.generic) else value
    elif isinstance(value, np.ndarray):
        if value.size <= 16:
            signature = (
                "array",
                tuple(value.shape),
                tuple(np.asarray(value).ravel().tolist()),
            )
        else:
            signature = ("array", tuple(value.shape), str(value.dtype), id(value))
    elif isinstance(value, Mapping):
        signature = {
            "__dict__": [
                [dynamic_value_signature(key), dynamic_value_signature(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ]
        }
    elif isinstance(value, (tuple, list)):
        signature = tuple(dynamic_value_signature(item) for item in value)
    else:
        signature = get_signature_safe(value, fallback_to_id=True)

    return signature


def dynamic_value_dependencies(value: Any) -> list[CalculatorBase[Any, Any]]:
    """Return calculator dependencies nested inside a dynamic value."""
    from .base import CalculatorBase

    if isinstance(value, CalculatorBase):
        return [value]
    if isinstance(value, Mapping):
        deps: list[CalculatorBase[Any, Any]] = []
        for item in value.values():
            deps.extend(dynamic_value_dependencies(item))
        return deps
    if isinstance(value, (tuple, list)):
        deps = []
        for item in value:
            deps.extend(dynamic_value_dependencies(item))
        return deps
    return []


class ValueResolver(Protocol):
    @property
    def active_sim(self) -> Any | None: ...

    def calculator_value(self, node: CalculatorBase[Any, TPublic]) -> TPublic: ...


@dataclass(slots=True)
class RuntimeValueResolver:
    ctx: ExecutionContext
    input: NodeInput

    @property
    def active_sim(self) -> Any:
        return self.input.active_sim

    def calculator_value(self, node: CalculatorBase[Any, TPublic]) -> TPublic:
        return self.ctx.public_value(node, self.input)


@dataclass(slots=True)
class StandaloneValueResolver:
    sim: Any | None
    options: RunOptions | None = None

    @property
    def active_sim(self) -> Any | None:
        return self.sim

    def calculator_value(self, node: CalculatorBase[Any, TPublic]) -> TPublic:
        if self.sim is None:
            raise ValueError("calculator-valued parameters require sim")
        return node.value(self.sim, options=self.options)


def _call_dynamic_callable(value: Any, resolver: ValueResolver) -> Any:
    sim = resolver.active_sim
    if sim is None:
        raise ValueError("callable parameters require sim")

    try:
        parameters = list(inspect_signature(value).parameters.values())
    except (TypeError, ValueError):
        return value(sim)

    required_positional = [
        parameter for parameter in parameters
        if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is Parameter.empty
    ]

    if len(required_positional) >= 2 and isinstance(resolver, RuntimeValueResolver):
        return value(resolver.ctx, resolver.input)

    return value(sim)


def _field_has_units(sim: Any | None, field_name: str | None) -> bool:
    if sim is None or field_name is None:
        return False
    try:
        return hasattr(sim[field_name], "units")
    except Exception:
        return False


def _convert_value_to_units(
    value: Any,
    sim: Any | None,
    *,
    field_name: str | None = None,
    target_units: Any | None = None,
    optional_units: bool = True,
    coerce_unit_string: bool = False,
) -> Any:
    if isinstance(value, str) and coerce_unit_string:
        value = units.Unit(value)

    if optional_units and target_units is None and not _field_has_units(sim, field_name):
        field_name = None

    if target_units is None:
        if field_name is None or sim is None:
            return value
        target_units = sim[field_name].units

    target_unit = units.Unit(target_units)
    context = sim.conversion_context() if sim is not None and hasattr(sim, "conversion_context") else {}

    if isinstance(value, str):
        value = units.Unit(value)
    if isinstance(value, units.UnitBase):
        return float(value.in_units(target_unit, **context))
    if isinstance(value, SimArray):
        converted = value.in_units(target_unit, **context)
        return converted.item() if converted.ndim == 0 or converted.size == 1 else converted
    if isinstance(value, np.ndarray) and (value.ndim == 0 or value.size == 1):
        return value.item()
    return value


def resolve_value_for(
    resolver: ValueResolver,
    value: Any,
    *,
    field_name: str | None = None,
    target_units: Any | None = None,
    optional_units: bool = True,
    allow_calculator: bool = True,
    allow_callable: bool = True,
    coerce_unit_string: bool = False,
) -> Any:
    from .base import CalculatorBase

    if isinstance(value, CalculatorBase):
        if resolver.active_sim is None and not allow_calculator:
            return value
        value = resolver.calculator_value(value)
    elif callable(value):
        if resolver.active_sim is None and not allow_callable:
            return value
        value = _call_dynamic_callable(value, resolver)

    return _convert_value_to_units(
        value,
        resolver.active_sim,
        field_name=field_name,
        target_units=target_units,
        optional_units=optional_units,
        coerce_unit_string=coerce_unit_string,
    )


def resolve_dynamic_value(
    ctx: ExecutionContext,
    input: NodeInput,
    value: Any,
    *,
    field_name: str | None = None,
    target_units: Any | None = None,
    optional_units: bool = True,
) -> Any:
    return resolve_value_for(
        RuntimeValueResolver(ctx, input),
        value,
        field_name=field_name,
        target_units=target_units,
        optional_units=optional_units,
    )
