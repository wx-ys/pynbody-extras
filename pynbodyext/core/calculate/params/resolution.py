"""Dynamic parameter resolution for calculator-valued arguments.

This module implements runtime resolution for dynamic constructor arguments.
It allows one parameter field to accept literals, callables, calculators, and
nested containers with consistent dependency and signature behavior.

Why This Exists
---------------
Without dynamic parameter support, each calculator would need custom logic to
differentiate static inputs from runtime-dependent inputs.

With this module, calculator authors can declare dynamic fields and rely on a
shared resolver pipeline.

What Can Be Dynamic
-------------------
A dynamic value may be:

- a plain literal (for example ``10.0``)
- a callable resolved against the active simulation
- another calculator whose public value is needed first
- a nested mapping or sequence containing any of the above

Authoring Surface
-----------------
Most users do not import this module directly. Instead, they use:

- :class:`Param` fields in dataclass-style calculators
- ``dynamic_param_specs`` on manually authored classes

Runtime resolution then happens through role-base or template lifecycle hooks.

Key Helpers
-----------
The most important helpers for framework-level code are:

- :func:`dynamic_value_signature` for stable signature fragments
- :func:`dynamic_value_dependencies` for nested calculator dependency discovery
- :func:`resolve_dynamic_value` for one-value resolution in runtime/standalone
  contexts

Debugging Orientation
---------------------
This module is the first place to inspect when:

- calculator-valued constructor arguments do not resolve as expected
- dynamic values are not contributing to signatures correctly
- nested container inputs lose dependency tracking

Notes
-----
Dynamic parameter identity and dependency propagation must remain stable for
cache correctness. If dynamic signatures are unstable, repeated runs may
recompute unexpectedly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import Parameter, signature as inspect_signature
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.core.calculate.runtime.context import ExecutionContext
from pynbodyext.core.calculate.runtime.input import NodeInput
from pynbodyext.util._type import get_signature_safe

if TYPE_CHECKING:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput
    from pynbodyext.core.calculate.runtime.options import RunOptions

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
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

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
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

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
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

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
