"""Declarative parameter helpers for dataclass-style calculators.

This module defines :class:`Param`, :class:`ParamSpec`, and :class:`ParamView`,
which support declarative calculator definitions built with
:meth:`CalculatorBase.dataclass` or :func:`dataclass_calc`.

Use this module when you want constructor fields to declare whether they are:

- dynamic runtime-resolved inputs
- static configuration values
- part of the calculator signature
- associated with a simulation field for unit-aware resolution

Most users encounter :class:`Param` first.

Basic Example
-------------
A calculator class can be declared with dataclass-style fields::

    from pynbodyext.core.calculate import Param, PropertyBase

    @PropertyBase.dataclass
    class MassInsideRadius(PropertyBase[float]):
        radius: Param[float] = Param(field_name="r")
        family: Param[str] = Param(default="star", dynamic=False)

        def calculate_with_params(self, sim, params=None):
            radius = params.radius
            mask = sim["r"] < radius
            return float(sim["mass"][mask].sum())

    result = MassInsideRadius(10.0, family="star").run(sim)
    print(result.value)

In this example, ``radius`` is resolved dynamically in the units of ``r``,
while ``family`` is treated as ordinary static configuration.

Dynamic Versus Static Fields
----------------------------
A :class:`Param` field is dynamic by default. This means it may accept:

- a literal value
- a callable resolved at run time
- another calculator whose public value is needed first

Mark a field with ``dynamic=False`` when it should stay as ordinary static
metadata rather than becoming part of runtime parameter resolution.

For example::

    label: Param[str] = Param(default="mass", dynamic=False)

Signature Control
-----------------
Use ``signature=False`` for fields that should not affect calculator identity,
such as display-only labels or cosmetic options::

    label: Param[str] = Param(default="mass", dynamic=False, signature=False)

ParamView
---------
Resolved parameters are usually presented to the calculator as a
:class:`ParamView`. It supports both attribute and item access::

    def calculate_with_params(self, sim, params=None):
        radius = params.radius
        same_radius = params["radius"]
        return float(sim["mass"][sim["r"] < radius].sum())

When To Use This Module
-----------------------
Reach for declarative fields when a calculator has several constructor
arguments and you want the class definition itself to show which ones are
runtime-resolved, unit-aware, or static.

For very small calculators, a manual ``__init__`` plus
``dynamic_param_specs`` may still be simpler.

Notes
-----
If a dataclass-style calculator is not resolving parameters as expected,
inspect the collected parameter specs and the dynamic resolution helpers in
:mod:`.params`.

If a field seems to affect caching unexpectedly, check whether its
``signature`` setting matches the intended identity semantics.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    get_origin,
    overload,
)

from .resolution import DynamicParamSpec, dynamic_value_dependencies

if TYPE_CHECKING:
    from collections.abc import Callable

    from pynbodyext.core.calculate.nodes.base import CalculatorBase

T = TypeVar("T")

if TYPE_CHECKING:
    DynamicParam: TypeAlias = T | Callable[[Any], T] | CalculatorBase[Any, T]


class Param(Generic[T]):
    """Unified field specifier for declarative calculator parameters.

    Parameters
    ----------
    default : Any, optional
        Default value for the parameter. If not provided, the parameter is required.
    dynamic : bool, default: True
        Whether the parameter is dynamic (resolved at runtime) or static (fixed at construction).
    field_name : str, optional
        Optional name of the simulation field associated with this parameter, used for automatic unit handling.
    target_units : Any, optional
        Optional target units for the parameter value, used for automatic unit handling. If not provided, no conversion is applied.
    optional_units : bool, default: True
        Whether units are optional for this parameter. If False, a unit-aware value is required.
    signature : bool, default: True
        Whether this parameter should be included in the calculator's signature for caching and hashing purposes.

    Examples
    --------
        .. code-block:: python

        @CalculatorBase.dataclass
        class MyCalculator(CalculatorBase[MyParams, MyResult]):
            # Required dynamic parameter with no default
            radius: Param[float]
            # Optional dynamic parameter with default
            mass: Param[float] = 1.0
            # Dynamic parameter with default and associated field for unit handling
            pos: Param[SimArray] = Param(default=(0, 0, 0), field_name="pos")
            # Not a Param, treated as a static parameter
            description: str = 'A simple calculator'
            #

    """

    @overload
    def __new__(
        cls,
        default: DynamicParam[T],
        *,
        field_name: str | None = None,
        target_units: Any | None = None,
        optional_units: bool = True,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any: ...

    @overload
    def __new__(
        cls,
        *,
        field_name: str | None = None,
        target_units: Any | None = None,
        optional_units: bool = True,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any: ...

    def __new__(
        cls,
        default: Any = MISSING,
        *,
        field_name: str | None = None,
        target_units: Any | None = None,
        optional_units: bool = True,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any:
        spec = ParamSpec(
            name="",
            kind="dynamic",
            field_name=field_name,
            target_units=target_units,
            optional_units=optional_units,
            signature=signature,
        )

        kwargs: dict[str, Any] = {
            "metadata": _merge_metadata(spec),
            "init": init,
            "kw_only": kw_only,
        }
        if default is not MISSING:
            kwargs["default"] = default
        return field(**kwargs)

    # ── static() classmethod ─────────────────────────────────────────────────

    @overload
    @classmethod
    def static(
        cls,
        default: T,
        *,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> T: ...

    @overload
    @classmethod
    def static(
        cls,
        *,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any: ...

    @classmethod
    def static(
        cls,
        default: Any = MISSING,
        *,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any:
        """Create a static (non-dynamic) dataclass field specifier.
        """
        spec = ParamSpec(name="", kind="static", signature=signature)
        kwargs: dict[str, Any] = {
            "metadata": _merge_metadata(spec),
            "init": init,
            "kw_only": kw_only,
        }
        if default is not MISSING:
            kwargs["default"] = default
        return field(**kwargs)


_PARAM_METADATA_KEY = "pynbodyext_calculate_param"


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Declarative metadata for one calculator constructor field."""

    name: str
    kind: Literal["static", "dynamic"]
    field_name: str | None = None
    target_units: Any | None = None
    optional_units: bool = True
    signature: bool = True

    def as_dynamic_param_spec(self) -> DynamicParamSpec:
        """Return runtime resolver metadata for dynamic parameters."""
        return DynamicParamSpec(
            field_name=self.field_name,
            target_units=self.target_units,
            optional_units=self.optional_units,
        )


@dataclass(frozen=True, slots=True)
class ParamView:
    """Attribute and item access wrapper for resolved calculator parameters."""

    dynamic: dict[str, Any]
    static: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        dynamic = dict(self.dynamic)
        static = dict(self.static)
        object.__setattr__(self, "dynamic", dynamic)
        object.__setattr__(self, "static", static)
        object.__setattr__(self, "data", {**static, **dynamic})

    @classmethod
    def from_calculator(cls, instance: Any, dynamic_values: dict[str, Any]) -> ParamView:
        """Build a parameter view from declarative fields and resolved values."""
        dynamic: dict[str, Any] = {}
        static: dict[str, Any] = {}
        specs = collect_param_specs(type(instance))

        if not specs:
            return cls(dict(dynamic_values))

        for spec in specs:
            if spec.kind == "dynamic":
                dynamic[spec.name] = dynamic_values[spec.name]
            else:
                static[spec.name] = getattr(instance, spec.name)
        return cls(dynamic=dynamic, static=static)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> Any:
        return self.data[name]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, name: str, default: Any = None) -> Any:
        return self.data.get(name, default)


def _merge_metadata(spec: ParamSpec) -> dict[str, Any]:
    return {_PARAM_METADATA_KEY: spec}

def _raw_annotations(cls: type[Any]) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        annotations.update(getattr(base, "__annotations__", {}))
    return annotations

def _is_param_annotation(annotation: Any) -> bool:
    if annotation is Param:
        return True
    if get_origin(annotation) is Param:
        return True
    if isinstance(annotation, str):
        text = annotation if " " not in annotation else annotation.replace(" ", "")
        return text == "Param" or text.startswith("Param[")
    return getattr(annotation, "__origin__", None) is Param

def collect_param_specs(cls: type[Any]) -> tuple[ParamSpec, ...]:
    """Collect declarative calculator field metadata from a dataclass class."""
    cached = getattr(cls, "__calculate_param_specs__", None)
    if cached is not None:
        return cached

    if not is_dataclass(cls):
        return ()

    hints = _raw_annotations(cls)
    specs: list[ParamSpec] = []

    for item in fields(cls):
        raw = item.metadata.get(_PARAM_METADATA_KEY)
        hint = hints.get(item.name)

        if raw is not None:
            specs.append(
                ParamSpec(
                    name=item.name,
                    kind=raw.kind,
                    field_name=raw.field_name,
                    target_units=raw.target_units,
                    optional_units=raw.optional_units,
                    signature=raw.signature,
                )
            )
            continue

        if _is_param_annotation(hint):
            specs.append(ParamSpec(name=item.name, kind="dynamic"))
            continue

        specs.append(ParamSpec(name=item.name, kind="static"))

    result = tuple(specs)
    try:
        type.__setattr__(cls, "__calculate_param_specs__", result)
    except Exception:
        pass
    return result


def declarative_dynamic_param_specs(cls: type[Any]) -> dict[str, DynamicParamSpec]:
    """Return dynamic resolver specs declared on a dataclass calculator."""
    return {
        spec.name: spec.as_dynamic_param_spec()
        for spec in collect_param_specs(cls)
        if spec.kind == "dynamic"
    }



def declarative_dependencies(instance: Any) -> list[CalculatorBase[Any, Any]]:
    """Return calculator dependencies nested in declarative dynamic fields."""
    deps: list[CalculatorBase[Any, Any]] = []
    for spec in collect_param_specs(type(instance)):
        if spec.kind == "dynamic":
            deps.extend(dynamic_value_dependencies(getattr(instance, spec.name)))
    return deps
