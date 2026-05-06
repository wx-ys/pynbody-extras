"""Declarative field helpers for calculator parameters."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    overload,
)

from .params import DynamicParamSpec, dynamic_value_dependencies, dynamic_value_signature

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import CalculatorBase

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
    """

    @overload
    def __new__(
        cls,
        default: DynamicParam[T],
        *,
        dynamic: Literal[True] = True,
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
        dynamic: Literal[True] = True,
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
        default: T,
        *,
        dynamic: Literal[False],
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any: ...

    @overload
    def __new__(
        cls,
        *,
        dynamic: Literal[False],
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any: ...

    def __new__(
        cls,
        default: Any = MISSING,
        *,
        dynamic: bool = True,
        field_name: str | None = None,
        target_units: Any | None = None,
        optional_units: bool = True,
        signature: bool = True,
        init: bool = True,
        kw_only: bool = False,
    ) -> Any:
        kind: Literal["static", "dynamic"] = "dynamic" if dynamic else "static"
        spec = ParamSpec(
            name="",
            kind=kind,
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


def collect_param_specs(cls: type[Any]) -> tuple[ParamSpec, ...]:
    """Collect declarative calculator field metadata from a dataclass class."""
    if not is_dataclass(cls):
        return ()

    specs: list[ParamSpec] = []
    for item in fields(cls):
        raw = item.metadata.get(_PARAM_METADATA_KEY)
        if raw is None:
            specs.append(ParamSpec(name=item.name, kind="static"))
            continue
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

    return tuple(specs)


def declarative_dynamic_param_specs(cls: type[Any]) -> dict[str, DynamicParamSpec]:
    """Return dynamic resolver specs declared on a dataclass calculator."""
    return {
        spec.name: spec.as_dynamic_param_spec()
        for spec in collect_param_specs(cls)
        if spec.kind == "dynamic"
    }


def declarative_instance_signature(instance: Any) -> tuple[Any, ...]:
    """Build a stable signature fragment from declarative calculator fields."""
    values: list[tuple[str, Any]] = []
    for spec in collect_param_specs(type(instance)):
        if spec.kind != "static" or not spec.signature:
            continue
        value = getattr(instance, spec.name)
        values.append((spec.name, value))
    return (type(instance).__name__, tuple(values))


def declarative_dynamic_param_signature(instance: Any) -> tuple[Any, ...]:
    """Build signature fragments for declarative dynamic parameters."""
    values: list[tuple[str, Any]] = []
    for spec in collect_param_specs(type(instance)):
        if spec.kind != "dynamic" or not spec.signature:
            continue
        values.append((spec.name, dynamic_value_signature(getattr(instance, spec.name))))
    return tuple(values)


def declarative_dependencies(instance: Any) -> list[CalculatorBase[Any, Any]]:
    """Return calculator dependencies nested in declarative dynamic fields."""
    deps: list[CalculatorBase[Any, Any]] = []
    for spec in collect_param_specs(type(instance)):
        if spec.kind == "dynamic":
            deps.extend(dynamic_value_dependencies(getattr(instance, spec.name)))
    return deps
