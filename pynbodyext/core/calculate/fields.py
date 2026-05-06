"""Declarative field helpers for calculator parameters."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from .params import DynamicParamSpec, dynamic_value_dependencies, dynamic_value_signature

if TYPE_CHECKING:
    from .base import CalculatorBase

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping as ABCMapping
    DynamicParam: TypeAlias = T | Callable[[Any], T] | CalculatorBase[Any, T]
else:
    class DynamicParam(Generic[T]):
        """Runtime marker used only for subscriptable type annotations."""


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
    """Attribute and item access wrapper for resolved dynamic parameters."""

    data: dict[str, Any]

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


def _merge_metadata(spec: ParamSpec) -> ABCMapping[str, Any]:
    return {_PARAM_METADATA_KEY: spec}


def dynamic(
    default: Any = MISSING,
    *,
    field_name: str | None = None,
    target_units: Any | None = None,
    optional_units: bool = True,
    signature: bool = True,
) -> Any:
    """Declare a dataclass calculator field as runtime-resolved."""
    spec = ParamSpec(
        name="",
        kind="dynamic",
        field_name=field_name,
        target_units=target_units,
        optional_units=optional_units,
        signature=signature,
    )
    if default is MISSING:
        return field(metadata=_merge_metadata(spec))
    return field(default=default, metadata=_merge_metadata(spec))


def static(default: Any = MISSING, *, signature: bool = True) -> Any:
    """Declare a dataclass calculator field as a static constructor value."""
    spec = ParamSpec(name="", kind="static", signature=signature)
    if default is MISSING:
        return field(metadata=_merge_metadata(spec))
    return field(default=default, metadata=_merge_metadata(spec))


def collect_param_specs(cls: type[Any]) -> tuple[ParamSpec, ...]:
    """Collect declarative calculator field metadata from a dataclass class."""
    if not is_dataclass(cls):
        return ()

    specs: list[ParamSpec] = []
    for item in fields(cls):
        raw = item.metadata.get(_PARAM_METADATA_KEY)
        if raw is None:
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
