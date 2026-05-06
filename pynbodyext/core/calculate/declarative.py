"""Dataclass-like calculator declaration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast, dataclass_transform, overload

from .base import CalculatorBase
from .fields import (
    collect_param_specs,
    declarative_dependencies,
    declarative_dynamic_param_signature,
    declarative_dynamic_param_specs,
    declarative_instance_signature,
    dynamic,
    static,
)

if TYPE_CHECKING:
    from collections.abc import Callable


TCalc = TypeVar("TCalc", bound=CalculatorBase[Any, Any])


def _instance_signature(instance: CalculatorBase[Any, Any]) -> tuple[Any, ...]:
    return declarative_instance_signature(instance)


def _declared_dependencies(instance: CalculatorBase[Any, Any]) -> list[CalculatorBase[Any, Any]]:
    return declarative_dependencies(instance)


def _dynamic_param_signature(instance: CalculatorBase[Any, Any]) -> tuple[Any, ...]:
    return declarative_dynamic_param_signature(instance)


@overload
def dataclass_calc(cls: type[TCalc], **dataclass_kwargs: Any) -> type[TCalc]: ...


@overload
def dataclass_calc(cls: None = None, **dataclass_kwargs: Any) -> Callable[[type[TCalc]], type[TCalc]]: ...


@dataclass_transform(field_specifiers=(dynamic, static))
def dataclass_calc(
    cls: type[TCalc] | None = None,
    **dataclass_kwargs: Any,
) -> type[TCalc] | Callable[[type[TCalc]], type[TCalc]]:
    """Decorate a calculator subclass with dataclass-style parameters.

    The decorator installs dynamic parameter metadata and default signature /
    dependency hooks from fields declared with :func:`dynamic` or
    :func:`static`.
    """

    def wrap(raw_cls: type[TCalc]) -> type[TCalc]:
        if not issubclass(raw_cls, CalculatorBase):
            raise TypeError("dataclass_calc can only decorate CalculatorBase subclasses.")

        original_post_init = raw_cls.__dict__.get("__post_init__")

        def __post_init__(self: Any) -> None:
            CalculatorBase.__init__(self)
            if callable(original_post_init):
                original_post_init(self)

        raw_cls.__post_init__ = __post_init__  # type: ignore[attr-defined]
        dc_cls = cast("type[TCalc]", dataclass(raw_cls, **dataclass_kwargs))
        dynamic_specs = declarative_dynamic_param_specs(dc_cls)
        inherited_specs = dict(getattr(dc_cls, "dynamic_param_specs", {}))
        inherited_specs.update(dynamic_specs)
        dc_cls.dynamic_param_specs = inherited_specs

        if "instance_signature" not in raw_cls.__dict__:
            type.__setattr__(dc_cls, "instance_signature", _instance_signature)

        if "declared_dependencies" not in raw_cls.__dict__:
            type.__setattr__(dc_cls, "declared_dependencies", _declared_dependencies)

        if "dynamic_param_signature" not in raw_cls.__dict__:
            type.__setattr__(dc_cls, "dynamic_param_signature", _dynamic_param_signature)

        type.__setattr__(dc_cls, "__calculate_param_specs__", collect_param_specs(dc_cls))
        return dc_cls

    if cls is None:
        return wrap
    return wrap(cls)
