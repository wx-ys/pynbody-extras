"""Dataclass-like calculator declaration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast, dataclass_transform, overload

from pynbodyext.core.calculate.nodes.base import CalculatorBase

from .fields import (
    Param,
    collect_param_specs,
    declarative_dependencies,
    declarative_dynamic_param_specs,
)

if TYPE_CHECKING:
    from collections.abc import Callable


TCalc = TypeVar("TCalc", bound=CalculatorBase[Any, Any])



def _declared_dependencies(instance: CalculatorBase[Any, Any]) -> list[CalculatorBase[Any, Any]]:
    return declarative_dependencies(instance)



@overload
def dataclass_calc(cls: type[TCalc], **dataclass_kwargs: Any) -> type[TCalc]: ...


@overload
def dataclass_calc(cls: None = None, **dataclass_kwargs: Any) -> Callable[[type[TCalc]], type[TCalc]]: ...


@dataclass_transform(field_specifiers=(Param,))
def dataclass_calc(
    cls: type[TCalc] | None = None,
    **dataclass_kwargs: Any,
) -> type[TCalc] | Callable[[type[TCalc]], type[TCalc]]:
    """Decorate a calculator subclass with dataclass-style parameters.

    The decorator installs dynamic parameter metadata and default signature /
    dependency hooks from fields declared with :class:`Param`. The decorated class must still
    implement the appropriate compute or role-specific hooks to be functional.
    """

    def wrap(raw_cls: type[TCalc]) -> type[TCalc]:
        if not issubclass(raw_cls, CalculatorBase):
            raise TypeError("dataclass_calc can only decorate CalculatorBase subclasses.")

        original_post_init = raw_cls.__dict__.get("__post_init__")

        def __post_init__(self: Any) -> None:
            self._init_dataclass_base()
            if callable(original_post_init):
                original_post_init(self)

        raw_cls.__post_init__ = __post_init__  # type: ignore[attr-defined]
        dc_cls = cast("type[TCalc]", dataclass(raw_cls, **dataclass_kwargs))
        dynamic_specs = declarative_dynamic_param_specs(dc_cls)
        inherited_specs = dict(getattr(dc_cls, "dynamic_param_specs", {}))
        inherited_specs.update(dynamic_specs)
        dc_cls.dynamic_param_specs = inherited_specs

        if raw_cls.declared_dependencies is CalculatorBase.declared_dependencies:
            type.__setattr__(dc_cls, "declared_dependencies", _declared_dependencies)


        type.__setattr__(dc_cls, "__calculate_param_specs__", collect_param_specs(dc_cls))
        return dc_cls

    if cls is None:
        return wrap
    return wrap(cls)
