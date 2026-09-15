"""Dataclass-like calculator declaration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast, dataclass_transform, overload

from pynbodyext.core.calculate.nodes.base import CalculatorBase

from .fields import (
    Param,
    capture_init_values,
    collect_base_init_fields,
    collect_param_specs,
    declarative_dependencies,
    declarative_dynamic_param_specs,
    declared_annotation,
    declared_param_field,
)

if TYPE_CHECKING:
    from collections.abc import Callable


TCalc = TypeVar("TCalc", bound=CalculatorBase[Any, Any])


def _declared_dependencies(instance: CalculatorBase[Any, Any]) -> list[CalculatorBase[Any, Any]]:
    return declarative_dependencies(instance)


def _materialise_base_init_fields(raw_cls: type[Any]) -> None:
    """Give a decorated subclass the constructor params its role base declares.

    ``CalculatorBase`` role bases (``TransformBase``, ...) are not dataclasses, so
    an attribute they declare — ``move_all`` on ``TransformBase`` — would never
    reach the ``__init__`` that ``dataclass`` generates here.  Each such
    declaration is appended as a field of the subclass instead, so the parameter
    is written once on the base while subclasses keep their own fields first
    (which keeps positional order and the rendered argument order stable) and the
    type checker, which reads the declaration on the base, sees the same
    signature.
    """
    for name, declared in collect_base_init_fields(raw_cls).items():
        annotations = raw_cls.__dict__.get("__annotations__")
        if annotations is None:
            annotations = {}
            raw_cls.__annotations__ = annotations
        if name in annotations or name in raw_cls.__dict__:
            # The subclass declares it itself (or already inherits a field).
            continue
        annotations[name] = declared_annotation(raw_cls, name)
        setattr(raw_cls, name, declared_param_field(declared))


@overload
def dataclass_calc(cls: type[TCalc], **dataclass_kwargs: Any) -> type[TCalc]: ...


@overload
def dataclass_calc(cls: None = None, **dataclass_kwargs: Any) -> Callable[[type[TCalc]], type[TCalc]]: ...


@dataclass_transform(field_specifiers=(Param, Param.static))
def dataclass_calc(
    cls: type[TCalc] | None = None, **dataclass_kwargs: Any
) -> type[TCalc] | Callable[[type[TCalc]], type[TCalc]]:
    """Decorate a calculator subclass with dataclass-style parameters.

    The decorator installs dynamic parameter metadata and default signature /
    dependency hooks from fields declared with :class:`Param`. The decorated class must still
    implement the appropriate compute or role-specific hooks to be functional.
    """

    def wrap(raw_cls: type[TCalc]) -> type[TCalc]:
        if not issubclass(raw_cls, CalculatorBase):
            raise TypeError("dataclass_calc can only decorate CalculatorBase subclasses.")

        _materialise_base_init_fields(raw_cls)

        original_post_init = raw_cls.__dict__.get("__post_init__")

        def __post_init__(self: Any) -> None:
            # Before the hook below (and the subclass's own) can normalise fields.
            capture_init_values(self)
            self._init_dataclass_base()
            if callable(original_post_init):
                original_post_init(self)

        raw_cls.__post_init__ = __post_init__  # type: ignore[attr-defined]
        dataclass_kwargs.setdefault("repr", False)
        dc_cls = cast("type[TCalc]", dataclass(raw_cls, **dataclass_kwargs))
        dynamic_specs = declarative_dynamic_param_specs(dc_cls)
        inherited_specs = dict(getattr(dc_cls, "dynamic_param_specs", {}))
        inherited_specs.update(dynamic_specs)
        dc_cls.dynamic_param_specs = inherited_specs

        if raw_cls.declared_dependencies is CalculatorBase.declared_dependencies:
            type.__setattr__(dc_cls, "declared_dependencies", _declared_dependencies)

        type.__setattr__(dc_cls, "__calculate_param_specs__", collect_param_specs(dc_cls))
        return dc_cls

    return wrap if cls is None else wrap(cls)
