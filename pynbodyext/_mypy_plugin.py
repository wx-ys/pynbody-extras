"""Minimal mypy plugin: unwrap ``Param[T]`` to ``T`` in calculator constructors.

``@CalculatorBase.dataclass`` / ``@dataclass_calc`` are ``dataclass_transform``
decorators, so mypy builds the constructor from the field annotations and yields
``__init__(prop: Param[float])``.  The built-in dataclass_transform re-generates
``__init__``, so instead of mutating it we hook ``get_function_signature_hook``
at type-check time and rewrite the constructor's argument types from ``Param[T]``
to their value type ``T``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.plugin import ClassDefContext, FunctionSigContext, Plugin
from mypy.types import CallableType, Instance, Type, get_proper_type

if TYPE_CHECKING:
    from collections.abc import Callable

PARAM_FULLNAME = "pynbodyext.core.calculate.params.fields.Param"

_DECORATORS = {
    "pynbodyext.core.calculate.nodes.base.CalculatorBase.dataclass",
    "pynbodyext.core.calculate.params.declarative.dataclass_calc",
}

_CALCULATOR_CLASSES: set[str] = set()


def _record_calculator(ctx: ClassDefContext) -> None:
    _CALCULATOR_CLASSES.add(ctx.cls.fullname)


def _unwrap_param_type(arg_type: Type) -> Type:
    proper = get_proper_type(arg_type)
    if isinstance(proper, Instance) and proper.type.fullname == PARAM_FULLNAME:
        if proper.args:
            return proper.args[0]
    return arg_type


def _rewrite_signature(ctx: FunctionSigContext) -> CallableType:
    fn = ctx.default_signature
    new_arg_types = [_unwrap_param_type(t) for t in fn.arg_types]
    return fn.copy_modified(arg_types=new_arg_types)


class CalculatorParamPlugin(Plugin):
    def get_class_decorator_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        if fullname in _DECORATORS:
            return _record_calculator
        return None

    def get_function_signature_hook(self, fullname: str) -> Callable[[FunctionSigContext], CallableType] | None:
        if fullname in _CALCULATOR_CLASSES:
            return _rewrite_signature
        return None


def plugin(version: str) -> type[CalculatorParamPlugin]:
    return CalculatorParamPlugin
