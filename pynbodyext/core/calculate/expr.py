"""Symbolic property expression nodes.

The framework represents arithmetic on properties as calculator nodes instead
of evaluating eagerly.  This allows expressions such as ``2 * ParamContain(...)``
to become dependencies that are resolved inside the same run context as the
calculator that consumes them.

Examples
--------
Build a symbolic aperture radius::

    re = ParamContain("r", frac=0.5, parameter="mass")
    aperture = Sphere(2 * re)
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

import numpy as np
from pynbody.array import SimArray

from .base import CalculatorBase
from .enums import BuiltinKinds
from .properties import PropertyBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from .context import ExecutionContext, NodeInput

TValue = TypeVar("TValue")

Addable: TypeAlias = SimArray | float | np.ndarray | int | bool | PropertyBase[Any] | None

_NUMERIC_TYPES = (int, float, np.integer, np.floating, bool)


def _is_numeric_scalar(x: Any) -> bool:
    if isinstance(x, _NUMERIC_TYPES):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _extract_numeric(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def as_property(other: object | None) -> PropertyBase[Any]:
    """Coerce a value or calculator into a property expression node."""
    if isinstance(other, PropertyBase):
        return other
    if isinstance(other, CalculatorBase):
        return CalculatorValueProperty(other)
    return ConstantProperty(other)


def make_binary_op(op_name: str, left: PropertyBase[Any], right: PropertyBase[Any]) -> PropertyBase[Any]:
    """Create a binary operation property."""
    return OpProperty(op_name, [left, right])


def make_unary_op(op_name: str, operand: PropertyBase[Any]) -> PropertyBase[Any]:
    """Create a unary operation property."""
    return OpProperty(op_name, [operand])


def make_clip_op(
    value: PropertyBase[Any],
    *,
    vmin: object | None = None,
    vmax: object | None = None,
) -> PropertyBase[Any]:
    """Create a symbolic ``numpy.clip`` property expression."""
    return OpProperty("clip", [value, as_property(vmin), as_property(vmax)])


def make_associative_op(
    op_name: str,
    left: PropertyBase[Any],
    right: PropertyBase[Any],
) -> PropertyBase[Any]:
    """Create an associative operation and fold constant operands."""
    operands: list[PropertyBase[Any]] = []

    def gather(node: PropertyBase[Any]) -> None:
        if isinstance(node, OpProperty) and node.op_name == op_name:
            for child in node.operands:
                gather(child)
        else:
            operands.append(node)

    gather(left)
    gather(right)

    # all_const = all(isinstance(operand, ConstantProperty) for operand in operands)
    const_operands: list[ConstantProperty[Any]] = [
        operand for operand in operands if isinstance(operand, ConstantProperty)
    ]
    if len(const_operands) == len(operands):
        raw_values = [operand._value for operand in const_operands]
        if all(_is_numeric_scalar(value) or isinstance(value, np.ndarray) for value in raw_values):
            func = operator.add if op_name == "add" else operator.mul
            acc = _extract_numeric(raw_values[0])
            for value in raw_values[1:]:
                acc = func(acc, _extract_numeric(value))
            return ConstantProperty(acc)

    return OpProperty(op_name, operands)


class ConstantProperty(PropertyBase[TValue]):
    """Property expression that always returns a constant value."""

    node_kind = BuiltinKinds.PROPERTY

    def __init__(self, value: TValue, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._value = value

    def instance_signature(self) -> tuple[Any, ...]:
        value = self._value
        if isinstance(value, (int, float, np.floating, bool, str)):
            return ("const", value)
        return ("const_obj", id(value))

    def calculate(self, sim: Any) -> TValue:
        return self._value

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("value", self._value)]
        if self.name is not None:
            fields.append(("name", self.name))
        return fields

    def __repr__(self) -> str:
        return f"ConstantProperty(value={self._value!r})"


class LambdaProperty(PropertyBase[TValue]):
    """Property expression backed by a callable ``func(sim)``."""

    node_kind = BuiltinKinds.PROPERTY

    def __init__(self, func: Callable[[Any], TValue], *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._func = func

    def instance_signature(self) -> tuple[Any, ...]:
        return ("lambda", id(self._func))

    def calculate(self, sim: Any) -> TValue:
        return self._func(sim)

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("func", self._func)]
        if self.name is not None:
            fields.append(("name", self.name))
        return fields

    def __repr__(self) -> str:
        return f"LambdaProperty(func={self._func!r})"


class CalculatorValueProperty(PropertyBase[TValue], Generic[TValue]):
    """Bridge that exposes any calculator as a property expression."""

    node_kind = BuiltinKinds.PROPERTY

    def __init__(self, calculator: CalculatorBase[Any, TValue], *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.calculator = calculator

    def instance_signature(self) -> tuple[Any, ...]:
        return ("calculator_value", self.calculator.signature())

    def declared_dependencies(self) -> list[CalculatorBase[Any,Any]]:
        return [self.calculator]

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("calculator", self.calculator)]
        if self.name is not None:
            fields.append(("name", self.name))
        return fields

    def calculate(self, sim: Any) -> Any:
        raise RuntimeError("CalculatorValueProperty must execute through the engine.")

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> Any:
        with ctx.phase(self, "calculate"):
            return ctx.public_value(self.calculator, input)


class OpProperty(PropertyBase[Any]):
    """Symbolic property operation evaluated by the execution engine."""

    node_kind = BuiltinKinds.OP

    def __init__(
        self,
        op_name: str,
        operands: list[PropertyBase[Any]],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.op_name = op_name
        self.operands = operands

    def instance_signature(self) -> tuple[Any, ...]:
        return ("op", self.op_name)

    def declared_dependencies(self) -> list[CalculatorBase[Any,Any]]:
        deps: list[CalculatorBase[Any,Any]] = []
        deps.extend(self.operands)
        return deps

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("op", self.op_name), ("operands", len(self.operands))]
        if self.name is not None:
            fields.append(("name", self.name))
        return fields

    def calculate(self, sim: Any) -> Any:
        raise RuntimeError("OpProperty must execute through the engine.")

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> Any:
        func_map: dict[str, Callable[[Any, Any], Any]] = {
            "add": operator.add,
            "mul": operator.mul,
            "sub": operator.sub,
            "truediv": operator.truediv,
            "pow": operator.pow,
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
            "eq": operator.eq,
            "ne": operator.ne,
        }
        unary_map: dict[str, Callable[[Any], Any]] = {
            "neg": operator.neg,
            "pos": lambda x: +x,
            "abs": operator.abs,
        }

        with ctx.phase(self, "calculate"):
            if self.op_name in unary_map:
                value = ctx.public_value(self.operands[0], input)
                return unary_map[self.op_name](value)

            if self.op_name == "clip":
                value = ctx.public_value(self.operands[0], input)
                vmin = ctx.public_value(self.operands[1], input)
                vmax = ctx.public_value(self.operands[2], input)
                if isinstance(value, SimArray):
                    clipped = value.copy()
                    clipped[...] = np.clip(value, vmin, vmax)
                    return clipped
                return np.clip(value, vmin, vmax)

            if self.op_name in ("add", "mul"):
                op_func = func_map[self.op_name]
                values = [ctx.public_value(operand, input) for operand in self.operands]
                acc = values[0]
                for value in values[1:]:
                    acc = op_func(acc, value)
                return acc

            if self.op_name in func_map:
                left = ctx.public_value(self.operands[0], input)
                right = ctx.public_value(self.operands[1], input)
                return func_map[self.op_name](left, right)

        raise ValueError(f"Unknown op_name {self.op_name!r}")

    def __repr__(self) -> str:
        return f"OpProperty(op={self.op_name!r}, operands={len(self.operands)})"
