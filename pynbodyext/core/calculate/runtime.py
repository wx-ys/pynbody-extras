"""Runtime helper objects for calculator subclass extension hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import CalculatorBase
    from .context import ExecutionContext, NodeInput


@dataclass(slots=True)
class CalcRuntime:
    """Small runtime facade passed to advanced calculator hooks."""

    ctx: ExecutionContext
    input: NodeInput
    node: CalculatorBase[Any, Any]

    @property
    def sim(self) -> Any:
        return self.input.active_sim

    def evaluate(self, child: CalculatorBase[Any, Any]) -> Any:
        return self.ctx.evaluate(child, self.input)

    def public_value(self, child: CalculatorBase[Any, Any]) -> Any:
        return self.ctx.public_value(child, self.input)

    def raw_value(self, child: CalculatorBase[Any, Any]) -> Any:
        return self.ctx.raw_value(child, self.input)

    def phase(self, name: str) -> Any:
        return self.ctx.phase(self.node, name)


@dataclass(slots=True)
class TransformRuntime(CalcRuntime):
    """Runtime facade for transform hooks."""

    measure_input: NodeInput
    measure_sim: Any
    target: Any
