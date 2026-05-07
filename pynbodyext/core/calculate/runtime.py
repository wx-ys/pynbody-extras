"""Runtime helper objects for calculator subclass extension hooks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .base import CalculatorBase
    from .context import ExecutionContext, NodeInput


_CURRENT_RUNTIME: ContextVar[Any] = ContextVar(
    "pynbodyext_calculate_current_runtime",
    default=None,
)


def current_runtime() -> CalcRuntime | None:
    """Return the calculator runtime currently executing on this context."""
    return _CURRENT_RUNTIME.get()


@contextmanager
def bind_runtime(runtime: CalcRuntime) -> Iterator[None]:
    """Temporarily expose ``runtime`` to simple subclass hooks."""
    token = _CURRENT_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _CURRENT_RUNTIME.reset(token)


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

    def log(self, level: str, message: str, *, phase: str | None = None) -> None:
        current = self.ctx.current_node
        self.ctx.log(level, message, node_id=current.node_id if current is not None else None, phase=phase)


@dataclass(slots=True)
class TransformRuntime(CalcRuntime):
    """Runtime facade for transform hooks."""

    measure_input: NodeInput
    measure_sim: Any
    target: Any
