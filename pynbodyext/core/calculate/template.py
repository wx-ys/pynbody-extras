"""Unified runtime template for calculator role base classes."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from .base import CalculatorBase
from .fields import ParamView
from .runtime import CalcRuntime

if TYPE_CHECKING:
    from .base import BoundCalculator
    from .context import ExecutionContext, NodeInput
    from .filters import FilterBase
    from .transforms import TransformBase

TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")
TRuntime = TypeVar("TRuntime", bound="RuntimeCalculatorBase[Any, Any]")


class RuntimeCalculatorBase(CalculatorBase[TRaw, TPublic], Generic[TRaw, TPublic], ABC):
    """Calculator base with a unified subclass lifecycle.

    The lifecycle is shared by property, filter, and transform nodes:

    ``make_runtime -> resolve_params -> prepare_params -> compute -> wrap_raw``.
    """

    def with_filter(self: TRuntime, filt: FilterBase) -> BoundCalculator[TRuntime, TRaw, TPublic]:
        """Return a scoped calculator while preserving the concrete base type."""
        return cast("BoundCalculator[TRuntime, TRaw, TPublic]", super().with_filter(filt))

    def filter(self: TRuntime, filt: FilterBase) -> BoundCalculator[TRuntime, TRaw, TPublic]:
        """Alias for :meth:`with_filter`."""
        return self.with_filter(filt)

    def with_transformation(
        self: TRuntime,
        transform: TransformBase[Any],
        *,
        revert: bool = True,
    ) -> BoundCalculator[TRuntime, TRaw, TPublic]:
        """Return a transformed calculator while preserving the concrete base type."""
        return cast(
            "BoundCalculator[TRuntime, TRaw, TPublic]",
            super().with_transformation(transform, revert=revert),
        )

    def transform(
        self: TRuntime,
        transform: TransformBase[Any],
        *,
        revert: bool = True,
    ) -> BoundCalculator[TRuntime, TRaw, TPublic]:
        """Alias for :meth:`with_transformation`."""
        return self.with_transformation(transform, revert=revert)

    def make_runtime(self, ctx: ExecutionContext, input: NodeInput) -> CalcRuntime:
        """Create the runtime facade passed to advanced hooks."""
        return CalcRuntime(ctx=ctx, input=input, node=self)

    def resolve_params(self, runtime: CalcRuntime) -> dict[str, Any]:
        """Resolve dynamic constructor values for this run."""
        return self.resolve_dynamic_params(runtime.ctx, runtime.input)

    def prepare_params(self, sim: Any, values: dict[str, Any]) -> Any:
        """Convert constructor parameters into the object passed to compute hooks.

        This keeps the existing ``prepare_params(sim, values)`` shape as a
        transition bridge.  New subclasses may override
        :meth:`prepare_resolved_params` when they need the full runtime facade.
        """
        params = ParamView.from_calculator(self, values)
        return params if params else None

    def prepare_resolved_params(self, runtime: CalcRuntime, values: dict[str, Any]) -> Any:
        """Prepare resolved parameters using the runtime facade."""
        return self.prepare_params(runtime.sim, values)

    def compute(self, runtime: CalcRuntime, params: Any) -> Any:
        """Compute the role-specific user value."""
        raise NotImplementedError(f"{type(self).__name__} must implement compute() or a role-specific hook.")

    def wrap_raw(self, runtime: CalcRuntime, computed: Any) -> TRaw:
        """Wrap the computed user value into the framework raw value."""
        return computed

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> TRaw:
        """Execute the unified runtime lifecycle."""
        runtime = self.make_runtime(ctx, input)

        with ctx.phase(self, "resolve_params"):
            values = self.resolve_params(runtime)
            params = self.prepare_resolved_params(runtime, values)

        with ctx.phase(self, "calculate"):
            computed = self.compute(runtime, params)

        return self.wrap_raw(runtime, computed)
