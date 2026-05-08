"""Unified runtime template for custom calculator roles.

This module defines :class:`RuntimeCalculatorBase`, the middle layer between
the specialized role bases and the lowest-level :class:`CalculatorBase`.

Inheritance Position
--------------------
Most custom nodes should start with one of the specialized role bases:

- :class:`PropertyBase` for read-only derived values
- :class:`FilterBase` for boolean masks
- :class:`TransformBase` for temporary mutations

All three inherit from :class:`RuntimeCalculatorBase`.

Use :class:`RuntimeCalculatorBase` directly when:

- the node does not naturally fit property, filter, or transform semantics
- the node still follows the standard runtime lifecycle
- you want dynamic parameter resolution and :class:`ParamView`
- you do not want to reimplement :meth:`CalculatorBase.execute`

Use :class:`CalculatorBase` instead only when you need full control over
:meth:`CalculatorBase.execute`.

Lifecycle
---------
:class:`RuntimeCalculatorBase` standardizes subclass execution as::

    make_runtime -> resolve_params -> prepare_resolved_params -> compute -> wrap_raw

This is the same lifecycle used by the built-in role bases. It keeps dynamic
parameter handling, prepared parameter views, and raw-value wrapping consistent
across custom calculator types.

Minimal Example
---------------
A :class:`RuntimeCalculatorBase` subclass usually uses the dataclass helper and
implements :meth:`compute`::

    @RuntimeCalculatorBase.dataclass
    class Ratio(RuntimeCalculatorBase[dict[str, float], float]):
        numerator: Param[float]
        denominator: Param[float]

        def compute(self, runtime, params):
            if params.denominator == 0:
                raise ValueError("denominator is zero")
            return {
                "numerator": float(params.numerator),
                "denominator": float(params.denominator),
                "ratio": float(params.numerator / params.denominator),
            }

        def public_value(self, value):
            return value["ratio"]

    result = Ratio(ParamSum("mass"), ParamContain()).run(sim)
    print(result.value)

Why Use This Instead Of CalculatorBase
--------------------------------------
:class:`RuntimeCalculatorBase` is appropriate when the node still looks like a
single computation after parameter resolution.

:class:`CalculatorBase` is more appropriate when the node must:

- evaluate child calculators manually through the execution context
- customize execution order directly
- manage custom orchestration or branching behavior
- define behavior that does not fit the standard lifecycle

Main Extension Points
---------------------
The main hooks are:

- :meth:`compute` for the main calculation
- :meth:`prepare_resolved_params` when :class:`ParamView` is not enough
- :meth:`wrap_raw` when the raw runtime payload needs an extra wrapper
- :meth:`make_runtime` when a specialized runtime facade is needed

Notes
-----
If a node can be described as a property, filter, or transform, prefer the
more specific role base. :class:`RuntimeCalculatorBase` is the generic template
layer, not the default first choice.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pynbodyext.core.calculate.params.fields import ParamView
from pynbodyext.core.calculate.runtime import CalcRuntime, bind_runtime

from .base import CalculatorBase

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput

    from .base import BoundCalculator
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

        with bind_runtime(runtime):
            with ctx.phase(self, "resolve_params"):
                values = self.resolve_params(runtime)
                params = self.prepare_resolved_params(runtime, values)

            with ctx.phase(self, "calculate"):
                computed = self.compute(runtime, params)

            return self.wrap_raw(runtime, computed)
