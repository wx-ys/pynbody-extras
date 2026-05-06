"""Base class for mutating calculators that return transform handles.

Subclasses implement :meth:`build_handle` or :meth:`_build_handle_runtime`.
The returned handle describes the applied mutation and may later be cleaned up
when the transform is used in a scoped calculator.

When To Use TransformBase
-------------------------
Subclass :class:`TransformBase` when your node:

- changes coordinates, velocities, or another mutable field
- should run before another calculator
- may expose a revertible handle
- should be tracked as a mutating execution step

Simple Subclass
---------------
The simplest transform implements :meth:`instance_signature`,
:meth:`build_handle`, and usually :meth:`cleanup`::

    class XShift(TransformBase[dict[str, object]]):
        def __init__(self, dx):
            super().__init__()
            self.dx = dx

        def instance_signature(self):
            return ("x_shift", self.dx)

        def build_handle(self, sim, target, params=None):
            original = target["x"].copy()
            target["x"] = target["x"] + self.dx
            return {"target": target, "original_x": original}

        def cleanup(self, ctx, handle):
            handle["target"]["x"] = handle["original_x"]

        def is_revertible(self, handle):
            return True

This is the preferred style when the transform can be fully described by the
active snapshot view, the chosen target, and constructor arguments.

Transform With Dynamic Parameters
---------------------------------
Use ``dynamic_param_specs`` when offsets or angles may be calculator-valued or
resolved at run time::

    class ShiftByRadius(TransformBase[dict[str, object]]):
        dynamic_param_specs = {"dx": "x"}

        def __init__(self, dx):
            super().__init__()
            self.dx = dx

        def instance_signature(self):
            return ("shift_by_radius", self.dx)

        def build_handle(self, sim, target, params=None):
            original = target["x"].copy()
            target["x"] = target["x"] + params["dx"]
            return {"target": target, "original_x": original}

        def cleanup(self, ctx, handle):
            handle["target"]["x"] = handle["original_x"]

        def is_revertible(self, handle):
            return True

Full Runtime Hook
-----------------
Override :meth:`_build_handle_runtime` when the transform needs access to
:class:`ExecutionContext`, :class:`NodeInput`, or child calculator evaluation::

    class ShiftToCentre(TransformBase[dict[str, object]]):
        def __init__(self, centre_calc):
            super().__init__()
            self.centre_calc = centre_calc

        def instance_signature(self):
            return ("shift_to_centre", self.centre_calc.signature())

        def declared_dependencies(self):
            return [self.centre_calc]

        def _build_handle_runtime(self, sim, target, params, ctx, input):
            centre = ctx.public_value(self.centre_calc, input)
            original = target["pos"].copy()
            target["pos"] = target["pos"] - centre
            return {"target": target, "original_pos": original}

        def cleanup(self, ctx, handle):
            handle["target"]["pos"] = handle["original_pos"]

        def is_revertible(self, handle):
            return True

Measure-On-Subset Example
-------------------------
Transforms can measure parameters on one subset while mutating the full target
through :meth:`measure_with`::

    centred = ShiftToCentre(centre_calc).measure_with(TemperatureAbove(1.0e5))
    result = StellarMass().transform(centred).run(sim)

Composition Example
-------------------
Transforms chain naturally and can be scoped onto any calculator::

    shift = XShift(1.0)
    calc = MeanTemperature().transform(shift)
    result = calc.run(sim)
    print(result.value)

Multiple transforms can be composed into one plan::

    plan = XShift(1.0).then(ShiftByRadius(2.0))
    result = StellarMass().transform(plan).run(sim)

Notes
-----
The default :meth:`cleanup` implementation only calls ``handle.revert()`` when
such a method exists.  If your transform returns a plain dict, tuple, or other
custom handle, override :meth:`cleanup` and usually :meth:`is_revertible` as
shown above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation

from .context import ExecutionContext, FilterResult, NodeInput, TransformResult
from .display import display_value
from .enums import BuiltinKinds, CachePolicy, EffectPolicy, RevertPolicy, normalize_revert_policy
from .runtime import CalcRuntime, TransformRuntime
from .template import RuntimeCalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .base import CalculatorBase
    from .filters import FilterBase

HandleT = TypeVar("HandleT")

TransformTarget: TypeAlias = SimSnap | Transformation

@dataclass(slots=True)
class TransformStep:
    """One step recorded inside a :class:`TransformChain` handle."""

    transform: TransformBase[Any]
    result: TransformResult[Any]


class TransformBase(
    RuntimeCalculatorBase[TransformResult[HandleT], HandleT],
    Generic[HandleT]
):
    """Base class for mutating calculators that return transform handles.

    Subclasses implement :meth:`build_handle`.  A handle may define ``revert()``
    to support automatic cleanup when the transform is used in a scoped
    calculator.
    """

    node_kind = BuiltinKinds.TRANSFORM
    effect = EffectPolicy.MUTATING
    cacheable = False
    parallel_safe = False
    cache_policy = CachePolicy.NONE

    def __init__(
        self,
        *,
        move_all: bool = True,
        name: str | None = None,
        revert_policy: RevertPolicy | str | bool = RevertPolicy.ALWAYS,
        measure_filter: FilterBase | None = None,
    ) -> None:
        super().__init__(name=name)
        self.move_all = move_all
        self.revert_policy = normalize_revert_policy(revert_policy)
        self.measure_filter = measure_filter

    def instance_signature(self) -> tuple[Any, ...]:
        return ("transform", self.move_all, self.revert_policy.value)

    def signature(self) -> tuple[Any, ...]:
        """Return a signature that includes transform metadata."""
        return (
            self.kind,
            self.__class__.__name__,
            (
                "transform_meta",
                self.move_all,
                self.revert_policy.value,
                self.measure_filter.signature() if self.measure_filter is not None else None,
                self.instance_signature(),
                ("dynamic", self.dynamic_param_signature()),
            ),
            tuple(dep.signature() for dep in self.dependencies()),
        )

    def declared_dependencies(self) -> list[CalculatorBase[Any,Any]]:
        if self.measure_filter is None:
            return []
        return [self.measure_filter]

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields = super()._repr_fields()
        if not self.move_all:
            fields.append(("move_all", self.move_all))
        if self.revert_policy != RevertPolicy.ALWAYS:
            fields.append(("revert", display_value(self.revert_policy)))
        if self.measure_filter is not None:
            fields.append(("measure", self.measure_filter))
        return fields

    def revert(self, policy: RevertPolicy | str | bool = RevertPolicy.ALWAYS) -> TransformBase[HandleT]:
        """Return a copy with a different revert policy."""
        return cast("TransformBase[HandleT]", self._clone(revert_policy=normalize_revert_policy(policy)))

    def measure_with(self, filt: FilterBase | None) -> TransformBase[HandleT]:
        """Return a copy that measures transform parameters on ``filt``."""
        return cast("TransformBase[HandleT]", self._clone(measure_filter=filt))

    def with_filter(self, filt: FilterBase) -> TransformBase[HandleT]:  # type: ignore[override]
        """Measure this transform on a filtered view without narrowing downstream sim."""
        return self.measure_with(filt)

    def filter(self, filt: FilterBase) -> TransformBase[HandleT]:   # type: ignore[override]
        """Alias for measure_with() on transforms."""
        return self.measure_with(filt)


    def public_value(self, value: TransformResult[HandleT]) -> HandleT:
        return value.handle

    def materialize_public(self, ctx: ExecutionContext, value: HandleT) -> HandleT:
        return value

    def resolve_target(self, input: NodeInput) -> Transformation | SimSnap:
        """Return the object that should be mutated by this transform."""
        if input.transform is not None:
            return input.transform.handle
        sim = input.active_sim
        if self.move_all and hasattr(sim, "ancestor"):
            return sim.ancestor
        return sim

    def make_runtime(self, ctx: ExecutionContext, input: NodeInput) -> TransformRuntime:
        measure_input = input
        if self.measure_filter is not None:
            with ctx.phase(self, "measure_filter"):
                filter_result = ctx.raw_value(self.measure_filter, input)
                if not isinstance(filter_result, FilterResult):
                    raise TypeError("measure filters must return FilterResult")
                measure_input = input.with_selection(filter_result)

        return TransformRuntime(
            ctx=ctx,
            input=input,
            node=self,
            measure_input=measure_input,
            measure_sim=measure_input.active_sim,
            target=self.resolve_target(input),
        )

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Convert resolved dynamic values into apply params."""
        return super().prepare_params(sim, dict(values))

    def resolve_params(self, runtime: CalcRuntime) -> dict[str, Any]:
        transform_runtime = cast("TransformRuntime", runtime)
        return self.resolve_dynamic_params(transform_runtime.ctx, transform_runtime.measure_input)

    def prepare_resolved_params(self, runtime: CalcRuntime, values: dict[str, Any]) -> Any:
        transform_runtime = cast("TransformRuntime", runtime)
        return self.prepare_params(transform_runtime.measure_sim, values)

    def _resolve_params_runtime(self, ctx: ExecutionContext, input: NodeInput) -> Any:
        values = self.resolve_dynamic_params(ctx, input)
        return self.prepare_params(input.active_sim, values)


    def build_handle(
        self,
        sim: SimSnap,
        target: TransformTarget,
        params: Mapping[str, Any] | None = None,
    ) -> HandleT:
        """Apply the transform and return a handle."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply(), build_handle(), _build_handle_runtime(), or compute()."
        )

    def apply(
        self,
        sim: SimSnap,
        target: TransformTarget,
        params: Mapping[str, Any] | None = None,
    ) -> HandleT:
        """Apply the transform and return a handle."""
        return self.build_handle(sim, target, params)

    def _build_handle_runtime(
        self,
        sim: SimSnap,
        target: TransformTarget,
        params: Any,
        ctx: ExecutionContext,
        input: NodeInput,
    ) -> HandleT:
        return self.apply(sim, target, params)

    def compute(self, runtime: CalcRuntime, params: Any) -> HandleT:
        transform_runtime = cast("TransformRuntime", runtime)
        return self._build_handle_runtime(
            transform_runtime.measure_sim,
            transform_runtime.target,
            params,
            transform_runtime.ctx,
            transform_runtime.measure_input,
        )

    def sim_after_transform(
        self,
        sim: SimSnap,
        target: TransformTarget,
        handle: HandleT,
    ) -> SimSnap:
        """Return the active simulation view after applying the transform."""
        return sim

    def _sim_after_transform_runtime(
        self,
        sim: SimSnap,
        target: TransformTarget,
        handle: HandleT,
        ctx: ExecutionContext,
        input: NodeInput,
    ) -> SimSnap:
        return self.sim_after_transform(sim, target, handle)

    def wrap_raw(self, runtime: CalcRuntime, computed: HandleT) -> TransformResult[HandleT]:
        transform_runtime = cast("TransformRuntime", runtime)
        sim_after = self._sim_after_transform_runtime(
            transform_runtime.input.active_sim,
            transform_runtime.target,
            computed,
            transform_runtime.ctx,
            transform_runtime.measure_input,
        )
        mutation_generation = transform_runtime.ctx.advance_mutation_generation(f"apply {self.log_label}")
        return TransformResult(
            handle=computed,
            target=transform_runtime.target,
            sim_after=sim_after,
            revertible=self.is_revertible(computed),
            artifacts={"mutation_generation": mutation_generation},
        )

    def is_revertible(self, handle: HandleT) -> bool:
        """Whether ``handle`` can be cleaned up automatically."""
        return self.revert_policy == RevertPolicy.ALWAYS and hasattr(handle, "revert")

    def cleanup(self, ctx: ExecutionContext, handle: HandleT) -> None:
        """Revert a transform handle when it is revertible."""
        if self.is_revertible(handle):
            handle.revert() # type: ignore[attr-defined]
            ctx.advance_mutation_generation(f"revert {self.log_label}")

    @classmethod
    def chain(cls, *transforms: TransformBase[Any]) -> TransformBase[Any]:
        """Create a transform chain from transform nodes."""
        return chain_transforms(*transforms)

    def then(self, transform: TransformBase[Any]) -> TransformBase[Any]:
        """Append another transform after this transform."""
        return chain_transforms(self, transform)


class TransformChain(TransformBase[tuple[TransformStep, ...]]):
    """Sequence of transform nodes executed in order and cleaned in reverse."""

    def __init__(self, *transforms: TransformBase[Any], name: str | None = None) -> None:
        if not transforms:
            raise ValueError("At least one transform must be provided.")
        super().__init__(name=name)
        for transform in transforms:
            if transform.kind != BuiltinKinds.TRANSFORM:
                raise TypeError(f"TransformChain only accepts transform nodes, got {type(transform)!r}")
        self.transforms = tuple(transforms)

    def instance_signature(self) -> tuple[Any, ...]:
        return ("transform_chain", tuple(transform.signature() for transform in self.transforms))

    def declared_dependencies(self) -> list[CalculatorBase[Any,Any]]:
        return list(self.transforms)

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("steps", len(self.transforms))]
        if self.name is not None:
            fields.append(("name", self.name))
        return fields

    def _repr_summary_rows(self) -> list[tuple[str, Any]]:
        rows = super()._repr_summary_rows()
        rows.append(("steps", " -> ".join(transform.log_label for transform in self.transforms)))
        return rows

    def build_handle(
        self,
        sim: SimSnap,
        target: TransformTarget,
        params: Any = None,
    ) -> tuple[TransformStep, ...]:
        raise RuntimeError("TransformChain executes child transforms directly.")

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> TransformResult[tuple[TransformStep, ...]]:
        work = input
        steps: list[TransformStep] = []
        try:
            with ctx.phase(self, "calculate"):
                for transform in self.transforms:
                    result = ctx.raw_value(transform, work)
                    if not isinstance(result, TransformResult):
                        raise TypeError("transform nodes must return TransformResult")
                    steps.append(TransformStep(transform=transform, result=result))
                    work = work.with_transform(result)

            last = steps[-1].result
            return TransformResult(
                handle=tuple(steps),
                target=last.target,
                sim_after=work.sim_current,
                revertible=any(step.result.revertible for step in steps),
                artifacts={"step_count": len(steps), "mutation_generation": ctx.mutation_generation},
            )
        except Exception:
            if steps:
                try:
                    self.cleanup(ctx, tuple(steps))
                    ctx.log("debug", f"cleaned up {len(steps)} steps after error in transform chain")
                except Exception as cleanup_error:
                    ctx.log("error", f"transform cleanup failed after error: {cleanup_error}")
            raise


    def is_revertible(self, handle: tuple[TransformStep, ...]) -> bool:
        """Whether any step in the chain is revertible."""
        return any(step.result.revertible for step in handle)

    def cleanup(self, ctx: ExecutionContext, handle: tuple[TransformStep, ...]) -> None:
        """Clean up chain steps in reverse execution order."""
        seen_handles: set[int] = set()

        for step in reversed(handle):
            if not step.result.revertible:
                continue

            handle_obj = step.result.handle
            handle_id = id(handle_obj)
            if handle_id in seen_handles:
                ctx.log("debug", f"skip duplicate revert {step.transform.log_label}")
                continue
            seen_handles.add(handle_id)

            if getattr(handle_obj, "_reverted", False):
                ctx.log("debug", f"skip already reverted {step.transform.log_label}")
                continue

            cleanup = getattr(step.transform, "cleanup", None)
            if cleanup is None:
                continue

            cleanup(ctx, handle_obj)

    def then(self, transform: TransformBase[Any]) -> TransformBase[Any]:
        """Append another transform after this chain."""
        return chain_transforms(*self.transforms, transform)


class TransformPlan(TransformChain):
    """Named alias for a planned transform chain."""

    def instance_signature(self) -> tuple[Any, ...]:
        return ("transform_plan", tuple(transform.signature() for transform in self.transforms))


def chain_transforms(*transforms: TransformBase[Any]) -> TransformBase[Any]:
    """Return a transform chain from one or more transform calculators."""
    if not transforms:
        raise ValueError("At least one transform must be provided.")
    return TransformChain(*transforms)
