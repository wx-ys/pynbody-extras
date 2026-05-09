"""Role base class for temporary simulation mutations.

:class:`TransformBase` is the standard base class for calculators that mutate
the active frame before downstream calculation and return a handle describing
the applied change.

In this project, most concrete transforms under :mod:`pynbodyext.transforms`
are written in dataclass style and implement :meth:`build_handle`. Real
examples include :class:`ShiftPosTo`, :class:`ShiftVelTo`,
:class:`AlignVec`, and :class:`WrapBox`.

Use :class:`TransformBase` when the node:

- changes coordinates, velocities, or another mutable field
- should run before another calculator
- returns a handle that can later be reverted or cleaned up
- represents a tracked mutating step in the execution graph

Recommended Authoring Style
---------------------------
For most new transforms in this codebase, prefer:

- :meth:`TransformBase.dataclass`
- :class:`Param` for runtime-resolved values
- :meth:`build_handle` as the main user hook
- returning a pynbody transformation handle when possible

If the returned handle already knows how to revert itself, custom cleanup is
often unnecessary.

Translation Example
-------------------
A simplified example in the same style as
:class:`pynbodyext.transforms.ShiftPosTo`::
    @TransformBase.dataclass
    class ShiftPosTo(TransformBase[Any]):
        mode: Param[Any] = Param(default="ssc", field_name="pos")
        move_all: bool = True

        def __post_init__(self):
            if isinstance(self.mode, str):
                self.mode = CenPos(self.mode)

        def build_handle(self, sim, target, params=None):
            cen = params.mode
            return GenericTranslation(
                target,
                "pos",
                -cen,
                description="PosToCenter",
            )

    result = KappaRot().transform(ShiftPosTo("ssc")).run(sim)
    print(result.value)

Rotation Example
----------------
A vector-alignment transform in the style of
:class:`pynbodyext.transforms.AlignVec`::

    @TransformBase.dataclass
    class AlignVec(TransformBase[Any]):
        vector: Param[np.ndarray]
        up: np.ndarray | None = None
        move_all: bool = True

        def build_handle(self, sim, target, params=None):
            vec = params.vector
            rotation = calc_faceon_matrix(vec, up=self.up)
            return target.rotate(rotation, description="AlignVec")

Measurement Versus Target
-------------------------
A transform may measure a parameter on one subset while mutating a wider
target. This is what :meth:`measure_with` is for::

    centred = ShiftPosTo("ssc").measure_with(FamilyFilter("star"))
    result = KappaRot().transform(centred).run(sim)
    print(result.value)

Revert Behavior
---------------
Transform cleanup is controlled by the selected revert policy.

Most transforms in this codebase return handles that already support
reversion, so the default cleanup path is often sufficient. Override
:meth:`cleanup` only when the handle is a plain custom object or needs
special reversal logic.

Which Hook To Implement
-----------------------
Most :class:`TransformBase` subclasses should implement :meth:`build_handle`.

Use the runtime-level hooks only when the transform needs direct access to
the execution context, a custom measurement workflow, or explicit dependency
evaluation.

Notes
-----
If a transform appears to leak state, inspect the handle type and revert
policy.

If a transform seems to measure on the wrong subset, inspect its measure
filter and the resulting scope composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation

from pynbodyext.core.calculate.display import display_value
from pynbodyext.core.calculate.result.enums import (
    BuiltinKinds,
    CachePolicy,
    EffectPolicy,
    RevertPolicy,
    normalize_revert_policy,
)
from pynbodyext.core.calculate.runtime import CalcRuntime, TransformRuntime
from pynbodyext.core.calculate.runtime.input import FilterResult, NodeInput, TransformResult

from .runtime_base import RuntimeCalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbodyext.core.calculate.runtime.context import ExecutionContext

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

    def _init_dataclass_base(self) -> None:
        TransformBase.__init__(
            self,
            move_all=getattr(self, "move_all", True),
            name=getattr(self, "name", None),
            revert_policy=getattr(self, "revert_policy", RevertPolicy.ALWAYS),
            measure_filter=getattr(self, "measure_filter", None),
        )
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

    def signature_payload(self) -> Mapping[str, Any] | None:
        payload: dict[str, Any] = {}
        if not self.move_all:
            payload["move_all"] = self.move_all
        return payload or None

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
        mutation_generation = transform_runtime.ctx.advance_mutation_generation(
            f"apply {self.log_label}",
            observed_phase="calculate",
        )
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
            from pynbodyext.core.calculate.diagnostics.observer import observation_phase

            with observation_phase("revert"):
                handle.revert() # type: ignore[attr-defined]
                ctx.advance_mutation_generation(
                    f"revert {self.log_label}",
                    observed_phase="revert",
                )

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
                    with ctx.phase(self, "revert"):
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

            from pynbodyext.core.calculate.diagnostics.observer import observation_phase

            with observation_phase("revert"):
                cleanup(ctx, handle_obj)

    def then(self, transform: TransformBase[Any]) -> TransformBase[Any]:
        """Append another transform after this chain."""
        return chain_transforms(*self.transforms, transform)


class TransformPlan(TransformChain):
    """Named alias for a planned transform chain."""


def chain_transforms(*transforms: TransformBase[Any]) -> TransformBase[Any]:
    """Return a transform chain from one or more transform calculators."""
    if not transforms:
        raise ValueError("At least one transform must be provided.")
    return TransformChain(*transforms)
