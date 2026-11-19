"""Ownership of reversible transformations applied inside a Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import IndexedSimArray

from pynbodyext.core.calculate.result.enums import RevertPolicy

from .input import FilterResult, TransformResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.nodes.transforms import TransformBase

    from .context import ExecutionContext
    from .runtime import TransformRuntime


@dataclass
class PipelineTransformScope:
    """Collect individual handles as they are applied, including partial chains."""

    restore: bool
    steps: list[tuple[TransformBase[Any], Any, bool]] = field(default_factory=list)

    def compute(self, runtime: TransformRuntime, transform: TransformBase[Any], params: Any) -> Any:
        try:
            handle = transform.compute(runtime, params)
        except Exception:
            # An apply hook may have changed particles before it could return a handle.
            runtime.ctx.pipeline_frame_failed = True
            raise
        self.applied(runtime.ctx, transform, handle)
        return handle

    def applied(
        self, ctx: ExecutionContext, transform: TransformBase[Any], handle: Any, *, revertible: bool = True
    ) -> None:
        ctx.mutation.pipeline_generation = ctx.advance_mutation_generation(
            f"apply {transform.log_label}", observed_phase="calculate"
        )
        self.steps.append((transform, handle, revertible))

    def confirm(
        self, ctx: ExecutionContext, transform: TransformBase[Any], result: TransformResult[Any], start: int
    ) -> None:
        """Use the final receipt, retaining early cleanup if wrapping ever fails."""
        for index in range(len(self.steps) - 1, start - 1, -1):
            if self.steps[index][0] is transform:
                self.steps[index] = (transform, result.handle, result.revertible)
                return
        self.applied(ctx, transform, result.handle, revertible=result.revertible)

    @contextmanager
    def lifetime(self, ctx: ExecutionContext, node: CalculatorBase) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:
            self._revert(ctx, node, exc)
            raise
        else:
            self._revert(ctx, node, None)

    def _revert(self, ctx: ExecutionContext, node: CalculatorBase, primary: BaseException | None) -> None:
        if not self.restore or not self.steps:
            return
        first_error: Exception | None = None
        seen_handles: set[int] = set()
        with ctx.phase(node, "revert"):
            for transform, handle, revertible in reversed(self.steps):
                if transform.revert_policy == RevertPolicy.NEVER or id(handle) in seen_handles:
                    continue
                seen_handles.add(id(handle))
                try:
                    if not revertible or not transform.is_revertible(handle):
                        raise RuntimeError(f"{transform.log_label} has no reversible handle for its Pipeline scope")
                    if not getattr(handle, "_reverted", False):
                        before = ctx.mutation.generation
                        transform.cleanup(ctx, handle)
                        if before == ctx.mutation.generation:
                            ctx.advance_mutation_generation(f"revert {transform.log_label}", observed_phase="revert")
                        ctx.mutation.pipeline_generation = ctx.mutation.generation
                except Exception as exc:
                    ctx.pipeline_frame_failed = True
                    ctx.log("error", f"Pipeline transform cleanup failed for {transform.log_label}: {exc}")
                    if primary is not None:
                        primary.add_note(f"Pipeline transform cleanup failed for {transform.log_label}: {exc}")
                    if first_error is None:
                        first_error = exc
        if primary is None and first_error is not None:
            raise first_error


def freeze_array_views(value: Any) -> Any:
    """Copy borrowed numerical values while their measured frame is still active."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        frozen_array = value.copy() if value.base is not None else value
        for index, item in enumerate(value.flat):
            frozen = freeze_array_views(item)
            if frozen is not item:
                if frozen_array is value:
                    frozen_array = value.copy()
                frozen_array.flat[index] = frozen
        return frozen_array
    if isinstance(value, IndexedSimArray) or isinstance(value, np.ndarray) and value.base is not None:
        return value.copy()
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        items = [freeze_array_views(item) for item in value]
        if any(item is not frozen for item, frozen in zip(value, items, strict=True)):
            return _replace_sequence_items(value, items)
    if is_dataclass(value) and not isinstance(value, (type, FilterResult, TransformResult)):
        # Transformation receipts and live handles retain their cleanup state.
        if not callable(getattr(value, "revert", None)):
            result = value
            for descriptor in fields(value):
                if not hasattr(value, descriptor.name):
                    continue
                item = getattr(value, descriptor.name)
                frozen = freeze_array_views(item)
                if frozen is not item:
                    if result is value:
                        result = copy(value)
                    object.__setattr__(result, descriptor.name, frozen)
            return result
    return value


def _replace_sequence_items(value: tuple | list, items: list[Any]) -> tuple | list:
    if isinstance(value, list):
        result = copy(value)
        result[:] = items
        return result
    frozen = tuple.__new__(type(value), items)
    if hasattr(value, "__dict__"):
        vars(frozen).update(vars(value))
    return frozen


def _freeze_mapping(value: Mapping) -> Mapping:
    changed = {}
    for key, item in value.items():
        frozen = freeze_array_views(item)
        if frozen is not item:
            changed[key] = frozen
    if not changed:
        return value
    if isinstance(value, dict):
        result = copy(value)
        result.update(changed)
        return result
    if isinstance(value, MappingProxyType):
        return MappingProxyType(dict(value) | changed)
    # A read-only Mapping need not accept a dict in its constructor. Its copy
    # protocol preserves the concrete type and any instance-specific behavior.
    memo = {id(key): key for key in value}
    memo.update({id(item): changed.get(key, item) for key, item in value.items()})
    return deepcopy(value, memo)
