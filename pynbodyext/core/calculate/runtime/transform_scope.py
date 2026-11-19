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
        self,
        ctx: ExecutionContext,
        transform: TransformBase[Any],
        handle: Any,
        *,
        revertible: bool = True,
        before: int | None = None,
    ) -> None:
        ctx.mutation.pipeline_generation = ctx.advance_mutation_generation(
            f"apply {transform.log_label}", observed_phase="calculate"
        )
        self.steps.insert(len(self.steps) if before is None else before, (transform, handle, revertible))

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
                        if ctx.selection_recorder is not None:
                            ctx.selection_recorder.reverted(handle)
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


_Copies = dict[int, tuple[Any, Any]]
_IN_PROGRESS = object()


def freeze_array_views(value: Any) -> Any:
    """Copy borrowed numerical values while their measured frame is still active."""
    return _freeze(value, {})


def _freeze(item: Any, copies: _Copies) -> Any:
    key = id(item)
    if isinstance(item, np.ndarray) and item.dtype != object:
        if item.base is None:
            return item
        if key not in copies:
            copies[key] = (item, item.copy())
        return copies[key][1]
    if type(item) in (bool, int, float, complex, str, bytes, type(None)):
        return item
    if key in copies:
        frozen = copies[key][1]
        if frozen is _IN_PROGRESS:
            raise ValueError("A cyclic measurement container cannot be detached from its Pipeline transform scope")
        return frozen
    # Containers can expose an item again through their instance attributes.
    copies[key] = (item, _IN_PROGRESS)
    frozen = _freeze_value(item, copies)
    if frozen is item:
        del copies[key]
    else:
        copies[key] = (item, frozen)
    return frozen


def _freeze_value(value: Any, copies: _Copies) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        frozen_array = value.copy() if value.base is not None else value
        for index, item in enumerate(value.flat):
            frozen = _freeze(item, copies)
            if frozen is not item:
                if frozen_array is value:
                    frozen_array = value.copy()
                frozen_array.flat[index] = frozen
        return frozen_array
    if isinstance(value, IndexedSimArray):
        return value.copy()
    if isinstance(value, (Mapping, tuple, list)):
        return _freeze_container(value, copies)
    if is_dataclass(value) and not isinstance(value, (type, FilterResult, TransformResult)):
        # Transformation receipts and live handles retain their cleanup state.
        if not callable(getattr(value, "revert", None)):
            result = value
            for descriptor in fields(value):
                if not hasattr(value, descriptor.name):
                    continue
                item = getattr(value, descriptor.name)
                frozen = _freeze(item, copies)
                if frozen is not item:
                    if result is value:
                        result = copy(value)
                    object.__setattr__(result, descriptor.name, frozen)
            return result
    return value


def _freeze_container(value: Mapping | tuple | list, copies: _Copies) -> Any:
    raw_state = object.__getstate__(value)
    slots: dict[str, Any] = {}
    if isinstance(raw_state, tuple):
        dictionary, slots = raw_state
        state = (dictionary or {}) | slots
    else:
        state = raw_state if isinstance(raw_state, dict) else {}
    # A direct self reference can be rebound once the new container exists.
    attributes = {name: item if item is value else _freeze(item, copies) for name, item in state.items()}
    changed = any(item is not state[name] for name, item in attributes.items())
    if isinstance(value, Mapping):
        frozen: Any = _freeze_mapping(value, copies, changed)
    else:
        frozen = _freeze_sequence(value, copies, changed)
    if frozen is not value and attributes:
        for name, item in attributes.items():
            member = frozen if item is value else item
            if name in slots:
                object.__setattr__(frozen, name, member)
            else:
                vars(frozen)[name] = member
    return frozen


def _freeze_sequence(value: tuple | list, copies: _Copies, attributes_changed: bool) -> tuple | list:
    items = [_freeze(item, copies) for item in value]
    if not attributes_changed and all(item is original for item, original in zip(items, value, strict=True)):
        return value
    if isinstance(value, list):
        result = copy(value)
        result[:] = items
        return result
    return tuple.__new__(type(value), items)


def _freeze_mapping(value: Mapping, copies: _Copies, attributes_changed: bool) -> Mapping:
    changed = {}
    for key, item in value.items():
        frozen = _freeze(item, copies)
        if frozen is not item:
            changed[key] = frozen
    if not changed and not attributes_changed:
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
