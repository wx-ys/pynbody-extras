"""Scoped node input and raw runtime result containers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from pynbodyext.core.calculate.result.result import ResultNode, ValueSummary

HandleT = TypeVar("HandleT")


@dataclass(slots=True)
class FilterResult:
    """Raw result of a filter calculation.

    The filtered simulation view is derived lazily from source_sim and mask
    so runtime cache entries do not keep a strong reference to a subsnap.
    """

    mask: Any
    source_sim: Any
    mask_summary: ValueSummary | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    _filtered_sim: Any = field(default=None, init=False, repr=False)
    _filtered_sim_ready: bool = field(default=False, init=False, repr=False)

    @property
    def filtered_sim(self) -> Any:
        """Build and reuse the filtered simulation view on demand."""
        if self._filtered_sim_ready:
            return self._filtered_sim

        mask = self.mask
        shape = getattr(mask, "shape", None)
        dtype = getattr(mask, "dtype", None)

        if (
            shape is not None
            and dtype is not None
            and len(shape) == 1
            and shape[0] == len(self.source_sim)
            and np.dtype(dtype) != np.dtype(np.bool_)
        ):
            try:
                mask = mask.astype(np.bool_, copy=False)
            except TypeError:
                mask = mask.astype(np.bool_)

        self._filtered_sim = self.source_sim[mask]
        self._filtered_sim_ready = True
        return self._filtered_sim

    @property
    def cache_token(self) -> tuple[int, int]:
        """Stable token for this selection within one run."""
        return (id(self.source_sim), id(self.mask))


@dataclass(slots=True)
class TransformResult(Generic[HandleT]):
    """Raw result of a transform calculation."""

    handle: HandleT
    target: Any
    sim_after: Any
    revertible: bool = True
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeInput:
    """Active simulation view and scope state for node evaluation."""

    sim_raw: Any
    sim_current: Any
    selection: FilterResult | None = None
    transform: TransformResult[Any] | None = None
    mutation_generation: int = 0
    upstream: dict[str, ResultNode] = field(default_factory=dict)

    @property
    def active_sim(self) -> Any:
        """Simulation object visible to the current node."""
        if self.selection is not None:
            return self.selection.filtered_sim
        return self.sim_current

    @property
    def cache_token(self) -> tuple[int, int, tuple[int, int] | None, int | None, int]:
        """Input-state token used as part of runtime cache keys."""
        transform_id = id(self.transform.handle) if self.transform is not None else None
        selection_token = self.selection.cache_token if self.selection is not None else None
        return (id(self.sim_raw), id(self.sim_current), selection_token, transform_id, self.mutation_generation)

    @property
    def scope_cache_token(self) -> tuple[int, int, tuple[int, int] | None, int | None]:
        """Input-scope token excluding mutation generations."""
        transform_id = id(self.transform.handle) if self.transform is not None else None
        selection_token = self.selection.cache_token if self.selection is not None else None
        return (id(self.sim_raw), id(self.sim_current), selection_token, transform_id)

    @property
    def observed_scope_cache_token(self) -> tuple[int, int, tuple[int, int] | None]:
        """Input-scope token for observer-validated cache entries.

        Transform identity is intentionally excluded: observer-derived field
        generations decide whether a transformed scope invalidates a cached
        value. The active simulation object and selection remain part of the
        scope because they change the population being measured.
        """
        selection_token = self.selection.cache_token if self.selection is not None else None
        return (id(self.sim_raw), id(self.sim_current), selection_token)

    def with_transform(self, result: TransformResult[Any]) -> NodeInput:
        """Return a copy after applying a transform result."""
        generation = result.artifacts.get("mutation_generation", self.mutation_generation)
        return replace(
            self,
            transform=result,
            sim_current=result.sim_after,
            mutation_generation=generation,
        )

    def with_selection(self, result: FilterResult) -> NodeInput:
        """Return a copy with an active filter selection."""
        return replace(self, selection=result)

    def with_mutation_generation(self, generation: int) -> NodeInput:
        """Return a copy with an updated mutation generation."""
        if self.mutation_generation == generation:
            return self
        return replace(self, mutation_generation=generation)
