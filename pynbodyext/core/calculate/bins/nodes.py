from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.params.fields import Param, declarative_dependencies

from .axes import BinAxis, materialize_axis, register_axis_property, register_bin_derived, resolve_axis_values
from .result import BinNDResult, SubBinNDResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput


@CalculatorBase.dataclass
class Bin1D(CalculatorBase[BinNDResult, BinNDResult]):
    prop: Param[Any]
    vmin: Param[float | None] = Param(default=None)
    vmax: Param[float | None] = Param(default=None)
    nbins: Param[int | None] = Param(default=None)
    mode: str = Param.static(default="linear", kw_only=True)
    edges: Param[Any] = Param(default=None, kw_only=True)
    lows: Param[Any] = Param(default=None, kw_only=True)
    highs: Param[Any] = Param(default=None, kw_only=True)
    alias: str | None = Param.static(default=None, kw_only=True)
    include_rightmost: bool = Param.static(default=True, kw_only=True)
    out_of_range: str = Param.static(default="drop", kw_only=True)
    units: Any | None = Param.static(default=None, kw_only=True)
    active: tuple[Any, ...] = Param.static(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.active, tuple):
            self.active = tuple(self.active or ())  # type: ignore[unreachable]

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps = declarative_dependencies(self)
        for key in self.active:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    @staticmethod
    def axis_property(*args: Any, **kwargs: Any) -> Callable[[Any], Bin1D]:
        return register_axis_property(*args, **kwargs)

    @staticmethod
    def derived(*args: Any, **kwargs: Any) -> Callable[[Any], Bin1D]:
        return register_bin_derived(*args, **kwargs)

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((self, *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((self, other))
        return NotImplemented

    def with_active(self, keys: Iterable[Any]) -> Bin1D:
        return cast("Bin1D", self._clone(active=tuple(keys)))

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return BinND((self,), active=self.active).execute(ctx, input)

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value


@CalculatorBase.dataclass
class BinND(CalculatorBase[BinNDResult, BinNDResult]):
    axes_specs: tuple[Bin1D, ...]
    active: tuple[Any, ...] = Param.static(default=(), kw_only=True)


    def __post_init__(self) -> None:
        if not isinstance(self.axes_specs, tuple):
            self.axes_specs = tuple(self.axes_specs)    # type: ignore[unreachable]
        if not self.axes_specs:
            raise ValueError("BinND requires at least one axis.")
        if not self.active:
            inherited: list[Any] = []
            for axis in self.axes_specs:
                inherited.extend(axis.active)
            self.active = tuple(inherited)

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((*self.axes_specs, *other.axes_specs), active=self.active + other.active)
        if isinstance(other, Bin1D):
            return BinND((*self.axes_specs, other), active=self.active + other.active)
        return NotImplemented

    def with_active(self, keys: Iterable[Any]) -> BinND:
        return cast("BinND", self._clone(active=tuple(keys)))

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps: list[CalculatorBase[Any, Any]] = []
        for axis in self.axes_specs:
            deps.append(axis)
        for key in self.active:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        sim = input.active_sim
        result = self._materialize_result(sim, ctx=ctx, input=input, source_sim=input.sim_raw, scope_signature=input.cache_token)
        for key in self.active:
            if isinstance(key, str):
                result[key]
            elif isinstance(key, CalculatorBase) or callable(key):
                result[key]
            else:
                raise TypeError(f"Unsupported active key {key!r}.")
        return result

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value

    def _materialize_result(
        self,
        sim: Any,
        *,
        ctx: ExecutionContext | None = None,
        input: NodeInput | None = None,
        source_sim: Any | None = None,
        scope_signature: Any = None,
        axes: tuple[BinAxis, ...] | None = None,
        parent: BinNDResult | None = None,
    ) -> BinNDResult:
        if axes is None:
            materialized_axes: list[BinAxis] = []
            values: list[Any] = []
            aliases: set[str] = set()
            for index, spec in enumerate(self.axes_specs):
                axis, axis_values = materialize_axis(spec, sim, ctx, input, index=index)
                if axis.alias in aliases:
                    raise ValueError(f"Duplicate bin axis alias {axis.alias!r}.")
                aliases.add(axis.alias)
                materialized_axes.append(axis)
                values.append(axis_values)
            axes = tuple(materialized_axes)
        else:
            values = [resolve_axis_values(spec.prop, sim, ctx, input) for spec in self.axes_specs]

        bin_indices, particle_bin, valid_mask = self._assign_particles(axes, values, len(sim))
        cls = BinNDResult if parent is None else SubBinNDResult
        return cls(
            sim=sim,
            source_sim=sim if source_sim is None else source_sim,
            axes=axes,
            bin_indices=bin_indices,
            particle_bin=particle_bin,
            valid_mask=valid_mask,
            calculator=self,
            scope_signature=scope_signature,
            parent=parent,
        )

    def _spawn_result(self, parent: BinNDResult, subset: Any) -> SubBinNDResult:
        result = self._materialize_result(
            subset,
            source_sim=parent.source_sim,
            scope_signature=parent.scope_signature,
            axes=parent.axes,
            parent=parent.root,
        )
        if not isinstance(result, SubBinNDResult):
            raise TypeError("spawned BinND result was not a SubBinNDResult")
        return result

    def _assign_particles(self, axes: tuple[BinAxis, ...], values: list[Any], n_particles: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        axis_bins: list[np.ndarray] = []
        valid_mask = np.ones(n_particles, dtype=bool)
        for axis, axis_values in zip(axes, values, strict=True):
            axis_bin, axis_valid = axis.assign(axis_values)
            if len(axis_bin) != n_particles:
                raise ValueError(f"axis {axis.alias!r} prop length must match sim length.")
            axis_bins.append(axis_bin)
            valid_mask &= axis_valid

        total_nbins = int(np.prod([axis.nbins for axis in axes], dtype=int))
        particle_bin = np.full(n_particles, -1, dtype=int)
        if not np.any(valid_mask):
            return [np.asarray([], dtype=int) for _ in range(total_nbins)], particle_bin, valid_mask

        valid_indices = np.nonzero(valid_mask)[0]
        multi = tuple(axis_bin[valid_indices] for axis_bin in axis_bins)
        shape = tuple(axis.nbins for axis in axes)
        flat = np.ravel_multi_index(multi, shape, order="C")
        particle_bin[valid_indices] = flat

        counts = np.bincount(flat, minlength=total_nbins).astype(int)
        order = np.argsort(flat, kind="stable")
        idx_sorted = valid_indices[order]
        starts = np.concatenate(([0], np.cumsum(counts)))
        bin_indices = [idx_sorted[starts[i] : starts[i + 1]] for i in range(total_nbins)]
        return bin_indices, particle_bin, valid_mask
