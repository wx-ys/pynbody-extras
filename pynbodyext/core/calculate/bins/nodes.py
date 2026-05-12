from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pynbodyext.core.calculate.nodes.base import CalculatorBase

from .axes import BinAxis, materialize_axis, register_axis_property, register_bin_derived, resolve_axis_values
from .result import BinNDResult, SubBinNDResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput


class Bin1D(CalculatorBase[BinNDResult, BinNDResult]):


    @staticmethod
    def axis_property(*args: Any, **kwargs: Any) -> Callable[[Any], Bin1D]:
        return register_axis_property(*args, **kwargs)

    @staticmethod
    def derived(*args: Any, **kwargs: Any) -> Callable[[Any], Bin1D]:
        return register_bin_derived(*args, **kwargs)

    def __init__(
        self,
        prop: Any,
        vmin: Any = None,
        vmax: Any = None,
        nbins: Any = None,
        *,
        mode: str = "linear",
        edges: Any = None,
        lows: Any = None,
        highs: Any = None,
        alias: str | None = None,
        include_rightmost: bool = True,
        out_of_range: str = "drop",
        units: Any | None = None,
        active: Iterable[Any] | None = None,
        **mode_kwargs: Any,
    ) -> None:
        super().__init__()
        self.prop = prop
        self.vmin = vmin
        self.vmax = vmax
        self.nbins = nbins
        self.mode = mode
        self.edges = edges
        self.lows = lows
        self.highs = highs
        self.alias = alias
        self.include_rightmost = include_rightmost
        self.out_of_range = out_of_range
        self.units = units
        self.mode_kwargs = dict(mode_kwargs)
        self.active_keys = tuple(active or ())

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((self, *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((self, other))
        return NotImplemented

    def active(self, keys: Iterable[Any]) -> Bin1D:
        clone = self._copy()
        clone.active_keys = tuple(keys)
        return clone

    def _copy(self) -> Bin1D:
        return Bin1D(
            self.prop,
            vmin=self.vmin,
            vmax=self.vmax,
            nbins=self.nbins,
            mode=self.mode,
            edges=self.edges,
            lows=self.lows,
            highs=self.highs,
            alias=self.alias,
            include_rightmost=self.include_rightmost,
            out_of_range=self.out_of_range,
            units=self.units,
            active=self.active_keys,
            **self.mode_kwargs,
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "prop": self.prop,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "nbins": self.nbins,
            "mode": self.mode,
            "edges": self.edges,
            "lows": self.lows,
            "highs": self.highs,
            "alias": self.alias,
            "include_rightmost": self.include_rightmost,
            "out_of_range": self.out_of_range,
            "units": self.units,
            "mode_kwargs": self.mode_kwargs,
            "active": self.active_keys,
        }

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps: list[CalculatorBase[Any, Any]] = []
        for value in (self.prop, self.vmin, self.vmax, self.nbins):
            if isinstance(value, CalculatorBase):
                deps.append(value)
        for key in self.active_keys:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return BinND((self,), active=self.active_keys).execute(ctx, input)

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value


class BinND(CalculatorBase[BinNDResult, BinNDResult]):
    def __init__(self, axes: Iterable[Bin1D], *, active: Iterable[Any] | None = None) -> None:
        super().__init__()
        axes_specs = tuple(axes)
        if not axes_specs:
            raise ValueError("BinND requires at least one axis.")
        self.axes_specs = axes_specs
        inherited_active: list[Any] = []
        for axis in axes_specs:
            inherited_active.extend(axis.active_keys)
        self.active_keys = tuple(active if active is not None else inherited_active)

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((*self.axes_specs, *other.axes_specs), active=self.active_keys + other.active_keys)
        if isinstance(other, Bin1D):
            return BinND((*self.axes_specs, other), active=self.active_keys + other.active_keys)
        return NotImplemented

    def active(self, keys: Iterable[Any]) -> BinND:
        return BinND(self.axes_specs, active=tuple(keys))

    def signature_payload(self) -> Mapping[str, Any]:
        return {"axes": self.axes_specs, "active": self.active_keys}

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps: list[CalculatorBase[Any, Any]] = []
        for axis in self.axes_specs:
            deps.extend(axis.declared_dependencies())
        for key in self.active_keys:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        deduped: list[CalculatorBase[Any, Any]] = []
        seen: set[int] = set()
        for dep in deps:
            if id(dep) in seen:
                continue
            seen.add(id(dep))
            deduped.append(dep)
        return deduped

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        sim = input.active_sim
        result = self._materialize_result(sim, ctx=ctx, input=input, source_sim=input.sim_raw, scope_signature=input.cache_token)
        for key in self.active_keys:
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
