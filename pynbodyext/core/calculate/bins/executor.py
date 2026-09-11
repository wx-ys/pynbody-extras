"""Execution of :class:`BinND` into a binned result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .axis_materializer import AxisMaterializer
from .model import BinResultModel
from .result import BinNDResult, SubBinNDResult

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput

    from .axes import BinAxis
    from .nodes import BinND


@dataclass(frozen=True)
class MaterializedBinAxes:
    axes: tuple[BinAxis, ...]
    values: tuple[Any, ...]


@dataclass(frozen=True)
class BinParticleAssignment:
    bin_data: np.ndarray
    bin_indptr: np.ndarray
    particle_bin: np.ndarray
    valid_mask: np.ndarray


class BinExecutor:
    def __init__(self, calculator: BinND) -> None:
        self._calculator = calculator
        self._axis_materializer = AxisMaterializer()

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        sim = input.active_sim
        materialized = self.resolve_axes(sim, ctx=ctx, input=input)
        result = self.build_result(
            sim, materialized.axes, materialized.values, source_sim=input.sim_raw, scope_signature=input.cache_token
        )
        for key in self._calculator.active:
            if isinstance(key, str):
                result[key]
            elif callable(key):
                result[key]
            else:
                raise TypeError(f"Unsupported active key {key!r}.")
        return result

    def resolve_axes(
        self, sim: Any, *, ctx: ExecutionContext | None = None, input: NodeInput | None = None
    ) -> MaterializedBinAxes:
        materialized_axes: list[BinAxis] = []
        values: list[Any] = []
        aliases: set[str] = set()

        for index, spec in enumerate(self._calculator.axes_specs):
            axis, axis_values = self._axis_materializer.materialize(spec, sim, ctx=ctx, input=input, index=index)
            if axis.alias in aliases:
                raise ValueError(f"Duplicate bin axis alias {axis.alias!r}.")
            aliases.add(axis.alias)
            materialized_axes.append(axis)
            values.append(axis_values)

        return MaterializedBinAxes(axes=tuple(materialized_axes), values=tuple(values))

    def build_result(
        self,
        sim: Any,
        axes: tuple[BinAxis, ...],
        values: tuple[Any, ...],
        *,
        source_sim: Any | None = None,
        scope_signature: Any = None,
        parent: BinNDResult | None = None,
    ) -> BinNDResult:
        assignment = self.assign_particles(axes, values, len(sim))
        model = BinResultModel(
            sim=sim,
            source_sim=sim if source_sim is None else source_sim,
            axes=axes,
            bin_data=assignment.bin_data,
            bin_indptr=assignment.bin_indptr,
            particle_bin=assignment.particle_bin,
            valid_mask=assignment.valid_mask,
            calculator=self._calculator,
            scope_signature=scope_signature,
            parent=parent._model if parent is not None else None,
            owner=None,
        )
        cls = BinNDResult if parent is None else SubBinNDResult
        return cls(model=model)

    def spawn_result(self, parent: BinNDResult, subset: Any) -> SubBinNDResult:
        values = tuple(
            self._axis_materializer._resolve_source(spec, subset).values for spec in self._calculator.axes_specs
        )
        result = self.build_result(
            subset,
            tuple(parent.axes),
            values,
            source_sim=parent.source_sim,
            scope_signature=parent._scope_signature,
            parent=parent.root,
        )
        if not isinstance(result, SubBinNDResult):
            raise TypeError("spawned BinND result was not a SubBinNDResult")
        return result

    def assign_particles(
        self, axes: tuple[BinAxis, ...], values: tuple[Any, ...], n_particles: int
    ) -> BinParticleAssignment:
        """Return BinParticleAssignment(bin_data, bin_indptr, particle_bin, valid_mask) in CSR format.

        ``bin_data[bin_indptr[i] : bin_indptr[i+1]]`` gives the particle indices
        assigned to flat bin *i*.
        """
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
            return BinParticleAssignment(
                bin_data=np.empty(0, dtype=int),
                bin_indptr=np.zeros(total_nbins + 1, dtype=int),
                particle_bin=particle_bin,
                valid_mask=valid_mask,
            )

        valid_indices = np.nonzero(valid_mask)[0]
        multi = tuple(axis_bin[valid_indices] for axis_bin in axis_bins)
        shape = tuple(axis.nbins for axis in axes)
        flat = np.ravel_multi_index(multi, shape, order="C")
        particle_bin[valid_indices] = flat

        counts = np.bincount(flat, minlength=total_nbins).astype(int)
        order = np.argsort(flat, kind="stable")
        bin_data = valid_indices[order]
        bin_indptr = np.concatenate(([0], np.cumsum(counts)))

        return BinParticleAssignment(
            bin_data=bin_data, bin_indptr=bin_indptr, particle_bin=particle_bin, valid_mask=valid_mask
        )
