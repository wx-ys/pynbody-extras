from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.params.fields import Param, declarative_dependencies
from pynbodyext.core.calculate.result.enums import BuiltinKinds, NodeKind

from .axes import AxisPropertyFunc, BinAxis, materialize_axis, resolve_axis_values
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

    mode: str = Param.static(default="linear", kw_only=False)
    edges: Param[Any] = Param(default=None, kw_only=True)
    lows: Param[Any] = Param(default=None, kw_only=True)
    highs: Param[Any] = Param(default=None, kw_only=True)
    alias: str | None = Param.static(default=None, kw_only=True)
    include_rightmost: bool = Param.static(default=True, kw_only=True)
    out_of_range: str = Param.static(default="drop", kw_only=True)
    units: Any | None = Param.static(default=None, kw_only=True)

    kind: ClassVar[NodeKind] = BuiltinKinds.BINND

    def __post_init__(self) -> None:
        self.active: tuple[Any, ...] = ()
        self._binnd_wrapper: BinND | None = None

    def _get_binnd(self) -> BinND:
        """Return a lazily-created single-axis BinND wrapper; built once per Bin1D instance."""
        if self._binnd_wrapper is None:
            self._binnd_wrapper = BinND((self,))
            self._binnd_wrapper.active = self.active
        return self._binnd_wrapper

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps = declarative_dependencies(self)
        for key in self.active:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    @staticmethod
    def axis_property(
        name: str | AxisPropertyFunc,
        func: AxisPropertyFunc | None = None,
        *,
        overwrite: bool = False,
    ) -> AxisPropertyFunc | Callable[[AxisPropertyFunc], AxisPropertyFunc]:
        return BinAxis.register_property(cast("Any", name), cast("Any", func), overwrite=overwrite)

    @staticmethod
    def derived(
        fn: Callable[[Any], Any] | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: Callable[[Any], bool] | None = None,
        overwrite: bool = False,
    ) -> Callable[[Any], Any] | Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        return BinNDResult.derived(cast("Any", fn), name=cast("Any", name), scope=scope, condition=condition, overwrite=overwrite)

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((self, *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((self, other))
        return NotImplemented

    def with_active(self, keys: Iterable[Any]) -> Bin1D:
        cl = cast("Bin1D", self._clone())
        cl.active = tuple(keys)
        return cl

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return self._get_binnd().execute(ctx, input)

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value

    def with_transformation(self, transform, revert = True):
        if revert:
            self.warning(f"Bin1D applies transform {transform} with revert=True; "
                         "the subsequent binned result will be derived from the original untransformed data. "
                         "Do you really want this? If so, consider setting revert=False to avoid such behavior.")
        return super().with_transformation(transform, revert=revert)


@CalculatorBase.dataclass
class BinND(CalculatorBase[BinNDResult, BinNDResult]):
    axes_specs: tuple[Bin1D, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes_specs, tuple):
            self.axes_specs = tuple(self.axes_specs)    # type: ignore[unreachable]
        if not self.axes_specs:
            raise ValueError("BinND requires at least one axis.")
        # Inherit active keys from constituent Bin1D specs
        inherited: list[Any] = []
        for axis in self.axes_specs:
            inherited.extend(axis.active)
        self.active = tuple(inherited)

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((*self.axes_specs, *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((*self.axes_specs, other))
        return NotImplemented

    def with_active(self, keys: Iterable[Any]) -> BinND:
        cl= cast("BinND", self._clone())
        cl.active = tuple(keys)
        return cl

    def with_transformation(self, transform, revert = True):
        if revert:
            self.warning(f"BinND applies transform {transform} with revert=True; "
                         "the subsequent binned result will be derived from the original untransformed data. "
                         "Do you really want this? If so, consider setting revert=False to avoid such behavior.")
        return super().with_transformation(transform, revert=revert)

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
        axes, values = self._resolve_axes(sim, ctx=ctx, input=input)
        result = self._build_result(sim, axes, values, source_sim=input.sim_raw, scope_signature=input.cache_token)
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

    def _resolve_axes(
        self,
        sim: Any,
        *,
        ctx: ExecutionContext | None = None,
        input: NodeInput | None = None,
    ) -> tuple[tuple[BinAxis, ...], list[Any]]:
        """Materialize axis specifications against *sim* and return ``(axes, values)``."""
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
        return tuple(materialized_axes), values

    def _build_result(
        self,
        sim: Any,
        axes: tuple[BinAxis, ...],
        values: list[Any],
        *,
        source_sim: Any | None = None,
        scope_signature: Any = None,
        parent: BinNDResult | None = None,
    ) -> BinNDResult:
        """Assign particles and construct a :class:`~.result.BinNDResult`."""
        bin_data, bin_indptr, particle_bin, valid_mask = self._assign_particles(axes, values, len(sim))
        cls = BinNDResult if parent is None else SubBinNDResult
        return cls(
            sim=sim,
            source_sim=sim if source_sim is None else source_sim,
            axes=axes,
            bin_data=bin_data,
            bin_indptr=bin_indptr,
            particle_bin=particle_bin,
            valid_mask=valid_mask,
            calculator=self,
            scope_signature=scope_signature,
            parent=parent,
        )

    def _spawn_result(self, parent: BinNDResult, subset: Any) -> SubBinNDResult:
        """Rebuild particle assignment for *subset* reusing the parent's axes."""
        values = [resolve_axis_values(spec.prop, subset, None, None) for spec in self.axes_specs]
        result = self._build_result(
            subset,
            parent.axes,
            values,
            source_sim=parent.source_sim,
            scope_signature=parent._scope_signature,
            parent=parent.root,
        )
        if not isinstance(result, SubBinNDResult):
            raise TypeError("spawned BinND result was not a SubBinNDResult")
        return result

    def _assign_particles(self, axes: tuple[BinAxis, ...], values: list[Any], n_particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (bin_data, bin_indptr, particle_bin, valid_mask) in CSR format.

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
            return (
                np.empty(0, dtype=int),
                np.zeros(total_nbins + 1, dtype=int),
                particle_bin,
                valid_mask,
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
        return bin_data, bin_indptr, particle_bin, valid_mask
