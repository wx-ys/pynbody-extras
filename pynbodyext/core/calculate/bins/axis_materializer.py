from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody import units as pynbody_units
from pynbody.array import SimArray

from .axes import (
    BIN_ALGORITHMS,
    BinAxis,
    _as_1d_array,
)

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput

    from .nodes import Bin1D

def _finite_minmax(values: Any) -> tuple[float, float]:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot infer bin range from an array with no finite values.")
    return float(np.min(finite)), float(np.max(finite))


def _coerce_bound_value(value: Any, reference_values: Any, sim: Any) -> float:
    """Convert a bound value (possibly a unit string or UnitBase) to a float.

    When *reference_values* carries pynbody units and *sim* supplies a
    conversion context, unit strings such as ``"13.8 Gyr"`` are converted
    to the same units as the axis values before being returned as a float.
    """
    if isinstance(value, str):
        value = pynbody_units.Unit(value)
    if isinstance(value, pynbody_units.UnitBase):
        if hasattr(reference_values, "units"):
            context = sim.conversion_context() if hasattr(sim, "conversion_context") else {}
            return float(value.in_units(reference_values.units, **context))
        return float(SimArray(1.0).in_units(value))
    if isinstance(value, SimArray):
        if hasattr(reference_values, "units"):
            context = sim.conversion_context() if hasattr(sim, "conversion_context") else {}
            return float(value.in_units(reference_values.units, **context))
        return float(value)
    return float(value)


def _coerce_edges_like(edges: Any, source: Any) -> Any:
    if isinstance(source, SimArray) and not isinstance(edges, SimArray):
        out = SimArray(edges)
        out.units = source.units
        out.sim = source.sim
        return out
    return edges

def infer_alias(prop: Any, alias: str | None, index: int = 0) -> str:
    if alias is not None:
        return alias
    if isinstance(prop, str):
        return prop
    return f"dim{index}"


def resolve_axis_values(prop: Any, sim: Any, ctx: ExecutionContext | None = None, input: NodeInput | None = None) -> Any:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

    if isinstance(prop, str):
        return sim[prop]
    if isinstance(prop, CalculatorBase):
        if ctx is None or input is None:
            return prop(sim)
        return ctx.public_value(prop, input)
    if callable(prop):
        return prop(sim)
    raise TypeError("Bin1D prop must be a string, callable, or CalculatorBase.")


@dataclass(frozen=True)
class ResolvedAxisSource:
    alias: str
    values: Any
    values_arr: np.ndarray


class AxisMaterializer:
    def materialize(
        self,
        spec: Bin1D,
        sim: Any,
        *,
        ctx: ExecutionContext | None = None,
        input: NodeInput | None = None,
        index: int = 0,
    ) -> tuple[BinAxis, Any]:
        source = self._resolve_source(spec, sim, ctx=ctx, input=input, index=index)

        if spec.edges is not None:
            return self._materialize_from_edges(spec, sim, source, ctx=ctx, input=input)

        if spec.lows is not None or spec.highs is not None: #  type: ignore[unreachable]
            return self._materialize_from_bounds(spec, sim, source, ctx=ctx, input=input)

        return self._materialize_generated(spec, sim, source, ctx=ctx, input=input)

    def _resolve_source(
        self,
        spec: Bin1D,
        sim: Any,
        *,
        ctx: ExecutionContext | None = None,
        input: NodeInput | None = None,
        index: int = 0,
    ) -> ResolvedAxisSource:
        values = resolve_axis_values(spec.prop, sim, ctx, input)
        values_arr = _as_1d_array(values, name=f"axis {index} prop")
        alias = infer_alias(spec.prop, spec.alias, index)

        if len(values_arr) != len(sim):
            raise ValueError(f"axis {alias!r} prop length must match active sim length.")

        return ResolvedAxisSource(alias=alias, values=values, values_arr=values_arr)

    def _resolve_spec_value(
        self,
        spec: Bin1D,
        name: str,
        *,
        sim: Any,
        ctx: ExecutionContext | None,
        input: NodeInput | None,
    ) -> Any:
        if ctx is not None and input is not None and spec.has_dynamic_param(name):
            return spec.resolve_dynamic_param(ctx, input, name)
        if spec.has_dynamic_param(name):
            return spec.resolve_param_for_sim(sim, name)
        return getattr(spec, name)

    def _build_axis(
        self,
        spec: Bin1D,
        source: ResolvedAxisSource,
        *,
        mins: Any,
        maxs: Any,
    ) -> tuple[BinAxis, Any]:
        return (
            BinAxis(
                alias=source.alias,
                prop=spec.prop,
                mins=mins,
                maxs=maxs,
                include_rightmost=spec.include_rightmost,
                units=spec.units,
            ),
            source.values,
        )

    def _materialize_from_edges(
        self,
        spec: Bin1D,
        sim: Any,
        source: ResolvedAxisSource,
        *,
        ctx: ExecutionContext | None,
        input: NodeInput | None,
    ) -> tuple[BinAxis, Any]:
        if any(value is not None for value in (spec.vmin, spec.vmax, spec.nbins)):
            raise ValueError("edges cannot be mixed with vmin/vmax/nbins.")
        if spec.lows is not None or spec.highs is not None: # type: ignore[unreachable]
            raise ValueError("edges cannot be mixed with lows/highs.")

        edges = self._resolve_spec_value(spec, "edges", sim=sim, ctx=ctx, input=input)  # type: ignore[unreachable]
        edges = _as_1d_array(edges, name="edges")
        if edges.shape[0] < 2:
            raise ValueError("edges must contain at least two values.")
        if not np.all(np.diff(np.asarray(edges)) > 0):
            raise ValueError("edges must be strictly increasing.")
        edges = _coerce_edges_like(edges, source.values)

        return self._build_axis(spec, source, mins=edges[:-1], maxs=edges[1:])

    def _materialize_from_bounds(
        self,
        spec: Bin1D,
        sim: Any,
        source: ResolvedAxisSource,
        *,
        ctx: ExecutionContext | None,
        input: NodeInput | None,
    ) -> tuple[BinAxis, Any]:
        if any(value is not None for value in (spec.vmin, spec.vmax, spec.nbins)):
            raise ValueError("lows/highs cannot be mixed with vmin/vmax/nbins.")
        if spec.lows is None or spec.highs is None:
            raise ValueError("lows and highs must be provided together.")

        lows = self._resolve_spec_value(spec, "lows", sim=sim, ctx=ctx, input=input)
        highs = self._resolve_spec_value(spec, "highs", sim=sim, ctx=ctx, input=input)
        lows = _coerce_edges_like(_as_1d_array(lows, name="lows"), source.values)
        highs = _coerce_edges_like(_as_1d_array(highs, name="highs"), source.values)

        return self._build_axis(spec, source, mins=lows, maxs=highs)

    def _materialize_generated(
        self,
        spec: Bin1D,
        sim: Any,
        source: ResolvedAxisSource,
        *,
        ctx: ExecutionContext | None,
        input: NodeInput | None,
    ) -> tuple[BinAxis, Any]:
        nbins = self._resolve_spec_value(spec, "nbins", sim=sim, ctx=ctx, input=input)
        if nbins is None:
            raise ValueError("nbins is required when edges or lows/highs are not provided.")
        nbins = int(nbins)
        if nbins <= 0:
            raise ValueError("nbins must be positive.")

        inferred_min, inferred_max = _finite_minmax(source.values_arr)
        vmin = self._resolve_spec_value(spec, "vmin", sim=sim, ctx=ctx, input=input)
        vmax = self._resolve_spec_value(spec, "vmax", sim=sim, ctx=ctx, input=input)

        vmin = inferred_min if vmin is None else _coerce_bound_value(vmin, source.values, sim)
        vmax = inferred_max if vmax is None else _coerce_bound_value(vmax, source.values, sim)
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax.")

        mode = spec.mode
        if callable(mode):
            edges = mode(source.values_arr, nbins, vmin, vmax)
        else:
            try:
                algorithm = BIN_ALGORITHMS[str(mode)]
            except KeyError as exc:
                raise KeyError(f"Unknown bin mode {mode!r}.") from exc
            edges = algorithm(source.values_arr, nbins, vmin, vmax)

        edges = _coerce_edges_like(_as_1d_array(edges, name="edges"), source.values)
        if len(edges) != nbins + 1:
            raise ValueError("bin algorithm must return nbins + 1 edges.")
        if not np.all(np.diff(np.asarray(edges)) > 0):
            raise ValueError("bin edges must be strictly increasing.")

        return self._build_axis(spec, source, mins=edges[:-1], maxs=edges[1:])
