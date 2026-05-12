from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np
from pynbody import units as pynbody_units
from pynbody.array import SimArray

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput


ArrayLike = Any
BinAlgorithm = Callable[[np.ndarray, int, float, float], ArrayLike]
AxisPropertyFunc = Callable[["BinAxis"], Any]
BinDerivedCondition = Callable[[Any], bool]
BinDerivedFunc = Callable[[Any], Any]


@dataclass(frozen=True)
class BinDerivedSpec:
    name: str
    func: BinDerivedFunc
    condition: BinDerivedCondition | None = None
    scope: str = "geometry"

    def is_available(self, bins: Any) -> bool:
        return True if self.condition is None else bool(self.condition(bins))


def _as_1d_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


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


def _linear_edges(values: np.ndarray, nbins: int, vmin: float, vmax: float) -> np.ndarray:
    return np.linspace(vmin, vmax, nbins + 1)


def _log_edges(values: np.ndarray, nbins: int, vmin: float, vmax: float) -> np.ndarray:
    if vmin <= 0 or vmax <= 0:
        raise ValueError("log bin mode requires positive vmin and vmax.")
    return np.logspace(np.log10(vmin), np.log10(vmax), nbins + 1)


def _equaln_edges(values: np.ndarray, nbins: int, vmin: float, vmax: float) -> np.ndarray:
    finite = np.asarray(values)[np.isfinite(values)]
    finite = finite[(finite >= vmin) & (finite <= vmax)]
    if finite.size == 0:
        raise ValueError("equaln bin mode requires at least one finite value in range.")
    quantiles = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(finite, quantiles)
    edges[0] = vmin
    edges[-1] = vmax
    return edges


BIN_ALGORITHMS: dict[str, BinAlgorithm] = {
    "linear": _linear_edges,
    "lin": _linear_edges,
    "log": _log_edges,
    "equaln": _equaln_edges,
    "quantile": _equaln_edges,
}

AXIS_PROPERTIES: dict[str, AxisPropertyFunc] = {}
BIN_DERIVED_PROPERTIES: dict[str, BinDerivedSpec] = {}


@overload
def register_bin_algorithm(name: str, func: None = None, *, overwrite: bool = False) -> Callable[[BinAlgorithm], BinAlgorithm]: ...


@overload
def register_bin_algorithm(name: str, func: BinAlgorithm, *, overwrite: bool = False) -> BinAlgorithm: ...


def register_bin_algorithm(name: str, func: BinAlgorithm | None = None, *, overwrite: bool = False) -> BinAlgorithm | Callable[[BinAlgorithm], BinAlgorithm]:
    def decorator(algorithm: BinAlgorithm) -> BinAlgorithm:
        if not overwrite and name in BIN_ALGORITHMS:
            raise KeyError(f"Bin algorithm {name!r} is already registered.")
        BIN_ALGORITHMS[name] = algorithm
        return algorithm

    if func is None:
        return decorator
    return decorator(func)


@overload
def register_axis_property(name: str, func: None = None, *, overwrite: bool = False) -> Callable[[AxisPropertyFunc], AxisPropertyFunc]: ...


@overload
def register_axis_property(name: str, func: AxisPropertyFunc, *, overwrite: bool = False) -> AxisPropertyFunc: ...


def register_axis_property(name: str, func: AxisPropertyFunc | None = None, *, overwrite: bool = False) -> AxisPropertyFunc | Callable[[AxisPropertyFunc], AxisPropertyFunc]:
    def decorator(property_func: AxisPropertyFunc) -> AxisPropertyFunc:
        if not overwrite and name in AXIS_PROPERTIES:
            raise KeyError(f"Axis property {name!r} is already registered.")
        AXIS_PROPERTIES[name] = property_func
        return property_func

    if func is None:
        return decorator
    return decorator(func)


@overload
def register_bin_derived(
    name: str,
    func: None = None,
    *,
    condition: BinDerivedCondition | None = None,
    scope: str = "geometry",
    overwrite: bool = False,
) -> Callable[[BinDerivedFunc], BinDerivedFunc]: ...


@overload
def register_bin_derived(
    name: str,
    func: BinDerivedFunc,
    *,
    condition: BinDerivedCondition | None = None,
    scope: str = "geometry",
    overwrite: bool = False,
) -> BinDerivedFunc: ...


def register_bin_derived(
    name: str,
    func: BinDerivedFunc | None = None,
    *,
    condition: BinDerivedCondition | None = None,
    scope: str = "geometry",
    overwrite: bool = False,
) -> BinDerivedFunc | Callable[[BinDerivedFunc], BinDerivedFunc]:
    def decorator(derived_func: BinDerivedFunc) -> BinDerivedFunc:
        if not overwrite and name in BIN_DERIVED_PROPERTIES:
            raise KeyError(f"Bin derived property {name!r} is already registered.")
        BIN_DERIVED_PROPERTIES[name] = BinDerivedSpec(name=name, func=derived_func, condition=condition, scope=scope)
        return derived_func

    if func is None:
        return decorator
    return decorator(func)


def axis_matches(axis: BinAxis, names: set[str]) -> bool:
    return axis.alias in names or (isinstance(axis.prop, str) and axis.prop in names)


def has_axis(names: set[str]) -> BinDerivedCondition:
    return lambda bins: any(axis_matches(axis, names) for axis in bins.axes)


def has_axes(*groups: set[str]) -> BinDerivedCondition:
    return lambda bins: all(has_axis(group)(bins) for group in groups)


@dataclass(frozen=True)
class BinAxis:
    alias: str
    prop: Any
    mins: Any
    maxs: Any
    include_rightmost: bool = True
    units: Any | None = None

    def __post_init__(self) -> None:
        mins = _as_1d_array(self.mins, name="mins")
        maxs = _as_1d_array(self.maxs, name="maxs")
        if mins.shape != maxs.shape:
            raise ValueError("mins and maxs must have the same shape.")
        if mins.size == 0:
            raise ValueError("an axis must contain at least one bin.")
        if not np.all(mins < maxs):
            raise ValueError("each bin must satisfy min < max.")
        if not np.all(np.diff(mins) >= 0) or not np.all(np.diff(maxs) >= 0):
            raise ValueError("bin bounds must be monotonic.")
        if mins.size > 1 and not np.all(maxs[:-1] <= mins[1:]):
            raise ValueError("bins must not overlap.")

    @property
    def nbins(self) -> int:
        return int(len(self.mins))

    @property
    def centers(self) -> Any:
        return 0.5 * (self.mins + self.maxs)

    @property
    def widths(self) -> Any:
        return self.maxs - self.mins

    @property
    def is_continuous(self) -> bool:
        return bool(self.nbins == 1 or np.all(np.asarray(self.maxs[:-1]) == np.asarray(self.mins[1:])))

    @property
    def edges(self) -> Any:
        if self.is_continuous:
            return np.concatenate((np.asarray(self.mins[:1]), np.asarray(self.maxs)))
        return np.column_stack((np.asarray(self.mins), np.asarray(self.maxs)))

    def assign(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"axis {self.alias!r} values must be one-dimensional.")

        mins = np.asarray(self.mins)
        maxs = np.asarray(self.maxs)
        axis_bin = np.full(arr.shape[0], -1, dtype=int)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return axis_bin, finite

        candidate = np.searchsorted(maxs, arr, side="right")
        if self.include_rightmost:
            candidate[arr == maxs[-1]] = self.nbins - 1

        candidate_valid = finite & (candidate >= 0) & (candidate < self.nbins)
        valid_positions = np.nonzero(candidate_valid)[0]
        if valid_positions.size:
            selected = candidate[valid_positions]
            in_interval = arr[valid_positions] >= mins[selected]
            if self.include_rightmost:
                in_interval &= (arr[valid_positions] < maxs[selected]) | (
                    (selected == self.nbins - 1) & (arr[valid_positions] == maxs[-1])
                )
            else:
                in_interval &= arr[valid_positions] < maxs[selected]
            accepted = valid_positions[in_interval]
            axis_bin[accepted] = candidate[accepted]

        return axis_bin, axis_bin >= 0


def _axis_min(axis: BinAxis) -> Any:
    return axis.mins


def _axis_max(axis: BinAxis) -> Any:
    return axis.maxs


def _axis_center(axis: BinAxis) -> Any:
    return axis.centers


def _axis_width(axis: BinAxis) -> Any:
    return axis.widths


def _axis_edges(axis: BinAxis) -> Any:
    edges = axis.edges
    return edges[:-1] if np.asarray(edges).ndim == 1 else edges


def _axis_is_continuous(axis: BinAxis) -> np.ndarray:
    return np.full(axis.nbins, axis.is_continuous, dtype=bool)


register_axis_property("min", _axis_min)
register_axis_property("max", _axis_max)
register_axis_property("center", _axis_center)
register_axis_property("width", _axis_width)
register_axis_property("edges", _axis_edges)
register_axis_property("is_continuous", _axis_is_continuous)


@register_bin_derived("multi_index")
def _bin_multi_index(bins: Any) -> np.ndarray:
    return bins.multi_index_array()


@register_bin_derived("cell_widths")
def _bin_cell_widths(bins: Any) -> np.ndarray:
    widths = [np.asarray(axis.widths) for axis in bins.axes]
    multi = bins.multi_index_array()
    return np.column_stack([widths[index][multi[:, index]] for index in range(bins.ndim)])


@register_bin_derived("cell_volume")
def _bin_cell_volume(bins: Any) -> np.ndarray:
    return np.prod(np.asarray(bins._resolve_query("cell_widths")), axis=1)


@register_bin_derived("area", condition=has_axis({"rxy", "R", "r"}))
def _bin_annulus_area(bins: Any) -> np.ndarray:
    axis = bins.find_axis({"rxy", "R", "r"})
    values = np.pi * (np.asarray(axis.maxs) ** 2 - np.asarray(axis.mins) ** 2)
    if bins.ndim > 1:
        axis_index = bins.axes.index(axis)
        values = values[bins.multi_index_array()[:, axis_index]]
    return values


def _volume_available(bins: Any) -> bool:
    return bins.ndim != 1 or has_axis({"r"})(bins)


@register_bin_derived("volume", condition=_volume_available)
def _bin_volume(bins: Any) -> np.ndarray:
    if bins.ndim == 1:
        axis = bins.find_axis({"r"})
        return 4.0 / 3.0 * np.pi * (np.asarray(axis.maxs) ** 3 - np.asarray(axis.mins) ** 3)
    return np.asarray(bins._resolve_query("cell_volume"))


@register_bin_derived("cylindrical_volume", condition=has_axes({"rxy", "R", "r"}, {"z"}))
def _bin_cylindrical_volume(bins: Any) -> np.ndarray:
    r_axis = bins.find_axis({"rxy", "R", "r"})
    z_axis = bins.find_axis({"z"})
    multi = bins.multi_index_array()
    r_index = bins.axes.index(r_axis)
    z_index = bins.axes.index(z_axis)
    area = np.pi * (np.asarray(r_axis.maxs) ** 2 - np.asarray(r_axis.mins) ** 2)
    z_width = np.asarray(z_axis.widths)
    return area[multi[:, r_index]] * z_width[multi[:, z_index]]



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


def resolve_runtime_value(value: Any, sim: Any, ctx: ExecutionContext | None, input: NodeInput | None) -> Any:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

    if isinstance(value, CalculatorBase):
        if ctx is None or input is None:
            return value(sim)
        return ctx.public_value(value, input)
    if callable(value) and not isinstance(value, (str, bytes)):
        return value(sim)
    return value


def materialize_axis(spec: Any, sim: Any, ctx: ExecutionContext | None = None, input: NodeInput | None = None, *, index: int = 0) -> tuple[BinAxis, Any]:
    values = resolve_axis_values(spec.prop, sim, ctx, input)
    arr = _as_1d_array(values, name=f"axis {index} prop")
    if len(arr) != len(sim):
        raise ValueError(f"axis {infer_alias(spec.prop, spec.alias, index)!r} prop length must match active sim length.")

    alias = infer_alias(spec.prop, spec.alias, index)
    if spec.edges is not None:
        if any(value is not None for value in (spec.vmin, spec.vmax, spec.nbins)):
            raise ValueError("edges cannot be mixed with vmin/vmax/nbins.")
        if spec.lows is not None or spec.highs is not None:
            raise ValueError("edges cannot be mixed with lows/highs.")
        edges = _as_1d_array(resolve_runtime_value(spec.edges, sim, ctx, input), name="edges")
        if edges.shape[0] < 2:
            raise ValueError("edges must contain at least two values.")
        if not np.all(np.diff(np.asarray(edges)) > 0):
            raise ValueError("edges must be strictly increasing.")
        edges = _coerce_edges_like(edges, values)
        return BinAxis(alias=alias, prop=spec.prop, mins=edges[:-1], maxs=edges[1:], include_rightmost=spec.include_rightmost, units=spec.units), values

    if spec.lows is not None or spec.highs is not None:
        if any(value is not None for value in (spec.vmin, spec.vmax, spec.nbins)):
            raise ValueError("lows/highs cannot be mixed with vmin/vmax/nbins.")
        if spec.lows is None or spec.highs is None:
            raise ValueError("lows and highs must be provided together.")
        lows = _coerce_edges_like(_as_1d_array(resolve_runtime_value(spec.lows, sim, ctx, input), name="lows"), values)
        highs = _coerce_edges_like(_as_1d_array(resolve_runtime_value(spec.highs, sim, ctx, input), name="highs"), values)
        return BinAxis(alias=alias, prop=spec.prop, mins=lows, maxs=highs, include_rightmost=spec.include_rightmost, units=spec.units), values

    nbins = resolve_runtime_value(spec.nbins, sim, ctx, input)
    if nbins is None:
        raise ValueError("nbins is required when edges or lows/highs are not provided.")
    nbins = int(nbins)
    if nbins <= 0:
        raise ValueError("nbins must be positive.")

    inferred_min, inferred_max = _finite_minmax(arr)
    vmin = resolve_runtime_value(spec.vmin, sim, ctx, input)
    vmax = resolve_runtime_value(spec.vmax, sim, ctx, input)
    vmin = inferred_min if vmin is None else _coerce_bound_value(vmin, values, sim)
    vmax = inferred_max if vmax is None else _coerce_bound_value(vmax, values, sim)
    if vmin >= vmax:
        raise ValueError("vmin must be smaller than vmax.")

    mode = spec.mode
    if callable(mode):
        edges = mode(arr, nbins, vmin, vmax)
    else:
        try:
            algorithm = BIN_ALGORITHMS[str(mode)]
        except KeyError as exc:
            raise KeyError(f"Unknown bin mode {mode!r}.") from exc
        edges = algorithm(np.asarray(arr), nbins, vmin, vmax)

    edges = _coerce_edges_like(_as_1d_array(edges, name="edges"), values)
    if len(edges) != nbins + 1:
        raise ValueError("bin algorithm must return nbins + 1 edges.")
    if not np.all(np.diff(np.asarray(edges)) > 0):
        raise ValueError("bin edges must be strictly increasing.")
    return BinAxis(alias=alias, prop=spec.prop, mins=edges[:-1], maxs=edges[1:], include_rightmost=spec.include_rightmost, units=spec.units), values
