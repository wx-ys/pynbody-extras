from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, overload

import numpy as np

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


def axis_matches(axis: BinAxis, names: set[str]) -> bool:
    return axis.alias in names or (isinstance(axis.prop, str) and axis.prop in names)


class BinAxisAccessor:
    """Accessor returned by ``bins.axis`` for convenient axis lookup.

    Supports attribute access by alias, string subscript, and integer subscript::

        bins.axis.r          # axis with alias "r"
        bins.axis["r"]       # same
        bins.axis[0]         # first axis
        list(bins.axis)      # iterate over all axes
    """

    def __init__(self, axes: tuple[BinAxis, ...]) -> None:
        self._axes = axes

    @property
    def extent(self) -> list[float]:
        ex: list[float] = []
        for axis in self._axes:
            ex.extend([float(axis.mins[0]), float(axis.maxs[-1])])
        return ex

    def __getitem__(self, key: int | str) -> BinAxis:
        if isinstance(key, (int, np.integer)):
            return self._axes[int(key)]
        for ax in self._axes:
            if ax.alias == key:
                return ax
        raise KeyError(f"No bin axis {key!r}.")

    def __getattr__(self, name: str) -> BinAxis:
        # Avoid recursing on internal dunder/private names
        if name.startswith("_"):
            raise AttributeError(name)
        for ax in self._axes:
            if ax.alias == name:
                return ax
        raise AttributeError(f"No bin axis {name!r}.")

    def __iter__(self):
        return iter(self._axes)

    def __len__(self) -> int:
        return len(self._axes)

    def __repr__(self) -> str:
        aliases = ", ".join(ax.alias for ax in self._axes)
        return f"<BinAxisAccessor [{aliases}]>"
    def __dir__(self) -> list[str]:
        # Include axis aliases in dir() for better auto-completion in interactive environments
        return [ax.alias for ax in self._axes] + list(super().__dir__())

    def _ipython_key_completions_(self) -> list[str]:
        return [ax.alias for ax in self._axes]



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

    _axis_properties: ClassVar[dict[str, AxisPropertyFunc]] = {}

    @overload
    @classmethod
    def register_property(cls, name: AxisPropertyFunc, func: None = None, *, overwrite: bool = False) -> AxisPropertyFunc: ...
    @overload
    @classmethod
    def register_property(cls, name: str, func: AxisPropertyFunc, *, overwrite: bool = False) -> AxisPropertyFunc: ...
    @overload
    @classmethod
    def register_property(cls, name: str, func: None = None, *, overwrite: bool = False) -> Callable[[AxisPropertyFunc], AxisPropertyFunc]: ...
    @classmethod
    def register_property(
        cls,
        name: str | AxisPropertyFunc,
        func: AxisPropertyFunc | None = None,
        *,
        overwrite: bool = False,
    ) -> AxisPropertyFunc | Callable[[AxisPropertyFunc], AxisPropertyFunc]:
        """Register a computed axis property accessible via attribute lookup on :class:`BinAxis`.

        Three calling conventions are supported::

            # bare decorator — property name inferred from function name
            @BinAxis.register_property
            def my_prop(axis: BinAxis):
                return axis.centers * 2

            # decorator factory — explicit name
            @BinAxis.register_property("my_prop")
            def _(axis: BinAxis):
                return axis.centers * 2

            # direct call
            BinAxis.register_property("my_prop", my_func)
        """
        # bare @BinAxis.register_property (name is actually the function)
        if callable(name):
            actual_func = name
            actual_name = actual_func.__name__
            if not overwrite and actual_name in cls._axis_properties:
                raise KeyError(f"Axis property {actual_name!r} is already registered.")
            cls._axis_properties[actual_name] = actual_func
            return actual_func

        def decorator(property_func: AxisPropertyFunc) -> AxisPropertyFunc:
            if not overwrite and name in cls._axis_properties:
                raise KeyError(f"Axis property {name!r} is already registered.")
            cls._axis_properties[name] = property_func
            return property_func

        if func is None:
            return decorator
        return decorator(func)

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
        return bool(self.nbins == 1 or np.all(self.maxs[:-1] == self.mins[1:]))

    @property
    def edges(self) -> Any:
        if self.is_continuous:
            return np.concatenate((self.mins[:1], self.maxs))
        return np.column_stack((np.asarray(self.mins), np.asarray(self.maxs)))

    @property
    def annulus_area(self) -> np.ndarray:
        """Per-bin annulus area ``π(max² − min²)`` regardless of axis alias."""
        return np.pi * (self.maxs** 2 - self.mins ** 2)

    @property
    def shell_volume(self) -> np.ndarray:
        """Per-bin spherical shell volume ``4/3 π(max³ − min³)`` regardless of axis alias."""
        return (4.0 / 3.0) * np.pi * (self.maxs ** 3 - self.mins ** 3)

    @property
    def measure(self) -> np.ndarray:
        """Physical measure per bin — auto-detected from the axis alias / prop.

        - ``rxy`` / ``R`` → annulus area ``π(max² − min²)``
        - ``r`` → spherical shell volume ``4/3π(max³ − min³)``
        - anything else → bin width ``max − min``

        Use :attr:`annulus_area` or :attr:`shell_volume` to force a specific
        formula regardless of alias.
        """
        alias = self.alias
        prop_str = self.prop if isinstance(self.prop, str) else ""
        if alias in {"rxy", "R"} or prop_str in {"rxy", "R"}:
            return self.annulus_area
        if alias == "r" or prop_str == "r":
            return self.shell_volume
        return self.widths

    def __getattr__(self, name: str) -> Any:
        # Fallback for dynamically registered axis properties.
        # Only called when normal attribute lookup (fields, @property, methods) fails.
        prop_func = type(self)._axis_properties.get(name)
        if prop_func is not None:
            return prop_func(self)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

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

@BinAxis.register_property("min")
def _axis_min(axis: BinAxis) -> Any:
    return axis.mins[0:1]

@BinAxis.register_property("max")
def _axis_max(axis: BinAxis) -> Any:
    return axis.maxs[-1:]

@BinAxis.register_property("center")
def _axis_center(axis: BinAxis) -> Any:
    return 0.5 * (axis.min + axis.max)

@BinAxis.register_property("width")
def _axis_width(axis: BinAxis) -> Any:
    return axis.max - axis.min


