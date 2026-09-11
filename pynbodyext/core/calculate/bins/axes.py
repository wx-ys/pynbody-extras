"""Bin axis definitions, edge-generation algorithms, and axis registries.

This module provides the :class:`BinAxis` value object (per-axis bin edges,
centers, widths, and physical measure), the :class:`BinAxisAccessor` used by
``bins.axes[...]``, the built-in edge algorithms (:func:`register_bin_algorithm`,
``BIN_ALGORITHMS``), and the extensible axis-property / measure registries.
"""

from __future__ import annotations

import logging
import weakref
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
def register_bin_algorithm(
    name: str, func: None = None, *, overwrite: bool = False
) -> Callable[[BinAlgorithm], BinAlgorithm]: ...


@overload
def register_bin_algorithm(name: str, func: BinAlgorithm, *, overwrite: bool = False) -> BinAlgorithm: ...


def register_bin_algorithm(
    name: str, func: BinAlgorithm | None = None, *, overwrite: bool = False
) -> BinAlgorithm | Callable[[BinAlgorithm], BinAlgorithm]:
    """Register a custom bin-edge algorithm.

    The algorithm must be ``f(values, nbins, vmin, vmax) -> ndarray`` returning
    ``nbins + 1`` strictly increasing edges.  Works as a decorator or as a direct
    call.

    Parameters
    ----------
    name : str
        Key used by ``Bin1D(mode=...)``.
    func : callable, optional
        The algorithm.  Omit to use as a decorator.
    overwrite : bool, default: False
        Whether to replace an existing algorithm with the same name.

    Returns
    -------
    callable
        The algorithm (when called directly) or a decorator.

    Examples
    --------
    >>> import numpy as np
    >>> @register_bin_algorithm("centroid_edges", overwrite=True)
    ... def centroid_edges(values, nbins, vmin, vmax):
    ...     return np.linspace(vmin, vmax, nbins + 1)
    """

    def decorator(algorithm: BinAlgorithm) -> BinAlgorithm:
        if not overwrite and name in BIN_ALGORITHMS:
            raise KeyError(f"Bin algorithm {name!r} is already registered.")
        BIN_ALGORITHMS[name] = algorithm
        return algorithm

    if func is None:
        return decorator
    return decorator(func)


# --- Axis physical measure registry ---
# Maps axis alias to callable (BinAxis) -> np.ndarray
_AXIS_MEASURE_REGISTRY: dict[str, Callable[[BinAxis], np.ndarray]] = {}

# Maps named measure types to callables, decoupled from alias
_AXIS_MEASURE_TYPE_REGISTRY: dict[str, Callable[[BinAxis], np.ndarray]] = {
    "spherical_shell": lambda axis: axis.shell_volume,
    "annulus": lambda axis: axis.annulus_area,
    "linear": lambda axis: axis.widths,
}

# Weak set of BinNDResult instances that should be notified when the global
# axis measure registry changes.  Each subscriber must expose a method
#  _on_axis_measure_change(alias: str, type_name: str) -> int
# that invalidates cached entries and returns the number cleared.
_measure_change_subscribers: weakref.WeakSet = weakref.WeakSet()


def axis_matches(axis: BinAxis, names: set[str]) -> bool:
    """Whether *axis* matches any of *names* (by alias or field prop)."""
    return axis.alias in names or (isinstance(axis.prop, str) and axis.prop in names)


class BinAxisAccessor:
    """Accessor returned by ``bins.axes`` for convenient axis lookup.

    Supports attribute access by alias, string subscript, and integer subscript.
    It also iterates over all axes and offers :meth:`find` and
    :meth:`set_measure_type`.

    Examples
    --------
    >>> import pynbody
    >>> sim = pynbody.new(dm=6)
    >>> sim["r"] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    >>> bins = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)
    >>> bins.axes.r.centers.tolist()
    [1.0, 3.0, 5.0]
    >>> bins.axes["r"] is bins.axes[0]
    True
    """

    def __init__(self, axes: tuple[BinAxis, ...], owner: Any = None) -> None:
        self._axes = axes
        self._owner = owner  # BinNDResult that owns this accessor

    @property
    def extent(self) -> list[float]:
        """``[min0, max0, min1, max1, ...]`` covering every axis."""
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

    def find(self, aliases: set[str]) -> BinAxis:
        """Return the axis matching any of *aliases* (alias or prop name)."""
        if self._owner is None:
            raise RuntimeError("BinAxisAccessor.find requires an owning BinNDResult.")
        return self._owner._find_axis(aliases)

    def set_measure_type(self, alias: str, type_name: str | None) -> None:
        """Per-instance override: assign *type_name* to *alias*.

        Only affects the owning :class:`BinNDResult` instance — its cached
        density entries are invalidated; no other instance is touched.

        Parameters
        ----------
        alias:
            The axis alias (or prop name) to assign the measure type to.
        type_name:
            A measure type name previously registered via
            :meth:`BinAxis.register_measure_type`.  Built-in types include
            ``"spherical_shell"``, ``"annulus"``, and ``"linear"``.  Pass
            ``None`` to clear a previously set per-instance override.
        """
        if self._owner is None:
            raise RuntimeError("BinAxisAccessor.set_measure_type requires an owning BinNDResult.")
        self._owner._set_axis_measure_type(alias, type_name)

    def set_axis_measure_type(self, alias: str, type_name: str | None) -> None:
        """Backwards-compatible alias for :meth:`set_measure_type`."""
        self.set_measure_type(alias, type_name)

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
    """Return a derived-property condition that is True when any axis matches."""
    return lambda bins: any(axis_matches(axis, names) for axis in bins.axes)


def has_axes(*groups: set[str]) -> BinDerivedCondition:
    """Return a condition that is True only when every alias group has a match."""
    return lambda bins: all(has_axis(group)(bins) for group in groups)


@dataclass(frozen=True)
class BinAxis:
    """A single binning axis (edges, centers, widths, and measure).

    Parameters
    ----------
    alias : str
        Short name used to look up the axis (e.g. ``"r"``).
    prop : str, callable, or CalculatorBase
        The source property/field used to assign particles.
    mins, maxs : array-like
        Lower/upper edge arrays; ``len(mins) == nbins``.
    include_rightmost : bool, default: True
        Whether the rightmost edge is inside the last bin.
    units : str or UnitBase, optional
        Units for the axis values.

    Examples
    --------
    >>> import numpy as np
    >>> axis = BinAxis("r", "r", mins=np.array([0.0, 2.0, 4.0]), maxs=np.array([2.0, 4.0, 6.0]))
    >>> axis.nbins
    3
    >>> axis.centers.tolist()
    [1.0, 3.0, 5.0]
    """

    alias: str
    prop: Any
    mins: np.ndarray
    maxs: np.ndarray
    include_rightmost: bool = True
    units: Any | None = None

    _axis_properties: ClassVar[dict[str, AxisPropertyFunc]] = {}

    @overload
    @classmethod
    def register_property(
        cls, name: AxisPropertyFunc, func: None = None, *, overwrite: bool = False
    ) -> AxisPropertyFunc: ...
    @overload
    @classmethod
    def register_property(cls, name: str, func: AxisPropertyFunc, *, overwrite: bool = False) -> AxisPropertyFunc: ...
    @overload
    @classmethod
    def register_property(
        cls, name: str, func: None = None, *, overwrite: bool = False
    ) -> Callable[[AxisPropertyFunc], AxisPropertyFunc]: ...
    @classmethod
    def register_property(
        cls, name: str | AxisPropertyFunc, func: AxisPropertyFunc | None = None, *, overwrite: bool = False
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
        """Number of bins along this axis."""
        return int(len(self.mins))

    @property
    def centers(self) -> np.ndarray:
        """Bin centers (midpoint of each bin's edges)."""
        return 0.5 * (self.mins + self.maxs)

    @property
    def widths(self) -> np.ndarray:
        """Bin widths (``maxs - mins``)."""
        return self.maxs - self.mins

    @property
    def is_continuous(self) -> bool:
        """Whether adjacent bins share edges (no gaps)."""
        return bool(self.nbins == 1 or np.all(self.maxs[:-1] == self.mins[1:]))

    @property
    def edges(self) -> np.ndarray:
        """Axis edges: a flat array when continuous, else ``(nbins, 2)`` bounds."""
        if self.is_continuous:
            return np.concatenate((self.mins[:1], self.maxs))
        return np.column_stack((np.asarray(self.mins), np.asarray(self.maxs)))

    @property
    def annulus_area(self) -> np.ndarray:
        """Per-bin annulus area ``π(max² − min²)`` regardless of axis alias."""
        return np.pi * (self.maxs**2 - self.mins**2)

    @property
    def shell_volume(self) -> np.ndarray:
        """Per-bin spherical shell volume ``4/3 π(max³ − min³)`` regardless of axis alias."""
        return (4.0 / 3.0) * np.pi * (self.maxs**3 - self.mins**3)

    @property
    def measure(self) -> np.ndarray:
        """Physical measure per bin — registry lookup with hardcoded fallback.

        Resolution order:

        1. :data:`_AXIS_MEASURE_REGISTRY` by ``.alias``
        2. Same registry by string ``.prop``
        3. Hardcoded legacy: ``rxy``/``R`` → annulus area, ``r`` → shell volume
        4. Default: bin width ``max − min``

        Use :attr:`annulus_area` or :attr:`shell_volume` to force a specific
        formula regardless of alias.
        """
        alias = self.alias
        prop_str = self.prop if isinstance(self.prop, str) else ""

        # 1. Registered function for this alias
        if alias in _AXIS_MEASURE_REGISTRY:
            return _AXIS_MEASURE_REGISTRY[alias](self)

        # 2. Registered function for this prop string
        if prop_str and prop_str in _AXIS_MEASURE_REGISTRY:
            return _AXIS_MEASURE_REGISTRY[prop_str](self)

        # 3. Hardcoded legacy fallback
        if alias in {"rxy", "R"} or prop_str in {"rxy", "R"}:
            return self.annulus_area
        if alias == "r" or prop_str == "r":
            return self.shell_volume

        # 4. Default: linear
        return self.widths

    # ------------------------------------------------------------------
    # Axis measure registry classmethods
    # ------------------------------------------------------------------

    @classmethod
    def register_axis_measure(
        cls, alias: str, func: Callable[[BinAxis], np.ndarray], *, overwrite: bool = False
    ) -> None:
        """Register a physical measure function for a specific axis alias.

        After registration, :attr:`measure` will call *func* for any axis
        whose ``.alias`` or string ``.prop`` matches *alias*.

        Parameters
        ----------
        alias:
            The axis alias (or prop name) to associate with this measure.
        func:
            A callable ``(BinAxis) -> np.ndarray`` returning per-bin measure.
        overwrite:
            If ``False`` (default), raise :exc:`KeyError` if *alias* is
            already registered.
        """
        if not overwrite and alias in _AXIS_MEASURE_REGISTRY:
            raise KeyError(f"Axis measure for alias {alias!r} is already registered.")
        _AXIS_MEASURE_REGISTRY[alias] = func

    @classmethod
    def register_measure_type(
        cls, name: str, func: Callable[[BinAxis], np.ndarray], *, overwrite: bool = False
    ) -> None:
        """Register a named physical measure type (global).

        Measure types can be assigned to axis aliases via
        :meth:`BinAxisAccessor.set_measure_type` (per-instance) or
        :meth:`BinAxis.set_axis_measure_type` (global).  Built-in types:
        ``"spherical_shell"``, ``"annulus"``, ``"linear"``.

        When *overwrite* is ``True`` and *func* differs from the
        currently registered function, all live :class:`BinNDResult`
        instances are notified so they can invalidate cached density
        entries that depend on the old measure.

        Parameters
        ----------
        name:
            Name for the measure type.
        func:
            A callable ``(BinAxis) -> np.ndarray`` returning per-bin measure.
        overwrite:
            If ``False`` (default), raise :exc:`KeyError` if *name* is
            already registered.
        """
        if not overwrite and name in _AXIS_MEASURE_TYPE_REGISTRY:
            raise KeyError(f"Measure type {name!r} is already registered.")

        old_func = _AXIS_MEASURE_TYPE_REGISTRY.get(name)
        _AXIS_MEASURE_TYPE_REGISTRY[name] = func

        # Notify subscribers if the function actually changed
        if overwrite and old_func is not None and old_func is not func:
            total_cleared = 0
            all_cleared: list[str] = []
            for subscriber in list(_measure_change_subscribers):
                try:
                    n, names = subscriber._on_axis_measure_change(name, name)
                except Exception:
                    continue
                total_cleared += n
                all_cleared.extend(names)
            if total_cleared > 0:
                unique = sorted(set(all_cleared))
                _logger = logging.getLogger("pynbody")
                _logger.warning(
                    "Measure type %r redefined (overwrite=True). Cleared %d cached entr%s: %s.",
                    name,
                    total_cleared,
                    "y" if total_cleared == 1 else "ies",
                    ", ".join(unique),
                )

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


class BinMeasureResolver:
    """Per-instance axis-measure overrides resolved against the global measure registries.

    The resolvers checks the per-instance override (keyed by ``axis.alias``,
    value is a measure-type name from :data:`_AXIS_MEASURE_TYPE_REGISTRY`) first,
    then falls back to the global :attr:`BinAxis.measure` property.
    """

    def __init__(self) -> None:
        self._overrides: dict[str, str] = {}

    def resolve(self, axis: BinAxis) -> np.ndarray:
        type_name = self._overrides.get(axis.alias)
        if type_name is not None:
            func = _AXIS_MEASURE_TYPE_REGISTRY.get(type_name)
            if func is not None:
                return func(axis)
        return axis.measure

    def set(self, alias: str, type_name: str | None) -> None:
        if type_name is None:
            self._overrides.pop(alias, None)
        else:
            self._overrides[alias] = type_name

    def get(self, alias: str) -> str | None:
        return self._overrides.get(alias)
