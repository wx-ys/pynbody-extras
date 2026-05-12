from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.snapshot import SimSnap

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.runtime.options import RunOptions

from .accessors import BinParticlesAccessor
from .arrays import BinsArray
from .axes import BIN_DERIVED_PROPERTIES, BinDerivedSpec, axis_matches
from .selectors import is_bool_array, is_int_sequence
from .statistics import BinNDStatAccessor, apply_pipeline, get_statistic, parse_pipeline_key

if TYPE_CHECKING:
    from collections.abc import Callable

    from .axes import BinAxis
    from .nodes import BinND


def _is_sim_like(value: Any) -> bool:
    return isinstance(value, SimSnap)


# Minimal RunOptions for per-bin apply loops: no cache, no progress, no perf,
# no observer.  Created once and reused across all apply() calls.
_BATCH_RUN_OPTIONS: RunOptions = RunOptions(
    cache=False,
    progress=False,
    perf_time=False,
    perf_memory=False,
    observe=False,
)


class BinsResultEngine:
    def __init__(self) -> None:
        self.diagnostics: list[dict[str, Any]] = []

    def record(self, event: str, **payload: Any) -> None:
        self.diagnostics.append({"event": event, **payload})


@dataclass(frozen=True)
class ResultDerivedSpec:
    name: str
    func: Callable[[Any], Any]
    scope: str = "derived"
    condition: Callable[[Any], bool] | None = None

    def is_available(self, bins: Any) -> bool:
        return True if self.condition is None else bool(self.condition(bins))


class _DerivedAccessor:
    def __get__(self, instance: BinNDResult | None, owner: type[BinNDResult]) -> Callable[..., Any]:
        if instance is None:
            return owner._register_derived
        return instance._resolve_query


class BinNDResult:
    _SHARED_SCOPES: ClassVar[set[str]] = {"axis", "geometry"}
    _derived_property_registry: ClassVar[defaultdict[type, dict[str, ResultDerivedSpec]]] = defaultdict(dict)
    derived = _DerivedAccessor()
    derived_property = _DerivedAccessor()

    def __init__(
        self,
        *,
        sim: Any,
        source_sim: Any,
        axes: tuple[BinAxis, ...],
        bin_indices: list[np.ndarray],
        particle_bin: np.ndarray,
        valid_mask: np.ndarray,
        calculator: BinND,
        scope_signature: Any = None,
        parent: BinNDResult | None = None,
    ) -> None:
        self.sim = sim
        self.source_sim = source_sim
        self.axes = axes
        self.shape_bins = tuple(axis.nbins for axis in axes)
        self.ndim = len(axes)
        self.bin_indices = bin_indices
        self.particle_bin = particle_bin
        self.valid_mask = valid_mask
        self.calculator = calculator
        self.scope_signature = scope_signature
        self.parent = parent
        self._cache: dict[Any, BinsArray] = {}
        self._engine = BinsResultEngine()
        if parent is None:
            self._subs_cache: dict[tuple[int, ...], SubBinNDResult] = {}

    @property
    def root(self) -> BinNDResult:
        return self if self.parent is None else self.parent.root

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def nbins(self) -> int:
        return int(np.prod(self.shape_bins, dtype=int))

    @property
    def total_nbins(self) -> int:
        return self.nbins

    @property
    def unassigned_count(self) -> int:
        return int(np.count_nonzero(~self.valid_mask))

    @property
    def count(self) -> BinsArray:
        return self._resolve_query("count")

    @property
    def particles_at_bin(self) -> BinParticlesAccessor:
        return BinParticlesAccessor(self)

    @property
    def num_cached_arr(self) -> int:
        return len(self._cache)

    @property
    def nsubs(self) -> int:
        return len(getattr(self.root, "_subs_cache", {}))

    @property
    def total_cached_arr(self) -> int:
        root = self.root
        total = root.num_cached_arr
        for subresult in getattr(root, "_subs_cache", {}).values():
            total += subresult.num_cached_arr
        return total

    @property
    def edges(self) -> Any:
        self._require_1d("edges")
        return self.axes[0].edges

    @property
    def mins(self) -> Any:
        self._require_1d("mins")
        return self.axes[0].mins

    @property
    def maxs(self) -> Any:
        self._require_1d("maxs")
        return self.axes[0].maxs

    @property
    def centers(self) -> Any:
        self._require_1d("centers")
        return self.axes[0].centers

    @property
    def widths(self) -> Any:
        self._require_1d("widths")
        return self.axes[0].widths

    def _require_1d(self, name: str) -> None:
        if self.ndim != 1:
            raise AttributeError(f"{name} is ambiguous for ND bins; use bins.axis(alias).{name}.")

    def axis(self, key: int | str) -> BinAxis:
        if isinstance(key, (int, np.integer)):
            return self.axes[int(key)]
        for axis in self.axes:
            if axis.alias == key:
                return axis
        raise KeyError(f"Unknown bin axis {key!r}.")

    def families(self) -> Any:
        return self.sim.families()

    @property
    def stat(self) -> BinNDStatAccessor:
        """Accessor for pipeline statistic queries, e.g. ``bins.stat["mass.sum"]``."""
        return BinNDStatAccessor(self)

    def keys(self) -> list[str]:
        keys: set[str] = set()
        keys.update(name for name, spec in BIN_DERIVED_PROPERTIES.items() if spec.is_available(self))
        keys.update(self.property_keys())
        for cache_key in self._cache:
            if isinstance(cache_key, tuple) and len(cache_key) >= 2 and isinstance(cache_key[1], str):
                keys.add(cache_key[1])
        return sorted(keys)

    def property_keys(self) -> list[str]:
        registry = type(self)._derived_property_registry
        keys: set[str] = set()
        for cls in type(self).mro():
            keys.update(name for name, spec in registry.get(cls, {}).items() if spec.is_available(self))
        return sorted(keys)

    def all_keys(self) -> list[str]:
        return self.keys()

    def _ipython_key_completions_(self) -> list[str]:
        return self.all_keys()

    def get_subresult(self, subset: Any) -> SubBinNDResult:
        if not self.is_root:
            return self.root.get_subresult(subset)
        key = self._subset_cache_key(subset)
        if key in self._subs_cache:
            return self._subs_cache[key]
        sub = self.spawn(subset)
        self._subs_cache[key] = sub
        return sub

    def spawn(self, subset: Any) -> SubBinNDResult:
        return self.calculator._spawn_result(self.root, subset)

    def _subset_cache_key(self, subset: Any) -> tuple[int, ...]:
        root_sim = self.root.sim
        if subset is root_sim:
            return tuple(range(len(root_sim)))
        if hasattr(subset, "get_index_list"):
            try:
                indices = subset.get_index_list(root_sim)
            except Exception as exc:
                raise TypeError("SimSnap subset cannot be mapped to the root BinNDResult sim.") from exc
            return tuple(int(index) for index in np.asarray(indices, dtype=int))
        raise TypeError("SubBinNDResult requires a SimSnap subset that can be mapped to the root sim.")

    def __getitem__(self, key: Any) -> SubBinNDResult | BinsArray:
        if isinstance(key, str):
            # Axis properties like "r.center" must be accessed via bins.axis("r").center
            # String queries only handle: geometry/derived properties and pipeline stat queries
            return self._resolve_query(key)
        if isinstance(key, CalculatorBase) or (callable(key) and not isinstance(key, (str, bytes))):
            return self.apply(key)
        if isinstance(key, tuple) or isinstance(key, (int, np.integer, slice)) or is_int_sequence(key):
            raise TypeError("Bin selectors must use bins.particles_at_bin[...], not BinNDResult.__getitem__.")
        if is_bool_array(key):
            mask = np.asarray(key, dtype=bool)
            if len(mask) == len(self.sim):
                return self.get_subresult(self.sim[mask])
            if len(mask) == self.nbins:
                raise TypeError("Bin boolean masks must use bins.particles_at_bin[mask].")
            raise ValueError("Boolean selector length must match the current sim length for particle selection.")
        try:
            subset = self.sim[key]
        except Exception as exc:
            raise TypeError(f"Unsupported BinNDResult selector: {key!r}.") from exc
        if _is_sim_like(subset):
            return self.get_subresult(subset)
        raise TypeError(f"Selector did not produce a SimSnap subset: {type(subset)!r}.")

    def __getattr__(self, name: str) -> Any:
        if name in {"center", "width", "min", "max", "edges"} and self.ndim == 1:
            # Convenience shortcuts for 1D results – delegate to the axis object
            return getattr(self.axes[0], {"center": "centers", "width": "widths", "min": "mins", "max": "maxs", "edges": "edges"}[name])
        try:
            sub = getattr(self.sim, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if _is_sim_like(sub):
            return self.get_subresult(sub)
        raise AttributeError(name)

    def _query_scope(self, key: str) -> str:
        bin_spec = self._get_bin_derived_spec(key)
        if bin_spec is not None:
            return bin_spec.scope
        result_spec = self._get_derived_spec(key)
        if result_spec is not None:
            return result_spec.scope
        return "particles"

    def _resolve_query(self, key: str) -> BinsArray:
        scope = self._query_scope(key)
        if not self.is_root and scope in self._SHARED_SCOPES:
            return self.root._resolve_query(key)

        cache_key = (scope, key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        arr = self._compute_query(key, scope=scope)
        self._cache[cache_key] = arr
        self._engine.record("query", key=key, scope=scope)
        return arr

    def _wrap(self, values: Any, *, name: str, field: str | None = None, mode: str | None = None) -> BinsArray:
        return BinsArray(self, values, name=name, field=field, mode=mode)

    def _compute_query(self, key: str, *, scope: str) -> BinsArray:  # noqa: PLR0911
        if key in BIN_DERIVED_PROPERTIES:
            return self._geometry_property(key)
        spec = self._get_derived_spec(key)
        if spec is not None:
            return self._compute_registered_derived(spec)
        # Dot-notation pipeline query: "field.transform*.stat"
        parsed = parse_pipeline_key(key)
        if parsed is not None:
            field, transforms, terminal_stat = parsed
            return self._stat_pipeline(field, transforms, terminal_stat, query_key=key)
        raise KeyError(
            f"Unknown BinND query {key!r}. "
            "Use 'field.stat' syntax (e.g. 'mass.sum') for particle statistics. "
            "For axis properties use bins.axis('r').center."
        )

    def _get_derived_spec(self, key: str) -> ResultDerivedSpec | None:
        registry = type(self)._derived_property_registry
        for cls in type(self).mro():
            bucket = registry.get(cls)
            if bucket and key in bucket:
                spec = bucket[key]
                if spec.is_available(self):
                    return spec
        return None

    def _compute_registered_derived(self, spec: ResultDerivedSpec) -> BinsArray:
        values = spec.func(self)
        if isinstance(values, BinsArray):
            result = values
        else:
            result = self._wrap(values, name=spec.name)
        if result.size != self.nbins:
            raise ValueError(f"Derived BinND query {spec.name!r} must return one value per bin.")
        return result

    def _get_bin_derived_spec(self, key: str) -> BinDerivedSpec | None:
        spec = BIN_DERIVED_PROPERTIES.get(key)
        if spec is None or not spec.is_available(self):
            return None
        return spec

    def _compute_bin_derived(self, spec: BinDerivedSpec) -> BinsArray:
        values = spec.func(self)
        result = values if isinstance(values, BinsArray) else self._wrap(values, name=spec.name)
        if result.size != self.nbins:
            raise ValueError(f"Bin derived query {spec.name!r} must return one value per bin.")
        return result

    def multi_index_array(self) -> np.ndarray:
        return np.column_stack(np.unravel_index(np.arange(self.nbins), self.shape_bins, order="C"))

    def _geometry_property(self, key: str) -> BinsArray:  # noqa: PLR0911
        spec = self._get_bin_derived_spec(key)
        if spec is None:
            raise KeyError(f"Unknown or unavailable bin derived query {key!r}.")
        return self._compute_bin_derived(spec)

    def find_axis(self, aliases: set[str]) -> Any:
        for axis in self.axes:
            if axis_matches(axis, aliases):
                return axis
        raise KeyError(f"No axis matching {sorted(aliases)!r}.")

    _find_axis = find_axis

    def _stat_pipeline(
        self,
        field: str,
        transforms: list[str],
        terminal_stat: Any,
        *,
        weight: str | Callable[[Any], Any] | Any | None = None,
        query_key: str | None = None,
    ) -> BinsArray:
        """Compute per-bin statistic with optional pipeline transforms.

        E.g. field="vz", transforms=["abs"], terminal_stat=Mean() computes
        the per-bin mean of |vz|.
        """
        weight_key = self._weight_cache_token(weight)
        stat_key_str = terminal_stat.key
        cache_key = ("stat", field, tuple(transforms), stat_key_str, weight_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        values = self.sim[field]
        weights = None
        if isinstance(weight, str):
            weights = self.sim[weight]
        elif callable(weight):
            weights = weight(self.sim)
        elif weight is not None:
            weights = weight

        out = np.full(self.nbins, np.nan, dtype=float)
        for index, particle_indices in enumerate(self.bin_indices):
            if len(particle_indices) == 0:
                continue
            sub = np.asarray(values[particle_indices], dtype=float)
            sub = apply_pipeline(sub, transforms)
            sub_weights = None if weights is None else np.asarray(weights[particle_indices], dtype=float)
            out[index] = terminal_stat(sub, sub_weights)

        name = query_key or (f"{field}.{'.' .join(transforms)}.{stat_key_str}" if transforms else f"{field}.{stat_key_str}")
        result: BinsArray = self._wrap(out, name=name, field=field, mode=stat_key_str)
        if isinstance(values, (SimArray, IndexedSimArray)):
            result.units = values.units
            result.sim = values.sim
        self._cache[cache_key] = result
        return result

    def stat_explicit(
        self,
        field: str,
        statistic: str,
        weight: str | Callable[[Any], Any] | Any | None = None,
        transforms: list[str] | None = None,
    ) -> BinsArray:
        """Explicitly compute a per-bin statistic.

        Equivalent to ``bins["field.stat"]`` but accepts optional transforms and
        weight.  Example::

            bins.stat_explicit("mass", "sum")
            bins.stat_explicit("vz", "mean", transforms=["abs"])
            bins.stat_explicit("mass", "mean", weight="mass")
        """
        stat_obj = get_statistic(statistic)
        if stat_obj is None:
            raise KeyError(f"Unknown statistic {statistic!r}.")
        return self._stat_pipeline(field, transforms or [], stat_obj, weight=weight)

    def _weight_cache_token(self, weight: str | Callable[[Any], Any] | Any | None) -> Any:
        if weight is None or isinstance(weight, str):
            return weight
        if callable(weight):
            return ("callable", id(weight))
        return ("array", id(weight))

    def apply(
        self,
        query: Callable[[Any], Any] | CalculatorBase[Any, Any],
        *,
        name: str | None = None,
        empty: float = np.nan,
        vectorized: bool = False,
    ) -> BinsArray:
        """Evaluate *query* on each bin's particle subset and return a :class:`BinsArray`.

        Parameters
        ----------
        query:
            A callable ``(SimSnap) -> scalar`` or a :class:`CalculatorBase`.  The
            callable receives the particle sub-snapshot for each bin and must return
            a scalar value.
        name:
            Optional name for the resulting array.
        empty:
            Fill value for empty bins (default ``nan``).
        vectorized:
            If ``True``, *query* is expected to accept the full simulation snap and
            a flat ``particle_bin`` index array (shape ``(n_particles,)``, ``-1`` for
            unassigned particles) and must return a 1-D array of length ``nbins``.  This
            path bypasses the per-bin loop entirely and is much faster for large grids.

            Signature: ``query(sim, particle_bin) -> np.ndarray``
        """
        cache_key = ("apply", self._callable_cache_token(query), name, empty, vectorized)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if vectorized:
            # Fast path: user supplies a vectorized callable (sim, particle_bin) -> array
            # CalculatorBase is not valid here – must be a plain callable
            if isinstance(query, CalculatorBase):
                raise TypeError("vectorized=True requires a plain callable, not a CalculatorBase.")
            raw = query(self.sim, self.particle_bin)  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue,reportArgumentType]
            values = np.asarray(raw, dtype=float)
            if values.shape != (self.nbins,):
                raise TypeError(
                    f"Vectorized apply callable must return shape ({self.nbins},), got {values.shape}."
                )
        else:
            values = np.full(self.nbins, empty, dtype=float)
            if isinstance(query, CalculatorBase):
                # Fast path 2: full CalculatorBase → reuse one EvalEngine, skip Result assembly
                _batch_opts = _BATCH_RUN_OPTIONS
                with query.batch(_batch_opts) as run_one:
                    for index, particle_indices in enumerate(self.bin_indices):
                        if len(particle_indices) == 0:
                            continue
                        sub = self.sim[particle_indices]
                        arr = np.asarray(run_one(sub))
                        if arr.ndim != 0:
                            raise TypeError("Callable or CalculatorBase bin query must return a scalar in phase 1.")
                        values[index] = arr.item()
            else:
                # Plain callable
                for index, particle_indices in enumerate(self.bin_indices):
                    if len(particle_indices) == 0:
                        continue
                    sub = self.sim[particle_indices]
                    arr = np.asarray(query(sub))
                    if arr.ndim != 0:
                        raise TypeError("Callable or CalculatorBase bin query must return a scalar in phase 1.")
                    values[index] = arr.item()

        result_name: str = str(name or getattr(query, "name", None) or getattr(query, "__name__", "apply"))
        result = self._wrap(values, name=result_name)
        self._cache[cache_key] = result
        self._engine.record("apply", name=name, query=repr(query), vectorized=vectorized)
        return result

    def _callable_cache_token(self, query: Any) -> Any:
        if isinstance(query, CalculatorBase):
            return query.signature()
        signature = getattr(query, "signature", None)
        if callable(signature):
            return signature()
        return id(query)

    def cache_report(self) -> dict[str, Any]:
        return {"queries": len(self._cache), "subresults": self.nsubs, "total_queries": self.total_cached_arr}

    def query_report(self) -> list[dict[str, Any]]:
        return list(self._engine.diagnostics)

    @classmethod
    def _register_derived(
        cls,
        fn: Callable[[Any], Any] | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: Callable[[Any], bool] | None = None,
        overwrite: bool = False,
    ) -> Callable[[Any], Any]:
        if isinstance(fn, str):
            return cls._register_derived(name=fn, scope=scope, condition=condition, overwrite=overwrite)

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            query_name = name or func.__name__
            bucket = cls._derived_property_registry[cls]
            if not overwrite and query_name in bucket:
                raise KeyError(f"BinNDResult derived property {query_name!r} is already registered.")
            bucket[query_name] = ResultDerivedSpec(name=query_name, func=func, scope=scope, condition=condition)
            return func

        if fn is None:
            return decorator
        return decorator(fn)

    def plot(self, x: str, y: str, ax: Any = None, *, kind: str | None = None, **kwargs: Any) -> Any:
        """Plot a 1-D or 2-D profile.

        For 1-D results::

            bins.plot("r", "density")       # x = axis centers of "r", y = density

        *x* must name a bin axis alias whose centers are used as the x-coordinate.
        *y* is any string query accepted by ``bins[...]``.

        Parameters
        ----------
        x:
            Axis alias to use as the x-coordinate (e.g. ``"r"``).  Its bin
            centers are used.
        y:
            String query for the y-data (e.g. ``"density"``, ``"mass.sum"``).
        ax:
            Matplotlib axes object.  A new figure/axes is created if ``None``.
        kind:
            Plot style: ``"plot"`` (default) or ``"scatter"``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        x_axis = self.axis(x)
        x_values = np.asarray(x_axis.centers)
        y_arr = np.asarray(self._resolve_query(y)).ravel()

        plot_kind = "plot" if kind is None else kind
        if plot_kind == "scatter":
            return ax.scatter(x_values, y_arr, **kwargs)
        if plot_kind == "plot":
            return ax.plot(x_values, y_arr, **kwargs)
        raise ValueError(f"Unknown plot kind {plot_kind!r}; use 'plot' or 'scatter'.")

    def imshow(self, field: str, ax: Any = None, **kwargs: Any) -> Any:
        """Show a 2-D bin grid as an image.

        Requires exactly two axes.  The first axis maps to the x-direction and
        the second to the y-direction::

            bins2d.imshow("mass.sum")

        Parameters
        ----------
        field:
            String query for the field to display.
        ax:
            Matplotlib axes object.  A new figure/axes is created if ``None``.
        """
        import matplotlib.pyplot as plt

        if self.ndim != 2:
            raise ValueError("imshow requires exactly 2 bin axes.")
        if ax is None:
            _, ax = plt.subplots()

        x_axis, y_axis = self.axes
        extent = [
            float(np.asarray(x_axis.mins)[0]),
            float(np.asarray(x_axis.maxs)[-1]),
            float(np.asarray(y_axis.mins)[0]),
            float(np.asarray(y_axis.maxs)[-1]),
        ]
        grid = np.asarray(self._resolve_query(field)).reshape(self.shape_bins)
        return ax.imshow(grid.T, origin="lower", aspect="auto", extent=extent, **kwargs)

    def __repr__(self) -> str:
        parent_flag = "root" if self.is_root else "sub"
        aliases = ",".join(axis.alias for axis in self.axes)
        return f"<{type(self).__name__} type={parent_flag} ndim={self.ndim} shape={self.shape_bins} axes={aliases} nsubs={self.nsubs} ncache={self.total_cached_arr}>"


class SubBinNDResult(BinNDResult):
    def get_subresult(self, subset: Any) -> SubBinNDResult:
        return self.root.get_subresult(subset)


def _has_family(name: str) -> Callable[[Any], bool]:
    def condition(bins: Any) -> bool:
        try:
            return any(getattr(family, "name", str(family)) == name for family in bins.families())
        except Exception:
            return False

    return condition


@BinNDResult.derived("count", scope="particles")
def _count(bins: BinNDResult) -> np.ndarray:
    return np.asarray([len(index) for index in bins.bin_indices], dtype=int)


@BinNDResult.derived("density", condition=lambda bins: BIN_DERIVED_PROPERTIES["volume"].is_available(bins))
def _density(bins: BinNDResult) -> np.ndarray:
    return np.asarray(bins._resolve_query("mass.sum") / bins._resolve_query("volume"))


@BinNDResult.derived("surface_density", condition=lambda bins: BIN_DERIVED_PROPERTIES["area"].is_available(bins))
def _surface_density(bins: BinNDResult) -> np.ndarray:
    return np.asarray(bins._resolve_query("mass.sum") / bins._resolve_query("area"))


@BinNDResult.derived("enclosed_mass")
def _enclosed_mass(bins: BinNDResult) -> np.ndarray:
    return np.cumsum(np.asarray(bins._resolve_query("mass.sum")))


@BinNDResult.derived("gas_fraction", condition=_has_family("gas"))
def gas_fraction(bins: BinNDResult) -> np.ndarray:
    gas_mass_sum = bins.gas["mass.sum"]
    total_mass_sum = bins._resolve_query("mass.sum")
    numerator = np.nan_to_num(np.asarray(gas_mass_sum), nan=0.0)
    denominator = np.asarray(total_mass_sum)
    values = np.full(bins.nbins, np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = numerator[valid] / denominator[valid]
    return values
