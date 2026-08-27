"""Query resolution, typed caching, statistics, density, and apply.

This module decomposes the former ``query_engine`` into focused collaborating
pieces: a typed :class:`QueryCache`, a :class:`StatPipeline` (vectorised
statistics with a per-bin fallback), a :class:`DensityResolver`, an
:class:`ApplyComposer`, and an orchestrating :class:`BinQueryService`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.units import NoUnit

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.runtime.options import RunOptions

from .arrays import BinsArray
from .extensions import BIN_RESULT_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Callable

    from .axes import BinDerivedSpec


# Minimal RunOptions for per-bin apply loops: no cache, no progress, no perf,
# no observer.  Created once and reused across all apply() calls.
_BATCH_RUN_OPTIONS: RunOptions = RunOptions(
    cache=False, progress=False, perf_time=False, perf_memory=False, observe=False
)


CacheKey = Any  # PipelineStatKey | DerivedKey | ApplyKey


@dataclass(frozen=True)
class PipelineStatKey:
    field: str
    transforms: tuple[str, ...]
    stat_key: str
    weight_token: Any


@dataclass(frozen=True)
class DerivedKey:
    scope: str
    name: str


@dataclass(frozen=True)
class ApplyKey:
    query_token: Any
    name: str | None
    empty: float
    vectorized: bool


class QueryCache:
    """Typed-key cache for resolved queries."""

    def __init__(self) -> None:
        self._cache: dict[CacheKey, BinsArray] = {}

    def get(self, key: CacheKey) -> BinsArray | None:
        return self._cache.get(key)

    def put(self, key: CacheKey, arr: BinsArray) -> None:
        self._cache[key] = arr

    @property
    def num_cached(self) -> int:
        return len(self._cache)

    def keys(self) -> list[CacheKey]:
        return list(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    def invalidate_measure_dependent(self) -> tuple[int, list[str]]:
        """Clear cached entries that depend on the axis measure.

        These are density-derived entries (canonical names ending in
        ``.density``) and the ``geometry.measure`` entry.  Returns
        ``(count, cleared_names)``.
        """
        to_del: list[CacheKey] = []
        names: list[str] = []
        for k in self._cache:
            if isinstance(k, DerivedKey):
                if (k.scope == "derived" and k.name.endswith(".density")) or (
                    k.scope == "geometry" and k.name == "measure"
                ):
                    to_del.append(k)
                    names.append(k.name)
        for k in to_del:
            del self._cache[k]
        return len(to_del), names


def _cache_key_for_stat(field: str, transforms: tuple[str, ...], stat_key: str, weight_token: Any) -> PipelineStatKey:
    return PipelineStatKey(field, transforms, stat_key, weight_token)


def _wrap(provider: Any, values: Any, *, name: str, field: str | None = None, mode: str | None = None) -> BinsArray:
    return BinsArray(provider, values, name=name, field=field, mode=mode)


def weight_cache_token(weight: str | Callable[[Any], Any] | Any | None) -> Any:
    if weight is None or isinstance(weight, str):
        return weight
    if callable(weight):
        return ("callable", id(weight))
    return ("array", id(weight))


def callable_cache_token(query: Any) -> Any:
    if isinstance(query, CalculatorBase):
        return query.signature()
    signature = getattr(query, "signature", None)
    if callable(signature):
        return signature()
    return id(query)


class StatPipeline:
    """Compute a per-bin statistic (vectorised fast path, then per-bin fallback)."""

    def __init__(self, provider: Any, cache: QueryCache) -> None:
        self._provider = provider
        self._cache = cache

    def compute(
        self,
        field: str,
        transforms: list[str],
        terminal_stat: Any,
        *,
        weight: str | Callable[[Any], Any] | Any | None = None,
        query_key: str | None = None,
    ) -> BinsArray:
        from .statistics import apply_pipeline

        provider = self._provider
        weight_token = weight_cache_token(weight)
        stat_key_str = terminal_stat.key
        key = _cache_key_for_stat(field, tuple(transforms), stat_key_str, weight_token)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        values = provider.sim[field]
        weights = None
        if isinstance(weight, str):
            weights = provider.sim[weight]
        elif callable(weight):
            weights = weight(provider.sim)
        elif weight is not None:
            weights = weight

        valid_mask = provider._valid_mask
        if valid_mask.any():
            v_valid = np.asarray(values[valid_mask], dtype=float)
            v_valid = apply_pipeline(v_valid, transforms)
            w_valid = None if weights is None else np.asarray(weights[valid_mask], dtype=float)
            bins_valid = provider._particle_bin[valid_mask]
            vec = terminal_stat.vectorized_call(v_valid, bins_valid, w_valid, provider.nbins)
        else:
            vec = None

        if vec is not None:
            out = vec
        else:
            out = np.full(provider.nbins, np.nan, dtype=float)
            for index in range(provider.nbins):
                start = int(provider._bin_indptr[index])
                stop = int(provider._bin_indptr[index + 1])
                if start == stop:
                    continue
                particle_indices = provider._bin_data[start:stop]
                sub = np.asarray(values[particle_indices], dtype=float)
                sub = apply_pipeline(sub, transforms)
                sub_weights = None if weights is None else np.asarray(weights[particle_indices], dtype=float)
                out[index] = terminal_stat(sub, sub_weights)

        name = query_key or (
            f"{field}.{'.'.join(transforms)}.{stat_key_str}" if transforms else f"{field}.{stat_key_str}"
        )
        result = _wrap(provider, out, name=name, field=field, mode=stat_key_str)
        if isinstance(values, (SimArray, IndexedSimArray)):
            result.units = values.units
            result.sim = values.sim
        self._cache.put(key, result)
        return result


class DensityResolver:
    """Resolve ``<field>.density`` queries to the canonical ``<field>.sum.density``."""

    _SUFFIX = ".density"

    def __init__(self, provider: Any, cache: QueryCache, resolve_query: Callable[[str], BinsArray]) -> None:
        self._provider = provider
        self._cache = cache
        self._resolve_query = resolve_query

    @staticmethod
    def strip(key: str) -> str | None:
        if key.endswith(DensityResolver._SUFFIX) and len(key) > len(DensityResolver._SUFFIX):
            base = key[: -len(DensityResolver._SUFFIX)]
            if base and not base.endswith("."):
                return base
        return None

    @staticmethod
    def canonical(base_field: str) -> str:
        if base_field.endswith(".sum"):
            return f"{base_field}.density"
        return f"{base_field}.sum.density"

    def resolve(self, key: str, base_field: str) -> BinsArray:
        provider = self._provider
        if base_field == "density":
            base_field = "mass"

        canonical_key = self.canonical(base_field)
        cache_key = DerivedKey("derived", canonical_key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if base_field == "count":
            numerator = self._resolve_query("count")
        elif base_field.endswith(".sum"):
            numerator = self._resolve_query(base_field)
        else:
            numerator = self._resolve_query(f"{base_field}.sum")

        denominator = self._resolve_query("measure")
        if isinstance(getattr(denominator, "units", None), NoUnit):
            denominator = np.asarray(denominator)

        with np.errstate(divide="ignore", invalid="ignore"):
            result_arr = numerator / denominator

        result = _wrap(provider, result_arr, name=canonical_key, field=base_field, mode="density")
        self._cache.put(cache_key, result)
        return result


class ApplyComposer:
    """Evaluate a query (callable or calculator) per bin into a :class:`BinsArray`."""

    def __init__(self, provider: Any, cache: QueryCache, diagnostics: Any) -> None:
        self._provider = provider
        self._cache = cache
        self._diagnostics = diagnostics

    def apply(
        self,
        query: Callable[[Any], Any] | CalculatorBase[Any, Any],
        *,
        name: str | None = None,
        empty: float = np.nan,
        vectorized: bool = False,
    ) -> BinsArray:
        provider = self._provider
        cache_key = ApplyKey(callable_cache_token(query), name, empty, vectorized)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if vectorized:
            if isinstance(query, CalculatorBase):
                raise TypeError("vectorized=True requires a plain callable, not a CalculatorBase.")
            raw = query(provider.sim, provider._particle_bin)  # type: ignore[call-arg]
            values = np.asarray(raw, dtype=float)
            if values.shape != (provider.nbins,):
                raise TypeError(f"Vectorized apply callable must return shape ({provider.nbins},), got {values.shape}.")
        else:
            values = self._value_nonvectorized(query)

        result_name = str(name or getattr(query, "name", None) or getattr(query, "__name__", "apply"))
        result = _wrap(provider, values, name=result_name)
        self._cache.put(cache_key, result)
        self._diagnostics.record("apply", name=name, query=repr(query), vectorized=vectorized)
        return result

    def _value_nonvectorized(self, query: Callable[[Any], Any] | CalculatorBase[Any, Any]) -> np.ndarray:
        provider = self._provider
        outputs: list[np.ndarray | None] = [None] * provider.nbins
        sample_shape: tuple[int, ...] | None = None
        if isinstance(query, CalculatorBase):
            with query.batch(_BATCH_RUN_OPTIONS) as run_one:
                for index in range(provider.nbins):
                    start = int(provider._bin_indptr[index])
                    stop = int(provider._bin_indptr[index + 1])
                    if start == stop:
                        continue
                    sub = provider.sim[provider._bin_data[start:stop]]
                    arr = np.asarray(run_one(sub))
                    if sample_shape is None:
                        sample_shape = arr.shape
                    elif arr.shape != sample_shape:
                        raise TypeError(
                            f"Inconsistent output shape from query: expected {sample_shape}, got {arr.shape}."
                        )
                    outputs[index] = arr
        else:
            for index in range(provider.nbins):
                start = int(provider._bin_indptr[index])
                stop = int(provider._bin_indptr[index + 1])
                if start == stop:
                    continue
                sub = provider.sim[provider._bin_data[start:stop]]
                arr = np.asarray(query(sub))
                if sample_shape is None:
                    sample_shape = arr.shape
                elif arr.shape != sample_shape:
                    raise TypeError(f"Inconsistent output shape from query: expected {sample_shape}, got {arr.shape}.")
                outputs[index] = arr
        if sample_shape is None:
            sample_shape = (0,)
        result_values = np.array([out if out is not None else np.full(sample_shape, np.nan) for out in outputs])
        return result_values


class BinQueryService:
    """Orchestrates query resolution over a provider (result or model)."""

    def __init__(self, provider: Any, diagnostics: Any, *, extensions: Any = None) -> None:
        self._provider = provider
        self._diagnostics = diagnostics
        self._extensions = extensions if extensions is not None else BIN_RESULT_EXTENSIONS
        self._cache = QueryCache()
        self._stat = StatPipeline(provider, self._cache)
        self._density = DensityResolver(provider, self._cache, self.resolve)
        self._apply = ApplyComposer(provider, self._cache, diagnostics)

    @property
    def num_cached_arr(self) -> int:
        return self._cache.num_cached

    def keys(self) -> list[str]:
        keys: set[str] = set()
        keys.update(self.property_keys())
        for cache_key in self._cache.keys():
            if isinstance(cache_key, PipelineStatKey):
                base = (
                    f"{cache_key.field}.{'.'.join(cache_key.transforms)}.{cache_key.stat_key}"
                    if cache_key.transforms
                    else f"{cache_key.field}.{cache_key.stat_key}"
                )
                keys.add(f"{base}@{cache_key.weight_token}" if isinstance(cache_key.weight_token, str) else base)
            elif isinstance(cache_key, DerivedKey):
                keys.add(cache_key.name)
        return sorted(keys)

    def property_keys(self) -> list[str]:
        return self._extensions.property_keys(self._provider)

    def resolve(self, key: str) -> BinsArray:
        provider = self._provider
        scope = self._extensions.query_scope(provider, key)
        if not provider.is_root and scope in provider._SHARED_SCOPES:
            return provider.root._query_engine.resolve(key)

        parsed = self._extensions.parse_pipeline_key(key)
        if parsed is not None:
            field, transforms, terminal_stat, weight_field = parsed
            result = self.stat_pipeline(field, transforms, terminal_stat, weight=weight_field, query_key=key)
            self._diagnostics.record("query", key=key, scope=scope)
            return result

        density_base = self._density.strip(key)
        if density_base is not None or key == "density":
            return self._density.resolve(key, density_base if density_base is not None else key)

        cache_key = DerivedKey(scope, key)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        arr = self.compute_query(key, scope=scope)
        self._cache.put(cache_key, arr)
        self._diagnostics.record("query", key=key, scope=scope)
        return arr

    def wrap(self, values: Any, *, name: str, field: str | None = None, mode: str | None = None) -> BinsArray:
        return _wrap(self._provider, values, name=name, field=field, mode=mode)

    def compute_query(self, key: str, *, scope: str) -> BinsArray:
        spec = self._extensions.get_derived_spec(self._provider, key)
        if spec is not None:
            return self.compute_derived(spec)
        raise KeyError(
            f"Unknown BinND query {key!r}. "
            "Use 'field.stat' syntax (e.g. 'mass.sum') for particle statistics. "
            "For axis properties use bins.axis('r').center."
        )

    def compute_derived(self, spec: BinDerivedSpec) -> BinsArray:
        provider = self._provider
        values = spec.func(provider)
        result = values if isinstance(values, BinsArray) else self.wrap(values, name=spec.name)
        if result.shape[: provider.ndim] != provider.shape_bins:
            raise ValueError(
                f"Derived query {spec.name!r} returned shape {result.shape!r}; "
                f"leading dimensions must match the bin grid {provider.shape_bins!r}."
            )
        return result

    def stat_pipeline(
        self,
        field: str,
        transforms: list[str],
        terminal_stat: Any,
        *,
        weight: str | Callable[[Any], Any] | Any | None = None,
        query_key: str | None = None,
    ) -> BinsArray:
        return self._stat.compute(field, transforms, terminal_stat, weight=weight, query_key=query_key)

    def stat_explicit(
        self,
        field: str,
        statistic: str,
        weight: str | Callable[[Any], Any] | Any | None = None,
        transforms: list[str] | None = None,
    ) -> BinsArray:
        stat_obj = self._extensions.get_statistic(statistic)
        if stat_obj is None:
            raise KeyError(f"Unknown statistic {statistic!r}.")
        return self.stat_pipeline(field, transforms or [], stat_obj, weight=weight)

    def apply(
        self,
        query: Callable[[Any], Any] | CalculatorBase[Any, Any],
        *,
        name: str | None = None,
        empty: float = np.nan,
        vectorized: bool = False,
    ) -> BinsArray:
        return self._apply.apply(query, name=name, empty=empty, vectorized=vectorized)

    @staticmethod
    def weight_cache_token(weight: str | Callable[[Any], Any] | Any | None) -> Any:
        return weight_cache_token(weight)

    @staticmethod
    def callable_cache_token(query: Any) -> Any:
        return callable_cache_token(query)

    def cache_report(self) -> dict[str, Any]:
        provider = self._provider
        return {
            "queries": self._cache.num_cached,
            "subresults": provider.nsubs,
            "total_queries": provider.total_cached_arr,
        }

    def query_report(self) -> list[dict[str, Any]]:
        return list(self._diagnostics.diagnostics)

    def invalidate_measure_dependent_cache(self) -> tuple[int, list[str]]:
        return self._cache.invalidate_measure_dependent()
