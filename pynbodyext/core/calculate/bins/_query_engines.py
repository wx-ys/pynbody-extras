"""The three ways a binned query is answered, and the helpers they share.

``StatPipeline`` reduces particle arrays to per-bin values (a vectorised statistic
first, a per-bin loop when the statistic cannot be vectorised); ``DensityResolver``
answers the ``.density`` suffix as a numerator over the cell measure;
``ApplyComposer`` runs a callable per bin, or once over the bin index.  Each owns
its cache reads and writes through the shared :class:`~._query_cache.QueryCache`.

They are separate from ``query.py`` (the service) so that "how is a statistic
computed" does not require reading the dispatcher, and so the service stays a short
list of which engine answers which spelling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.units import NoUnit

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.runtime.options import RunOptions

from ._query_cache import ApplyKey, DerivedKey, QueryCache, _cache_key_for_stat
from .arrays import BinsArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from .model import BinResultModel
    from .registries import BinDerivedSpec
    from .result import BinNDResult


# Minimal RunOptions for per-bin apply loops: no cache, no progress, no perf,
# no observer.  Created once and reused across all apply() calls.
_BATCH_RUN_OPTIONS: RunOptions = RunOptions(
    cache=False, progress=False, perf_time=False, perf_memory=False, observe=False
)

def _wrap(
    owner: BinNDResult, values: Any, *, name: str, field: str | None = None, mode: str | None = None
) -> BinsArray:
    return BinsArray(owner, values, name=name, field=field, mode=mode)

def evaluate_derived(owner: BinNDResult, spec: BinDerivedSpec, model: Any) -> BinsArray:
    """Run a derived callback against *model* and wrap its answer for *owner*.

    Derived callbacks operate on a :class:`~.model.BinResultModel`; they reach the
    result-level read API (``bins[...]``, ``.gas``, ``.centers``) through it.  The
    sph view calls this too, with a model whose queries come back smoothed, so the
    same callback answers both the strict and the smoothed version of itself.
    """
    values = spec.func(model)
    result = values if isinstance(values, BinsArray) else _wrap(owner, values, name=spec.name)
    if result.shape[: model.ndim] != model.shape_bins:
        raise ValueError(
            f"Derived query {spec.name!r} returned shape {result.shape!r}; "
            f"leading dimensions must match the bin grid {model.shape_bins!r}."
        )
    return result

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

    def __init__(self, model: BinResultModel, owner: BinNDResult, cache: QueryCache) -> None:
        self._model = model
        self._owner = owner
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
        stat_key = terminal_stat.key
        key = _cache_key_for_stat(field, tuple(transforms), stat_key, weight_cache_token(weight))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        values, weights = self._particles(field, weight)
        out = self._reduce(terminal_stat, values, weights, transforms)
        result = _wrap(self._owner, out, name=self._name(field, transforms, stat_key, query_key), field=field, mode=stat_key)
        self._carry_units(result, values)
        self._cache.put(key, result)
        return result

    def _particles(
        self, field: str, weight: str | Callable[[Any], Any] | Any | None
    ) -> tuple[Any, np.ndarray | None]:
        """The sampled array and the weight array (if any), in the snapshot's units."""
        model = self._model
        values = model.sim[field]
        if isinstance(weight, str):
            weights: np.ndarray | None = model.sim[weight]
        elif callable(weight):
            weights = weight(model.sim)
        else:
            weights = weight
        return values, weights

    def _reduce(
        self,
        terminal_stat: Any,
        values: Any,
        weights: np.ndarray | None,
        transforms: list[str],
    ) -> np.ndarray:
        """One value per bin: the vectorised call when the statistic has one, else a loop."""
        from .statistics import apply_pipeline

        model = self._model
        assert model.valid_mask is not None
        assert model.particle_bin is not None
        assert model.bin_indptr is not None
        assert model.bin_data is not None
        if model.valid_mask.any():
            valid_values = apply_pipeline(np.asarray(values[model.valid_mask], dtype=float), transforms)
            valid_weights = None if weights is None else np.asarray(weights[model.valid_mask], dtype=float)
            vectorised = terminal_stat.vectorized_call(
                valid_values, model.particle_bin[model.valid_mask], valid_weights, model.nbins
            )
            if vectorised is not None:
                return vectorised

        out = np.full(model.nbins, np.nan, dtype=float)
        for index in range(model.nbins):
            start = int(model.bin_indptr[index])
            stop = int(model.bin_indptr[index + 1])
            if start == stop:
                continue
            particle_indices = model.bin_data[start:stop]
            sub = apply_pipeline(np.asarray(values[particle_indices], dtype=float), transforms)
            sub_weights = None if weights is None else np.asarray(weights[particle_indices], dtype=float)
            out[index] = terminal_stat(sub, sub_weights)
        return out

    @staticmethod
    def _name(field: str, transforms: list[str], stat_key: str, query_key: str | None) -> str:
        """The query spelling a caller asked for, or the one this statistic means."""
        if query_key:
            return query_key
        return f"{field}.{'.'.join(transforms)}.{stat_key}" if transforms else f"{field}.{stat_key}"

    @staticmethod
    def _carry_units(result: BinsArray, values: Any) -> None:
        """Give the answer the sampled array's units and snapshot, when it has any."""
        if isinstance(values, (SimArray, IndexedSimArray)):
            result.units = values.units
            result.sim = values.sim

class DensityResolver:
    """Resolve ``<field>.density`` queries to the canonical ``<field>.sum.density``."""

    _SUFFIX = ".density"

    def __init__(self, owner: BinNDResult, cache: QueryCache, resolve_query: Callable[[str], BinsArray]) -> None:
        self._owner = owner
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

        measure = self._resolve_query("measure")
        # A unitless measure divides as a plain array, so the result has no NoUnit.
        denominator = np.asarray(measure) if isinstance(getattr(measure, "units", None), NoUnit) else measure

        with np.errstate(divide="ignore", invalid="ignore"):
            result_arr = numerator / denominator

        result = _wrap(self._owner, result_arr, name=canonical_key, field=base_field, mode="density")
        self._cache.put(cache_key, result)
        return result

class ApplyComposer:
    """Evaluate a query (callable or calculator) per bin into a :class:`BinsArray`."""

    def __init__(self, model: BinResultModel, owner: BinNDResult, cache: QueryCache, diagnostics: Any) -> None:
        self._model = model
        self._owner = owner
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
        model = self._model
        cache_key = ApplyKey(callable_cache_token(query), name, empty, vectorized)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if vectorized:
            assert model.particle_bin is not None
            if isinstance(query, CalculatorBase):
                raise TypeError("vectorized=True requires a plain callable, not a CalculatorBase.")
            raw = query(model.sim, model.particle_bin)  # type: ignore[call-arg]
            values = np.asarray(raw, dtype=float)
            if values.shape != (model.nbins,):
                raise TypeError(f"Vectorized apply callable must return shape ({model.nbins},), got {values.shape}.")
        else:
            values = self._value_nonvectorized(query)

        result_name = str(name or getattr(query, "name", None) or getattr(query, "__name__", "apply"))
        result = _wrap(self._owner, values, name=result_name)
        self._cache.put(cache_key, result)
        self._diagnostics.record("apply", name=name, query=repr(query), vectorized=vectorized)
        return result

    def _value_nonvectorized(self, query: Callable[[Any], Any] | CalculatorBase[Any, Any]) -> np.ndarray:
        if isinstance(query, CalculatorBase):
            with query.batch(_BATCH_RUN_OPTIONS) as run_one:
                return self._per_bin(run_one)
        return self._per_bin(query)

    def _per_bin(self, run: Callable[[Any], Any]) -> np.ndarray:
        """Call *run* once per non-empty bin; every output must have the same shape."""
        model = self._model
        assert model.bin_indptr is not None
        assert model.bin_data is not None
        outputs: list[np.ndarray | None] = [None] * model.nbins
        sample_shape: tuple[int, ...] | None = None
        for index in range(model.nbins):
            start = int(model.bin_indptr[index])
            stop = int(model.bin_indptr[index + 1])
            if start == stop:
                continue
            arr = np.asarray(run(model.sim[model.bin_data[start:stop]]))
            if sample_shape is None:
                sample_shape = arr.shape
            elif arr.shape != sample_shape:
                raise TypeError(f"Inconsistent output shape from query: expected {sample_shape}, got {arr.shape}.")
            outputs[index] = arr
        if sample_shape is None:
            sample_shape = (0,)
        return np.array([out if out is not None else np.full(sample_shape, np.nan) for out in outputs])
