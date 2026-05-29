from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import IndexedSimArray, SimArray

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.runtime.options import RunOptions

from .arrays import BinsArray

if TYPE_CHECKING:
    from collections.abc import Callable

    from .axes import BinDerivedSpec
    from .result import BinNDResult

# Minimal RunOptions for per-bin apply loops: no cache, no progress, no perf,
# no observer.  Created once and reused across all apply() calls.
_BATCH_RUN_OPTIONS: RunOptions = RunOptions(
    cache=False,
    progress=False,
    perf_time=False,
    perf_memory=False,
    observe=False,
)


class BinQueryEngine:
    def __init__(self, owner: BinNDResult, diagnostics: Any) -> None:
        self._owner = owner
        self._diagnostics = diagnostics
        self._cache: dict[Any, BinsArray] = {}

    @property
    def num_cached_arr(self) -> int:
        return len(self._cache)

    def keys(self) -> list[str]:
        keys: set[str] = set()
        keys.update(self.property_keys())
        for cache_key in self._cache:
            if not isinstance(cache_key, tuple):
                continue
            # (scope, "query_string") — geometry / derived results
            if len(cache_key) == 2 and isinstance(cache_key[1], str):
                keys.add(cache_key[1])
            # ("stat", field, transforms, stat_key, weight) — pipeline stat results
            elif len(cache_key) >= 5 and cache_key[0] == "stat":
                _, field, transforms, stat_key, weight_key = cache_key[:5]
                base = f"{field}.{'.'.join(transforms)}.{stat_key}" if transforms else f"{field}.{stat_key}"
                keys.add(f"{base}@{weight_key}" if isinstance(weight_key, str) else base)
        return sorted(keys)

    def property_keys(self) -> list[str]:
        return type(self._owner)._extensions.property_keys(self._owner)

    def resolve(self, key: str) -> BinsArray:
        owner = self._owner
        scope = type(owner)._extensions.query_scope(owner, key)
        if not owner.is_root and scope in owner._SHARED_SCOPES:
            return owner.root._query_engine.resolve(key)

        # Pipeline stat queries ("mass.sum", "vz.abs.mean", …) are delegated
        # directly to _stat_pipeline, which owns the authoritative cache entry
        # keyed on (stat, field, transforms, stat_name, weight).  This ensures
        # bins["mass.sum"] and stat_explicit("mass", "sum") always share one
        # cache entry regardless of call order.
        parsed = type(owner)._extensions.parse_pipeline_key(key)
        if parsed is not None:
            field, transforms, terminal_stat, weight_field = parsed
            result = self.stat_pipeline(
                field,
                transforms,
                terminal_stat,
                weight=weight_field,
                query_key=key,
            )
            self._diagnostics.record("query", key=key, scope=scope)
            return result

        cache_key = (scope, key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        arr = self.compute_query(key, scope=scope)
        self._cache[cache_key] = arr
        self._diagnostics.record("query", key=key, scope=scope)
        return arr

    def wrap(
        self,
        values: Any,
        *,
        name: str,
        field: str | None = None,
        mode: str | None = None,
    ) -> BinsArray:
        return BinsArray(self._owner, values, name=name, field=field, mode=mode)

    def compute_query(self, key: str, *, scope: str) -> BinsArray:
        spec = type(self._owner)._extensions.get_derived_spec(self._owner, key)
        if spec is not None:
            return self.compute_derived(spec)
        raise KeyError(
            f"Unknown BinND query {key!r}. "
            "Use 'field.stat' syntax (e.g. 'mass.sum') for particle statistics. "
            "For axis properties use bins.axis('r').center."
        )

    def compute_derived(self, spec: BinDerivedSpec) -> BinsArray:
        owner = self._owner
        values = spec.func(owner)
        result = values if isinstance(values, BinsArray) else self.wrap(values, name=spec.name)
        if result.shape[: owner.ndim] != owner.shape_bins:
            raise ValueError(
                f"Derived query {spec.name!r} returned shape {result.shape!r}; "
                f"leading dimensions must match the bin grid {owner.shape_bins!r}."
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
        """Compute per-bin statistic with optional pipeline transforms.

        E.g. field="vz", transforms=["abs"], terminal_stat=Mean() computes
        the per-bin mean of |vz|.
        """
        owner = self._owner
        weight_key = self.weight_cache_token(weight)
        stat_key_str = terminal_stat.key
        cache_key = ("stat", field, tuple(transforms), stat_key_str, weight_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        from .statistics import apply_pipeline

        values = owner.sim[field]
        weights = None
        if isinstance(weight, str):
            weights = owner.sim[weight]
        elif callable(weight):
            weights = weight(owner.sim)
        elif weight is not None:
            weights = weight

        # --- fast vectorised path ---
        valid_mask = owner._valid_mask
        if valid_mask.any():
            v_valid = np.asarray(values[valid_mask], dtype=float)
            v_valid = apply_pipeline(v_valid, transforms)
            w_valid = None if weights is None else np.asarray(weights[valid_mask], dtype=float)
            bins_valid = owner._particle_bin[valid_mask]
            vec = terminal_stat.vectorized_call(v_valid, bins_valid, w_valid, owner.nbins)
        else:
            vec = None

        if vec is not None:
            out = vec
        else:
            # --- fallback to per-bin loop ---
            out = np.full(owner.nbins, np.nan, dtype=float)
            for index in range(owner.nbins):
                start = int(owner._bin_indptr[index])
                stop = int(owner._bin_indptr[index + 1])
                if start == stop:
                    continue
                particle_indices = owner._bin_data[start:stop]
                sub = np.asarray(values[particle_indices], dtype=float)
                sub = apply_pipeline(sub, transforms)
                sub_weights = None if weights is None else np.asarray(weights[particle_indices], dtype=float)
                out[index] = terminal_stat(sub, sub_weights)

        name = query_key or (
            f"{field}.{'.'.join(transforms)}.{stat_key_str}" if transforms else f"{field}.{stat_key_str}"
        )
        result = self.wrap(out, name=name, field=field, mode=stat_key_str)
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

            .stat_explicit("mass", "sum")
            .stat_explicit("vz", "mean", transforms=["abs"])
            .stat_explicit("mass", "mean", weight="mass")
        """
        stat_obj = type(self._owner)._extensions.get_statistic(statistic)
        if stat_obj is None:
            raise KeyError(f"Unknown statistic {statistic!r}.")
        return self.stat_pipeline(field, transforms or [], stat_obj, weight=weight)

    @staticmethod
    def weight_cache_token(weight: str | Callable[[Any], Any] | Any | None) -> Any:
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
        owner = self._owner
        cache_key = ("apply", self.callable_cache_token(query), name, empty, vectorized)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if vectorized:
            # Fast path: user supplies a vectorized callable (sim, particle_bin) -> array
            # CalculatorBase is not valid here – must be a plain callable
            if isinstance(query, CalculatorBase):
                raise TypeError("vectorized=True requires a plain callable, not a CalculatorBase.")
            raw = query(owner.sim, owner._particle_bin)  # type: ignore[call-arg]
            values = np.asarray(raw, dtype=float)
            if values.shape != (owner.nbins,):
                raise TypeError(
                    f"Vectorized apply callable must return shape ({owner.nbins},), got {values.shape}."
                )
        else:
            values = np.full(owner.nbins, empty, dtype=float)
            if isinstance(query, CalculatorBase):
                # Fast path 2: full CalculatorBase → reuse one EvalEngine, skip Result assembly
                with query.batch(_BATCH_RUN_OPTIONS) as run_one:
                    for index in range(owner.nbins):
                        start = int(owner._bin_indptr[index])
                        stop = int(owner._bin_indptr[index + 1])
                        if start == stop:
                            continue
                        sub = owner.sim[owner._bin_data[start:stop]]
                        arr = np.asarray(run_one(sub))
                        if arr.ndim != 0:
                            raise TypeError("Callable or CalculatorBase bin query must return a scalar in phase 1.")
                        values[index] = arr.item()
            else:
                # Plain callable
                for index in range(owner.nbins):
                    start = int(owner._bin_indptr[index])
                    stop = int(owner._bin_indptr[index + 1])
                    if start == stop:
                        continue
                    sub = owner.sim[owner._bin_data[start:stop]]
                    arr = np.asarray(query(sub))
                    if arr.ndim != 0:
                        raise TypeError("Callable or CalculatorBase bin query must return a scalar in phase 1.")
                    values[index] = arr.item()

        result_name = str(name or getattr(query, "name", None) or getattr(query, "__name__", "apply"))
        result = self.wrap(values, name=result_name)
        self._cache[cache_key] = result
        self._diagnostics.record("apply", name=name, query=repr(query), vectorized=vectorized)
        return result

    @staticmethod
    def callable_cache_token(query: Any) -> Any:
        if isinstance(query, CalculatorBase):
            return query.signature()
        signature = getattr(query, "signature", None)
        if callable(signature):
            return signature()
        return id(query)

    def cache_report(self) -> dict[str, Any]:
        owner = self._owner
        return {
            "queries": len(self._cache),
            "subresults": owner.nsubs,
            "total_queries": owner.total_cached_arr,
        }

    def query_report(self) -> list[dict[str, Any]]:
        return list(self._diagnostics.diagnostics)
