from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.units import NoUnit

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
    cache=False, progress=False, perf_time=False, perf_memory=False, observe=False
)

# Density suffix pattern — only .density (dot notation)
_DENSITY_SUFFIX: str = ".density"


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
            result = self.stat_pipeline(field, transforms, terminal_stat, weight=weight_field, query_key=key)
            self._diagnostics.record("query", key=key, scope=scope)
            return result

        # Intercept density-suffix queries ("mass.density", "mass.sum.density", …)
        # Bare "density" is treated as "mass.density".
        # Only .density (dot notation) is recognised — no underscore aliases.
        density_base = self._strip_density_suffix(key)
        if density_base is not None or key == "density":
            return self._resolve_density(key, density_base if density_base is not None else key)

        cache_key = (scope, key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        arr = self.compute_query(key, scope=scope)
        self._cache[cache_key] = arr
        self._diagnostics.record("query", key=key, scope=scope)
        return arr

    def wrap(self, values: Any, *, name: str, field: str | None = None, mode: str | None = None) -> BinsArray:
        return BinsArray(self._owner, values, name=name, field=field, mode=mode)

    # ------------------------------------------------------------------
    # Density-suffix resolution ("mass.density" → canonical "mass.sum.density")
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_density_key(base_field: str) -> str:
        """Normalize *base_field* to the canonical density key ``{field}.sum.density``.

        ``"mass"`` → ``"mass.sum.density"``
        ``"mass.sum"`` → ``"mass.sum.density"``
        ``"vz"`` → ``"vz.sum.density"``
        """
        if base_field.endswith(".sum"):
            return f"{base_field}.density"
        return f"{base_field}.sum.density"

    @staticmethod
    def _strip_density_suffix(key: str) -> str | None:
        """If *key* ends with ``.density``, return the base field (stripped suffix).

        Returns ``None`` if the suffix does not match.
        Only the dot-notation ``.density`` is recognised (no underscore alias).
        """
        if key.endswith(_DENSITY_SUFFIX) and len(key) > len(_DENSITY_SUFFIX):
            base = key[: -len(_DENSITY_SUFFIX)]
            if base and not base.endswith("."):
                return base
        return None

    def _resolve_density(self, key: str, base_field: str) -> BinsArray:
        """Resolve a ``"<field>.density"`` query via the canonical form.

        All density queries normalise to ``{field}.sum.density`` so that
        ``bins["density"]``, ``bins["mass.density"]``, and
        ``bins["mass.sum.density"]`` share a single cache entry.
        """
        owner = self._owner

        # Handle bare "density" — default field is "mass"
        if base_field == "density":
            base_field = "mass"

        canonical_key = self._canonical_density_key(base_field)
        cache_key = ("derived", canonical_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Determine the numerator: for "count" use the derived property
        # directly (count is already per-bin); for any other field compute
        # "<field>.sum".
        if base_field == "count":
            numerator = owner._resolve_query("count")
        elif base_field.endswith(".sum"):
            # Already has a stat suffix — don't double-wrap
            numerator = owner._resolve_query(base_field)
        else:
            numerator = owner._resolve_query(f"{base_field}.sum")

        denominator = owner._resolve_query("measure")

        # A dimensionless (NoUnit) measure — e.g. a bin axis with no physical
        # units — must not strip the numerator's units.  pynbody's
        # ``SimArray / NoUnit`` degrades the quotient to ``NoUnit``, so drop the
        # denominator's wrapper and divide by a bare array to retain the mass
        # units (``mass / dimensionless_measure`` keeps ``Msol``).
        if isinstance(getattr(denominator, "units", None), NoUnit):
            denominator = np.asarray(denominator)

        # Divide BinsArray objects directly to preserve units
        with np.errstate(divide="ignore", invalid="ignore"):
            result_arr = numerator / denominator

        result = self.wrap(result_arr, name=canonical_key, field=base_field, mode="density")
        self._cache[cache_key] = result
        self._diagnostics.record("query", key=key, scope="derived")
        return result

    def invalidate_measure_dependent_cache(self) -> tuple[int, list[str]]:
        """Clear all cached entries that depend on the axis measure.

        This includes density-related entries (all whose canonical name
        ends with ``.density``) and the ``"measure"`` geometry entry
        itself.  Returns ``(count, cleared_names)``.
        """
        keys_to_clear: list[tuple] = []
        cleared_names: list[str] = []
        for k in self._cache:
            if not isinstance(k, tuple) or len(k) != 2:
                continue
            scope, name = k[0], k[1]
            if not isinstance(name, str):
                continue
            # Density entries: ("derived", "*density")
            if scope == "derived" and name.endswith(_DENSITY_SUFFIX):
                keys_to_clear.append(k)
                cleared_names.append(name)
            # Measure entry: ("geometry", "measure") — the root geometry cache
            elif scope == "geometry" and name == "measure":
                keys_to_clear.append(k)
                cleared_names.append(name)
        for k in keys_to_clear:
            del self._cache[k]
        return len(keys_to_clear), cleared_names

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
                raise TypeError(f"Vectorized apply callable must return shape ({owner.nbins},), got {values.shape}.")
        else:
            values = self._value_nonvectorized(query)

        result_name = str(name or getattr(query, "name", None) or getattr(query, "__name__", "apply"))
        result = self.wrap(values, name=result_name)
        self._cache[cache_key] = result
        self._diagnostics.record("apply", name=name, query=repr(query), vectorized=vectorized)
        return result

    def _value_nonvectorized(self, query: Callable[[Any], Any] | CalculatorBase[Any, Any]) -> np.ndarray:
        owner = self._owner
        outputs: list[np.ndarray | None] = [None] * owner.nbins
        sample_shape: tuple[int, ...] | None = None
        if isinstance(query, CalculatorBase):
            # Fast path: full CalculatorBase → reuse one EvalEngine, skip Result assembly
            with query.batch(_BATCH_RUN_OPTIONS) as run_one:
                for index in range(owner.nbins):
                    start = int(owner._bin_indptr[index])
                    stop = int(owner._bin_indptr[index + 1])
                    if start == stop:
                        continue
                    sub = owner.sim[owner._bin_data[start:stop]]
                    arr = np.asarray(run_one(sub))
                    if sample_shape is None:
                        sample_shape = arr.shape
                    elif arr.shape != sample_shape:
                        raise TypeError(
                            f"Inconsistent output shape from query: expected {sample_shape}, got {arr.shape}."
                        )
                    outputs[index] = arr
        else:
            # Plain callable
            for index in range(owner.nbins):
                start = int(owner._bin_indptr[index])
                stop = int(owner._bin_indptr[index + 1])
                if start == stop:
                    continue
                sub = owner.sim[owner._bin_data[start:stop]]
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
        return {"queries": len(self._cache), "subresults": owner.nsubs, "total_queries": owner.total_cached_arr}

    def query_report(self) -> list[dict[str, Any]]:
        return list(self._diagnostics.diagnostics)
