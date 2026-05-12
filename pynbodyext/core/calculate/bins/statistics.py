"""
Per-bin statistic plug-in system for BinNDResult.

Design
------
Statistics are registered as :class:`BinStatisticBase` subclasses.  Each class
auto-registers itself via ``__init_subclass__``.  :meth:`BinStatisticBase.valid`
is the matching gate: given a raw key token it returns an instance of the
statistic if the token is recognised, or ``None`` otherwise.

Pipeline query syntax
---------------------
String queries in :class:`~.result.BinNDResult` use ``"."`` as separator:

    ``"mass.sum"``       → field=``"mass"``, pipeline=[``Sum``]
    ``"vz.abs.mean"``    → field=``"vz"``,   pipeline=[``Abs``, ``Mean``]
    ``"vz.p16"``         → field=``"vz"``,   pipeline=[``Percentile(16)``]

Parsing rules
-------------
Given a dot-separated query string, the parser splits at the *last* position
such that the suffix is a recognised statistic token.  The prefix is the field
name; the remaining intermediate tokens (if any) are treated as a *transform
pipeline* applied to the raw particle array before the final reduction.

Currently supported pipeline transforms: ``abs`` (takes |x| element-wise).

Empty bin → each statistic returns ``np.nan`` (or 0 for ``count``).
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

StatisticFunc = Callable[[Any, Any | None], float]

__all__ = [
    "BinStatisticBase",
    "BinNDStatAccessor",
    "parse_pipeline_key",
    "apply_pipeline",
    "get_statistic",
    "evaluate_statistic",
    "is_statistic_name",
    # built-in statistics
    "Mean",
    "Sum",
    "SumWeighted",
    "Percentile",
    "Median",
    "RMS",
    "Dispersion",
    "Abs",
]

_REGISTRY: list[type[BinStatisticBase]] = []


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BinStatisticBase:
    """Abstract base for per-bin statistics.

    Subclass and implement ``__call__``.  Override ``valid`` to control which
    string key(s) this statistic matches.  Set ``example_name`` to a
    representative key string (used for autocomplete).

    Subclasses auto-register in ``_REGISTRY`` via ``__init_subclass__``.
    """

    example_name: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "example_name", None) is None:
            warnings.warn(
                f"BinStatisticBase subclass {cls.__name__!r} is missing example_name; "
                "add one for completions to work correctly.",
                stacklevel=2,
            )
        _REGISTRY.append(cls)

    def __init__(self, key: str) -> None:
        self.key = key

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        raise NotImplementedError

    @classmethod
    def valid(cls, key: str) -> BinStatisticBase | None:
        """Return an instance if *key* matches this statistic, else ``None``."""
        if cls.example_name is not None and key == cls.example_name:
            return cls(key)
        return None


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def get_statistic(key: str) -> BinStatisticBase | None:
    """Return the first registered statistic matching *key*, or ``None``."""
    for cls in reversed(_REGISTRY):
        inst = cls.valid(key)
        if inst is not None:
            return inst
    return None


# ---------------------------------------------------------------------------
# Pipeline key parsing
# ---------------------------------------------------------------------------


def _known_transform_token(token: str) -> bool:
    return token in {"abs"}


def parse_pipeline_key(key: str) -> tuple[str, list[str], BinStatisticBase] | None:
    """Parse a dot-notation query into ``(field, transforms, terminal_stat)``.

    ``"vz.abs.mean"`` → field=``"vz"``, transforms=``["abs"]``, stat=``Mean``
    ``"mass.sum"``    → field=``"mass"``, transforms=``[]``, stat=``Sum``

    Returns ``None`` if the key has no ``"."`` or the suffix is not a
    recognised statistic.
    """
    if "." not in key:
        return None
    parts = key.split(".")
    if len(parts) < 2:
        return None

    terminal_token = parts[-1]
    stat = get_statistic(terminal_token)
    if stat is None:
        return None

    field = parts[0]
    transforms = parts[1:-1]

    for t in transforms:
        if not _known_transform_token(t):
            return None

    return field, transforms, stat


def apply_pipeline(arr: np.ndarray, transforms: list[str]) -> np.ndarray:
    """Apply named element-wise transforms to *arr*."""
    out = arr
    for t in transforms:
        if t == "abs":
            out = np.abs(out)
        else:
            raise KeyError(f"Unknown pipeline transform {t!r}.")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_float(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _weighted_mean(arr: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(arr))
    w = _as_float(weights)
    denom = float(np.sum(w))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(arr * w) / denom)


# ---------------------------------------------------------------------------
# Built-in statistics
# ---------------------------------------------------------------------------


class Mean(BinStatisticBase):
    """Weighted arithmetic mean."""

    example_name = "mean"

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        if len(arr) == 0:
            return float("nan")
        return _weighted_mean(_as_float(arr), weight)

    @classmethod
    def valid(cls, key: str) -> Mean | None:
        return cls(key) if key.lower() == "mean" else None


class Sum(BinStatisticBase):
    """Unweighted sum."""

    example_name = "sum"

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        if len(arr) == 0:
            return float("nan")
        return float(np.sum(_as_float(arr)))

    @classmethod
    def valid(cls, key: str) -> Sum | None:
        return cls(key) if key.lower() == "sum" else None


class SumWeighted(BinStatisticBase):
    """Weighted sum (weight × value)."""

    example_name = "sum_w"

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        if len(arr) == 0:
            return float("nan")
        a = _as_float(arr)
        if weight is None:
            return float(np.sum(a))
        return float(np.sum(a * _as_float(weight)))

    @classmethod
    def valid(cls, key: str) -> SumWeighted | None:
        return cls(key) if key.lower() == "sum_w" else None


class Percentile(BinStatisticBase):
    """Weighted/unweighted percentile.  Matches keys like ``p16``, ``p84``."""

    example_name = "p16"

    def __init__(self, key: str, percentile: float) -> None:
        super().__init__(key)
        self.percentile = percentile

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        if len(arr) == 0:
            return float("nan")
        a = _as_float(arr)
        if weight is None:
            return float(np.percentile(a, self.percentile))
        idx = np.argsort(a)
        a_sorted = a[idx]
        w_sorted = _as_float(weight)[idx]
        cdf = np.cumsum(w_sorted)
        cdf -= cdf[0]
        total = float(cdf[-1])
        if total == 0.0:
            return float("nan")
        cdf /= total
        return float(np.interp(self.percentile / 100.0, cdf, a_sorted))

    @classmethod
    def valid(cls, key: str) -> Percentile | None:
        m = re.fullmatch(r"p(\d{1,3}(?:\.\d+)?)", key.lower())
        if m:
            pct = float(m.group(1))
            if 0.0 <= pct <= 100.0:
                return cls(key, pct)
        return None


class Median(BinStatisticBase):
    """Median (equivalent to p50)."""

    example_name = "median"

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        return Percentile("p50", 50.0)(arr, weight)

    @classmethod
    def valid(cls, key: str) -> Median | None:
        return cls(key) if key.lower() in {"median", "med"} else None


class RMS(BinStatisticBase):
    """Root-mean-square."""

    example_name = "rms"

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        if len(arr) == 0:
            return float("nan")
        a = _as_float(arr)
        if weight is None:
            return float(np.sqrt(np.mean(a * a)))
        w = _as_float(weight)
        denom = float(np.sum(w))
        if denom == 0.0:
            return float("nan")
        return float(np.sqrt(np.sum(a * a * w) / denom))

    @classmethod
    def valid(cls, key: str) -> RMS | None:
        return cls(key) if key.lower() == "rms" else None


class Dispersion(BinStatisticBase):
    """Velocity-like dispersion: ``sqrt(E[v²] - E[v]²)``."""

    example_name = "disp"

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        if len(arr) == 0:
            return float("nan")
        a = _as_float(arr)
        sq_mean = _weighted_mean(a * a, weight)
        mean_sq = _weighted_mean(a, weight) ** 2
        diff = sq_mean - mean_sq
        if diff < 0 and diff > -1e-12:
            diff = 0.0
        return float(np.sqrt(diff)) if diff >= 0 else float("nan")

    @classmethod
    def valid(cls, key: str) -> Dispersion | None:
        return cls(key) if key.lower() in {"disp", "dispersion"} else None


class Abs(BinStatisticBase):
    """Standalone statistic: mean of absolute values.

    In pipeline queries (e.g. ``"vz.abs.mean"``) ``abs`` is handled by
    :func:`apply_pipeline` as an element-wise transform, not as a terminal
    statistic.  This class only handles the standalone key ``"abs"``.
    """

    example_name = "abs"

    def __init__(self, key: str, substat: BinStatisticBase | None = None) -> None:
        super().__init__(key)
        self._substat = substat or Mean("mean")

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        return self._substat(np.abs(arr), weight)

    @classmethod
    def valid(cls, key: str) -> Abs | None:
        return cls(key, Mean("mean")) if key.lower() == "abs" else None


# ---------------------------------------------------------------------------
# BinNDStatAccessor
# ---------------------------------------------------------------------------


class BinNDStatAccessor:
    """Accessor returned by ``bins.stat`` that dispatches string pipeline keys.

    Usage::

        bins.stat["mass.sum"]     # equivalent to bins["mass.sum"]
        bins.stat["vz.abs.mean"]  # pipeline query
        bins.stat.keys()          # example keys for all registered statistics
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def __getitem__(self, key: str) -> Any:
        return self._owner._resolve_query(key)

    def keys(self) -> list[str]:
        return [cls.example_name for cls in _REGISTRY if cls.example_name is not None]

    def _ipython_key_completions_(self) -> list[str]:
        return self.keys()


# ---------------------------------------------------------------------------
# Compatibility shims for old code that imported from this module
# ---------------------------------------------------------------------------


def evaluate_statistic(values: Any, statistic: str, weights: Any | None = None) -> float:
    """Compatibility shim.  Prefer :func:`get_statistic`."""
    stat = get_statistic(statistic)
    if stat is None:
        raise KeyError(f"Unknown statistic {statistic!r}.")
    arr = np.asarray(values, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    return stat(arr, w)


def is_statistic_name(name: str) -> bool:
    """Return True if *name* is a recognised statistic key."""
    return get_statistic(name) is not None


# Legacy: kept so old imports of parse_stat_key still work but now uses dot syntax.
def parse_stat_key(key: str) -> tuple[str, str] | None:
    """Deprecated. Use :func:`parse_pipeline_key` instead.

    For backwards-compat only: tries to find a ``"field.stat"`` pattern.
    """
    result = parse_pipeline_key(key)
    if result is None:
        return None
    field, transforms, stat = result
    if transforms:
        return None  # old API didn't support pipeline transforms
    return field, stat.key
