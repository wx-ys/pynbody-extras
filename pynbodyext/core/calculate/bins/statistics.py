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

Supported pipeline transforms are stored in ``_PIPELINE_TRANSFORMS`` and can be
extended with :func:`register_pipeline_transform`.  Built-in transforms:
``abs``, ``log``, ``log10``, ``sqrt``, ``square``.

Empty bin → each statistic returns ``np.nan`` (or 0 for ``count``).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .arrays import BinsArray
    from .result import BinNDResult


StatisticFunc = Callable[[Any, Any | None], float]

__all__ = [
    "BinStatisticBase",
    "BinNDStatAccessor",
    "parse_pipeline_key",
    "apply_pipeline",
    "get_statistic",
    "evaluate_statistic",
    "is_statistic_name",
    "weighted_percentiles",
    "register_pipeline_transform",
    # built-in statistics
    "Mean",
    "Sum",
    "Percentile",
    "Median",
    "RMS",
    "Dispersion",
]

_REGISTRY: list[type[BinStatisticBase]] = []

# ---------------------------------------------------------------------------
# Pipeline transform registry
# ---------------------------------------------------------------------------

_PIPELINE_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "abs": np.abs,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "square": np.square,
}


class StatRegistry:
    """Registry of registered :class:`BinStatisticBase` subclasses."""

    def __init__(self, classes: list[type[BinStatisticBase]] | None = None) -> None:
        self._classes = classes if classes is not None else []

    def register(self, cls: type[BinStatisticBase]) -> None:
        self._classes.append(cls)

    def get(self, key: str) -> BinStatisticBase | None:
        for cls in reversed(self._classes):
            inst = cls.valid(key)
            if inst is not None:
                return inst
        return None

    def keys(self) -> list[str]:
        return [cls.example_name for cls in self._classes if cls.example_name is not None]


class PipelineParser:
    """Registry of element-wise transforms and dot-notation query parsing."""

    def __init__(self, transforms: dict[str, Callable[[np.ndarray], np.ndarray]]) -> None:
        self._transforms = transforms

    def known(self, token: str) -> bool:
        return token in self._transforms

    def register(self, name: str, func: Callable[[np.ndarray], np.ndarray], *, overwrite: bool = False) -> None:
        if not overwrite and name in self._transforms:
            raise KeyError(f"Pipeline transform {name!r} is already registered.")
        self._transforms[name] = func

    def apply(self, arr: np.ndarray, transforms: list[str]) -> np.ndarray:
        out = arr
        for t in transforms:
            fn = self._transforms.get(t)
            if fn is None:
                raise KeyError(f"Unknown pipeline transform {t!r}. Known transforms: {sorted(self._transforms)}.")
            out = fn(out)
        return out

    def parse(self, key: str) -> tuple[str, list[str], BinStatisticBase, str | None] | None:
        weight_field: str | None = None
        core_key = key
        if "@" in key:
            core_key, weight_field = key.rsplit("@", 1)
            if not weight_field:
                return None

        if "." not in core_key:
            return None
        parts = core_key.split(".")
        if len(parts) < 2:
            return None

        terminal_token = parts[-1]
        stat = get_statistic(terminal_token)
        if stat is None:
            return None

        field = parts[0]
        transforms = parts[1:-1]
        for t in transforms:
            if not self.known(t):
                return None

        return field, transforms, stat, weight_field

    def transform_keys(self) -> list[str]:
        return sorted(self._transforms.keys())


STAT_REGISTRY = StatRegistry(_REGISTRY)
PIPELINE_PARSER = PipelineParser(_PIPELINE_TRANSFORMS)


def register_pipeline_transform(
    name: str, func: Callable[[np.ndarray], np.ndarray], *, overwrite: bool = False
) -> None:
    """Register a named element-wise transform for use in pipeline queries.

    After registration, queries like ``"field.{name}.mean"`` apply *func*
    element-wise to the particle array before the terminal statistic.

    Parameters
    ----------
    name:
        Token string used in dot-notation queries (e.g. ``"log10"``).
    func:
        A callable ``(np.ndarray) -> np.ndarray`` applied element-wise.
    overwrite:
        If ``False`` (default), raise :exc:`KeyError` if *name* is already registered.

    Examples
    --------
    >>> import numpy as np
    >>> register_pipeline_transform("double", lambda a: a * 2)
    >>> bins["mass.double.sum"]  # per-bin sum of doubled mass
    """
    PIPELINE_PARSER.register(name, func, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BinStatisticBase:
    """Abstract base for per-bin statistics.

    Subclass and implement ``__call__``.  Override ``valid`` to control which
    string key(s) this statistic matches.  Set ``example_name`` to a
    representative key string (used for autocomplete).

    Subclasses auto-register in ``_REGISTRY`` via ``__init_subclass__``.
    Pass ``abstract=True`` to declare an intermediate base class that should
    *not* be registered::

        class MyWeightedBase(BinStatisticBase, abstract=True): ...
    """

    example_name: str | None = None

    def __init_subclass__(cls, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        if getattr(cls, "example_name", None) is None:
            warnings.warn(
                f"BinStatisticBase subclass {cls.__name__!r} is missing example_name; "
                "add one for completions to work correctly.",
                stacklevel=2,
            )
        STAT_REGISTRY.register(cls)

    def __init__(self, key: str) -> None:
        self.key = key

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        raise NotImplementedError

    def vectorized_call(
        self, values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None, nbins: int
    ) -> np.ndarray | None:
        """Compute statistic over all bins simultaneously without a Python loop.

        Parameters
        ----------
        values:
            Float array for valid (assigned) particles only, with any pipeline
            transforms already applied.  Shape ``(n_valid,)``.
        bins:
            Flat bin index for each valid particle.  All values are ``>= 0``.
            Shape ``(n_valid,)``.
        weights:
            Weight array for valid particles, or ``None``.  Shape ``(n_valid,)``.
        nbins:
            Total number of bins.

        Returns
        -------
        np.ndarray or None
            1-D float array of length ``nbins`` (``np.nan`` for empty bins),
            or ``None`` to fall back to the per-bin Python loop.
        """
        return None

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
    return STAT_REGISTRY.get(key)


# ---------------------------------------------------------------------------
# Pipeline key parsing
# ---------------------------------------------------------------------------


def _known_transform_token(token: str) -> bool:
    return PIPELINE_PARSER.known(token)


def parse_pipeline_key(key: str) -> tuple[str, list[str], BinStatisticBase, str | None] | None:
    """Parse a dot-notation query into ``(field, transforms, terminal_stat, weight_field)``.

    ``"vz.abs.mean"``     → ``("vz", ["abs"], Mean, None)``
    ``"mass.sum"``        → ``("mass", [], Sum, None)``
    ``"age.mean@mass"``   → ``("age", [], Mean, "mass")``   # mass-weighted mean

    The optional ``@weight_field`` suffix specifies the SimSnap field to use as
    per-particle weights.  Any string that is a valid SimSnap array key is
    accepted.

    Returns ``None`` if the key has no ``"."`` or the suffix is not a
    recognised statistic.
    """
    return PIPELINE_PARSER.parse(key)


def apply_pipeline(arr: np.ndarray, transforms: list[str]) -> np.ndarray:
    """Apply named element-wise transforms to *arr*."""
    return PIPELINE_PARSER.apply(arr, transforms)


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


def _bincount_nan(bins: np.ndarray, values: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (counts, weighted_sums) of length *nbins*; counts dtype is intp."""
    counts = np.bincount(bins, minlength=nbins).astype(np.intp)
    sums = np.bincount(bins, weights=values, minlength=nbins)
    return counts, sums


def _flat_samples(values: Any, weights: Any, segments: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Flatten a batch of samples into values, weights and a sample index each.

    Accepts one sample (1-D), one per row (2-D), or flat arrays with a length per
    segment; the caller gets the same three things either way.
    """
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    if values_array.shape != weights_array.shape:
        raise ValueError(
            f"values and weights must have the same shape, got {values_array.shape} and {weights_array.shape}."
        )
    if segments is not None:
        lengths = np.asarray(segments, dtype=int)
        sample_of = np.repeat(np.arange(len(lengths)), lengths)
        if len(values_array) != len(sample_of):
            raise ValueError(f"segments describe {len(sample_of)} values, got {len(values_array)}.")
        return values_array.ravel(), weights_array.ravel(), sample_of, len(lengths)
    if values_array.ndim == 1:
        values_array = values_array[None, :]
        weights_array = weights_array[None, :]
    if values_array.ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got shape {values_array.shape}.")
    sample_of = np.repeat(np.arange(len(values_array)), values_array.shape[1])
    return values_array.ravel(), weights_array.ravel(), sample_of, len(values_array)


def weighted_percentiles(values: Any, weights: Any, percentile: float, *, segments: Any = None) -> np.ndarray | float:
    """Weighted percentile of one sample, of every row of a 2-D batch — or of a
    flat batch split into ``segments``.

    The definition is the one the weighted :class:`Percentile` has always used: the
    pairs are sorted by value, the cumulative weight is shifted so the lightest
    value sits at zero, normalised by the total, and the percentile is
    interpolated along that curve.  Two details matter at the edges:

    - a sample whose weight sits on a single value *is* that value (there is no
      curve to interpolate, and this used to come back as ``NaN``);
    - values that are not finite carry no information and are dropped, as the
      kernel sums drop them — weights and values have to be finite together.

    Parameters
    ----------
    values, weights : array_like
        Per-value values and weights, the same shape.
    percentile : float
        Percentile in ``[0, 100]``.
    segments : array_like of int, optional
        For flat input: the length of each consecutive sample.  This is how a
        scattering of particles onto cells is reduced — one segment per cell,
        however many particles landed in it — without padding every cell to the
        largest one.  The values are sorted within their segment here, so the
        caller need not order them.

    Returns
    -------
    numpy.ndarray or float
        A float for a single 1-D sample, else one value per sample (row, or
        segment), ``NaN`` where nothing carries weight.

    Examples
    --------
    >>> weighted_percentiles([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 50.0)
    2.0
    >>> weighted_percentiles([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 3.0]], 50.0)
    array([1.5, 3.5])
    >>> weighted_percentiles([6.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], 50.0, segments=[1, 3])
    array([ 6., nan])
    """
    single = segments is None and np.asarray(values).ndim == 1
    values_array, weights_array, sample_of, n_samples = _flat_samples(values, weights, segments)

    carries = (weights_array > 0.0) & np.isfinite(values_array)
    samples = sample_of[carries]
    counts = np.bincount(samples, minlength=n_samples)
    out = np.full(n_samples, np.nan)

    # Cell-major, value-sorted: the segments come out contiguous, and the sorting
    # puts the weighted, finite values of each before the rest.
    order = np.lexsort((values_array[carries], samples))
    sorted_values = values_array[carries][order]
    sorted_weights = weights_array[carries][order]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)

    one_point = counts == 1
    if one_point.any():
        out[one_point] = sorted_values[starts[one_point]]

    several = counts > 1
    if several.any():
        # The lightest value sits at zero and the total excludes it, exactly as the
        # scalar definition does; "base" is the weight accumulated before each
        # segment starts.
        cumulative = np.cumsum(sorted_weights)
        last = max(len(sorted_weights) - 1, 0)
        safe_starts = np.clip(starts, 0, last)
        base = np.concatenate([[0.0], cumulative[:-1]])[safe_starts]
        first_weight = sorted_weights[safe_starts]
        ends = np.clip(starts + np.maximum(counts, 1) - 1, 0, last)
        segment_total = (cumulative[ends] - base) - first_weight
        shifted = (cumulative - np.repeat(base, counts)) - np.repeat(first_weight, counts)
        # The lightest value *is* the zero of the curve; cancellation in the line
        # above can leave it a few ulps off, which is enough to move a percentile
        # of 0 onto the next value.
        shifted[starts[counts > 0]] = 0.0
        total = np.repeat(segment_total, counts)
        with np.errstate(invalid="ignore", divide="ignore"):
            cdf = shifted / total

        # np.interp's lookup, done for every segment at once: the bracket is the
        # first entry whose cumulative weight reaches the percentile, and a target
        # on a knot interpolates to it.
        target = float(percentile) / 100.0
        reached = np.flatnonzero(cdf >= target)
        first = np.unique(samples[order][reached], return_index=True)[1]
        hit = reached[first]
        previous = np.maximum(hit - 1, starts[samples[order][hit]])
        low, high = cdf[previous], cdf[hit]
        below, above = sorted_values[previous], sorted_values[hit]
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(high > low, (target - low) / (high - low), 0.0)
        interpolated = below + np.nan_to_num(fraction) * (above - below)
        out[samples[order][hit]] = interpolated

    return float(out[0]) if single else out


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

    def vectorized_call(
        self, values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None, nbins: int
    ) -> np.ndarray:
        v = values.astype(float, copy=False)
        if weights is None:
            counts, sums = _bincount_nan(bins, v, nbins)
            with np.errstate(invalid="ignore"):
                return np.where(counts > 0, sums / counts, np.nan)
        w = weights.astype(float, copy=False)
        counts = np.bincount(bins, minlength=nbins).astype(np.intp)
        w_sums = np.bincount(bins, weights=w, minlength=nbins)
        vw_sums = np.bincount(bins, weights=v * w, minlength=nbins)
        with np.errstate(invalid="ignore"):
            return np.where((counts > 0) & (w_sums != 0.0), vw_sums / w_sums, np.nan)

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

    def vectorized_call(
        self, values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None, nbins: int
    ) -> np.ndarray:
        v = values.astype(float, copy=False)
        counts, sums = _bincount_nan(bins, v, nbins)
        return np.where(counts > 0, sums, np.nan)

    @classmethod
    def valid(cls, key: str) -> Sum | None:
        return cls(key) if key.lower() == "sum" else None


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
        return float(weighted_percentiles(a, _as_float(weight), self.percentile))

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
    percentile = 50.0

    def __call__(self, arr: np.ndarray, weight: np.ndarray | None) -> float:
        return Percentile("p50", self.percentile)(arr, weight)

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

    def vectorized_call(
        self, values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None, nbins: int
    ) -> np.ndarray:
        v = values.astype(float, copy=False)
        if weights is None:
            counts, sq_sums = _bincount_nan(bins, v * v, nbins)
            with np.errstate(invalid="ignore"):
                return np.where(counts > 0, np.sqrt(sq_sums / counts), np.nan)
        w = weights.astype(float, copy=False)
        counts = np.bincount(bins, minlength=nbins).astype(np.intp)
        w_sums = np.bincount(bins, weights=w, minlength=nbins)
        sq_w_sums = np.bincount(bins, weights=v * v * w, minlength=nbins)
        with np.errstate(invalid="ignore"):
            return np.where((counts > 0) & (w_sums != 0.0), np.sqrt(sq_w_sums / w_sums), np.nan)

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

    def vectorized_call(
        self, values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None, nbins: int
    ) -> np.ndarray:
        # Var(v) = E[v²] - E[v]² computed fully vectorised via Mean
        mean_obj = Mean("mean")
        sq_mean = mean_obj.vectorized_call(values * values, bins, weights, nbins)
        mean_sq = mean_obj.vectorized_call(values, bins, weights, nbins) ** 2
        diff = sq_mean - mean_sq
        diff = np.where(np.abs(diff) < 1e-12, 0.0, diff)
        return np.where(diff >= 0.0, np.sqrt(diff), np.nan)

    @classmethod
    def valid(cls, key: str) -> Dispersion | None:
        return cls(key) if key.lower() in {"disp", "dispersion"} else None


# ---------------------------------------------------------------------------
# BinNDStatAccessor
# ---------------------------------------------------------------------------


class BinNDStatAccessor:
    """Accessor returned by ``bins.stat`` that dispatches string pipeline keys.

    For extended documentation and examples of supported query syntax, see
    :class:`BinStatisticBase` and :func:`parse_pipeline_key`.
    """

    def __init__(self, owner: BinNDResult) -> None:
        self._owner = owner

    def __repr__(self) -> str:
        return f"<BinNDStatAccessor for {self._owner}>"

    def __getitem__(self, key: str) -> BinsArray:
        return self._owner._resolve_query(key)

    def __call__(
        self,
        field: str,
        statistic: str,
        *,
        weight: str | Callable[[Any], Any] | Any | None = None,
        transforms: list[str] | None = None,
    ) -> BinsArray:
        """Evaluate an explicit per-bin statistic and return a cached :class:`BinsArray`.

        Equivalent to ``bins["field.stat"]`` but accepts optional transforms and
        weight::

            bins.stat("mass", "mean", weight="mass")
            bins.stat("vz", "mean", transforms=["abs"])
        """
        return self._owner._stat_explicit(field, statistic, weight=weight, transforms=transforms)

    def keys(self) -> list[str]:
        return STAT_REGISTRY.keys()

    def transform_keys(self) -> list[str]:
        return PIPELINE_PARSER.transform_keys()

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
