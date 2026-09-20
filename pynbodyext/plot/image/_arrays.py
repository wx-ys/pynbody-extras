"""Array and grid helpers shared by the image-processing modules.

Private to :mod:`pynbodyext.plot.image`: the public surface lives in the
modules that use these helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "FWHM_PER_SIGMA",
    "as_2d",
    "as_pair",
    "bin_centers",
    "checked_edges",
    "edges_are_uniform",
    "masked_filter",
    "pixel_width",
    "resolve_edges",
    "resolve_sigma",
    "typical_width",
    "uniform_edges",
    "validity_mask",
    "value_limits",
]

#: Full width at half maximum of a Gaussian in units of its standard deviation.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def as_2d(data: Any, *, name: str = "data") -> np.ndarray:
    """Return *data* as a float 2-D array, rejecting anything else."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D image, got shape {array.shape}.")
    return array


def shape_hint(expected: tuple[int, int], got: tuple[int, int], *, name: str = "data") -> str:
    """Message for a shape mismatch, naming the likely transposition mistake.

    A binned array is laid out ``(x, y)`` while an image is ``(row=y, column=x)``,
    so a raw grid of the right *size* is wrong *way round* — and when it happens to
    be square the mistake is silent, which is worse.  Anything that compares two
    images says so explicitly.
    """
    if got == (expected[1], expected[0]) and expected != got:
        return (
            f"{name} has shape {got}, which is this image's shape {expected} transposed: a binned array is "
            "(x, y) while an image is (row=y, column=x). Pass the binned array itself (a BinsArray knows its "
            "orientation), an ImageData, or transpose the array you have."
        )
    return f"{name} has shape {got}, expected {expected}."


def as_pair(value: Any, *, name: str) -> int | tuple[int, int]:
    """Normalise a scalar-or-``(y, x)`` pair of integers."""
    array = np.atleast_1d(np.asarray(value, dtype=int))
    if array.size == 1:
        return int(array[0])
    if array.size != 2:
        raise ValueError(f"{name} must be a scalar or a (y, x) pair, got {value!r}.")
    return int(array[0]), int(array[1])


def resolve_sigma(sigma: Any, fwhm: Any, pixel_scale: Any) -> float | tuple[float, float]:
    """Turn a ``sigma``/``fwhm`` pair (optionally in physical units) into pixels."""
    if (sigma is None) == (fwhm is None):
        raise ValueError("Pass exactly one of 'sigma' or 'fwhm' (pixel or 'pixel_scale' units).")
    width = np.asarray(fwhm if fwhm is not None else sigma, dtype=float)
    if fwhm is not None:
        width = width / FWHM_PER_SIGMA
    if pixel_scale is not None:
        scale = np.asarray(pixel_scale, dtype=float)
        if np.any(scale <= 0.0):
            raise ValueError(f"pixel_scale must be positive, got {pixel_scale!r}.")
        width = width / scale
    width = np.atleast_1d(width)
    if np.any(width < 0.0):
        raise ValueError(f"Kernel width must be non-negative, got {sigma if sigma is not None else fwhm!r}.")
    if width.size == 1:
        return float(width[0])
    if width.size != 2:
        raise ValueError("Kernel width must be a scalar or a (y, x) pair.")
    return float(width[0]), float(width[1])


def validity_mask(data: np.ndarray, mask: Any) -> np.ndarray:
    """Boolean mask of pixels that may contribute to a neighbourhood average."""
    valid = np.isfinite(data)
    if mask is None:
        return valid
    extra = np.asarray(mask, dtype=bool)
    if extra.shape != data.shape:
        raise ValueError(f"mask must have shape {data.shape}, got {extra.shape}.")
    return valid & extra


def masked_filter(data: np.ndarray, valid: np.ndarray, apply: Any) -> np.ndarray:
    """Apply *apply* to ``data * valid`` and to ``valid``, then renormalise.

    This is the trick that makes a smoother or a convolution NaN-aware: only
    valid pixels enter the average of a pixel, and pixels with no valid
    neighbour come out non-finite.
    """
    numerator = apply(np.where(valid, data, 0.0))
    denominator = apply(valid.astype(float))
    averaged = np.full(data.shape, np.nan)
    np.divide(numerator, denominator, out=averaged, where=denominator > 0.0)
    return np.where(valid, averaged, np.nan)


# ---------------------------------------------------------------------------
# bin grid geometry
# ---------------------------------------------------------------------------


def checked_edges(edges: Any, name: str, count: int) -> np.ndarray | None:
    """Validate one array of bin edges against *count* bins, and freeze it.

    Returns ``None`` when *edges* is ``None``, i.e. when the direction is simply
    indexed by pixels.

    Raises
    ------
    ValueError
        If the edges do not number ``count + 1``, are not finite, or do not
        increase monotonically.
    """
    if edges is None:
        return None
    array = np.array(edges, dtype=float)  # copy, so freezing never touches the caller's array
    if array.shape != (count + 1,):
        raise ValueError(f"{name} must hold {count + 1} edges for {count} pixels, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array}.")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must increase monotonically, got {array}.")
    array.setflags(write=False)
    return array


def uniform_edges(bounds: tuple[float, float], count: int) -> np.ndarray:
    """Evenly spaced edges spanning *bounds* in *count* steps."""
    return np.linspace(float(bounds[0]), float(bounds[1]), count + 1)


def resolve_edges(
    shape: tuple[int, int], *, extent: Any = None, x_edges: Any = None, y_edges: Any = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Bin edges of both directions, from explicit edges and/or an extent.

    An ``extent`` of ``(xmin, xmax, ymin, ymax)`` is shorthand for evenly spaced
    bins and is expanded into edges; explicit edges may be unevenly spaced and
    take precedence.  Giving an extent *and* edges checks that the two agree, so a
    copy of a grid (``dataclasses.replace``) round-trips.

    Raises
    ------
    ValueError
        If only one direction has edges, or an extent contradicts given edges.
    """
    resolved_x = checked_edges(x_edges, "x_edges", shape[1])
    resolved_y = checked_edges(y_edges, "y_edges", shape[0])
    if (resolved_x is None) != (resolved_y is None):
        raise ValueError("Give both x_edges and y_edges, or neither.")
    if resolved_x is None or resolved_y is None:
        if extent is None:
            return None, None
        if len(extent) != 4:
            raise ValueError(f"extent must be (xmin, xmax, ymin, ymax), got {extent!r}.")
        xmin, xmax, ymin, ymax = (float(bound) for bound in extent)
        return uniform_edges((xmin, xmax), shape[1]), uniform_edges((ymin, ymax), shape[0])
    if extent is not None:
        span = (float(resolved_x[0]), float(resolved_x[-1]), float(resolved_y[0]), float(resolved_y[-1]))
        if len(extent) != 4:
            raise ValueError(f"extent must be (xmin, xmax, ymin, ymax), got {extent!r}.")
        requested = tuple(float(bound) for bound in extent)
        if not np.allclose(requested, span):
            raise ValueError(f"extent {requested} contradicts the bin edges, which span {span}.")
    return resolved_x, resolved_y


def bin_centers(edges: np.ndarray | None, count: int) -> np.ndarray:
    """Centres of *count* bins: midpoints of the edges, or pixel centres without edges."""
    if edges is None:
        return np.arange(count, dtype=float) + 0.5
    return 0.5 * (edges[:-1] + edges[1:])


def edges_are_uniform(edges: np.ndarray | None) -> bool:
    """Whether an edge array is evenly spaced (a missing one means pixel indices)."""
    if edges is None:
        return True
    widths = np.diff(edges)
    return bool(np.allclose(widths, widths[0]))


def pixel_width(edges: np.ndarray | None, count: int) -> float:
    """Width of one bin along a direction, requiring evenly spaced edges."""
    if edges is None:
        return 1.0
    if not edges_are_uniform(edges):
        raise ValueError("This direction has bins that are not uniformly spaced, so it has no single pixel size.")
    return float(edges[1] - edges[0])


def typical_width(edges: np.ndarray | None) -> float:
    """A representative bin width, for an algorithm that needs one number."""
    if edges is None:
        return 1.0
    return float(np.median(np.diff(edges)))


def value_limits(
    data: Any, *, vmin: float | None = None, vmax: float | None = None, percentiles: tuple[float, float] | None = None
) -> tuple[float, float]:
    """The ``(vmin, vmax)`` an image of *data* would be drawn with.

    Shared by :func:`~pynbodyext.plot.image.postprocess.normalize`, by the colour
    bars and by :meth:`~pynbodyext.plot.image.ops.ImageOps.limits`, so every colour
    scale in the package is derived by the same rules.

    Parameters
    ----------
    data : array_like
        Values to take the limits from; non-finite entries are ignored.
    vmin, vmax : float, optional
        Explicit limits; each falls back to the requested percentile, then to the
        data range.
    percentiles : (float, float), optional
        Percentiles used for whichever of *vmin*/*vmax* is not given.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)``.

    Raises
    ------
    ValueError
        If ``vmax`` is smaller than ``vmin``.
    """
    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    if percentiles is not None:
        if vmin is None:
            vmin = float(np.nanpercentile(array, percentiles[0])) if finite.any() else 0.0
        if vmax is None:
            vmax = float(np.nanpercentile(array, percentiles[1])) if finite.any() else 1.0
    if vmin is None:
        vmin = float(np.nanmin(array)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(array)) if finite.any() else 1.0
    if vmax < vmin:
        raise ValueError(f"vmax ({vmax}) must not be smaller than vmin ({vmin}).")
    return float(vmin), float(vmax)
