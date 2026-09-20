"""Numeric post-processing of 2-D images: smoothing, stretching and resampling.

All functions take a 2-D array (rows are the y-direction, columns the
x-direction) and return a new array of the same shape, so a map can be passed
through several steps without bookkeeping.  Empty bins produced by
:class:`~pynbodyext.core.calculate.BinND` are ``NaN``; every smoother here is
NaN-aware: non-finite pixels are left out of the average instead of spreading a
``NaN`` over their neighbourhood, and stay non-finite in the result.

Kernel sizes are given in pixels by default.  Pass ``pixel_scale`` (the physical
size of one pixel, or a ``(dy, dx)`` pair) to give ``sigma``/``fwhm`` in physical
units instead — ``ImageData.pixel_size`` supplies exactly that::

    smoothed = gaussian_smooth(image.data, fwhm=0.5, pixel_scale=image.pixel_size)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import ndimage

from ._arrays import aligned_values, as_2d, as_pair, masked_filter, resolve_sigma, validity_mask, value_limits
from .ops import ImageOps, register_ops

if TYPE_CHECKING:
    from .data import ImageData

__all__ = [
    "STRETCHES",
    "SmoothOps",
    "box_smooth",
    "downsample",
    "gaussian_smooth",
    "median_filter",
    "normalize",
    "value_limits",
]

#: Accepted values of the ``stretch`` argument of :func:`normalize`.
STRETCHES = ("linear", "sqrt", "log", "asinh", "hist")

_LOG_GAIN = 1e3


def gaussian_smooth(
    data: Any,
    sigma: Any = None,
    *,
    fwhm: Any = None,
    truncate: float = 4.0,
    mode: str = "reflect",
    mask: Any = None,
    pixel_scale: Any = None,
) -> np.ndarray:
    """Smooth a 2-D image with a Gaussian kernel.

    Parameters
    ----------
    data : array_like
        2-D image; non-finite pixels are excluded from the average.
    sigma : float or (float, float), optional
        Gaussian width in pixels (a ``(y, x)`` pair for an anisotropic kernel).
        Exactly one of *sigma* and *fwhm* must be given.
    fwhm : float or (float, float), optional
        Full width at half maximum, converted to ``sigma = fwhm / 2.3548``.
    truncate : float, default: 4.0
        Kernel is cut at this many standard deviations.
    mode : str, default: "reflect"
        Edge handling, any mode accepted by ``scipy.ndimage.gaussian_filter``.
    mask : array_like of bool, optional
        Additional per-pixel validity mask; ``False`` pixels are left out of the
        average (and are non-finite in the result).
    pixel_scale : float or (float, float), optional
        Physical size of one pixel.  When given, *sigma*/*fwhm* are interpreted
        in those units and divided by the scale before filtering.

    Returns
    -------
    numpy.ndarray
        Smoothed image with the same shape as *data*.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.zeros((9, 9))
    >>> image[4, 4] = 1.0
    >>> smoothed = gaussian_smooth(image, fwhm=2.0)  # doctest: +SKIP
    """
    array = as_2d(data)
    valid = validity_mask(array, mask)
    width = resolve_sigma(sigma, fwhm, pixel_scale)
    return masked_filter(
        array, valid, lambda values: ndimage.gaussian_filter(values, width, mode=mode, truncate=truncate)
    )


def box_smooth(data: Any, size: Any = 3, *, mode: str = "reflect", mask: Any = None) -> np.ndarray:
    """Smooth a 2-D image with a top-hat (moving average) kernel.

    Parameters
    ----------
    data : array_like
        2-D image; non-finite pixels are excluded from the average.
    size : int or (int, int), default: 3
        Kernel size in pixels (a ``(y, x)`` pair for an anisotropic kernel).
    mode : str, default: "reflect"
        Edge handling, any mode accepted by ``scipy.ndimage.uniform_filter``.
    mask : array_like of bool, optional
        Additional per-pixel validity mask.

    Returns
    -------
    numpy.ndarray
        Smoothed image with the same shape as *data*.
    """
    array = as_2d(data)
    valid = validity_mask(array, mask)
    kernel = as_pair(size, name="size")
    return masked_filter(array, valid, lambda values: ndimage.uniform_filter(values, size=kernel, mode=mode))


_PAD_MODES = {"nearest": "edge", "reflect": "reflect", "mirror": "symmetric", "constant": "constant"}


def median_filter(data: Any, size: Any = 3, *, mode: str = "nearest", mask: Any = None) -> np.ndarray:
    """Median-filter a 2-D image, ignoring non-finite pixels.

    Useful for removing cosmic-ray-like hot pixels before smoothing.  The
    NaN-aware path pads the image and takes a median over every sliding window;
    that is memory-hungry for large ``size``, so keep the kernel small.

    Parameters
    ----------
    data : array_like
        2-D image; non-finite pixels are excluded from the median.
    size : int or (int, int), default: 3
        Window size in pixels.
    mode : str, default: "nearest"
        Edge handling: ``"nearest"``, ``"reflect"``, ``"mirror"`` or
        ``"constant"`` (padded with ``NaN``).
    mask : array_like of bool, optional
        Additional per-pixel validity mask.

    Returns
    -------
    numpy.ndarray
        Filtered image with the same shape as *data*.
    """
    array = as_2d(data)
    valid = validity_mask(array, mask)
    kernel = as_pair(size, name="size")
    kernel_y, kernel_x = (kernel, kernel) if isinstance(kernel, int) else kernel
    if valid.all():
        return ndimage.median_filter(array, size=kernel, mode="nearest" if mode == "mirror" else mode)
    if mode not in _PAD_MODES:
        raise ValueError(f"mode must be one of {', '.join(sorted(_PAD_MODES))}, got {mode!r}.")
    filled = np.where(valid, array, np.nan)
    pad_y, pad_x = kernel_y // 2, kernel_x // 2
    padded = np.pad(filled, ((pad_y, pad_x), (pad_y, pad_x)), mode=_PAD_MODES[mode])
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_y, kernel_x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are expected at the edges
        medians = np.nanmedian(windows, axis=(-1, -2))
    return np.where(valid, medians, np.nan)


def _hist_stretch(unit: np.ndarray) -> np.ndarray:
    """Rank-based (histogram equalisation) stretch of values already in [0, 1]."""
    flat = unit.ravel()
    if flat.size < 2:
        return np.zeros_like(unit)
    ranks = np.empty(flat.size, dtype=float)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size, dtype=float)
    return (ranks / (flat.size - 1)).reshape(unit.shape)


def _apply_stretch(unit: np.ndarray, stretch: str, asinh_a: float) -> np.ndarray:
    """Map values already clipped to [0, 1] through *stretch* (keeping [0, 1])."""
    if stretch == "linear":
        return unit
    if stretch == "sqrt":
        return np.sqrt(unit)
    if stretch == "log":
        return np.log1p(_LOG_GAIN * unit) / np.log1p(_LOG_GAIN)
    if stretch == "hist":
        return _hist_stretch(unit)
    return 0.5 * (1.0 + np.arcsinh(asinh_a * (2.0 * unit - 1.0)) / np.arcsinh(asinh_a))


def normalize(
    data: Any,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    stretch: str = "linear",
    percentiles: tuple[float, float] | None = None,
    asinh_a: float = 10.0,
) -> np.ndarray:
    """Map image values to ``[0, 1]`` for display, applying a *stretch*.

    Parameters
    ----------
    data : array_like
        Values to map; non-finite entries stay non-finite.
    vmin, vmax : float, optional
        Limits of the mapping.  Defaults to the data range, or to the requested
        *percentiles* when those are given.
    stretch : {"linear", "sqrt", "log", "asinh", "hist"}, default: "linear"
        Display stretch.  ``"log"`` is a ``log1p`` stretch that lifts faint
        structure, ``"asinh"`` is the usual astronomical alternative, and
        ``"hist"`` is a rank-based histogram equalisation.
    percentiles : (float, float), optional
        Percentiles used for whichever of *vmin*/*vmax* is not given, e.g.
        ``(1, 99)`` to keep a bright source from flattening the rest.
    asinh_a : float, default: 10.0
        Softening parameter of the ``"asinh"`` stretch.

    Returns
    -------
    numpy.ndarray
        Float array in ``[0, 1]`` with the same shape as *data*.

    Examples
    --------
    >>> import numpy as np
    >>> normalize(np.array([0.0, 5.0, 10.0])).round(2).tolist()
    [0.0, 0.5, 1.0]
    >>> normalize(np.array([0.0, 5.0, 10.0]), percentiles=(0, 100)).round(2).tolist()
    [0.0, 0.5, 1.0]
    """
    if stretch not in STRETCHES:
        raise ValueError(f"Unknown stretch {stretch!r}; choose from {', '.join(STRETCHES)}.")
    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    vmin, vmax = value_limits(array, vmin=vmin, vmax=vmax, percentiles=percentiles)
    if vmax == vmin:
        return np.where(finite, 0.0, np.nan)
    with np.errstate(invalid="ignore"):  # non-finite entries are masked out below
        unit = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.where(finite, _apply_stretch(unit, stretch, asinh_a), np.nan)


_REDUCERS = {"mean": np.nanmean, "sum": np.nansum, "median": np.nanmedian, "max": np.nanmax}


def downsample(data: Any, factor: Any = 2, *, func: str = "mean") -> np.ndarray:
    """Average neighbouring pixels into larger blocks.

    Each ``factor_y x factor_x`` block of the input becomes one output pixel.
    Non-finite pixels are skipped by the reduction.

    Parameters
    ----------
    data : array_like
        2-D image whose shape is divisible by *factor* in both directions.
    factor : int or (int, int), default: 2
        Block size in pixels.
    func : {"mean", "sum", "median", "max"}, default: "mean"
        Reduction applied inside each block.

    Returns
    -------
    numpy.ndarray
        Image reduced by *factor* in each direction.
    """
    if func not in _REDUCERS:
        raise ValueError(f"Unknown func {func!r}; choose from {', '.join(sorted(_REDUCERS))}.")
    array = as_2d(data)
    pair = as_pair(factor, name="factor")
    factor_y, factor_x = (pair, pair) if isinstance(pair, int) else pair
    if factor_y < 1 or factor_x < 1:
        raise ValueError(f"factor must be positive, got {factor!r}.")
    if array.shape[0] % factor_y or array.shape[1] % factor_x:
        raise ValueError(f"Image shape {array.shape} must be divisible by factor {factor!r}.")
    blocks = array.reshape(array.shape[0] // factor_y, factor_y, array.shape[1] // factor_x, factor_x).transpose(
        0, 2, 1, 3
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks are expected
        return _REDUCERS[func](blocks, axis=(2, 3))


def _factor_pair(factor: Any) -> tuple[int, int]:
    """Normalise a ``factor`` into an explicit ``(y, x)`` pair."""
    pair = as_pair(factor, name="factor")
    return (pair, pair) if isinstance(pair, int) else pair


def _mask(mask: Any) -> np.ndarray | None:
    """Bring a mask into image orientation, accepting a binned array as well."""
    return None if mask is None else aligned_values(mask)


@dataclass(frozen=True)
class SmoothOps(ImageOps):
    """The smoothing family of an image: ``image.smooth.gaussian(fwhm=2)``.

    Each method returns a new :class:`~pynbodyext.plot.image.data.ImageData`, so
    calls chain and the geometry, units and provenance travel along with the
    values.  ``sigma``/``fwhm`` are in the units of the axes whenever the grid is
    evenly spaced (the image's own ``pixel_size`` is used for the conversion), and
    in pixels otherwise.
    """

    def gaussian(
        self, sigma: Any = None, *, fwhm: Any = None, truncate: float = 4.0, mode: str = "reflect", mask: Any = None
    ) -> ImageData:
        """Smooth with a Gaussian kernel; see :func:`gaussian_smooth`."""
        smoothed = gaussian_smooth(
            self.image.data,
            sigma,
            fwhm=fwhm,
            truncate=truncate,
            mode=mode,
            mask=_mask(mask),
            pixel_scale=self.kernel_scale(),
        )
        return self.image._derived(
            smoothed,
            "gaussian_smooth",
            {"sigma": sigma, "fwhm": fwhm, "truncate": truncate, "mode": mode, "mask": mask},
        )

    def box(self, size: Any = 3, *, mode: str = "reflect", mask: Any = None) -> ImageData:
        """Smooth with a top-hat kernel; see :func:`box_smooth`."""
        smoothed = box_smooth(self.image.data, size, mode=mode, mask=_mask(mask))
        return self.image._derived(smoothed, "box_smooth", {"size": size, "mode": mode, "mask": mask})

    def median(self, size: Any = 3, *, mode: str = "nearest", mask: Any = None) -> ImageData:
        """Median-filter the image; see :func:`median_filter`."""
        filtered = median_filter(self.image.data, size, mode=mode, mask=_mask(mask))
        return self.image._derived(filtered, "median_filter", {"size": size, "mode": mode, "mask": mask})

    def downsample(self, factor: Any = 2, *, func: str = "mean") -> ImageData:
        """Block-average the image, shrinking it by *factor*; see :func:`downsample`."""
        reduced = downsample(self.image.data, factor, func=func)
        factor_y, factor_x = _factor_pair(factor)
        overrides: dict[str, Any] = {}
        if self.image.x_edges is not None and self.image.y_edges is not None:
            overrides["x_edges"] = self.image.x_edges[::factor_x]
            overrides["y_edges"] = self.image.y_edges[::factor_y]
        return self.image._derived(reduced, "downsample", {"factor": factor, "func": func}, **overrides)


register_ops("smooth", SmoothOps)
