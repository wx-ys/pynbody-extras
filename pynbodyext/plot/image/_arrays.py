"""Array helpers shared by the image-processing modules.

Private to :mod:`pynbodyext.plot.image`: the public surface lives in the
modules that use these helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["FWHM_PER_SIGMA", "as_image", "as_pair", "masked_filter", "resolve_sigma", "validity_mask"]

#: Full width at half maximum of a Gaussian in units of its standard deviation.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def as_image(data: Any, *, name: str = "data") -> np.ndarray:
    """Return *data* as a float 2-D array, rejecting anything else."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D image, got shape {array.shape}.")
    return array


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
