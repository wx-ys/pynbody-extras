"""
2-Dimensional Galaxy/Halo Image Post-Processing and Visualization
=================================================================

This package turns the 2-D arrays produced by the calculator layer
(:class:`~pynbodyext.core.calculate.BinND`, and later SPH rendering or
tessellation) into publication-ready figures:

- :mod:`.data` wraps an array with the metadata needed to display it
  (``extent``, units, axis aliases) and bridges :class:`BinNDResult` queries.
- :mod:`.postprocess` smooths, resamples and stretches images, keeping empty
  bins (``NaN``) out of the averages.
- :mod:`.psf` applies an observational point-spread function, and inverts one.
- :mod:`.adaptive` adaptively bins a noisy map by signal (PowerBin) so that each
  region of the figure carries a comparable amount of signal.
- :mod:`.cmaps` provides the ``K_B_C_G_Y_R_W`` velocity colour map.
- :mod:`.compose` joins two maps drawn with different colour maps through a soft
  transition mask.

Everything here operates on plain 2-D ``numpy`` arrays; nothing in this package
requires a live simulation object.
"""

from .adaptive import AdaptiveMap, adaptive_bin_map, adaptive_map_from_bins
from .cmaps import (
    K_B_C_G_Y_R_W,
    K_B_C_G_Y_R_W_COLORS,
    K_B_C_G_Y_R_W_POSITIONS,
    cmap_from_colors,
    get_cmap,
    register_cmap,
    to_rgba,
    vel_cmap,
    vel_cmap_r,
)
from .compose import blend_images, blend_stack, compose_maps, create_map_mask, imshow_compose
from .data import ImageData
from .postprocess import STRETCHES, box_smooth, downsample, gaussian_smooth, median_filter, normalize
from .psf import convolve_psf, deconvolve_psf, gaussian_psf, normalize_psf, richardson_lucy, wiener_deconvolve

__all__ = [
    "AdaptiveMap",
    "ImageData",
    "K_B_C_G_Y_R_W",
    "K_B_C_G_Y_R_W_COLORS",
    "K_B_C_G_Y_R_W_POSITIONS",
    "STRETCHES",
    "adaptive_bin_map",
    "adaptive_map_from_bins",
    "blend_images",
    "blend_stack",
    "box_smooth",
    "cmap_from_colors",
    "compose_maps",
    "convolve_psf",
    "create_map_mask",
    "deconvolve_psf",
    "downsample",
    "gaussian_psf",
    "gaussian_smooth",
    "get_cmap",
    "imshow_compose",
    "median_filter",
    "normalize",
    "normalize_psf",
    "register_cmap",
    "richardson_lucy",
    "to_rgba",
    "vel_cmap",
    "vel_cmap_r",
    "wiener_deconvolve",
]
