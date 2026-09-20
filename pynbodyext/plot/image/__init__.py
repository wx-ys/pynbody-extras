"""
2-Dimensional Galaxy/Halo Image Post-Processing and Visualization
=================================================================

This package turns the 2-D arrays produced by the calculator layer
(:class:`~pynbodyext.core.calculate.BinND`, and later SPH rendering or
tessellation) into publication-ready figures:

- :mod:`.data` holds :class:`ImageData` — the array, its geometry and metadata,
  and the provenance of the operations applied to it.  It also bridges
  :class:`BinNDResult` queries, which is what ``BinNDResult.imshow`` uses.
- :mod:`.postprocess` smooths, resamples and stretches images, keeping empty
  bins (``NaN``) out of the averages.
- :mod:`.psf` applies an observational point-spread function, and inverts one.
- :mod:`.adaptive` adaptively bins a noisy map by signal (PowerBin) so that each
  region of the figure carries a comparable amount of signal.
- :mod:`.cmaps` provides the ``K_B_C_G_Y_R_W`` velocity colour map.
- :mod:`.compose` joins two maps drawn with different colour maps through a soft
  transition mask (:class:`MapStyle` says how each one becomes colours).
- :mod:`.display` is the matplotlib side: artist choice, axis labels, colour bars.

Each free function stays available for plain 2-D ``numpy`` arrays — nothing in
this package requires a live simulation object — and :class:`ImageData` offers the
same operations as methods, grouped by family::

    image.ImageData.from_bins(bins2d, "vz.mean").smooth.gaussian(fwhm=2).imshow(colorbar=True)
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
from .compose import MapStyle, blend_images, blend_stack, compose_maps, create_map_mask, imshow_compose
from .data import OPERATIONS, ImageData, ImageOp, register_ops
from .display import COLORBAR_LOCATIONS, add_colorbar
from .ops import ImageDataView, ImageOps
from .postprocess import STRETCHES, box_smooth, downsample, gaussian_smooth, median_filter, normalize
from .psf import convolve_psf, deconvolve_psf, gaussian_psf, normalize_psf, richardson_lucy, wiener_deconvolve

__all__ = [
    "COLORBAR_LOCATIONS",
    "AdaptiveMap",
    "ImageData",
    "ImageDataView",
    "ImageOp",
    "ImageOps",
    "K_B_C_G_Y_R_W",
    "K_B_C_G_Y_R_W_COLORS",
    "K_B_C_G_Y_R_W_POSITIONS",
    "MapStyle",
    "OPERATIONS",
    "STRETCHES",
    "add_colorbar",
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
    "register_ops",
    "richardson_lucy",
    "to_rgba",
    "vel_cmap",
    "vel_cmap_r",
    "wiener_deconvolve",
]
