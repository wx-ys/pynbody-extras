"""
Plotting helpers for :mod:`pynbodyext`.

Plot *families* live in their own subpackage so that they stay separable as the
library grows:

- :mod:`pynbodyext.plot.image` — 2-D image post-processing and visualization
  (smoothing, PSF effects, adaptive binning, colour maps, map compositing).
"""

from . import image

__all__ = ["image"]
