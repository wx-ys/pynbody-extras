"""Public API for the binning framework.

Exports the calculator nodes (:class:`Bin1D`, :class:`BinND`), the binned
result (:class:`BinNDResult`, :class:`SubBinNDResult`), the per-bin array
(:class:`BinsArray`), axis utilities (:class:`BinAxis`, :class:`BinAxisAccessor`),
registries (``register_bin_algorithm``, ``register_pipeline_transform``), and
helpers (``has_axis``, ``has_axes``).
"""

from .accessors import BinParticlesAccessor
from .arrays import BinsArray
from .axes import BIN_ALGORITHMS, BinAxis, BinAxisAccessor, has_axes, has_axis, register_bin_algorithm
from .nodes import Bin1D, BinND
from .plot import BinPlotMixin
from .result import BinNDResult, BinsResultEngine, SubBinNDResult
from .statistics import register_pipeline_transform

__all__ = [
    "BIN_ALGORITHMS",
    "register_bin_algorithm",
    "register_pipeline_transform",
    "has_axis",
    "has_axes",
    "Bin1D",
    "BinND",
    "BinAxis",
    "BinAxisAccessor",
    "BinNDResult",
    "SubBinNDResult",
    "BinsArray",
    "BinsResultEngine",
    "BinParticlesAccessor",
    "BinPlotMixin",
]
