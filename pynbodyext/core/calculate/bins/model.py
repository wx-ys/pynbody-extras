"""Pure data container for a binned result (root or sub-result)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .axes import BinAxis


class BinResultModel:
    """Immutable-ish holder of the per-run binning data.

    Holds no query/cache/geometry logic; services (``BinGeometry``,
    ``BinQueryService``, ``BinSubresultService``) operate on it.
    """

    def __init__(
        self,
        *,
        sim: Any,
        source_sim: Any,
        axes: tuple[BinAxis, ...],
        bin_data: np.ndarray | None,
        bin_indptr: np.ndarray | None,
        particle_bin: np.ndarray | None,
        valid_mask: np.ndarray | None,
        calculator: Any,
        scope_signature: Any,
        parent: BinResultModel | None = None,
    ) -> None:
        self.sim = sim
        self.source_sim = source_sim
        self.axes = axes
        self.shape_bins = tuple(axis.nbins for axis in axes)
        self.ndim = len(axes)
        self.bin_data = bin_data
        self.bin_indptr = bin_indptr
        self.particle_bin = particle_bin
        self.valid_mask = valid_mask
        self.calculator = calculator
        self.scope_signature = scope_signature
        self.parent = parent

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def root(self) -> BinResultModel:
        return self if self.parent is None else self.parent.root

    @property
    def nbins(self) -> int:
        return int(np.prod(self.shape_bins, dtype=int))

    @property
    def total_nbins(self) -> int:
        return self.nbins

    @property
    def unassigned_count(self) -> int:
        mask = self.valid_mask
        if mask is None:
            return 0
        return int(np.count_nonzero(~mask))
