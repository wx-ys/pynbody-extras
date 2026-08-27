"""Pure data container for a binned result (root or sub-result)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .axes import BinAxis


class BinResultModel:
    """Holder of the per-run binning data plus a read-only result view.

    The raw fields are the source of truth for the data services
    (``BinGeometry``, ``StatPipeline``, ``ApplyComposer``).  The result-level
    read-only accessors (``__getitem__``, ``.gas``, ``.centers``, …) delegate to
    the owning :class:`~.result.BinNDResult` so derived properties can operate on
    the model without reaching into the facade.
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
        owner: Any = None,
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
        self._owner = owner

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

    # ------------------------------------------------------------------
    # Result-level read-only accessors (delegate to the owning result)
    # so derived properties can operate on the model.
    # ------------------------------------------------------------------

    @property
    def owner(self) -> Any:
        return self._owner

    def _require_owner(self) -> Any:
        if self._owner is None:
            raise RuntimeError("BinResultModel is not bound to a BinNDResult.")
        return self._owner

    def __getitem__(self, key: str) -> Any:
        return self._require_owner()._resolve_query(key)

    @property
    def axis(self) -> Any:
        return self._require_owner().axis

    @property
    def centers(self) -> Any:
        return self._require_owner().centers

    @property
    def mins(self) -> Any:
        return self._require_owner().mins

    @property
    def maxs(self) -> Any:
        return self._require_owner().maxs

    @property
    def widths(self) -> Any:
        return self._require_owner().widths

    @property
    def edges(self) -> Any:
        return self._require_owner().edges

    def find_axis(self, aliases: set[str]) -> Any:
        return self._require_owner().find_axis(aliases)

    def multi_index_array(self) -> np.ndarray:
        return self._require_owner().multi_index_array()

    def _resolve_axis_measure(self, axis: Any) -> np.ndarray:
        return self._require_owner()._resolve_axis_measure(axis)

    def families(self) -> Any:
        return self._require_owner().families()

    @property
    def gas(self) -> Any:
        return self._require_owner().gas

    @property
    def dm(self) -> Any:
        return self._require_owner().dm

    @property
    def star(self) -> Any:
        return self._require_owner().star

    @property
    def g(self) -> Any:
        return self._require_owner().g

    @property
    def s(self) -> Any:
        return self._require_owner().s
