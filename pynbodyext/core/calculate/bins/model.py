"""Pure data container for a binned result (root or sub-result)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .arrays import BinsArray
    from .axes import BinAxis, BinAxisAccessor
    from .result import BinNDResult


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
        owner: BinNDResult | None = None,
    ) -> None:
        self.sim = sim
        self.source_sim = source_sim
        self.axes = axes
        self.shape_bins = tuple(axis.nbins for axis in axes)
        self.nbins = int(np.prod(self.shape_bins, dtype=int))
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
        """Whether this model is the root result (has no parent)."""
        return self.parent is None

    @property
    def root(self) -> BinResultModel:
        """The root model of this sub-result's tree."""
        return self if self.parent is None else self.parent.root

    @property
    def total_nbins(self) -> int:
        """Alias for :attr:`nbins`."""
        return self.nbins

    @property
    def unassigned_count(self) -> int:
        """Number of particles that fall outside every bin."""
        mask = self.valid_mask
        if mask is None:
            return 0
        return int(np.count_nonzero(~mask))

    # ------------------------------------------------------------------
    # Result-level read-only accessors (delegate to the owning result)
    # so derived properties can operate on the model.
    # ------------------------------------------------------------------

    @property
    def owner(self) -> BinNDResult | None:
        """The :class:`~.result.BinNDResult` facade that owns this model, if any."""
        return self._owner

    def _require_owner(self) -> BinNDResult:
        if self._owner is None:
            raise RuntimeError("BinResultModel is not bound to a BinNDResult.")
        return self._owner

    def __getitem__(self, key: Any) -> BinsArray | BinResultModel:
        result = self._require_owner()[key]
        sub = getattr(result, "_model", None)
        return sub if sub is not None else result

    @property
    def axis(self) -> BinAxisAccessor:
        """Accessor over this result's axes, keyed by alias or index."""
        return self._require_owner().axis

    @property
    def centers(self) -> np.ndarray:
        """Per-axis bin centers."""
        return self._require_owner().centers

    @property
    def mins(self) -> np.ndarray:
        """Per-axis lower bin edges."""
        return self._require_owner().mins

    @property
    def maxs(self) -> np.ndarray:
        """Per-axis upper bin edges."""
        return self._require_owner().maxs

    @property
    def widths(self) -> np.ndarray:
        """Per-axis bin widths."""
        return self._require_owner().widths

    @property
    def edges(self) -> np.ndarray:
        """Per-axis bin edges (length ``nbins + 1``)."""
        return self._require_owner().edges

    def find_axis(self, aliases: set[str]) -> BinAxis:
        """Return the first axis matching any alias in *aliases*."""
        return self._require_owner().find_axis(aliases)

    def multi_index_array(self) -> np.ndarray:
        """Return an integer array of per-particle bin indices."""
        return self._require_owner().multi_index_array()

    def _resolve_axis_measure(self, axis: BinAxis) -> np.ndarray:
        return self._require_owner()._resolve_axis_measure(axis)

    def families(self) -> Any:
        """Return the pynbody families present in this result's snapshot."""
        return self._require_owner().families()

    @property
    def gas(self) -> BinResultModel:
        """Sub-result restricted to the gas family."""
        return self._require_owner().gas._model

    @property
    def dm(self) -> BinResultModel:
        """Sub-result restricted to the dark-matter family."""
        return self._require_owner().dm._model

    @property
    def star(self) -> BinResultModel:
        """Sub-result restricted to the star family."""
        return self._require_owner().star._model

    @property
    def g(self) -> BinResultModel:
        """Short alias for :attr:`gas`."""
        return self._require_owner().g._model

    @property
    def s(self) -> BinResultModel:
        """Short alias for :attr:`star`."""
        return self._require_owner().s._model
