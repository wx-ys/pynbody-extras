from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import SimArray

if TYPE_CHECKING:
    from .result import BinNDResult


class BinsArray(SimArray):
    """Per-bin result array with ND grid shape.

    The stored values are shaped as ``(axis0_nbins, axis1_nbins, ...)`` (C-order,
    last axis varies fastest), matching the bin grid.  Flat indexing is still
    available via ``np.asarray(arr).ravel()``.

    Use :meth:`grid` or :meth:`reshape_bins` for an explicit ND view.
    """

    __slots__ = ["_bins", "_name", "_field", "_mode", "_shape_bins", "_axis_aliases", "_provenance"]

    _bins: Any  # set by __new__; kept Any for numpy __array_finalize__ quirks
    _name: str | None
    _field: str | None
    _mode: str | None
    _shape_bins: tuple[int, ...] | None
    _axis_aliases: tuple[str, ...] | None
    _provenance: dict[str, Any]

    def __new__(
        cls,
        bins: BinNDResult,
        values: Any,
        *,
        name: str | None = None,
        field: str | None = None,
        mode: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> BinsArray:
        shape_bins = tuple(bins.shape_bins)
        # Reshape flat values to ND grid shape.
        # Arrays of shape (nbins, *extra) are reshaped to (*shape_bins, *extra)
        # so that per-bin 2-D results (e.g. multi_index, cell_widths) work correctly.
        arr = np.asarray(values)
        if arr.shape != shape_bins:
            nbins = int(np.prod(shape_bins))
            if arr.ndim > 1 and arr.shape[0] == nbins:
                arr = arr.reshape((*shape_bins, *arr.shape[1:]))
            else:
                arr = arr.reshape(shape_bins)
        obj = super().__new__(cls, arr)
        # Preserve units and sim from the source array so that BinsArray
        # retains unit-tracking when wrapping a SimArray/IndexedSimArray.
        if hasattr(values, "units") and values.units is not None:
            obj.units = values.units
        if hasattr(values, "sim") and values.sim is not None:
            obj.sim = values.sim
        obj._bins = bins
        obj._name = name
        obj._field = field
        obj._mode = mode
        obj._shape_bins = shape_bins
        obj._axis_aliases = tuple(axis.alias for axis in bins.axes)
        obj._provenance = provenance or {}
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        super().__array_finalize__(obj)
        if obj is None:
            return
        self._bins = getattr(obj, "_bins", None)
        self._name = getattr(obj, "_name", None)
        self._field = getattr(obj, "_field", None)
        self._mode = getattr(obj, "_mode", None)
        self._shape_bins = getattr(obj, "_shape_bins", None)
        self._axis_aliases = getattr(obj, "_axis_aliases", None)
        self._provenance = getattr(obj, "_provenance", {})

    @property
    def bins(self) -> BinNDResult:
        return self._bins

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def field(self) -> str | None:
        return self._field

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    @property
    def shape_bins(self) -> tuple[int, ...]:
        shape_bins = self._shape_bins
        if shape_bins is None:
            return ()
        return tuple(shape_bins)

    @property
    def axis_aliases(self) -> tuple[str, ...]:
        axis_aliases = self._axis_aliases
        if axis_aliases is None:
            return ()
        return tuple(axis_aliases)

    @property
    def grid(self) -> np.ndarray:
        return np.asarray(self).reshape(self.shape_bins)

    def reshape_bins(self, copy: bool = False) -> np.ndarray:
        arr = np.asarray(self)
        if copy:
            arr = arr.copy()
        return arr.reshape(self.shape_bins)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot this 1-D per-bin array against the first axis's bin centers.

        Parameters
        ----------
        ax:
            Matplotlib axes object.  A new figure/axes is created if ``None``.
        **kwargs:
            Extra keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        list[Line2D]
            The line objects returned by ``ax.plot`` (one line for a 1-D array).

        Raises
        ------
        ValueError
            If the owning result is not one-dimensional.
        """
        import matplotlib.pyplot as plt

        bins = self._bins
        if bins is None or bins.ndim != 1:
            raise ValueError("BinsArray.plot requires a one-dimensional bin result.")

        if ax is None:
            _, ax = plt.subplots()
        x = np.asarray(bins.centers)
        y = np.asarray(self)
        return ax.plot(x, y, **kwargs)
