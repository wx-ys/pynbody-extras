"""Visualization mixin for BinNDResult.

Extracted from result.py to reduce the size of the god class.
``BinNDResult`` inherits from :class:`BinPlotMixin`; no public API changes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .arrays import BinsArray
    from .axes import BinAxis


@runtime_checkable
class _BinQueryable(Protocol):
    """Minimal interface required by :class:`BinPlotMixin`.

    Depends only on the public API of :class:`~.result.BinNDResult`; no
    private methods are accessed.  Any object satisfying this protocol can
    inherit :class:`BinPlotMixin` without importing the concrete class.
    """

    axes: tuple[BinAxis, ...]
    shape_bins: tuple[int, ...]
    ndim: int

    def axis(self, key: int | str) -> BinAxis: ...
    def __getitem__(self, key: str) -> BinsArray: ...


class BinPlotMixin:
    """Provides ``plot`` and ``imshow`` methods for :class:`~.result.BinNDResult`."""

    def plot(
        self: _BinQueryable,
        x: str,
        y: str,
        ax: Any = None,
        *,
        kind: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot a 1-D profile.

        Parameters
        ----------
        x:
            Axis alias whose bin centers are used as the x-coordinate.
        y:
            String query for the y-data (e.g. ``"density"``, ``"mass.sum"``).
        ax:
            Matplotlib axes object.  A new figure/axes is created if ``None``.
        kind:
            Plot style: ``"plot"`` (default) or ``"scatter"``.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        x_axis = self.axis(x)
        x_values = np.asarray(x_axis.centers)
        y_arr = np.asarray(self[y]).ravel()

        plot_kind = "plot" if kind is None else kind
        if plot_kind == "scatter":
            return ax.scatter(x_values, y_arr, **kwargs)
        if plot_kind == "plot":
            return ax.plot(x_values, y_arr, **kwargs)
        raise ValueError(f"Unknown plot kind {plot_kind!r}; use 'plot' or 'scatter'.")

    def imshow(
        self: _BinQueryable,
        field: str,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Show a 2-D bin grid as an image.

        Requires exactly two axes.  The first axis maps to the x-direction and
        the second to the y-direction::

            bins2d.imshow("mass.sum")

        Parameters
        ----------
        field:
            String query for the field to display.
        ax:
            Matplotlib axes object.  A new figure/axes is created if ``None``.
        """
        import matplotlib.pyplot as plt

        if self.ndim != 2:
            raise ValueError("imshow requires exactly 2 bin axes.")
        if ax is None:
            _, ax = plt.subplots()

        x_axis, y_axis = self.axes
        extent = [
            float(np.asarray(x_axis.mins)[0]),
            float(np.asarray(x_axis.maxs)[-1]),
            float(np.asarray(y_axis.mins)[0]),
            float(np.asarray(y_axis.maxs)[-1]),
        ]
        grid = np.asarray(self[field]).reshape(self.shape_bins)
        return ax.imshow(grid.T, origin="lower", aspect="auto", extent=extent, **kwargs)
