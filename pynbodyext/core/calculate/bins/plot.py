"""Visualization mixin for BinNDResult.

Extracted from result.py to reduce the size of the god class.
``BinNDResult`` inherits from :class:`BinPlotMixin`; no public API changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

    from .arrays import BinsArray
    from .axes import BinAxisAccessor
    from .result import BinNDResult


@runtime_checkable
class _BinQueryable(Protocol):
    """Minimal interface required by :class:`BinPlotMixin`.

    Depends only on the public API of :class:`~.result.BinNDResult`; no
    private methods are accessed.  Any object satisfying this protocol can
    inherit :class:`BinPlotMixin` without importing the concrete class.
    """

    axes: BinAxisAccessor
    shape_bins: tuple[int, ...]
    ndim: int

    def __getitem__(self, key: str) -> BinsArray: ...


class BinPlotMixin:
    """Provides ``plot`` and ``imshow`` methods for :class:`~.result.BinNDResult`."""

    def plot(
        self: _BinQueryable, x: str, y: str, ax: Axes | None = None, *, kind: str | None = None, **kwargs: Any
    ) -> list[Line2D] | PathCollection:
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

        Returns
        -------
        list[Line2D] or PathCollection
            The matplotlib artists returned by ``ax.plot`` / ``ax.scatter``.

        Examples
        --------
        >>> bins.plot("r", "mass.sum")  # profile of mass.sum vs bin centers
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> lines = bins.plot("r", "density", ax=ax, kind="scatter")
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        x_axis = self.axes[x]
        x_values = np.asarray(x_axis.centers)
        y_arr = np.asarray(self[y]).ravel()

        plot_kind = "plot" if kind is None else kind
        if plot_kind == "scatter":
            return ax.scatter(x_values, y_arr, **kwargs)
        if plot_kind == "plot":
            return ax.plot(x_values, y_arr, **kwargs)
        raise ValueError(f"Unknown plot kind {plot_kind!r}; use 'plot' or 'scatter'.")

    def imshow(self: _BinQueryable, field: str, ax: Axes | None = None, **kwargs: Any) -> AxesImage | QuadMesh:
        """Show a 2-D bin grid as an image.

        Requires exactly two axes, the first mapping to x and the second to y::

            bins2d.imshow("mass.sum")

        The drawing itself belongs to :mod:`pynbodyext.plot.image`: this method
        builds an :class:`~pynbodyext.plot.image.ImageData` from the query — which
        carries the bin edges, units and axis names of the result — and lets it
        draw.  Logarithmic or otherwise uneven bins therefore come out as a
        ``pcolormesh`` instead of being forced onto a regular pixel grid.

        Parameters
        ----------
        field:
            String query for the field to display.
        ax:
            Matplotlib axes object.  A new figure/axes is created if ``None``.
        **kwargs:
            Forwarded to the image artist, e.g. ``cmap``, ``colorbar``,
            ``aspect`` (``"auto"`` by default, as for the older implementation),
            or ``figsize`` when *ax* is ``None``.

        Returns
        -------
        AxesImage or QuadMesh
            The artist drawing the grid.

        Raises
        ------
        ValueError
            If the result does not have exactly two bin axes.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> im = bins2d.imshow("mass.sum", ax=ax)
        """
        from pynbodyext.plot.image import ImageData

        if self.ndim != 2:
            raise ValueError("imshow requires exactly 2 bin axes.")
        kwargs.setdefault("aspect", "auto")
        return ImageData.from_bins(cast("BinNDResult", self), field).display.draw(ax=ax, **kwargs)
