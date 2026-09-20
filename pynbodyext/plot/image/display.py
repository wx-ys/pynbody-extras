"""Drawing an image: the display and colour side of :class:`ImageData`.

Kept out of ``data.py`` so that the data object stays about geometry, metadata and
provenance, and every matplotlib detail lives in one place: axis labels, colour
bars, and the choice between ``imshow`` and ``pcolormesh``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._arrays import edges_are_uniform
from .cmaps import K_B_C_G_Y_R_W, get_cmap, to_rgba
from .postprocess import normalize

if TYPE_CHECKING:
    from .data import ImageData

__all__ = ["DisplayMixin"]

#: Unit spellings that mean "no units at all", so they never reach a figure label.
_EMPTY_UNITS = {"", "1", "NoUnit()", "dimensionless", "unitless"}


def _unit_text(units: Any) -> str | None:
    """Render *units* for a figure label, or ``None`` when there is nothing to show."""
    if units is None or type(units).__name__ == "NoUnit":
        return None
    text = str(units)
    return None if text in _EMPTY_UNITS else text


def _annotation(label: str | None, units: Any) -> str | None:
    """Combine a quantity label and its units for an axis or colour bar."""
    unit_text = _unit_text(units)
    if label is None:
        return None if unit_text is None else f"[{unit_text}]"
    return str(label) if unit_text is None else f"{label} [{unit_text}]"


def _apply_axis_labels(ax: Any, image: ImageData) -> None:
    """Label the axes from the image metadata, without overwriting the caller's labels."""
    if not ax.get_xlabel():
        annotation = _annotation(image.x_label, image.x_units)
        if annotation is not None:
            ax.set_xlabel(annotation)
    if not ax.get_ylabel():
        annotation = _annotation(image.y_label, image.y_units)
        if annotation is not None:
            ax.set_ylabel(annotation)


class DisplayMixin:
    """Display and colour methods of :class:`~pynbodyext.plot.image.data.ImageData`.

    The mixin reads the image's own attributes; it holds no state of its own.
    """

    if TYPE_CHECKING:
        data: np.ndarray
        extent: tuple[float, float, float, float] | None
        x_edges: np.ndarray | None
        y_edges: np.ndarray | None
        label: str | None
        units: Any
        x_units: Any
        y_units: Any
        x_label: str | None
        y_label: str | None

        def _derived(
            self, data: Any, op_name: str, params: dict[str, Any] | None = None, **overrides: Any
        ) -> ImageData: ...

    def normalize(
        self,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        stretch: str = "linear",
        percentiles: tuple[float, float] | None = None,
        asinh_a: float = 10.0,
    ) -> ImageData:
        """Map the values to ``[0, 1]`` for display, keeping the geometry.

        Parameters
        ----------
        vmin, vmax, stretch, percentiles, asinh_a :
            As in :func:`~pynbodyext.plot.image.postprocess.normalize`.  Unlike the
            free function, this returns an image (so it can be chained) rather
            than a bare array.

        Returns
        -------
        ImageData
            The stretched image; non-finite pixels stay non-finite.

        Examples
        --------
        >>> stretched = image.normalize(stretch="asinh", percentiles=(1, 99))  # doctest: +SKIP
        """
        stretched = normalize(
            self.data, vmin=vmin, vmax=vmax, stretch=stretch, percentiles=percentiles, asinh_a=asinh_a
        )
        return self._derived(
            stretched, "normalize", {"vmin": vmin, "vmax": vmax, "stretch": stretch, "percentiles": percentiles}
        )

    def to_rgba(
        self,
        cmap: Any = None,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        stretch: str = "linear",
        percentiles: tuple[float, float] | None = None,
        norm: Any = None,
        alpha: Any = None,
        bad: Any = None,
    ) -> np.ndarray:
        """Map the values to an ``(ny, nx, 4)`` RGBA array.

        Parameters
        ----------
        cmap : str or Colormap, optional
            Colour map; defaults to the velocity map ``K_B_C_G_Y_R_W``.
        vmin, vmax, stretch, percentiles, norm, alpha, bad :
            As in :func:`~pynbodyext.plot.image.cmaps.to_rgba`.

        Returns
        -------
        numpy.ndarray
            Float RGBA image, with non-finite pixels transparent by default.
        """
        return to_rgba(
            self.data,
            get_cmap(cmap) if cmap is not None else K_B_C_G_Y_R_W,
            vmin=vmin,
            vmax=vmax,
            stretch=stretch,
            percentiles=percentiles,
            norm=norm,
            alpha=alpha,
            bad=bad,
        )

    def draw(self, ax: Any = None, *, colorbar: bool = False, aspect: Any = None, **kwargs: Any) -> Any:
        """Draw the image, picking the right artist for the bin spacing.

        Evenly spaced bins go through :meth:`imshow`; unevenly spaced ones through
        :meth:`pcolormesh`, so a logarithmic or quantile grid is drawn where its
        bins really are.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        colorbar : bool, default: False
            Add a colour bar labelled with the image's ``label`` and ``units``.
        aspect : optional
            Axes aspect, e.g. ``"auto"``; matplotlib's default when omitted.
        **kwargs
            Forwarded to the chosen artist.

        Returns
        -------
        matplotlib.image.AxesImage or matplotlib.collections.QuadMesh
            The artist.
        """
        uniform = edges_are_uniform(getattr(self, "x_edges", None)) and edges_are_uniform(
            getattr(self, "y_edges", None)
        )
        draw = self.imshow if uniform else self.pcolormesh
        return draw(ax=ax, colorbar=colorbar, aspect=aspect, **kwargs)

    def imshow(self, ax: Any = None, *, colorbar: bool = False, aspect: Any = None, **kwargs: Any) -> Any:
        """Draw the image with ``imshow``, labelling it from the metadata.

        Use this for evenly spaced bins; :meth:`pcolormesh` handles arbitrary bin
        edges, and :meth:`draw` picks between them for you.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        colorbar : bool, default: False
            Add a colour bar labelled with the image's ``label`` and ``units``.
        aspect : optional
            Axes aspect, e.g. ``"auto"``.
        **kwargs
            Forwarded to ``matplotlib.axes.Axes.imshow``; ``extent`` defaults to
            the image's own and ``origin`` to ``"lower"``.

        Returns
        -------
        matplotlib.image.AxesImage
            The artist.
        """
        import matplotlib.pyplot as plt

        if not (edges_are_uniform(self.x_edges) and edges_are_uniform(self.y_edges)):
            raise ValueError("These bins are not evenly spaced; use pcolormesh() or draw() instead.")
        figsize = kwargs.pop("figsize", (5.0, 5.0))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("extent", self.extent)
        artist = ax.imshow(self.data, **kwargs)
        _finish(ax, self, artist, colorbar, aspect)
        return artist

    def pcolormesh(
        self, ax: Any = None, *, colorbar: bool = False, aspect: Any = None, shading: str = "flat", **kwargs: Any
    ) -> Any:
        """Draw the image as quadrilateral cells, honouring arbitrary bin edges.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        colorbar : bool, default: False
            Add a colour bar labelled with the image's ``label`` and ``units``.
        aspect : optional
            Axes aspect, e.g. ``"auto"``.
        shading : str, default: "flat"
            Matplotlib shading mode; ``"flat"`` pairs the data with the given edges.
        **kwargs
            Forwarded to ``matplotlib.axes.Axes.pcolormesh``.

        Returns
        -------
        matplotlib.collections.QuadMesh
            The artist.
        """
        import matplotlib.pyplot as plt

        figsize = kwargs.pop("figsize", (5.0, 5.0))
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        x_edges = self.x_edges if self.x_edges is not None else np.arange(self.data.shape[1] + 1, dtype=float)
        y_edges = self.y_edges if self.y_edges is not None else np.arange(self.data.shape[0] + 1, dtype=float)
        artist = ax.pcolormesh(x_edges, y_edges, self.data, shading=shading, **kwargs)
        _finish(ax, self, artist, colorbar, aspect)
        return artist


def _finish(ax: Any, image: Any, artist: Any, colorbar: bool, aspect: Any) -> None:
    """Shared tail of the drawing methods: labels, aspect and colour bar."""
    _apply_axis_labels(ax, image)
    if aspect is not None:
        ax.set_aspect(aspect)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, label=_annotation(image.label, image.units))
