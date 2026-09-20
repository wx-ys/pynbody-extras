"""``ImageData``: a 2-D array plus the metadata needed to display and measure it.

Everything in :mod:`pynbodyext.plot.image` works on plain arrays, but a map
carries more than numbers: where its axes sit, what they are measured in, what to
call them, and what the values are.  :class:`ImageData` keeps those together and
bridges the calculator layer::

    from pynbodyext.plot import image

    density = image.ImageData.from_bins(bins2d, "mass.sum")
    smoothed = density.with_data(image.gaussian_smooth(density.data, fwhm=0.5, pixel_scale=density.pixel_size))
    smoothed.imshow(colorbar=True)

Three things vary between maps, and all three are supported here:

1. **The simple map** — both axes in the same units, evenly spaced (linear) bins.
   Give the ``extent`` and, if you like, one set of units and labels.
2. **Per-axis metadata** — ``x_units``/``y_units`` and ``x_label``/``y_label`` are
   independent, so an image whose axes are, say, ``kpc`` and ``K`` is ordinary.
3. **Bin edges that are not evenly spaced** — give ``x_edges``/``y_edges``
   (logarithmic, quantile, or any explicit edges) and the image knows its real
   geometry: ``pcolormesh`` draws it correctly, and anything that needs a single
   pixel size (``pixel_size``, ``imshow``) refuses rather than lying.

Pass either ``extent`` or both ``x_edges``/``y_edges``; the ``extent`` is derived
from the edges when only the latter are given.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._arrays import bin_centers, edges_are_uniform, pixel_width, resolve_edges

__all__ = ["ImageData"]

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


@dataclass(frozen=True)
class ImageData:
    """A 2-D image with the metadata needed to display it.

    Parameters
    ----------
    data : array_like
        2-D array of values; the first axis is the y-direction, the second the
        x-direction (the convention of ``matplotlib.imshow``).
    extent : (float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` of a uniformly sampled image, matching
        ``matplotlib.axes.Axes.imshow(extent=...)``.  Mutually exclusive with
        *x_edges*/*y_edges* (but derived from them when only those are given).
    x_edges, y_edges : array_like, optional
        Bin edges along each direction, with ``len(edges) == n + 1`` for ``n``
        columns or rows.  Edges need not be evenly spaced.  Give both or neither.
    x_units, y_units : object, optional
        Units of each axis; they may differ, and each is shown next to the axis
        label.  Without them the axes are taken to be pixel indices.
    x_label, y_label : str, optional
        Names of the two axes.
    label : str, optional
        Name of the quantity the values represent (``"mass.sum"``, ``"vz.mean"``).
    units : object, optional
        Units of the values, shown on the colour bar.

    Examples
    --------
    The simple map — one extent, one set of units:

    >>> image = ImageData(np.zeros((4, 5)), extent=(0, 5, 0, 4), x_units="kpc", y_units="kpc")
    >>> image.shape
    (4, 5)
    >>> image.pixel_size
    (1.0, 1.0)

    The general map — per-axis units and logarithmic bins:

    >>> image = ImageData(np.zeros((3, 3)), x_edges=[1, 2, 4, 8], x_units="kpc", y_edges=[0, 1, 2, 3], y_units="K")
    >>> image.x_centers
    array([1.5, 3. , 6. ])
    >>> image.uniform
    False
    """

    data: np.ndarray
    extent: tuple[float, float, float, float] | None = None
    x_edges: np.ndarray | None = None
    y_edges: np.ndarray | None = None
    x_units: Any = None
    y_units: Any = None
    x_label: str | None = None
    y_label: str | None = None
    label: str | None = None
    units: Any = None

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 2:
            raise ValueError(f"ImageData expects a 2-D array, got shape {array.shape}.")
        object.__setattr__(self, "data", array)
        x_edges, y_edges = resolve_edges(array.shape, extent=self.extent, x_edges=self.x_edges, y_edges=self.y_edges)
        object.__setattr__(self, "x_edges", x_edges)
        object.__setattr__(self, "y_edges", y_edges)
        span = None
        if x_edges is not None and y_edges is not None:
            span = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
        object.__setattr__(self, "extent", span)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the image."""
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @property
    def ndim(self) -> int:
        """Number of dimensions: always 2."""
        return 2

    @property
    def x_centers(self) -> np.ndarray:
        """Centre of every column, in the units of the x axis."""
        return bin_centers(self.x_edges, self.shape[1])

    @property
    def y_centers(self) -> np.ndarray:
        """Centre of every row, in the units of the y axis."""
        return bin_centers(self.y_edges, self.shape[0])

    @property
    def x_uniform(self) -> bool:
        """Whether the columns are evenly spaced (pixel indices count as even)."""
        return edges_are_uniform(self.x_edges)

    @property
    def y_uniform(self) -> bool:
        """Whether the rows are evenly spaced (pixel indices count as even)."""
        return edges_are_uniform(self.y_edges)

    @property
    def uniform(self) -> bool:
        """Whether both axes are evenly spaced, i.e. whether ``imshow`` applies."""
        return self.x_uniform and self.y_uniform

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Size of one pixel as ``(dy, dx)``, in the units of the axes.

        Ready to pass as ``pixel_scale`` to the smoothing and PSF helpers.  Works
        when the bins are evenly spaced (or absent, in which case the axes are
        pixel indices and the size is 1).

        Raises
        ------
        ValueError
            If a direction has bin edges that are not evenly spaced, where no
            single pixel size exists.
        """
        return (pixel_width(self.y_edges, self.shape[0]), pixel_width(self.x_edges, self.shape[1]))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Expose the raw values, so ``np.asarray(image)`` does the obvious thing."""
        return np.array(self.data, dtype=dtype, copy=copy)

    def with_data(self, data: Any, **overrides: Any) -> ImageData:
        """Return a copy with new values, keeping (or overriding) the metadata.

        Parameters
        ----------
        data : array_like
            Replacement values, with the same shape as this image.
        **overrides
            Any other field to replace, e.g. ``label`` or ``y_units``.

        Returns
        -------
        ImageData
            The new image.
        """
        replacement = np.asarray(data)
        if replacement.shape != self.shape:
            raise ValueError(f"New data has shape {replacement.shape}, expected {self.shape}.")
        return replace(self, data=replacement, **overrides)

    # ------------------------------------------------------------------
    # calculator bridge
    # ------------------------------------------------------------------

    @classmethod
    def from_bins(
        cls,
        bins: Any,
        query: str,
        *,
        label: str | None = None,
        units: Any = None,
        x_label: str | None = None,
        y_label: str | None = None,
        x_units: Any = None,
        y_units: Any = None,
    ) -> ImageData:
        """Build an image from one query of a 2-D :class:`BinNDResult`.

        The bin edges, per-axis units and axis names come from the binned result,
        so non-uniform bins (``mode="log"``, ``equaln``, explicit ``edges``) and
        axes measured in different units both work without further input.

        Parameters
        ----------
        bins : BinNDResult
            A binned result with exactly two axes.
        query : str
            Query to display, e.g. ``"mass.sum"`` or ``"vz.mean"``.
        label : str, optional
            Name of the values; defaults to *query*.
        units : object, optional
            Units of the values; default to the ones the binned array carries.
        x_label, y_label, x_units, y_units : optional
            Overrides for the metadata read from the axes.

        Returns
        -------
        ImageData
            The query on the bin grid, with the geometry of that grid.

        Raises
        ------
        ValueError
            If the result is not 2-D, or an axis has gaps between its bins (such
            a grid is not an image).

        Examples
        --------
        >>> image = ImageData.from_bins(bins2d, "mass.sum")  # doctest: +SKIP
        >>> image.pcolormesh()  # doctest: +SKIP
        """
        ndim = getattr(bins, "ndim", None)
        if ndim != 2:
            raise ValueError(f"from_bins needs a 2-D binned result, got ndim={ndim}.")
        axes = bins.axes
        edges = [_axis_edges(axis, index) for index, axis in enumerate(axes)]
        array = bins[query]
        return cls(
            data=np.asarray(array.grid),
            x_edges=edges[0],
            y_edges=edges[1],
            x_units=axes[0].units if x_units is None else x_units,
            y_units=axes[1].units if y_units is None else y_units,
            x_label=_axis_name(axes[0]) if x_label is None else x_label,
            y_label=_axis_name(axes[1]) if y_label is None else y_label,
            label=query if label is None else label,
            units=getattr(array, "units", None) if units is None else units,
        )

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def imshow(self, ax: Any = None, *, colorbar: bool = False, **kwargs: Any) -> Any:
        """Draw the image with matplotlib, labelling it from the metadata.

        Use this for evenly spaced bins; for arbitrary bin edges use
        :meth:`pcolormesh` instead.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        colorbar : bool, default: False
            Add a colour bar labelled with :attr:`label` and :attr:`units`.
        **kwargs
            Forwarded to ``matplotlib.axes.Axes.imshow``; ``extent`` defaults to
            :attr:`extent` and ``origin`` to ``"lower"``.

        Returns
        -------
        matplotlib.image.AxesImage
            The artist.
        """
        import matplotlib.pyplot as plt

        if not self.uniform:
            raise ValueError("These bins are not evenly spaced; use pcolormesh() to draw them correctly.")
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (5.0, 5.0)))
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("extent", self.extent)
        artist = ax.imshow(self.data, **kwargs)
        _apply_axis_labels(ax, self)
        if colorbar:
            ax.figure.colorbar(artist, ax=ax, label=_annotation(self.label, self.units))
        return artist

    def pcolormesh(self, ax: Any = None, *, colorbar: bool = False, shading: str = "flat", **kwargs: Any) -> Any:
        """Draw the image as quadrilateral cells, honouring arbitrary bin edges.

        Unlike :meth:`imshow`, this places every bin where it really is, so it is
        the right call whenever the bin widths are uneven (logarithmic, quantile,
        explicit edges).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        colorbar : bool, default: False
            Add a colour bar labelled with :attr:`label` and :attr:`units`.
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

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (5.0, 5.0)))
        x_edges = self.x_edges if self.x_edges is not None else np.arange(self.shape[1] + 1, dtype=float)
        y_edges = self.y_edges if self.y_edges is not None else np.arange(self.shape[0] + 1, dtype=float)
        artist = ax.pcolormesh(x_edges, y_edges, self.data, shading=shading, **kwargs)
        _apply_axis_labels(ax, self)
        if colorbar:
            ax.figure.colorbar(artist, ax=ax, label=_annotation(self.label, self.units))
        return artist


def _axis_edges(axis: Any, index: int) -> np.ndarray:
    """Edges of a binned axis, rejecting grids whose bins do not tile the axis."""
    if not getattr(axis, "is_continuous", False):
        raise ValueError(
            f"Axis {getattr(axis, 'alias', index)!r} has gaps between its bins, so its grid is not contiguous "
            "and cannot be drawn as an image."
        )
    edges = np.asarray(axis.edges, dtype=float)
    if edges.ndim != 1:
        raise ValueError(f"Axis {getattr(axis, 'alias', index)!r} does not have a single set of edges.")
    return edges


def _axis_name(axis: Any) -> str:
    """Display name of an axis: its property when that is a plain name, else its alias."""
    prop = getattr(axis, "prop", None)
    return prop if isinstance(prop, str) else str(getattr(axis, "alias", ""))


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
