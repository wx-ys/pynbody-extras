"""``ImageData``: a 2-D array plus the metadata and the tools to work on it.

Most of this package operates on plain arrays, but a map carries more than
numbers: where its axes sit, what they are measured in, what to call them, what
the values are, and how they were produced.  :class:`ImageData` keeps those
together and is the entry point for the whole image layer::

    from pynbodyext.plot import image

    velocity = image.ImageData.from_bins(bins2d, "vz.mean")
    velocity.smooth.gaussian(fwhm=2.0).psf.convolve(fwhm=3.0).imshow(colorbar=True)

How it is put together:

- :class:`ImageData` itself owns the values, the geometry, the metadata and the
  provenance (``.ops``).
- The *families* of operations live in one mixin each, next to the free functions
  they wrap: :class:`~pynbodyext.plot.image.postprocess.SmoothMixin` (``.smooth``),
  :class:`~pynbodyext.plot.image.psf.PsfMixin` (``.psf``),
  :class:`~pynbodyext.plot.image.compose.ComposeMixin` (``create_mask``,
  ``compose``), :class:`~pynbodyext.plot.image.adaptive.AdaptiveMixin`
  (``adaptive_bin``) and :class:`~pynbodyext.plot.image.display.DisplayMixin`
  (``normalize``, ``to_rgba``, ``draw``, ``imshow``, ``pcolormesh``).
  Single-call, everyday operations are plain methods; families with several
  variants are grouped behind an accessor (``image.smooth.gaussian(...)``), so no
  operation has two spellings.
- Methods that produce an image return a new :class:`ImageData` with the
  operation appended to :attr:`ops`; everything else returns what the free
  function returns (an array, a mask pair, an
  :class:`~pynbodyext.plot.image.adaptive.AdaptiveMap`).  The free functions stay
  the implementation and the tested contract.

Three things vary between maps, and all three are supported here:

1. **The simple map** — both axes in the same units, evenly spaced (linear) bins.
   Give the ``extent`` and, if you like, one set of units and labels.
2. **Per-axis metadata** — ``x_units``/``y_units`` and ``x_label``/``y_label`` are
   independent, so an image whose axes are, say, ``kpc`` and ``K`` is ordinary.
3. **Bin edges that are not evenly spaced** — give ``x_edges``/``y_edges``
   (logarithmic, quantile, or any explicit edges) and the image knows its real
   geometry: ``pcolormesh``/``draw`` place the bins correctly, and anything that
   needs a single pixel size (``pixel_size``, ``imshow``) refuses rather than lying.

Pass either ``extent`` or both ``x_edges``/``y_edges``; the ``extent`` is derived
from the edges when only the latter are given.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._arrays import bin_centers, edges_are_uniform, pixel_width, resolve_edges
from .adaptive import AdaptiveMixin
from .compose import ComposeMixin
from .display import DisplayMixin, _unit_text
from .postprocess import SmoothMixin
from .psf import PsfMixin

__all__ = ["ImageData", "ImageOp"]


def _describe(value: Any) -> str:
    """Compact rendering of one recorded operation parameter."""
    if isinstance(value, np.ndarray):
        return f"<array {value.shape}>"
    if isinstance(value, ImageData):
        return f"<ImageData {value.shape}>"
    return repr(value)


@dataclass(frozen=True)
class ImageOp:
    """One operation applied to an image, recorded for provenance.

    Parameters
    ----------
    name : str
        Name of the free function that did the work, e.g. ``"gaussian_smooth"``.
    params : dict
        The arguments it was called with; treat as read-only.

    Examples
    --------
    >>> op = ImageOp("gaussian_smooth", {"fwhm": 2.0})
    >>> repr(op)
    'gaussian_smooth(fwhm=2.0)'
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        arguments = ", ".join(f"{key}={_describe(value)}" for key, value in self.params.items())
        return f"{self.name}({arguments})"


@dataclass(frozen=True)
class ImageData(SmoothMixin, PsfMixin, ComposeMixin, AdaptiveMixin, DisplayMixin):
    """A 2-D image with the metadata needed to display, measure and process it.

    Parameters
    ----------
    data : array_like
        2-D array of values; the first axis is the y-direction, the second the
        x-direction (the convention of ``matplotlib.imshow``).
    extent : (float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` of a uniformly sampled image.  Shorthand for
        evenly spaced bin edges, and mutually exclusive with *x_edges*/*y_edges*
        (but derived from them when only those are given).
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
    ops : tuple of ImageOp, optional
        Operations already applied, oldest first.  Filled in by the processing
        methods; plain construction starts empty.

    Examples
    --------
    The simple map — one extent, one set of units:

    >>> image = ImageData(np.zeros((4, 5)), extent=(0, 5, 0, 4), x_units="kpc", y_units="kpc")
    >>> image.shape
    (4, 5)
    >>> image.pixel_size
    (1.0, 1.0)
    >>> image.imshow(colorbar=True)  # doctest: +SKIP

    The general map — per-axis units and logarithmic bins:

    >>> image = ImageData(np.zeros((3, 3)), x_edges=[1, 2, 4, 8], x_units="kpc", y_edges=[0, 1, 2, 3], y_units="K")
    >>> image.x_centers
    array([1.5, 3. , 6. ])
    >>> image.uniform
    False
    >>> image.draw()  # picks pcolormesh on its own  # doctest: +SKIP
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
    ops: tuple[ImageOp, ...] = ()

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
        object.__setattr__(self, "ops", tuple(self.ops))

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

    # ------------------------------------------------------------------
    # derivation
    # ------------------------------------------------------------------

    def _derived(self, data: Any, op_name: str, params: dict[str, Any] | None = None, **overrides: Any) -> ImageData:
        """Return a copy carrying *data*, with *op_name* appended to :attr:`ops`.

        Used by the processing methods; everything that changes shape must pass
        its own ``x_edges``/``y_edges`` through ``overrides``.
        """
        recorded = {key: value for key, value in (params or {}).items() if value is not None}
        overrides.setdefault("ops", (*self.ops, ImageOp(op_name, recorded)))
        return replace(self, data=np.asarray(data), **overrides)

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
            The new image, with the same provenance as this one.
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

        The bin grid arrives as ``(x, y)`` — ``bins.shape_bins`` follows the axis
        order — and is transposed into the image convention of rows ``= y``,
        columns ``= x``.  Bin edges, per-axis units and axis names come from the
        binned result, so non-uniform bins (``mode="log"``, ``equaln``, explicit
        ``edges``) and axes measured in different units both work directly.

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
            The query as an image, with the geometry of that grid.

        Raises
        ------
        ValueError
            If the result is not 2-D, or an axis has gaps between its bins (such
            a grid is not an image).

        Examples
        --------
        >>> image = ImageData.from_bins(bins2d, "mass.sum")  # doctest: +SKIP
        >>> image.draw()  # doctest: +SKIP
        """
        ndim = getattr(bins, "ndim", None)
        if ndim != 2:
            raise ValueError(f"from_bins needs a 2-D binned result, got ndim={ndim}.")
        axes = bins.axes
        edges = [_axis_edges(axis, index) for index, axis in enumerate(axes)]
        array = bins[query]
        return cls(
            data=np.asarray(array.grid).T,  # (x, y) grid -> (row=y, column=x)
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
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"shape={self.shape}"]
        if self.extent is not None:
            parts.append(f"extent={self.extent}")
        if self.label is not None or self.units is not None:
            unit_text = _unit_text(self.units)
            parts.append(f"label={self.label!r}" if unit_text is None else f"label={self.label!r} [{unit_text}]")
        if self.ops:
            parts.append("ops=" + " → ".join(repr(op) for op in self.ops))
        return f"ImageData({', '.join(parts)})"


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
