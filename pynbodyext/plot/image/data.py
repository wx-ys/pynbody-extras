"""``ImageData``: a 2-D array plus the metadata needed to display and measure it.

Everything in :mod:`pynbodyext.plot.image` works on plain arrays, but a map
carries more than numbers: the physical extent of the axes, the units, and which
axes the two directions correspond to.  :class:`ImageData` keeps those together
and bridges the calculator layer::

    from pynbodyext.plot import image

    density = image.ImageData.from_bins(bins2d, "mass.sum")
    smoothed = density.with_data(image.gaussian_smooth(density.data, fwhm=0.5, pixel_scale=density.pixel_size))
    smoothed.imshow()
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["ImageData"]


@dataclass(frozen=True)
class ImageData:
    """A 2-D image with the metadata needed to display it.

    Parameters
    ----------
    data : array_like
        2-D array of pixel values; the first axis is the y-direction, the second
        the x-direction (the convention of ``matplotlib.imshow``).
    extent : (float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` of the image, matching
        ``matplotlib.axes.Axes.imshow(extent=...)``.
    units : object, optional
        Units of the values, e.g. a pynbody unit.
    label : str, optional
        Human-readable name of the quantity, used to label a figure.
    x_axis, y_axis : str, optional
        Names of the two axes, carried through from a binned result.

    Examples
    --------
    >>> image = ImageData(np.zeros((4, 5)), extent=(0, 5, 0, 4), label="mass.sum")
    >>> image.shape
    (4, 5)
    >>> image.pixel_size
    (1.0, 1.0)
    """

    data: np.ndarray
    extent: tuple[float, float, float, float] | None = None
    units: Any = None
    label: str | None = None
    x_axis: str | None = None
    y_axis: str | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim != 2:
            raise ValueError(f"ImageData expects a 2-D array, got shape {array.shape}.")
        object.__setattr__(self, "data", array)
        if self.extent is not None:
            if len(self.extent) != 4:
                raise ValueError(f"extent must be (xmin, xmax, ymin, ymax), got {self.extent!r}.")
            object.__setattr__(self, "extent", tuple(float(bound) for bound in self.extent))

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the image."""
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @property
    def ndim(self) -> int:
        """Number of dimensions: always 2."""
        return 2

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Physical size of one pixel as ``(dy, dx)``.

        In the units of :attr:`extent`, and ready to pass as ``pixel_scale`` to the
        smoothing and PSF helpers.

        Raises
        ------
        ValueError
            If the image has no :attr:`extent`.
        """
        if self.extent is None:
            raise ValueError("ImageData.pixel_size needs an extent; this image has none.")
        xmin, xmax, ymin, ymax = self.extent
        return ((ymax - ymin) / self.shape[0], (xmax - xmin) / self.shape[1])

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
            Any other field to replace, e.g. ``label``.

        Returns
        -------
        ImageData
            The new image.
        """
        replacement = np.asarray(data)
        if replacement.shape != self.shape:
            raise ValueError(f"New data has shape {replacement.shape}, expected {self.shape}.")
        return replace(self, data=replacement, **overrides)

    @classmethod
    def from_bins(cls, bins: Any, query: str, *, label: str | None = None, units: Any = None) -> ImageData:
        """Build an image from one query of a 2-D :class:`BinNDResult`.

        Parameters
        ----------
        bins : BinNDResult
            A binned result with exactly two axes.
        query : str
            Query to display, e.g. ``"mass.sum"`` or ``"vz.mean"``.
        label : str, optional
            Name to display instead of *query*.
        units : object, optional
            Units to record instead of the ones carried by the binned array.

        Returns
        -------
        ImageData
            Values on the bin grid, with the axes' physical extent.

        Examples
        --------
        >>> image = ImageData.from_bins(bins2d, "mass.sum")  # doctest: +SKIP
        >>> image.imshow()  # doctest: +SKIP
        """
        ndim = getattr(bins, "ndim", None)
        if ndim != 2:
            raise ValueError(f"from_bins needs a 2-D binned result, got ndim={ndim}.")
        array = bins[query]
        axes = bins.axes
        return cls(
            data=np.asarray(array.grid),
            extent=tuple(axes.extent),
            units=getattr(array, "units", None) if units is None else units,
            label=query if label is None else label,
            x_axis=axes[0].alias,
            y_axis=axes[1].alias,
        )

    def imshow(self, ax: Any = None, **kwargs: Any) -> Any:
        """Draw the image with matplotlib, labelling it from the metadata.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        **kwargs
            Forwarded to ``matplotlib.axes.Axes.imshow``; ``extent`` defaults to
            :attr:`extent` and ``origin`` to ``"lower"``.

        Returns
        -------
        matplotlib.image.AxesImage
            The artist.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (5.0, 5.0)))
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("extent", self.extent)
        artist = ax.imshow(self.data, **kwargs)
        if self.label is not None and not ax.get_ylabel():
            ax.set_ylabel(self.label if self.units is None else f"{self.label} [{self.units}]")
        return artist
