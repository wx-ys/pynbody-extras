"""Adaptive binning of a noisy map by signal strength.

A fine regular grid (a ``BinND`` map, an SPH render, a tessellation) shows a
quantity per pixel — say ``vz.mean`` — that is noisy where little mass sits and
clean where there is a lot.  Instead of smoothing over that uneven noise, this
module merges pixels into regions of comparable *capacity* with PowerBin
(Cappellari 2025, MNRAS 544, 1432: adaptive binning with centroidal power
diagrams, the modern successor of Voronoi binning), and paints every pixel with
the value of its region::

    from pynbodyext.plot import image

    binned = image.adaptive_map_from_bins(bins2d, "vz.mean", "mass.sum", target_nbins=200)
    binned.imshow(cmap="K_B_C_G_Y_R_W", symmetric=True)

The capacity is the signal itself (``mass.sum``), or ``(signal / noise)**2`` when
a noise map is supplied.  Pixels that carry no signal — empty bins (``NaN``),
masked pixels, or everything below ``min_signal`` — are left out of the binning
and stay transparent in the figure, so the result is exactly the "high-signal"
map: the faint outskirts are not painted with noise.

The tessellation itself is built in grid-cell space rather than in the units of
the axes, so it stays meaningful when the two axes are not commensurable (``kpc``
against ``K``) or when the bins are unevenly spaced: a region that looks compact
in the figure is compact, whatever the units say.

PowerBin is an optional dependency: install it with ``pip install
pynbodyext[image]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pynbodyext.util.deps import POWERBIN_AVAILABLE

from ._arrays import as_image, bin_centers, resolve_edges, typical_width
from .ops import ImageOps, register_ops

if TYPE_CHECKING:
    from .data import ImageData

__all__ = ["AdaptiveMap", "AdaptiveOps", "adaptive_bin_map", "adaptive_map_from_bins"]

_METHODS = ("mean", "median", "sum", "weighted")


@dataclass(frozen=True)
class AdaptiveMap:
    """The result of :func:`adaptive_bin_map`.

    The painted map is an ordinary :class:`~pynbodyext.plot.image.data.ImageData`
    (``.image``), so it keeps the geometry, units and labels of the grid it was
    binned from and can be processed further; the attributes below describe the
    partition itself.

    Attributes
    ----------
    image : ImageData
        The painted map: every binned pixel carries its bin's value, unbinned
        pixels stay non-finite.  ``.value``, ``.extent``, ``.x_edges`` … are
        shortcuts to it.
    value : numpy.ndarray
        Per-pixel image painted with its bin's value; non-finite outside the
        binned region.
    bin_num : numpy.ndarray of int
        Bin index of every pixel, ``-1`` outside the binned region.
    bin_value, bin_capacity, bin_count, bin_signal : numpy.ndarray
        Per-bin value, capacity, pixel count and total signal.
    xybin : numpy.ndarray
        ``(n_bins, 2)`` centres of the bins, in the units of the two axes.
    rbin : numpy.ndarray
        Effective radius of each bin.  The tessellation is built in cell space, so
        this is a cell count scaled by the typical bin width: exact for evenly
        spaced axes, indicative when the two axes have very different scales.
    mask : numpy.ndarray of bool
        Pixels that were binned.
    target_capacity : float
        Capacity each bin was asked to reach.
    method : str
        How the per-bin value was aggregated.
    rms_frac : float
        Percent scatter of the achieved bin capacities (from PowerBin).
    label, units : str, object
        Description and units of ``value``, taken from ``image``.
    """

    image: ImageData
    bin_num: np.ndarray
    bin_value: np.ndarray
    bin_capacity: np.ndarray
    bin_count: np.ndarray
    bin_signal: np.ndarray
    xybin: np.ndarray
    rbin: np.ndarray
    mask: np.ndarray
    target_capacity: float
    method: str
    rms_frac: float

    @property
    def value(self) -> np.ndarray:
        """Per-pixel image painted with its bin's value; non-finite outside the binned region."""
        return self.image.data

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """``(xmin, xmax, ymin, ymax)`` of the binned map."""
        return self.image.extent

    @property
    def x_edges(self) -> np.ndarray | None:
        """Bin edges of the input grid along x, which need not be evenly spaced."""
        return self.image.x_edges

    @property
    def y_edges(self) -> np.ndarray | None:
        """Bin edges of the input grid along y, which need not be evenly spaced."""
        return self.image.y_edges

    @property
    def x_centers(self) -> np.ndarray:
        """Centre of every column of the input grid, in the units of the x axis."""
        return self.image.x_centers

    @property
    def y_centers(self) -> np.ndarray:
        """Centre of every row of the input grid, in the units of the y axis."""
        return self.image.y_centers

    @property
    def x_uniform(self) -> bool:
        """Whether the input columns are evenly spaced."""
        return self.image.x_uniform

    @property
    def y_uniform(self) -> bool:
        """Whether the input rows are evenly spaced."""
        return self.image.y_uniform

    @property
    def uniform(self) -> bool:
        """Whether the whole input grid is evenly spaced."""
        return self.image.uniform

    @property
    def label(self) -> str | None:
        """Name of the binned quantity."""
        return self.image.label

    @property
    def units(self) -> Any:
        """Units of the binned quantity."""
        return self.image.units

    @property
    def x_label(self) -> str | None:
        """Name of the x axis, taken from ``image``."""
        return self.image.x_label

    @property
    def y_label(self) -> str | None:
        """Name of the y axis, taken from ``image``."""
        return self.image.y_label

    @property
    def x_units(self) -> Any:
        """Units of the x axis, taken from ``image``."""
        return self.image.x_units

    @property
    def y_units(self) -> Any:
        """Units of the y axis, taken from ``image``."""
        return self.image.y_units

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return int(self.bin_value.size)

    @property
    def single(self) -> np.ndarray:
        """Mask of bins that hold a single pixel (their value is not an average)."""
        return self.bin_count <= 1

    def to_image_data(self) -> ImageData:
        """The painted map as an :class:`~pynbodyext.plot.image.data.ImageData`.

        This is the image the binned values were painted onto; because it is an
        ordinary image, it can be smoothed, stretched or composited like any other.
        """
        return self.image

    def imshow(self, ax: Any = None, *, symmetric: bool = False, **kwargs: Any) -> Any:
        """Draw :attr:`value` as an image, leaving unbinned pixels transparent.

        Evenly spaced grids are drawn with ``imshow`` and unevenly spaced ones with
        ``pcolormesh``, so the bins stay where they belong either way.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        symmetric : bool, default: False
            Place zero in the middle of the colour range — the right choice for a
            velocity map.  Ignored if ``vmin``/``vmax`` are passed explicitly.
        **kwargs
            Forwarded to :meth:`ImageData.draw`, e.g. ``cmap``, ``colorbar``
            (``True`` or a location such as ``"left"``/``"bottom"``),
            ``colorbar_kwargs``.

        Returns
        -------
        matplotlib.image.AxesImage or matplotlib.collections.QuadMesh
            The artist.
        """
        if symmetric and "vmin" not in kwargs and "vmax" not in kwargs:
            limit = float(np.nanmax(np.abs(self.value)))
            kwargs["vmin"], kwargs["vmax"] = -limit, limit
        return self.image.draw(ax=ax, **kwargs)

    def add_colorbar(self, mappable: Any = None, ax: Any = None, **kwargs: Any) -> Any:
        """Dock a colour bar describing the painted map to its panel.

        Shorthand for :func:`pynbodyext.plot.image.display.add_colorbar` on
        :attr:`image`; with no *mappable* the artist drawn from the painted map in
        *ax* is used, so ``binned.imshow(); binned.add_colorbar()`` just works.

        Returns
        -------
        matplotlib.colorbar.Colorbar
            The colour bar.
        """
        return self.image.add_colorbar(mappable, ax=ax, **kwargs)


def _aggregate(values: np.ndarray, bin_num: np.ndarray, weights: np.ndarray, n_bins: int, method: str) -> np.ndarray:
    """Reduce the pixels of each bin to one value."""
    counts = np.bincount(bin_num, minlength=n_bins)
    if method == "sum":
        return np.bincount(bin_num, weights=values, minlength=n_bins)
    if method == "weighted":
        return np.bincount(bin_num, weights=weights * values, minlength=n_bins) / np.bincount(
            bin_num, weights=weights, minlength=n_bins
        )
    if method == "mean":
        return np.bincount(bin_num, weights=values, minlength=n_bins) / counts
    order = np.argsort(bin_num, kind="stable")
    starts = np.searchsorted(bin_num[order], np.arange(n_bins))
    ends = np.append(starts[1:], len(order))
    sorted_values = values[order]
    return np.array([np.median(sorted_values[start:end]) for start, end in zip(starts, ends, strict=True)])


def _resolve_target(
    *,
    target_capacity: float | None,
    target_signal: float | None,
    target_nbins: int | None,
    total_capacity: float,
    has_noise: bool,
) -> float:
    """Turn the requested target resolution into one capacity value."""
    given = {"target_capacity": target_capacity, "target_signal": target_signal, "target_nbins": target_nbins}
    requested = [name for name, value in given.items() if value is not None]
    if len(requested) != 1:
        raise ValueError(
            "Pass exactly one of target_capacity, target_signal or target_nbins "
            f"(got {', '.join(requested) if requested else 'none'})."
        )
    if target_capacity is not None:
        capacity = float(target_capacity)
    elif target_signal is not None:
        if has_noise:
            raise ValueError(
                "target_signal is ambiguous once a noise map is given "
                "(capacity is (S/N)^2); pass target_capacity = (S/N)^2 instead."
            )
        capacity = float(target_signal)
    else:
        assert target_nbins is not None  # guaranteed by the check above
        if target_nbins < 1:
            raise ValueError(f"target_nbins must be positive, got {target_nbins!r}.")
        capacity = total_capacity / float(target_nbins)
    if capacity <= 0.0:
        raise ValueError(f"target capacity must be positive, got {capacity!r}.")
    return capacity


def _cell_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Centre of every grid cell, in cell units: ``0.5, 1.5, ...``.

    The power diagram is built in cell space rather than in the units of the
    axes, because the two axes need not be commensurable (``kpc`` against ``K``):
    in cell space a bin that looks round in the figure is round, whatever units
    the axes carry.
    """
    rows, cols = np.indices(shape, dtype=float)
    return cols + 0.5, rows + 0.5


def _to_axis_units(cell_positions: np.ndarray, edges: np.ndarray | None, count: int) -> np.ndarray:
    """Map positions given in cell units onto the axis the bin edges describe."""
    if edges is None:
        return np.asarray(cell_positions, dtype=float)
    return np.interp(cell_positions, bin_centers(None, count), bin_centers(edges, count))


def adaptive_bin_map(
    value: Any,
    signal: Any,
    *,
    noise: Any = None,
    target_capacity: float | None = None,
    target_signal: float | None = None,
    target_nbins: int | None = None,
    mask: Any = None,
    min_signal: float | None = None,
    method: str = "mean",
    extent: tuple[float, float, float, float] | None = None,
    x_edges: Any = None,
    y_edges: Any = None,
    x_units: Any = None,
    y_units: Any = None,
    x_label: str | None = None,
    y_label: str | None = None,
    regul: bool = True,
    maxiter: int = 50,
    verbose: int = 0,
    label: str | None = None,
    units: Any = None,
) -> AdaptiveMap:
    """Bin a 2-D map into regions of comparable signal, PowerBin-style.

    Parameters
    ----------
    value : array_like
        The quantity to display (e.g. ``vz.mean`` from a ``BinND`` result).
        Non-finite pixels are not binned.
    signal : array_like
        Signal strength driving the binning (e.g. ``mass.sum``); the capacity of
        a bin is its total signal, so each bin reaches the same signal.
    noise : array_like, optional
        Per-pixel noise; when given, capacity becomes ``(signal / noise) ** 2``,
        the usual ``(S/N)**2``.
    target_capacity, target_signal, target_nbins :
        How to fix the bin size.  Give exactly one: the capacity per bin, the
        total signal per bin (equivalent when there is no ``noise``), or the
        number of bins to aim for.
    mask : array_like of bool, optional
        Restrict the binning to these pixels; the rest stay unbinned.
    min_signal : float, optional
        Drop pixels fainter than this — the "only show the high-signal region"
        switch.
    method : {"mean", "median", "sum", "weighted"}, default: "mean"
        How the pixels of a bin are reduced to the bin value; ``"weighted"``
        weights pixels by their capacity.
    extent : (float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` of an evenly spaced map.  When given,
        ``xybin``/``rbin`` are in these coordinates instead of pixels.
    x_edges, y_edges : array_like, optional
        Bin edges of the map, for a grid that is not evenly spaced (logarithmic,
        quantile, or explicit edges).  Give both or neither; takes precedence
        over *extent*.
    x_units, y_units : object, optional
        Units of the two axes, which may differ.
    x_label, y_label : str, optional
        Names of the two axes.
    regul : bool, default: True
        Let PowerBin regularise the bin shapes (off means accretion only).
    maxiter : int, default: 50
        Maximum regularisation iterations.
    verbose : int, default: 0
        PowerBin verbosity; the default keeps the library quiet.
    label, units : str, object, optional
        Stored on the result for labelling.

    Returns
    -------
    AdaptiveMap
        The binned map, its per-bin statistics and its geometry.

    Examples
    --------
    >>> velocity, mass = np.ones((20, 20)), np.ones((20, 20))  # doctest: +SKIP
    >>> binned = adaptive_bin_map(velocity, mass, target_nbins=4)  # doctest: +SKIP
    >>> binned.n_bins  # doctest: +SKIP
    4
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(_METHODS)}.")
    if not POWERBIN_AVAILABLE:
        raise ImportError(
            "Adaptive binning needs the optional dependency 'powerbin'. "
            "Install it with 'pip install pynbodyext[image]' (or 'pip install powerbin')."
        )
    from powerbin import PowerBin

    from .data import ImageData  # local import: data.py composes this module

    values = as_image(value, name="value")
    signals = as_image(signal, name="signal")
    if signals.shape != values.shape:
        raise ValueError(f"signal shape {signals.shape} does not match value shape {values.shape}.")
    if noise is None:
        capacity = signals.copy()
    else:
        noise_array = as_image(noise, name="noise")
        if noise_array.shape != values.shape:
            raise ValueError(f"noise shape {noise_array.shape} does not match value shape {values.shape}.")
        if np.any(noise_array <= 0.0):
            raise ValueError("noise must be positive everywhere.")
        capacity = (signals / noise_array) ** 2

    valid = np.isfinite(values) & np.isfinite(capacity) & (capacity > 0.0)
    if mask is not None:
        extra = np.asarray(mask, dtype=bool)
        if extra.shape != values.shape:
            raise ValueError(f"mask must have shape {values.shape}, got {extra.shape}.")
        valid &= extra
    if min_signal is not None:
        valid &= signals >= float(min_signal)
    if not valid.any():
        raise ValueError("No pixels left to bin; check mask, min_signal and the input maps.")

    target = _resolve_target(
        target_capacity=target_capacity,
        target_signal=target_signal,
        target_nbins=target_nbins,
        total_capacity=float(capacity[valid].sum()),
        has_noise=noise is not None,
    )

    grid_x, grid_y = resolve_edges(values.shape, extent=extent, x_edges=x_edges, y_edges=y_edges)
    x_coords, y_coords = _cell_coordinates(values.shape)
    coordinates = np.column_stack([x_coords[valid], y_coords[valid]])
    with warnings.catch_warnings():
        # PowerBin reports the scatter of its bin capacities; with a single bin
        # (or when every bin holds one pixel) that scatter is undefined and
        # numpy warns.  The outcome is still well defined, so the warning is noise.
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
        binned = PowerBin(
            coordinates, capacity[valid], target, pixelsize=1.0, verbose=verbose, regul=regul, maxiter=maxiter
        )
    n_bins = int(binned.rbin.size)
    bin_num = np.asarray(binned.bin_num, dtype=int)
    cell_width = 0.5 * (typical_width(grid_x) + typical_width(grid_y))
    xybin = np.column_stack(
        [
            _to_axis_units(binned.xybin[:, 0], grid_x, values.shape[1]),
            _to_axis_units(binned.xybin[:, 1], grid_y, values.shape[0]),
        ]
    )

    painted_bins = np.full(values.shape, -1, dtype=int)
    painted_bins[valid] = bin_num
    bin_value = _aggregate(values[valid], bin_num, capacity[valid], n_bins, method)
    painted = np.where(painted_bins >= 0, bin_value[np.clip(painted_bins, 0, None)], np.nan)
    return AdaptiveMap(
        image=ImageData(
            painted,
            x_edges=grid_x,
            y_edges=grid_y,
            x_units=x_units,
            y_units=y_units,
            x_label=x_label,
            y_label=y_label,
            label=label,
            units=units,
        ),
        bin_num=painted_bins,
        bin_value=bin_value,
        bin_capacity=np.asarray(binned.bin_capacity, dtype=float),
        bin_count=np.bincount(bin_num, minlength=n_bins),
        bin_signal=np.bincount(bin_num, weights=signals[valid], minlength=n_bins),
        xybin=xybin,
        rbin=np.asarray(binned.rbin, dtype=float) * cell_width,
        mask=valid,
        target_capacity=target,
        method=method,
        rms_frac=float(binned.rms_frac),
    )


def adaptive_map_from_bins(
    bins: Any, value: str, signal: str, *, noise: str | None = None, **kwargs: Any
) -> AdaptiveMap:
    """Adaptively bin one query of a :class:`BinNDResult` against another.

    Parameters
    ----------
    bins : BinNDResult
        A 2-D binned result.
    value, signal : str
        Queries for the quantity to display and the signal driving the binning,
        e.g. ``"vz.mean"`` and ``"mass.sum"``.
    noise : str, optional
        Query for a per-bin noise map, turning the capacity into ``(S/N)**2``.
    **kwargs
        Forwarded to :func:`adaptive_bin_map`.

    Returns
    -------
    AdaptiveMap
        The binned map, with the axes' physical extent attached.

    Examples
    --------
    >>> binned = adaptive_map_from_bins(bins2d, "vz.mean", "mass.sum", target_nbins=200)  # doctest: +SKIP
    """
    from .data import ImageData  # local import: data.py composes this module

    value_image = ImageData.from_bins(bins, value)  # rejects anything but a 2-D result
    kwargs.setdefault("x_edges", value_image.x_edges)
    kwargs.setdefault("y_edges", value_image.y_edges)
    kwargs.setdefault("x_units", value_image.x_units)
    kwargs.setdefault("y_units", value_image.y_units)
    kwargs.setdefault("x_label", value_image.x_label)
    kwargs.setdefault("y_label", value_image.y_label)
    kwargs.setdefault("label", value_image.label)
    kwargs.setdefault("units", value_image.units)
    if noise is not None:
        kwargs.setdefault("noise", np.asarray(ImageData.from_bins(bins, noise).data, dtype=float))
    return adaptive_bin_map(
        np.asarray(value_image.data, dtype=float),
        np.asarray(ImageData.from_bins(bins, signal).data, dtype=float),
        **kwargs,
    )


@dataclass(frozen=True)
class AdaptiveOps(ImageOps):
    """The adaptive-binning family of an image: ``image.adaptive.bin(signal, …)``."""

    def bin(
        self,
        signal: Any,
        *,
        noise: Any = None,
        target_capacity: float | None = None,
        target_signal: float | None = None,
        target_nbins: int | None = None,
        mask: Any = None,
        min_signal: float | None = None,
        method: str = "mean",
        regul: bool = True,
        maxiter: int = 50,
        verbose: int = 0,
    ) -> AdaptiveMap:
        """Bin this image adaptively by *signal*, keeping its geometry.

        Parameters
        ----------
        signal : ImageData or array_like
            Signal strength driving the bin size, e.g. the ``mass.sum`` map that
            goes with this ``vz.mean`` map.
        noise : ImageData or array_like, optional
            Per-pixel noise, turning the capacity into ``(S/N)**2``.
        target_capacity, target_signal, target_nbins, mask, min_signal, method, regul, maxiter, verbose :
            As in :func:`adaptive_bin_map`; give exactly one target.

        Returns
        -------
        AdaptiveMap
            The partition, and the painted map as ``.image``.

        Examples
        --------
        >>> velocity = ImageData.from_bins(bins2d, "vz.mean")  # doctest: +SKIP
        >>> mass = ImageData.from_bins(bins2d, "mass.sum")  # doctest: +SKIP
        >>> binned = velocity.adaptive.bin(mass, target_nbins=200)  # doctest: +SKIP
        >>> binned.imshow(cmap="K_B_C_G_Y_R_W", symmetric=True)  # doctest: +SKIP
        """
        return adaptive_bin_map(
            self.data,
            _as_map_data(signal),
            noise=None if noise is None else _as_map_data(noise),
            target_capacity=target_capacity,
            target_signal=target_signal,
            target_nbins=target_nbins,
            mask=mask,
            min_signal=min_signal,
            method=method,
            x_edges=self.x_edges,
            y_edges=self.y_edges,
            x_units=self.x_units,
            y_units=self.y_units,
            x_label=self.x_label,
            y_label=self.y_label,
            regul=regul,
            maxiter=maxiter,
            verbose=verbose,
            label=self.label,
            units=self.units,
        )


def _as_map_data(value: Any) -> np.ndarray:
    """Raw values of an image or of an array, so both can drive the binning."""
    from .data import ImageData  # local import: data.py composes this module

    if isinstance(value, ImageData):
        return np.asarray(value.data, dtype=float)
    return np.asarray(value, dtype=float)


register_ops("adaptive", AdaptiveOps)
