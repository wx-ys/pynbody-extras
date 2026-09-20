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

PowerBin is an optional dependency: install it with ``pip install
pynbodyext[image]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from pynbodyext.util.deps import POWERBIN_AVAILABLE

from ._arrays import as_image
from .data import ImageData

__all__ = ["AdaptiveMap", "adaptive_bin_map", "adaptive_map_from_bins"]

_METHODS = ("mean", "median", "sum", "weighted")


@dataclass(frozen=True)
class AdaptiveMap:
    """The result of :func:`adaptive_bin_map`.

    Attributes
    ----------
    value : numpy.ndarray
        Per-pixel image painted with its bin's value; non-finite outside the
        binned region.
    bin_num : numpy.ndarray of int
        Bin index of every pixel, ``-1`` outside the binned region.
    bin_value, bin_capacity, bin_count, bin_signal : numpy.ndarray
        Per-bin value, capacity, pixel count and total signal.
    xybin : numpy.ndarray
        ``(n_bins, 2)`` centres of the bins, in the coordinates of ``extent``.
    rbin : numpy.ndarray
        Effective radius of each bin, in the same coordinates.
    mask : numpy.ndarray of bool
        Pixels that were binned.
    target_capacity : float
        Capacity each bin was asked to reach.
    method : str
        How the per-bin value was aggregated.
    rms_frac : float
        Percent scatter of the achieved bin capacities (from PowerBin).
    extent : tuple of 4 floats, optional
        ``(xmin, xmax, ymin, ymax)`` of the input map.
    label, units : str, object, optional
        Description and units of ``value``, for labelling a figure.
    """

    value: np.ndarray
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
    extent: tuple[float, float, float, float] | None = None
    label: str | None = None
    units: Any = None

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return int(self.bin_value.size)

    @property
    def single(self) -> np.ndarray:
        """Mask of bins that hold a single pixel (their value is not an average)."""
        return self.bin_count <= 1

    def imshow(self, ax: Any = None, *, symmetric: bool = False, **kwargs: Any) -> Any:
        """Draw :attr:`value` as an image, leaving unbinned pixels transparent.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        symmetric : bool, default: False
            Place zero in the middle of the colour range — the right choice for a
            velocity map.  Ignored if ``vmin``/``vmax`` are passed explicitly.
        **kwargs
            Forwarded to ``matplotlib.axes.Axes.imshow``.

        Returns
        -------
        matplotlib.image.AxesImage
            The artist.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (5.0, 5.0)))
        if symmetric and "vmin" not in kwargs and "vmax" not in kwargs:
            limit = float(np.nanmax(np.abs(self.value)))
            kwargs["vmin"], kwargs["vmax"] = -limit, limit
        extent = kwargs.pop("extent", self.extent)
        artist = ax.imshow(self.value, origin="lower", extent=extent, **kwargs)
        if self.label is not None and not ax.get_ylabel():
            ax.set_ylabel(self.label if self.units is None else f"{self.label} [{self.units}]")
        return artist


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


def _pixel_coordinates(
    shape: tuple[int, int], extent: tuple[float, float, float, float] | None, pixelsize: float | None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre of every pixel, in physical coordinates when *extent* is given.

    Returns the x and y coordinates of all pixels plus the pixel size PowerBin
    should use internally.
    """
    rows, cols = np.indices(shape, dtype=float)
    if extent is None:
        return cols + 0.5, rows + 0.5, 1.0 if pixelsize is None else float(pixelsize)
    xmin, xmax, ymin, ymax = (float(bound) for bound in extent)
    step_x = (xmax - xmin) / shape[1]
    step_y = (ymax - ymin) / shape[0]
    size = float(np.mean([abs(step_x), abs(step_y)])) if pixelsize is None else float(pixelsize)
    return xmin + (cols + 0.5) * step_x, ymin + (rows + 0.5) * step_y, size


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
    pixelsize: float | None = None,
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
        ``(xmin, xmax, ymin, ymax)`` of the map.  When given, ``xybin``/``rbin``
        are in these physical coordinates instead of pixels.
    pixelsize : float, optional
        Pixel size used internally for numerical stability; estimated from
        *extent* when that is given, else 1.
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

    x_coords, y_coords, pixel_size = _pixel_coordinates(values.shape, extent, pixelsize)
    coordinates = np.column_stack([x_coords[valid], y_coords[valid]])
    with warnings.catch_warnings():
        # PowerBin reports the scatter of its bin capacities; with a single bin
        # (or when every bin holds one pixel) that scatter is undefined and
        # numpy warns.  The outcome is still well defined, so the warning is noise.
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
        binned = PowerBin(
            coordinates, capacity[valid], target, pixelsize=pixel_size, verbose=verbose, regul=regul, maxiter=maxiter
        )
    n_bins = int(binned.rbin.size)
    bin_num = np.asarray(binned.bin_num, dtype=int)

    painted_bins = np.full(values.shape, -1, dtype=int)
    painted_bins[valid] = bin_num
    bin_value = _aggregate(values[valid], bin_num, capacity[valid], n_bins, method)
    return AdaptiveMap(
        value=np.where(painted_bins >= 0, bin_value[np.clip(painted_bins, 0, None)], np.nan),
        bin_num=painted_bins,
        bin_value=bin_value,
        bin_capacity=np.asarray(binned.bin_capacity, dtype=float),
        bin_count=np.bincount(bin_num, minlength=n_bins),
        bin_signal=np.bincount(bin_num, weights=signals[valid], minlength=n_bins),
        xybin=np.asarray(binned.xybin, dtype=float),
        rbin=np.asarray(binned.rbin, dtype=float),
        mask=valid,
        target_capacity=target,
        method=method,
        rms_frac=float(binned.rms_frac),
        extent=None if extent is None else (float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3])),
        label=label,
        units=units,
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
    value_image = ImageData.from_bins(bins, value)  # rejects anything but a 2-D result
    kwargs.setdefault("extent", value_image.extent)
    kwargs.setdefault("label", value_image.label)
    kwargs.setdefault("units", value_image.units)
    if noise is not None:
        kwargs.setdefault("noise", np.asarray(ImageData.from_bins(bins, noise).data, dtype=float))
    return adaptive_bin_map(
        np.asarray(value_image.data, dtype=float),
        np.asarray(ImageData.from_bins(bins, signal).data, dtype=float),
        **kwargs,
    )
