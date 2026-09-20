"""Drawing an image: the display and colour side of :class:`ImageData`.

Kept out of ``data.py`` so that the data object stays about geometry, metadata and
provenance, and every matplotlib detail lives in one place: axis labels, colour
bars, and the choice between ``imshow`` and ``pcolormesh``.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._arrays import edges_are_uniform, value_limits
from .cmaps import get_cmap
from .ops import ImageOps, register_ops
from .smooth import normalize

if TYPE_CHECKING:
    from .data import ImageData

__all__ = [
    "COLORBAR_LOCATIONS",
    "DisplayOps",
    "add_colorbar",
    "draw_contour",
    "draw_image",
    "draw_imshow",
    "draw_pcolormesh",
]

#: Unit spellings that mean "no units at all", so they never reach a figure label.
_EMPTY_UNITS = {"", "1", "NoUnit()", "dimensionless", "unitless"}

#: Places :func:`add_colorbar` can dock a colour bar to.
COLORBAR_LOCATIONS = ("right", "left", "top", "bottom")

#: One divider per panel, so that several colour bars can be docked to it.
_DIVIDERS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()

#: Label of each artist we drew, so ``add_colorbar(artist)`` can name it too.
_ARTIST_LABELS: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()


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


def add_colorbar(
    mappable: Any = None,
    ax: Any = None,
    *,
    loc: str = "right",
    size: str = "5%",
    pad: float = 0.05,
    label_pad: float = 2.0,
    tick_label_size: float = 10.0,
    label: str | None = None,
    cmap: Any = None,
    norm: Any = None,
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> Any:
    """Dock a colour bar to the panel that holds *mappable*.

    The colour bar is placed with ``mpl_toolkits.axes_grid1.make_axes_locatable``,
    so it hugs the image at the requested relative *size* instead of floating at
    matplotlib's default distance: compact, aligned with the panel, and easy to
    control.  Tick labels and the axis label go on the outside edge for every
    location.

    Parameters
    ----------
    mappable : artist, ImageData or AdaptiveMap, optional
        What to describe.  A drawn artist (``AxesImage``/``QuadMesh``/…) is used
        exactly as it was drawn.  An image is accepted too: the artist already
        drawn from it in *ax* is used when there is one, otherwise a colour bar is
        built from the image's own values (with *cmap*/*vmin*/*vmax*).  ``None``
        means "the image this method was called on".
    ax : matplotlib.axes.Axes, optional
        Panel to dock to; defaults to the artist's axes, then to the current axes.
    loc : {"right", "left", "top", "bottom"}, default: "right"
        Side of the panel to dock to; top/bottom give a horizontal bar.
    size : str, default: "5%"
        Thickness of the colour bar relative to the panel.
    pad : float, default: 0.05
        Gap between the panel and the colour bar, in inches.
    label_pad : float, default: 2.0
        Padding between the ticks and their labels.
    tick_label_size : float, default: 10.0
        Font size of the tick labels.
    label : str, optional
        Axis label of the colour bar; defaults to the image's ``label`` and
        ``units`` when the mappable is one of ours.
    cmap, vmin, vmax :
        Used only when *mappable* is an image that has not been drawn yet.
    norm : matplotlib.colors.Normalize, optional
        Colour scale for that bar, e.g. a ``LogNorm``; instead of ``log=True``.  A
        drawn artist brings its own norm, so this applies only when the bar is built
        from an image.
    log : bool, default: False
        Build a logarithmic bar over the positive part of the image (or *vmin* /
        *vmax*).  It must agree with an artist passed in: a linear artist cannot be
        relabelled logarithmically, and saying so is an error rather than a bar whose
        ticks do not match the colours.
    **kwargs
        Forwarded to ``Figure.colorbar``.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colour bar.

    Raises
    ------
    ValueError
        If no axes can be found to dock to, or *loc* is unknown.
    TypeError
        If *mappable* is neither an artist nor an image.

    Examples
    --------
    >>> art = image.imshow()  # doctest: +SKIP
    >>> add_colorbar(art, loc="bottom", size="8%")  # doctest: +SKIP
    >>> add_colorbar(image, loc="left")  # the image may be passed directly  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize

    if loc not in COLORBAR_LOCATIONS:
        raise ValueError(f"loc must be one of {', '.join(COLORBAR_LOCATIONS)}, got {loc!r}.")
    artist, source = _as_mappable(mappable)
    if (norm is not None or log) and artist is not None:
        requested_log = log or isinstance(norm, LogNorm)
        drawn_log = isinstance(getattr(artist, "norm", None), LogNorm)
        if requested_log != drawn_log:
            raise ValueError(
                f"This artist was drawn on a {'logarithmic' if drawn_log else 'linear'} colour scale, so its "
                "colour bar cannot be labelled differently; pass norm=/log= where you draw "
                "(image.display.imshow(log=True)), or pass the image instead of the artist."
            )
    if label is None and artist is not None:
        label = _ARTIST_LABELS.get(artist)
    axes = ax if ax is not None else getattr(artist, "axes", None)
    if axes is None:
        figure = plt.gcf()
        axes = figure.axes[-1] if figure.axes else None
    if axes is None:
        raise ValueError("No axes found for a colour bar; pass ax=... or draw the image first.")
    if artist is None:  # an image that has not been drawn yet
        if source is None:  # pragma: no cover - _as_mappable returns a mappable or an image
            raise TypeError("Nothing to describe with a colour bar.")
        artist = _artist_in(axes, source)
        if artist is None:
            resolved, _ = resolve_norm(source.data, norm=norm, log=log, vmin=vmin, vmax=vmax)
            artist = ScalarMappable(
                norm=resolved if resolved is not None else Normalize(*value_limits(source.data, vmin=vmin, vmax=vmax)),
                cmap=get_cmap(cmap),
            )

    horizontal = loc in ("top", "bottom")
    cax = _divider_for(axes).append_axes(loc, size=size, pad=pad)
    colorbar = axes.figure.colorbar(artist, cax=cax, orientation="horizontal" if horizontal else "vertical", **kwargs)
    axis = cax.xaxis if horizontal else cax.yaxis
    axis.set_ticks_position(loc)
    axis.set_label_position(loc)
    colorbar.ax.tick_params(labelsize=tick_label_size, pad=label_pad)
    if label is None and source is not None:
        label = _annotation(source.label, source.units)
    if label is not None:
        colorbar.set_label(label)
    return colorbar


def _divider_for(axes: Any) -> Any:
    """The divider installed on *axes*, shared so several bars can share a panel.

    ``make_axes_locatable`` installs a fresh divider each time it is called, and
    two of them on one panel fight over its box (one bar ends up underneath the
    image), so the divider is cached per panel.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = _DIVIDERS.get(axes)
    if divider is None:
        divider = make_axes_locatable(axes)
        _DIVIDERS[axes] = divider
    return divider


def _as_mappable(mappable: Any) -> tuple[Any, ImageData | None]:
    """Split a colour-bar argument into a drawn artist and the image behind it."""
    from matplotlib.cm import ScalarMappable

    if mappable is None or isinstance(mappable, ScalarMappable) or hasattr(mappable, "get_array"):
        return mappable, None
    image = getattr(mappable, "image", mappable)  # AdaptiveMap carries an ImageData
    if hasattr(image, "data") and hasattr(image, "extent"):
        return None, image
    raise TypeError(f"add_colorbar expects a drawn artist or an ImageData/AdaptiveMap, got {type(mappable).__name__}.")


def _artist_in(axes: Any, image: ImageData) -> Any:
    """The artist in *axes* that was drawn from *image*, if it is still there."""
    candidates = list(getattr(axes, "images", ())) + list(getattr(axes, "collections", ()))
    for artist in reversed(candidates):
        array = getattr(artist, "get_array", lambda: None)()
        if array is not None and np.shape(array) == image.shape:
            return artist
    return None


def _finish(
    ax: Any, image: Any, artist: Any, colorbar: bool | str, aspect: Any, colorbar_kwargs: dict[str, Any] | None = None
) -> None:
    """Shared tail of the drawing methods: labels, aspect and colour bar."""
    _apply_axis_labels(ax, image)
    if aspect is not None:
        ax.set_aspect(aspect)
    annotation = _annotation(image.label, image.units)
    if annotation is not None:
        _ARTIST_LABELS[artist] = annotation  # so a bar added later can name it too
    if not colorbar:
        return
    options = dict(colorbar_kwargs or {})
    if isinstance(colorbar, str):
        options.setdefault("loc", colorbar)
    add_colorbar(artist, ax=ax, label=annotation, **options)


def draw_image(
    image: Any,
    ax: Any = None,
    *,
    colorbar: bool | str = False,
    colorbar_kwargs: dict[str, Any] | None = None,
    aspect: Any = None,
    norm: Any = None,
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> Any:
    """Draw *image*, picking the right artist for the bin spacing.

    Evenly spaced bins go through :func:`draw_imshow`; unevenly spaced ones through
    :func:`draw_pcolormesh`, so a logarithmic or quantile grid is drawn where its
    bins really are.

    Parameters
    ----------
    image : ImageData
        The image to draw.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    colorbar : bool or str, default: False
        Add a colour bar labelled with the image's ``label`` and ``units``.
        ``True`` docks it on the right; ``"left"``/``"top"``/``"bottom"`` docks it
        there.
    colorbar_kwargs : dict, optional
        Forwarded to :func:`add_colorbar`, e.g. ``{"size": "8%", "pad": 0.1}``.
    aspect : optional
        Axes aspect, e.g. ``"auto"``; matplotlib's default when omitted.
    norm : matplotlib.colors.Normalize, optional
        Colour scale to draw with, e.g. a ``LogNorm``; instead of ``log=True``.
    log : bool, default: False
        Draw on a logarithmic colour scale over the positive part of the data — the
        right choice for a map that spans decades, such as a density.
    vmin, vmax : float, optional
        Limits for the logarithmic scale, when *log* is set.
    **kwargs
        Forwarded to the chosen artist.

    Returns
    -------
    matplotlib.image.AxesImage or matplotlib.collections.QuadMesh
        The artist.
    """
    uniform = edges_are_uniform(image.x_edges) and edges_are_uniform(image.y_edges)
    draw = draw_imshow if uniform else draw_pcolormesh
    resolved, _ = resolve_norm(image.data, norm=norm, log=log, vmin=vmin, vmax=vmax)
    return draw(
        image, ax=ax, colorbar=colorbar, colorbar_kwargs=colorbar_kwargs, aspect=aspect, norm=resolved, **kwargs
    )


def draw_imshow(
    image: Any,
    ax: Any = None,
    *,
    colorbar: bool | str = False,
    colorbar_kwargs: dict[str, Any] | None = None,
    aspect: Any = None,
    norm: Any = None,
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> Any:
    """Draw *image* with ``imshow``, labelling it from its metadata.

    Use this for evenly spaced bins; :func:`draw_pcolormesh` handles arbitrary bin
    edges, and :func:`draw_image` picks between them.

    Parameters
    ----------
    image : ImageData
        The image to draw.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    colorbar : bool or str, default: False
        Add a docked colour bar; see :func:`draw_image`.
    colorbar_kwargs : dict, optional
        Forwarded to :func:`add_colorbar`.
    aspect : optional
        Axes aspect, e.g. ``"auto"``.
    norm : matplotlib.colors.Normalize, optional
        Colour scale to draw with, e.g. a ``LogNorm``; instead of ``log=True``.
    log : bool, default: False
        Draw on a logarithmic colour scale over the positive part of the data.
    vmin, vmax : float, optional
        Limits for the logarithmic scale, when *log* is set.
    **kwargs
        Forwarded to ``matplotlib.axes.Axes.imshow``; ``extent`` defaults to the
        image's own and ``origin`` to ``"lower"``.

    Returns
    -------
    matplotlib.image.AxesImage
        The artist.
    """
    import matplotlib.pyplot as plt

    if not (edges_are_uniform(image.x_edges) and edges_are_uniform(image.y_edges)):
        raise ValueError("These bins are not evenly spaced; use pcolormesh() or draw() instead.")
    figsize = kwargs.pop("figsize", (5.0, 5.0))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    resolved, _ = resolve_norm(image.data, norm=norm, log=log, vmin=vmin, vmax=vmax)
    if resolved is not None:
        kwargs.setdefault("norm", resolved)
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("extent", image.extent)
    artist = ax.imshow(image.data, **kwargs)
    _finish(ax, image, artist, colorbar, aspect, colorbar_kwargs)
    return artist


def draw_pcolormesh(
    image: Any,
    ax: Any = None,
    *,
    colorbar: bool | str = False,
    colorbar_kwargs: dict[str, Any] | None = None,
    aspect: Any = None,
    norm: Any = None,
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    shading: str = "flat",
    **kwargs: Any,
) -> Any:
    """Draw *image* as quadrilateral cells, honouring arbitrary bin edges.

    Parameters
    ----------
    image : ImageData
        The image to draw.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    colorbar : bool or str, default: False
        Add a docked colour bar; see :func:`draw_image`.
    colorbar_kwargs : dict, optional
        Forwarded to :func:`add_colorbar`.
    aspect : optional
        Axes aspect, e.g. ``"auto"``.
    norm : matplotlib.colors.Normalize, optional
        Colour scale to draw with, e.g. a ``LogNorm``; instead of ``log=True``.
    log : bool, default: False
        Draw on a logarithmic colour scale over the positive part of the data.
    vmin, vmax : float, optional
        Limits for the logarithmic scale, when *log* is set.
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
    resolved, _ = resolve_norm(image.data, norm=norm, log=log, vmin=vmin, vmax=vmax)
    if resolved is not None:
        kwargs.setdefault("norm", resolved)
    x_edges = image.x_edges if image.x_edges is not None else np.arange(image.shape[1] + 1, dtype=float)
    y_edges = image.y_edges if image.y_edges is not None else np.arange(image.shape[0] + 1, dtype=float)
    artist = ax.pcolormesh(x_edges, y_edges, image.data, shading=shading, **kwargs)
    _finish(ax, image, artist, colorbar, aspect, colorbar_kwargs)
    return artist


def draw_contour(
    image: Any,
    ax: Any = None,
    *,
    levels: Any = 8,
    filled: bool = False,
    colorbar: bool | str = False,
    colorbar_kwargs: dict[str, Any] | None = None,
    aspect: Any = None,
    norm: Any = None,
    log: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> Any:
    """Draw contour lines of *image*, on the grid the image actually samples.

    The contours are placed on the bin centres, so they are correct for unevenly
    spaced bins (logarithmic, quantile, explicit edges) rather than assuming a
    regular pixel grid.  Pass the same *ax* as an :func:`draw_image` call to overlay
    contours on the map.

    Parameters
    ----------
    image : ImageData
        The image to contour.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    levels : int or array_like, default: 8
        Number of levels, or the levels themselves — e.g. ``[-2, 0, 2]`` to mark
        zero crossings.  With a logarithmic scale an integer means that many
        *intervals*, spaced geometrically between *vmin* and *vmax* (or across the
        positive data range) — a linear set of levels would bunch all of them into
        the brightest decade of a density map.
    filled : bool, default: False
        Fill the bands (``contourf``) instead of drawing lines (``contour``).
    colorbar : bool or str, default: False
        Add a docked colour bar; see :func:`draw_image`.
    colorbar_kwargs : dict, optional
        Forwarded to :func:`add_colorbar`.
    aspect : optional
        Axes aspect, e.g. ``"auto"``.
    norm : matplotlib.colors.Normalize, optional
        Colour scale for the lines or bands, e.g. a ``LogNorm``; instead of
        ``log=True``.  A ``LogNorm`` here also makes the levels geometric.
    log : bool, default: False
        Space the levels geometrically, and colour them on a logarithmic scale.
    vmin, vmax : float, optional
        Limits of the logarithmic scale, when *log* is set.
    **kwargs
        Forwarded to ``matplotlib.axes.Axes.contour`` / ``contourf``, e.g.
        ``colors``, ``linewidths``, ``cmap``.

    Returns
    -------
    matplotlib.contour.ContourSet
        The artist, which :func:`add_colorbar` accepts directly.
    """
    import matplotlib.pyplot as plt

    figsize = kwargs.pop("figsize", (5.0, 5.0))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    resolved, log_limits = resolve_norm(image.data, norm=norm, log=log, vmin=vmin, vmax=vmax)
    if log_limits is not None and isinstance(levels, (int, np.integer)):
        levels = np.geomspace(log_limits[0], log_limits[1], int(levels) + 1)
    if resolved is not None:
        kwargs.setdefault("norm", resolved)
    method = ax.contourf if filled else ax.contour
    values = np.asarray(image.data)
    if log_limits is not None:  # a log contour cannot include the non-positive pixels
        values = np.ma.masked_where(~(np.isfinite(values) & (values > 0.0)), values)
    artist = method(image.x_centers, image.y_centers, values, levels, **kwargs)
    _finish(ax, image, artist, colorbar, aspect, colorbar_kwargs)
    return artist


def _symmetric_limits(data: np.ndarray, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add symmetric ``vmin``/``vmax`` unless the caller set them."""
    if "vmin" in kwargs or "vmax" in kwargs:
        return kwargs
    limit = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.0
    return {**kwargs, "vmin": -limit, "vmax": limit}


def positive_limits(data: Any, *, vmin: float | None = None, vmax: float | None = None) -> tuple[float, float]:
    """The strictly positive range of *data*, for a logarithmic scale.

    Parameters
    ----------
    data : array_like
        Values to measure; non-positive and non-finite entries are ignored, since a
        logarithmic scale cannot show them.
    vmin, vmax : float, optional
        Explicit limits, which must be positive and increasing.

    Returns
    -------
    tuple of float
        ``(vmin, vmax)``.

    Raises
    ------
    ValueError
        If no positive values exist, or the given limits are not positive.
    """
    array = np.asarray(data, dtype=float)
    positive = np.isfinite(array) & (array > 0.0)
    low = vmin if vmin is not None else (float(np.min(array[positive])) if positive.any() else None)
    high = vmax if vmax is not None else (float(np.max(array[positive])) if positive.any() else None)
    if low is None or high is None:
        raise ValueError("A logarithmic scale needs positive values; this map has none (pass vmin=/vmax=).")
    if low <= 0.0 or high <= low:
        raise ValueError(f"A logarithmic scale needs 0 < vmin < vmax, got vmin={low}, vmax={high}.")
    return float(low), float(high)


def resolve_norm(
    data: Any, *, norm: Any = None, log: bool = False, vmin: float | None = None, vmax: float | None = None
) -> tuple[Any, tuple[float, float] | None]:
    """Work out the norm to draw *data* with, and its limits when it is logarithmic.

    ``norm=`` and ``log=True`` are two ways of saying the same thing and cannot be
    combined.  ``log=True`` builds a ``LogNorm`` over the positive part of the data
    (or over ``vmin``/``vmax`` when they are given); a ``LogNorm`` handed in through
    ``norm=`` is recognised so that callers can compute geometric contour levels.

    Returns
    -------
    tuple
        ``(norm, limits)`` — the norm to pass to matplotlib (``None`` for its
        default linear scale), and the ``(vmin, vmax)`` to space levels over when
        the scale is logarithmic (``None`` otherwise).
    """
    from matplotlib.colors import LogNorm

    if norm is not None and log:
        raise ValueError("Pass either 'norm' or log=True, not both: they describe the same scale twice.")
    if norm is None and not log:
        return None, None
    if norm is None:
        limits = positive_limits(data, vmin=vmin, vmax=vmax)
        return LogNorm(vmin=limits[0], vmax=limits[1]), limits
    if isinstance(norm, LogNorm):
        low = vmin if vmin is not None else norm.vmin
        high = vmax if vmax is not None else norm.vmax
        if low is None or high is None:
            limits = positive_limits(data)
        else:
            limits = positive_limits(data, vmin=low, vmax=high)
        return norm, limits
    return norm, None


def reject_log_and_symmetric(symmetric: bool, log: bool) -> None:
    """Refuse ``symmetric=True`` together with ``log=True``, which cannot both hold."""
    if symmetric and log:
        raise ValueError("symmetric=True centres the colour scale on zero, which log=True cannot do; pass one of them.")


@dataclass(frozen=True)
class DisplayOps(ImageOps):
    """The display family of an image: ``image.display.imshow(...)``.

    Everything that turns values into a figure lives here — the stretch
    (:meth:`normalize`), the colours (:meth:`to_rgba`), the artists
    (:meth:`imshow`, :meth:`pcolormesh`, :meth:`draw`, :meth:`contour`) and
    :meth:`add_colorbar` — so "what does this image look like" is one namespace,
    next to the families that change the values.
    """

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

        Unlike the free :func:`~pynbodyext.plot.image.smooth.normalize`, this
        returns an image, so it can be chained.
        """
        stretched = normalize(
            self.data, vmin=vmin, vmax=vmax, stretch=stretch, percentiles=percentiles, asinh_a=asinh_a
        )
        return self.derive(
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

        See :func:`~pynbodyext.plot.image.cmaps.to_rgba`; the default colour map is
        the SAURON map ``sauron_cmap``.
        """
        from .cmaps import sauron_cmap, to_rgba

        return to_rgba(
            self.data,
            sauron_cmap if cmap is None else cmap,
            vmin=vmin,
            vmax=vmax,
            stretch=stretch,
            percentiles=percentiles,
            norm=norm,
            alpha=alpha,
            bad=bad,
        )

    def draw(
        self,
        ax: Any = None,
        *,
        colorbar: bool | str = False,
        colorbar_kwargs: dict[str, Any] | None = None,
        aspect: Any = None,
        symmetric: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Draw the image, picking the right artist for the bin spacing.

        Parameters
        ----------
        symmetric : bool, default: False
            Centre the colour scale on zero — the right choice for a velocity map.
            Ignored when ``vmin``/``vmax`` are given; cannot be combined with
            ``log=True``.
        **kwargs
            As in :func:`draw_image`, e.g. ``cmap``, ``colorbar``, ``log=True``.
        """
        reject_log_and_symmetric(symmetric, bool(kwargs.get("log")))
        kwargs = _symmetric_limits(self.data, kwargs) if symmetric else kwargs
        return draw_image(
            self.image, ax=ax, colorbar=colorbar, colorbar_kwargs=colorbar_kwargs, aspect=aspect, **kwargs
        )

    def imshow(
        self,
        ax: Any = None,
        *,
        colorbar: bool | str = False,
        colorbar_kwargs: dict[str, Any] | None = None,
        aspect: Any = None,
        symmetric: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Draw the image with ``imshow``; requires evenly spaced bins.

        Accepts the arguments of :func:`draw_imshow`, in particular ``log=True`` (or
        a ``norm=``) for a map that spans decades, and ``symmetric=True`` for a
        velocity map.
        """
        reject_log_and_symmetric(symmetric, bool(kwargs.get("log")))
        kwargs = _symmetric_limits(self.data, kwargs) if symmetric else kwargs
        return draw_imshow(
            self.image, ax=ax, colorbar=colorbar, colorbar_kwargs=colorbar_kwargs, aspect=aspect, **kwargs
        )

    def pcolormesh(
        self,
        ax: Any = None,
        *,
        colorbar: bool | str = False,
        colorbar_kwargs: dict[str, Any] | None = None,
        aspect: Any = None,
        symmetric: bool = False,
        shading: str = "flat",
        **kwargs: Any,
    ) -> Any:
        """Draw the image as cells, honouring arbitrary bin edges.

        Accepts the arguments of :func:`draw_pcolormesh`, in particular ``log=True``
        (or a ``norm=``) and ``symmetric=True``.
        """
        reject_log_and_symmetric(symmetric, bool(kwargs.get("log")))
        kwargs = _symmetric_limits(self.data, kwargs) if symmetric else kwargs
        return draw_pcolormesh(
            self.image,
            ax=ax,
            colorbar=colorbar,
            colorbar_kwargs=colorbar_kwargs,
            aspect=aspect,
            shading=shading,
            **kwargs,
        )

    def contour(
        self,
        ax: Any = None,
        *,
        levels: Any = 8,
        filled: bool = False,
        colorbar: bool | str = False,
        colorbar_kwargs: dict[str, Any] | None = None,
        aspect: Any = None,
        symmetric: bool = False,
        count: int = 6,
        **kwargs: Any,
    ) -> Any:
        """Draw contour lines (or ``filled=True`` bands) on the image's own grid.

        Parameters
        ----------
        symmetric : bool, default: False
            With ``levels`` left as a count, place the levels symmetrically about
            zero, *count* on each side — the right choice for a velocity map.
        count : int, default: 6
            How many levels on each side of zero when *symmetric* is set.
        **kwargs
            As in :func:`draw_contour`, e.g. ``colors``, ``linewidths``, and
            ``log=True`` (or a ``norm=``) to space the levels geometrically.
        """
        reject_log_and_symmetric(symmetric, bool(kwargs.get("log")))
        if symmetric and "levels" not in kwargs:
            limit = float(np.nanmax(np.abs(self.data)))
            levels = np.linspace(-limit, limit, 2 * count + 1)
        return draw_contour(
            self.image,
            ax=ax,
            levels=levels,
            filled=filled,
            colorbar=colorbar,
            colorbar_kwargs=colorbar_kwargs,
            aspect=aspect,
            **kwargs,
        )

    def add_colorbar(self, mappable: Any = None, ax: Any = None, **kwargs: Any) -> Any:
        """Dock a colour bar to the panel showing this image.

        Shorthand for :func:`add_colorbar`: with no *mappable*, the artist drawn
        from this image in *ax* is used, so ``img.display.imshow();
        img.display.add_colorbar(loc="bottom")`` works.
        """
        return add_colorbar(self.image if mappable is None else mappable, ax=ax, **kwargs)


register_ops("display", DisplayOps)
