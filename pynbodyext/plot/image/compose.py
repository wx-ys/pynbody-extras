"""Stitch two maps drawn with different colour maps into one figure.

The example this exists for: a cosmological simulation where gas density and
dark-matter density share the same footprint but need different colour maps.  A
**map mask** gives a soft, straight-line split of the image; each map is
rendered through its own colour map, and the two are crossfaded in a band around
the dividing line so there is no seam::

    from pynbodyext.plot import image

    image.imshow_compose(
        gas_density,
        dm_density,
        style1=image.MapStyle(cmap="inferno", stretch="log"),
        style2=image.MapStyle(cmap="cividis"),
        label1="gas",
        label2="dark matter",
        extent=(-50, 50, -50, 50),
    )

The masks are plain arrays in ``[0, 1]``, so they also drive :func:`blend_images`
and :func:`blend_stack` directly when the two layers are already rendered.
How a map becomes colours lives in one place, :class:`MapStyle`, which is also
what the colour bars are built from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._arrays import aligned_values, shape_hint, value_limits
from .cmaps import as_norm, get_cmap, norm_from_stretch, to_rgba
from .display import add_colorbar
from .ops import ImageOps
from .smooth import STRETCHES

__all__ = ["ComposeOps", "MapStyle", "blend_images", "blend_stack", "compose_maps", "create_map_mask", "imshow_compose"]


@dataclass(frozen=True)
class MapStyle:
    """How one map is turned into colours.

    Bundles the choices that a compose call would otherwise spell out twice
    (``cmap1``/``vmin1``/``vmax1``/``stretch1`` …), and knows the limits its own
    colour bar should show.

    Parameters
    ----------
    cmap : str or Colormap, optional
        Colour map; defaults to the SAURON map ``sauron_cmap``.
    vmin, vmax : float, optional
        Value limits; default to the data range, or to *percentiles*.
    stretch : {"linear", "sqrt", "log", "asinh", "hist"}, default: "linear"
        Display stretch, see :func:`~pynbodyext.plot.image.smooth.normalize`.
    norm : matplotlib.colors.Normalize, optional
        An explicit colour scale — a ``LogNorm`` for a map that spans decades, say.
        Mutually exclusive with *stretch*: they describe the same scale, so give one
        of them.
    percentiles : (float, float), optional
        Percentiles used for whichever limit is not given, e.g. ``(1, 99)``.
    bad : color, optional
        Colour for non-finite pixels; fully transparent by default.

    Examples
    --------
    >>> style = MapStyle(cmap="inferno", vmin=0.0, vmax=1.0)
    >>> style.limits([0.0, 0.5, 1.0])
    (0.0, 1.0)
    """

    cmap: Any = None
    vmin: float | None = None
    vmax: float | None = None
    stretch: str = "linear"
    norm: Any = None
    percentiles: tuple[float, float] | None = None
    bad: Any = None

    def __post_init__(self) -> None:
        if self.stretch not in STRETCHES:
            raise ValueError(f"Unknown stretch {self.stretch!r}; choose from {', '.join(STRETCHES)}.")
        if self.norm is not None and self.stretch != "linear":
            raise ValueError("Pass either norm= or stretch=, not both: they describe the same scale twice.")
        if self.percentiles is not None and self.percentiles[0] > self.percentiles[1]:
            raise ValueError(f"percentiles must be increasing, got {self.percentiles!r}.")

    def limits(self, data: Any) -> tuple[float, float]:
        """The value limits this style uses for *data*, colour bar included."""
        return value_limits(data, vmin=self.vmin, vmax=self.vmax, percentiles=self.percentiles)

    def to_rgba(self, data: Any) -> np.ndarray:
        """Colour *data* with this style."""
        return to_rgba(
            data,
            get_cmap(self.cmap),
            vmin=self.vmin,
            vmax=self.vmax,
            stretch=self.stretch,
            norm=self.norm,
            percentiles=self.percentiles,
            bad=self.bad,
        )

    def norm_for(self, data: Any) -> Any:
        """The norm a colour bar of this style should use over *data*.

        Without this a stretched map would be labelled by a linear scale, putting the
        ticks where the colours are not: ``stretch="log"`` returns a ``FuncNorm`` that
        matches the stretch, an explicit ``norm`` is used as given, and ``"hist"``
        (which equalises over the whole image and so has no per-value inverse) falls
        back to a linear scale.
        """
        if self.norm is not None:
            return as_norm(self.norm)
        vmin, vmax = self.limits(data)
        return norm_from_stretch(self.stretch, vmin=vmin, vmax=vmax)


def as_map_style(style: MapStyle | str | dict[str, Any] | None) -> MapStyle:
    """Coerce a style argument into a :class:`MapStyle`.

    ``None`` gives the default style, a string or colour map is taken as ``cmap``,
    and a mapping is expanded into the fields.
    """
    if style is None:
        return MapStyle()
    if isinstance(style, MapStyle):
        return style
    if isinstance(style, str) or hasattr(style, "name") and not isinstance(style, dict):
        return MapStyle(cmap=style)
    if isinstance(style, dict):
        return MapStyle(**style)
    raise TypeError(f"Cannot read {style!r} as a MapStyle.")


def create_map_mask(
    show_array: Any, line_angle: float = 45.0, width: float = 0.1, *, center: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Create two complementary soft masks split by a line at *line_angle*.

    Each mask transitions gradually from 0 to 1 across the dividing line.
    *width* is the ramp width as a fraction of the image diagonal (0-1), so the
    smooth transition looks the same at any resolution: a larger width gives a
    wider, softer fade, a smaller one a sharper split, and ``width <= 0`` a hard
    binary split.

    Parameters
    ----------
    show_array : array_like or (int, int)
        The image the masks are for — only its first two dimensions are used, so
        an RGB array works — or the shape to build masks for.
    line_angle : float, default: 45.0
        Angle of the dividing line in degrees.  It is measured from the ``+x``
        direction (the second array axis, to the right in the default
        ``origin="lower"`` display) and increases counter-clockwise, so
        ``0`` splits left/right, ``90`` splits bottom/top and ``180`` puts the
        first map on the left.
    width : float, default: 0.1
        Width of the transition band as a fraction of the image diagonal.
    center : (float, float), optional
        Point the line passes through, in pixel indices; defaults to the image
        centre.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mask1, mask2)`` float arrays in ``[0, 1]`` with the leading shape of
        *show_array*.  ``mask1`` is ``~1`` on one side of the line, ``mask2`` is
        its complement.

    Examples
    --------
    >>> mask1, mask2 = create_map_mask((4, 6), line_angle=0, width=0.0)
    >>> mask1.tolist()  # the right-hand half
    [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
    """
    shape = _leading_shape(show_array)
    theta = np.radians(line_angle)
    n_y, n_x = shape
    x = np.arange(n_x)
    y = np.arange(n_y)
    grid_x, grid_y = np.meshgrid(x, y)
    if center is None:
        centre_x, centre_y = (n_x - 1) / 2.0, (n_y - 1) / 2.0
    else:
        centre_x, centre_y = (float(value) for value in center)

    # Signed distance from the dividing line, in pixels (positive on one side).
    distance = (grid_x - centre_x) * np.cos(theta) + (grid_y - centre_y) * np.sin(theta)

    if width <= 0.0:  # hard binary split
        mask1 = (distance > 0).astype(float)
    else:
        diagonal = np.hypot(n_x, n_y)
        half = width * diagonal / 2.0  # half-ramp width in pixels
        ramp = np.clip((distance + half) / (2.0 * half), 0.0, 1.0)
        mask1 = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep: zero slope at both ends
    return mask1, 1.0 - mask1


def _leading_shape(show_array: Any) -> tuple[int, int]:
    """Shape of the first two dimensions of an image, or of a shape argument."""
    if (
        isinstance(show_array, (tuple, list))
        and len(show_array) >= 2
        and all(isinstance(v, int) for v in show_array[:2])
    ):
        return int(show_array[0]), int(show_array[1])
    array = np.asarray(show_array)
    if array.ndim < 2:
        raise ValueError(f"show_array must be at least 2-D, got shape {array.shape}.")
    return int(array.shape[0]), int(array.shape[1])


def blend_images(image1: Any, image2: Any, mask: Any) -> np.ndarray:
    """Crossfade two equally shaped images with a per-pixel *mask*.

    Parameters
    ----------
    image1, image2 : array_like
        Images or data arrays of the same shape; trailing dimensions (colour
        channels) are broadcast over by the mask.
    mask : array_like
        Weight of *image1*, in ``[0, 1]``, with the leading shape of the images.

    Returns
    -------
    numpy.ndarray
        ``mask * image1 + (1 - mask) * image2``.
    """
    first = np.asarray(image1, dtype=float)
    second = np.asarray(image2, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"image1 shape {first.shape} does not match image2 shape {second.shape}.")
    weights = np.asarray(mask, dtype=float)
    if weights.shape != first.shape[:2]:
        raise ValueError(f"mask shape {weights.shape} does not match the image shape {first.shape}.")
    if first.ndim > weights.ndim:
        weights = weights[(...,) + (None,) * (first.ndim - weights.ndim)]
    return weights * first + (1.0 - weights) * second


def blend_stack(images: Any, weights: Any) -> np.ndarray:
    """Combine N layers with per-pixel weights that are normalised to sum to one.

    Parameters
    ----------
    images : sequence of array_like
        Layers, all with the same shape.
    weights : sequence of array_like
        One array per layer, with the leading shape of the images.  Pixels whose
        weights sum to zero come out as zeros (nothing constrains them).

    Returns
    -------
    numpy.ndarray
        The weighted average of the layers.
    """
    layers = [np.asarray(image, dtype=float) for image in images]
    if not layers:
        raise ValueError("blend_stack needs at least one image.")
    weight_arrays = [np.asarray(weight, dtype=float) for weight in weights]
    if len(weight_arrays) != len(layers):
        raise ValueError(f"Got {len(weight_arrays)} weights for {len(layers)} images.")
    stacked = np.stack(
        [layer * _broadcast_weight(layer, weight) for layer, weight in zip(layers, weight_arrays, strict=True)]
    )
    total = np.sum(weight_arrays, axis=0)
    total = total[(...,) + (None,) * (layers[0].ndim - total.ndim)] if layers[0].ndim > total.ndim else total
    return np.divide(stacked.sum(axis=0), total, out=np.zeros_like(stacked[0]), where=total != 0.0)


def _broadcast_weight(layer: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Validate a layer/weight pair and return the weight broadcast over channels."""
    if layer.shape[:2] != weight.shape:
        raise ValueError(f"weight shape {weight.shape} does not match the image shape {layer.shape}.")
    if layer.ndim > weight.ndim:
        return weight[(...,) + (None,) * (layer.ndim - weight.ndim)]
    return weight


def compose_maps(
    data1: Any,
    data2: Any,
    *,
    style1: MapStyle | str | dict[str, Any] | None = None,
    style2: MapStyle | str | dict[str, Any] | None = None,
    mask: Any = None,
    line_angle: float = 45.0,
    width: float = 0.1,
) -> np.ndarray:
    """Render two data maps with different colour maps and stitch them together.

    Each map is normalised and coloured on its own, then the two are composited
    through :func:`create_map_mask` (or an explicit *mask*).  Non-finite pixels —
    empty bins, say — stay transparent.

    Parameters
    ----------
    data1, data2 : array_like
        The two maps, with the same shape.
    style1, style2 : MapStyle, str or dict, optional
        How each map becomes colours.  A bare string or colour map is read as
        ``cmap``, a mapping is expanded into the fields, and the default is a
        plain ``sauron_cmap`` rendering of that map's own range.
    mask : array_like, optional
        Weight of the first map in ``[0, 1]``.  Defaults to the soft split from
        :func:`create_map_mask` with *line_angle* and *width*.
    line_angle, width :
        Passed to :func:`create_map_mask` when *mask* is not given.

    Returns
    -------
    numpy.ndarray
        Float RGBA image of shape ``(*data1.shape, 4)``.  The alpha channel is
        the coverage of the two maps, so it can be composited over any background.

    Examples
    --------
    >>> composite = compose_maps(gas_density, dm_density, style1="inferno", style2="cividis")  # doctest: +SKIP
    """
    first = _values(data1)
    second = _values(data2)
    if first.shape != second.shape:
        raise ValueError(shape_hint(first.shape, second.shape, name="data2"))
    if mask is None:
        weights = create_map_mask(first, line_angle=line_angle, width=width)[0]
    else:
        weights = np.asarray(mask, dtype=float)
        if weights.shape != first.shape:
            raise ValueError(f"mask shape {weights.shape} does not match the data shape {first.shape}.")
        weights = np.clip(weights, 0.0, 1.0)

    rgba1 = as_map_style(style1).to_rgba(first)
    rgba2 = as_map_style(style2).to_rgba(second)
    weight1 = weights[..., None]
    alpha1 = rgba1[..., 3:4] * weight1
    alpha2 = rgba2[..., 3:4] * (1.0 - weight1)
    out_alpha = alpha1 + alpha2 * (1.0 - alpha1)
    color = np.divide(
        rgba1[..., :3] * alpha1 + rgba2[..., :3] * alpha2 * (1.0 - alpha1),
        out_alpha,
        out=np.zeros_like(rgba1[..., :3]),
        where=out_alpha > 0.0,
    )
    return np.concatenate([color, out_alpha], axis=-1)


def imshow_compose(
    data1: Any,
    data2: Any,
    *,
    ax: Any = None,
    extent: tuple[float, float, float, float] | None = None,
    colorbars: bool = True,
    style1: MapStyle | str | dict[str, Any] | None = None,
    style2: MapStyle | str | dict[str, Any] | None = None,
    label1: str | None = None,
    label2: str | None = None,
    **kwargs: Any,
) -> Any:
    """Draw :func:`compose_maps` into axes, with one colour bar per map.

    Parameters
    ----------
    data1, data2 : array_like
        The two maps.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    extent : (float, float, float, float), optional
        ``(xmin, xmax, ymin, ymax)`` of the maps.
    colorbars : bool, default: True
        Draw one colour bar per map, on the left and right of the image.
    style1, style2 : MapStyle, str or dict, optional
        How each map becomes colours; also what the colour bars are labelled with.
    label1, label2 : str, optional
        Colour bar labels.
    **kwargs
        Forwarded to :func:`compose_maps`.

    Returns
    -------
    matplotlib.image.AxesImage
        The image artist.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    first = _values(data1)
    second = _values(data2)
    styles = (as_map_style(style1), as_map_style(style2))
    figsize = kwargs.pop("figsize", (6.0, 5.0))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    composite = compose_maps(first, second, style1=styles[0], style2=styles[1], **kwargs)
    artist = ax.imshow(composite, origin="lower", extent=extent)
    if colorbars:
        bars = (("left", styles[0], label1, first), ("right", styles[1], label2, second))
        for side, style, label, data in bars:
            add_colorbar(
                ScalarMappable(norm=style.norm_for(data), cmap=get_cmap(style.cmap)), ax=ax, loc=side, label=label
            )
    return artist


@dataclass(frozen=True)
class ComposeOps(ImageOps):
    """The stitching family of an image: ``image.process.compose(other, …)``.

    Calling the view is how you stitch: ``gas.compose(dm, style=…, other_style=…)``
    returns the composited RGBA image.  :meth:`masks` exposes the transition masks on
    their own and :meth:`imshow` draws the pair with one colour bar each.
    """

    def __call__(
        self,
        other: Any,
        *,
        style: MapStyle | str | dict[str, Any] | None = None,
        other_style: MapStyle | str | dict[str, Any] | None = None,
        mask: Any = None,
        line_angle: float = 45.0,
        width: float = 0.1,
    ) -> np.ndarray:
        """Stitch this image together with *other* into one RGBA array.

        Each map is rendered through its own colour map and the two are crossfaded
        through :meth:`masks`, so a gas density map and a dark-matter map can share
        one figure with no seam between them.

        Parameters
        ----------
        other : ImageData or array_like
            The second map; an image must have the same shape as this one.
        style, other_style : MapStyle, str or dict, optional
            How each map becomes colours; *style* describes this image.  A bare
            colour-map name is accepted, so ``style="inferno"`` is enough.
        mask : array_like, optional
            Weight of this image in ``[0, 1]``; defaults to the soft line split.
        line_angle : float, default: 45.0
            Angle of that split, in degrees (see :meth:`masks`).
        width : float, default: 0.1
            Width of the transition band, as a fraction of the image diagonal.

        Returns
        -------
        numpy.ndarray
            Float RGBA image whose alpha channel is the coverage of the two maps.

        Examples
        --------
        >>> composite = gas.compose(dm, style="inferno", other_style="cividis")  # doctest: +SKIP
        >>> image.imshow_compose(gas, dm, extent=gas.extent)  # doctest: +SKIP

        See Also
        --------
        masks, imshow :
            The transition masks on their own, and the ready-made figure.
        :func:`~pynbodyext.plot.image.compose.compose_maps` :
            The array-level form.
        """
        return compose_maps(
            self.data, _values(other), style1=style, style2=other_style, mask=mask, line_angle=line_angle, width=width
        )

    def masks(
        self, line_angle: float = 45.0, width: float = 0.1, *, center: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split this image with two complementary soft masks.

        Useful on its own — to weight two layers, or to cut a region out of a map —
        as well as behind ``image.compose(other, …)``.

        Parameters
        ----------
        line_angle : float, default: 45.0
            Angle of the dividing line in degrees, measured from the ``+x``
            direction and increasing counter-clockwise: ``0`` splits left/right,
            ``90`` bottom/top, ``180`` puts this image on the left.
        width : float, default: 0.1
            Width of the transition band as a fraction of the image diagonal, so the
            softness looks the same at any resolution.  ``0`` gives a hard split.
        center : (float, float), optional
            Point the line passes through, in pixels; the image centre by default.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mask1, mask2)``, complementary floats in ``[0, 1]`` with the image's
            shape.

        Examples
        --------
        >>> mask1, mask2 = gas.compose.masks(line_angle=45, width=0.15)  # doctest: +SKIP
        >>> image.blend_images(rgb_gas, rgb_dm, mask1)  # doctest: +SKIP

        See Also
        --------
        :func:`~pynbodyext.plot.image.compose.create_map_mask` :
            The array-level form.
        """
        return create_map_mask(self.data, line_angle=line_angle, width=width, center=center)

    def imshow(
        self,
        other: Any,
        *,
        ax: Any = None,
        extent: tuple[float, float, float, float] | None = None,
        colorbars: bool = True,
        style: MapStyle | str | dict[str, Any] | None = None,
        other_style: MapStyle | str | dict[str, Any] | None = None,
        label1: str | None = None,
        label2: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Draw this image stitched with *other*, with one colour bar each.

        Parameters
        ----------
        other : ImageData or array_like
            The second map.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.
        extent : (float, float, float, float), optional
            ``(xmin, xmax, ymin, ymax)`` of the maps.
        colorbars : bool, default: True
            Draw one colour bar per map, on the left and right of the image.
        style, other_style : MapStyle, str or dict, optional
            How each map becomes colours; *style* describes this image.  The same
            styles decide what the two colour bars show, including their norm.
        label1, label2 : str, optional
            Colour-bar labels, e.g. ``"gas"`` and ``"dark matter"``.
        **kwargs
            Forwarded to :func:`imshow_compose` beyond the above, e.g. ``line_angle``,
            ``width``, ``mask`` or ``figsize``.

        Returns
        -------
        matplotlib.image.AxesImage
            The artist.

        Examples
        --------
        >>> gas.compose.imshow(
        ...     dm,
        ...     style="inferno",
        ...     other_style="cividis",  # doctest: +SKIP
        ...     label1="gas",
        ...     label2="dark matter",
        ... )

        See Also
        --------
        :func:`~pynbodyext.plot.image.compose.imshow_compose` :
            The array-level form.
        """
        return imshow_compose(
            self.data,
            _values(other),
            ax=ax,
            extent=extent,
            colorbars=colorbars,
            style1=style,
            style2=other_style,
            label1=label1,
            label2=label2,
            **kwargs,
        )


def _values(value: Any) -> np.ndarray:
    """Values of anything image-like — an ImageData, a binned array, or an array.

    Binned arrays are transposed into image orientation on the way (see
    :func:`~pynbodyext.plot.image.data.as_image`), so ``image.process.compose(bins.s[q])``
    lines up instead of silently turning the map on its side.
    """
    return aligned_values(value)
