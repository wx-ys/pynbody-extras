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
        cmap1="inferno",
        cmap2="cividis",
        label1="gas",
        label2="dark matter",
        line_angle=45,
        width=0.15,
        extent=(-50, 50, -50, 50),
    )

The masks are plain arrays in ``[0, 1]``, so they also drive :func:`blend_images`
and :func:`blend_stack` directly when the two layers are already rendered.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .cmaps import K_B_C_G_Y_R_W, get_cmap, to_rgba

__all__ = ["blend_images", "blend_stack", "compose_maps", "create_map_mask", "imshow_compose"]


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
    cmap1: Any = K_B_C_G_Y_R_W,
    cmap2: Any = K_B_C_G_Y_R_W,
    vmin1: float | None = None,
    vmax1: float | None = None,
    vmin2: float | None = None,
    vmax2: float | None = None,
    stretch1: str = "linear",
    stretch2: str = "linear",
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
    cmap1, cmap2 : str or Colormap, optional
        Colour map of each map; defaults to ``K_B_C_G_Y_R_W``.
    vmin1, vmax1, vmin2, vmax2 : float, optional
        Display limits of each map; default to that map's own range.
    stretch1, stretch2 : str, default: "linear"
        Display stretch of each map, see
        :func:`~pynbodyext.plot.image.postprocess.normalize`.
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
    >>> composite = compose_maps(gas_density, dm_density, cmap1="inferno", cmap2="cividis")  # doctest: +SKIP
    """
    first = np.asarray(data1, dtype=float)
    second = np.asarray(data2, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"data1 shape {first.shape} does not match data2 shape {second.shape}.")
    if mask is None:
        weights = create_map_mask(first, line_angle=line_angle, width=width)[0]
    else:
        weights = np.asarray(mask, dtype=float)
        if weights.shape != first.shape:
            raise ValueError(f"mask shape {weights.shape} does not match the data shape {first.shape}.")
        weights = np.clip(weights, 0.0, 1.0)

    rgba1 = to_rgba(first, get_cmap(cmap1), vmin=vmin1, vmax=vmax1, stretch=stretch1)
    rgba2 = to_rgba(second, get_cmap(cmap2), vmin=vmin2, vmax=vmax2, stretch=stretch2)
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
    from matplotlib.colors import Normalize

    first = np.asarray(data1, dtype=float)
    second = np.asarray(data2, dtype=float)
    figsize = kwargs.pop("figsize", (6.0, 5.0))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    composite = compose_maps(first, second, **kwargs)
    artist = ax.imshow(composite, origin="lower", extent=extent)
    if colorbars:
        bars = (
            ("left", kwargs.get("cmap1", K_B_C_G_Y_R_W), kwargs.get("vmin1"), kwargs.get("vmax1"), label1, first),
            ("right", kwargs.get("cmap2", K_B_C_G_Y_R_W), kwargs.get("vmin2"), kwargs.get("vmax2"), label2, second),
        )
        for side, colors, vmin, vmax, label, data in bars:
            bar = ax.figure.colorbar(
                ScalarMappable(norm=Normalize(*_display_limits(data, vmin, vmax)), cmap=get_cmap(colors)),
                ax=ax,
                location=side,
                fraction=0.046,
                pad=0.04,
            )
            if label is not None:
                bar.set_label(label)
    return artist


def _display_limits(data: np.ndarray, vmin: float | None, vmax: float | None) -> tuple[float, float]:
    """Colour limits of one map, mirroring the defaults used by ``compose_maps``."""
    finite = np.isfinite(data)
    if vmin is None:
        vmin = float(np.nanmin(data)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(data)) if finite.any() else 1.0
    return float(vmin), float(vmax)
