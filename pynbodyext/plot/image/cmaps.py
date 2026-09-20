"""Colour maps and colour helpers for image display.

The headline entry is :data:`K_B_C_G_Y_R_W` — black → blue → cyan → green →
yellow → red → white — the classic kinematic colour map, built with green on
zero velocity (``0.50``), cyan/yellow on the shoulders and black/white at the
extremes so that the sign of a velocity is readable at a glance.  It is
registered with matplotlib, so ``cmap="K_B_C_G_Y_R_W"`` also works::

    ax.imshow(velocity_map, cmap="K_B_C_G_Y_R_W", vmin=-200, vmax=200)

Note that a 256-entry look-up table samples the declared colour stops
approximately: :data:`K_B_C_G_Y_R_W_POSITIONS` are exact in the continuous
definition, and matplotlib interpolates them onto ``N`` levels.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap

from .postprocess import normalize

__all__ = [
    "K_B_C_G_Y_R_W",
    "K_B_C_G_Y_R_W_COLORS",
    "K_B_C_G_Y_R_W_POSITIONS",
    "cmap_from_colors",
    "get_cmap",
    "register_cmap",
    "to_rgba",
    "vel_cmap",
    "vel_cmap_r",
]

#: Colour stops of the velocity map, black → blue → cyan → green → yellow → red → white.
K_B_C_G_Y_R_W_COLORS = ["#000000", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000", "#FFFFFF"]

#: Positions of :data:`K_B_C_G_Y_R_W_COLORS`, symmetric about green at ``0.50``.
K_B_C_G_Y_R_W_POSITIONS = [0.00, 0.18, 0.44, 0.50, 0.56, 0.84, 1.00]


def cmap_from_colors(name: str, colors: Any, positions: Any = None, *, N: int = 256) -> LinearSegmentedColormap:
    """Build a :class:`~matplotlib.colors.LinearSegmentedColormap` from stops.

    Parameters
    ----------
    name : str
        Name of the colour map.
    colors : sequence
        Colours, anything matplotlib accepts (hex strings, RGB tuples, names).
    positions : sequence of float, optional
        One position in ``[0, 1]`` per colour, non-decreasing.  Defaults to
        evenly spaced stops.
    N : int, default: 256
        Number of look-up table entries.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The colour map (not registered; use :func:`register_cmap`).
    """
    color_list = [mcolors.to_rgba(color) for color in colors]
    if positions is None:
        position_array = np.linspace(0.0, 1.0, len(color_list))
    else:
        position_array = np.asarray(positions, dtype=float)
    if position_array.size != len(color_list):
        raise ValueError(f"positions must hold one entry per colour ({len(color_list)}), got {position_array.size}.")
    if position_array[0] < 0.0 or position_array[-1] > 1.0 or np.any(np.diff(position_array) < 0.0):
        raise ValueError("positions must be non-decreasing and inside [0, 1].")
    return LinearSegmentedColormap.from_list(name, list(zip(position_array, color_list, strict=True)), N=N)


def register_cmap(cmap: Colormap, name: str | None = None, *, overwrite: bool = False) -> Colormap:
    """Register *cmap* with matplotlib so it can be named in ``cmap=`` arguments.

    Registering an already-known name is a no-op unless *overwrite* is set, which
    keeps module re-imports (and repeated test runs) harmless.

    Returns
    -------
    matplotlib.colors.Colormap
        The colour map passed in, for chaining.
    """
    try:
        matplotlib.colormaps.register(cmap, name=name or cmap.name, force=overwrite)
    except ValueError:
        if overwrite:
            raise
    return cmap


def get_cmap(cmap: Colormap | str | None = None) -> Colormap:
    """Resolve *cmap* to a colour map object.

    ``None`` yields the default :data:`K_B_C_G_Y_R_W`; a string is looked up in
    the matplotlib registry; a colour map is returned unchanged.

    Raises
    ------
    ValueError
        If a name is not registered with matplotlib.
    """
    if cmap is None:
        return K_B_C_G_Y_R_W
    if isinstance(cmap, Colormap):
        return cmap
    try:
        return matplotlib.colormaps[cmap]
    except KeyError as exc:
        raise ValueError(f"Unknown colour map {cmap!r}.") from exc


def to_rgba(
    data: Any,
    cmap: Colormap | str | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    stretch: str = "linear",
    percentiles: tuple[float, float] | None = None,
    norm: Any = None,
    alpha: Any = None,
    bad: Any = None,
) -> np.ndarray:
    """Map a 2-D array of numbers to an ``(ny, nx, 4)`` RGBA image.

    Parameters
    ----------
    data : array_like
        2-D image.  Non-finite pixels come out transparent (or *bad*).
    cmap : str or Colormap, optional
        Colour map; defaults to :data:`K_B_C_G_Y_R_W`.
    vmin, vmax, stretch, percentiles :
        Passed to :func:`~pynbodyext.plot.image.postprocess.normalize` to map the
        values onto ``[0, 1]``.
    norm : matplotlib.colors.Normalize, optional
        A ready-made norm, used instead of ``vmin``/``vmax``/``stretch``.
    alpha : float or array_like, optional
        Constant opacity, or a per-pixel array with the shape of *data*.
    bad : color, optional
        Colour for non-finite pixels; the default is fully transparent.

    Returns
    -------
    numpy.ndarray
        Float RGBA array with the shape ``(*data.shape, 4)``.

    Examples
    --------
    >>> import numpy as np
    >>> rgba = to_rgba(np.linspace(0, 1, 4).reshape(1, 4))
    >>> rgba.shape
    (1, 4, 4)
    """
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"data must be a 2-D image, got shape {array.shape}.")
    colors = get_cmap(cmap)
    if norm is not None:
        values = np.asarray(norm(array), dtype=float)
    else:
        values = normalize(array, vmin=vmin, vmax=vmax, stretch=stretch, percentiles=percentiles)
    if bad is not None:
        colors = colors.with_extremes(bad=bad)
    rgba = np.asarray(colors(values), dtype=float)
    if alpha is not None:
        alpha_array = np.asarray(alpha, dtype=float)
        if alpha_array.ndim == 0:
            rgba[..., 3] = float(alpha_array)
        elif alpha_array.shape == array.shape:
            rgba[..., 3] = alpha_array
        else:
            raise ValueError(f"alpha must be a scalar or have shape {array.shape}, got {alpha_array.shape}.")
    return rgba


#: The velocity colour map, registered so that ``cmap="K_B_C_G_Y_R_W"`` works.
K_B_C_G_Y_R_W: LinearSegmentedColormap = cmap_from_colors(
    "K_B_C_G_Y_R_W", K_B_C_G_Y_R_W_COLORS, K_B_C_G_Y_R_W_POSITIONS, N=256
)

#: Alias of :data:`K_B_C_G_Y_R_W` under the name it usually goes by in plotting code.
vel_cmap: LinearSegmentedColormap = K_B_C_G_Y_R_W

#: Reversed velocity colour map (white → red → … → black).
vel_cmap_r: LinearSegmentedColormap = vel_cmap.reversed(name="K_B_C_G_Y_R_W_r")

register_cmap(vel_cmap)
register_cmap(vel_cmap_r)
