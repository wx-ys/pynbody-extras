"""Colour maps and colour helpers for image display.

The headline entry is :data:`sauron_cmap` — the **SAURON colormap**: black → blue
→ cyan → green → yellow → red → light grey, with green on zero.  It is the map
used for the stellar velocity fields of the SAURON and ATLAS³D integral-field
surveys, which is why it is the default for velocity maps here::

    ax.imshow(velocity_map, cmap="sauron", vmin=-200, vmax=200)

Source and citation
-------------------
The colours are Michele Cappellari & Eric Emsellem's SAURON colormap (Leiden,
2001).  :data:`SAURON_POSITIONS` and :data:`SAURON_RGB` below are that published
table verbatim — eleven control points — and the map is built from them exactly as
the reference implementation does, so the two look-up tables are identical (a test
asserts it whenever ``plotbin`` is installed).

The reference implementation is ``plotbin/sauron_colormap.py`` (Copyright (C)
2014-2024 Michele Cappellari, https://purl.org/cappellari, PyPI ``plotbin``), which
registers the map with matplotlib as ``sauron``/``sauron_r``.  If that package has
registered them first, :func:`register_cmap` keeps its entry rather than ours — the
colours are the same either way, so ``cmap="sauron"`` is unambiguous.  To cite the
map in a publication, credit Cappellari & Emsellem's SAURON colormap and the
SAURON/ATLAS³D kinematic figures it comes from.

Note that a 256-entry look-up table samples the control points approximately: the
table is exact in the continuous definition, and matplotlib interpolates it onto
``N`` levels.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap

from .smooth import normalize, stretch_functions

__all__ = [
    "SAURON_POSITIONS",
    "SAURON_RGB",
    "as_norm",
    "cmap_from_colors",
    "get_cmap",
    "is_log_norm",
    "norm_from_stretch",
    "register_cmap",
    "sauron_cmap",
    "sauron_cmap_r",
    "to_rgba",
    "vel_cmap",
    "vel_cmap_r",
]

#: Matplotlib scale names accepted where a norm is expected (``norm="log"`` and
#: friends), mapped to the matching public norm class — the ones this matplotlib
#: has.  Anything more exotic — a ``PowerNorm`` with a custom exponent, say — has to
#: be passed as an instance.
_NAMED_NORMS: dict[str, type[matplotlib.colors.Normalize]] = {
    name: getattr(mcolors, class_name)
    for name, class_name in (
        ("linear", "Normalize"),
        ("log", "LogNorm"),
        ("symlog", "SymLogNorm"),
        ("logit", "LogitNorm"),  # matplotlib >= 3.9
        ("asinh", "AsinhNorm"),
    )
    if hasattr(mcolors, class_name)
}


def as_norm(norm: Any) -> Any:
    """Resolve a norm that may be given as a matplotlib scale *name*.

    Matplotlib lets ``norm=`` be a string (``"log"``, ``"symlog"``, ``"linear"``, …)
    as well as a ``Normalize`` instance, and plotting code uses both.  This layer
    needs an instance: it has to know whether the scale is logarithmic (to space
    contour levels geometrically) and it has to apply the norm itself when mapping
    values to colours.

    Parameters
    ----------
    norm : str, matplotlib.colors.Normalize or None
        The scale to resolve.

    Returns
    -------
    matplotlib.colors.Normalize or None
        The same instance when one was given, an equivalent instance for a known
        name, and ``None`` unchanged.

    Raises
    ------
    ValueError
        If a string is not the name of a scale this layer knows.
    """
    if norm is None or not isinstance(norm, str):
        return norm
    try:
        return _NAMED_NORMS[norm]()
    except KeyError:
        raise ValueError(
            f"Unknown colour scale {norm!r}; pass one of {', '.join(_NAMED_NORMS)} "
            "or a matplotlib.colors.Normalize instance."
        ) from None


def is_log_norm(norm: Any) -> bool:
    """Whether *norm* asks for a logarithmic scale, as a name or an instance."""
    return norm == "log" or isinstance(norm, mcolors.LogNorm)


#: Positions of the SAURON control points, from the published table (``x/255``).
#: The table is symmetric about ``0.5`` (green, the zero of the velocity field):
#: ``(1 - x)[::-1] - x == 0``.
SAURON_POSITIONS = np.array([0, 42.5, 85, 105, 117.5, 127.5, 137.5, 150, 170, 212.5, 255]) / 255.0

#: Red, green and blue values at :data:`SAURON_POSITIONS` — Cappellari & Emsellem's
#: SAURON colormap table, as distributed in ``plotbin`` (see the module docstring).
#: Black at the negative end, light grey (0.9) at the positive end, green on zero.
SAURON_RGB = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.4, 0.85, 1.0],
        [0.5, 1.0, 1.0],
        [0.3, 1.0, 0.7],
        [0.0, 0.9, 0.0],
        [0.7, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.85, 0.0],
        [1.0, 0.0, 0.0],
        [0.9, 0.9, 0.9],
    ]
)


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


def norm_from_stretch(
    stretch: str = "linear", *, vmin: float, vmax: float, asinh_a: float = 10.0
) -> matplotlib.colors.Normalize:
    """A matplotlib norm whose mapping matches the *stretch* of :func:`to_rgba`.

    Drawing a stretched image with the matching norm is what makes its colour bar
    tell the truth: without it a ``stretch="log"`` image would be labelled by a
    linear scale, and the ticks would sit where the colours are not.

    Parameters
    ----------
    stretch : {"linear", "sqrt", "log", "asinh", "hist"}, default: "linear"
        The display stretch, as in
        :func:`~pynbodyext.plot.image.smooth.normalize`.
    vmin, vmax : float
        Limits the stretch was applied over.
    asinh_a : float, default: 10.0
        Softening parameter of the ``"asinh"`` stretch.

    Returns
    -------
    matplotlib.colors.Normalize
        A ``FuncNorm`` for the invertible stretches, and a plain ``Normalize`` for
        ``"linear"`` and for ``"hist"`` — the latter equalises over the whole image,
        so no per-value scale can represent it (stretch the image with
        ``display.normalize`` first if the bar must match exactly).
    """
    functions = stretch_functions(stretch, asinh_a=asinh_a)
    if functions is None or stretch == "linear":
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    forward, inverse = functions
    endpoints = np.asarray(forward(np.array([0.0, 1.0])), dtype=float)
    if not np.all(np.isfinite(endpoints)) or not np.all(np.isfinite(inverse(endpoints))):
        return mcolors.Normalize(vmin=vmin, vmax=vmax)  # not representable on this range
    # A ``FuncNorm`` transforms *values*, not positions, so the unit-interval stretch
    # is wrapped into the data range; the norm then reproduces ``normalize`` exactly,
    # including its clamping, which is what makes the colour bar truthful.
    span = float(vmax) - float(vmin)
    to_stretched = lambda value: forward((np.asarray(value, dtype=float) - vmin) / span)  # noqa: E731
    from_stretched = lambda position: vmin + inverse(np.asarray(position, dtype=float)) * span  # noqa: E731
    return mcolors.FuncNorm(functions=(to_stretched, from_stretched), vmin=vmin, vmax=vmax, clip=True)


def get_cmap(cmap: Colormap | str | None = None) -> Colormap:
    """Resolve *cmap* to a colour map object.

    ``None`` yields the default :data:`sauron_cmap`; a string is looked up in
    the matplotlib registry; a colour map is returned unchanged.

    Raises
    ------
    ValueError
        If a name is not registered with matplotlib.
    """
    if cmap is None:
        return sauron_cmap
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
        Colour map; defaults to :data:`sauron_cmap`.
    vmin, vmax, stretch, percentiles :
        Passed to :func:`~pynbodyext.plot.image.smooth.normalize` to map the
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
        values = np.asarray(as_norm(norm)(array), dtype=float)
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


def _sauron_cdict(reverse: bool = False) -> dict[str, np.ndarray]:
    """Segment data of the SAURON map, built the way ``plotbin`` builds it.

    Mirroring that construction (rather than re-deriving the colours) is what keeps
    the two look-up tables identical; ``reverse`` flips the colour columns, which is
    exactly how ``plotbin`` defines ``sauron_r``.
    """
    red, green, blue = (SAURON_RGB[:, channel][::-1] if reverse else SAURON_RGB[:, channel] for channel in range(3))
    return {
        "red": np.column_stack([SAURON_POSITIONS, red, red]),
        "green": np.column_stack([SAURON_POSITIONS, green, green]),
        "blue": np.column_stack([SAURON_POSITIONS, blue, blue]),
    }


#: The SAURON colormap, registered so that ``cmap="sauron"`` works.
sauron_cmap: LinearSegmentedColormap = LinearSegmentedColormap("sauron", _sauron_cdict())

#: Reversed SAURON colormap (light grey → red → … → black), registered as ``"sauron_r"``.
sauron_cmap_r: LinearSegmentedColormap = LinearSegmentedColormap("sauron_r", _sauron_cdict(reverse=True))

#: Alias of :data:`sauron_cmap` under the name it usually goes by in plotting code.
vel_cmap: LinearSegmentedColormap = sauron_cmap

#: Alias of :data:`sauron_cmap_r`.
vel_cmap_r: LinearSegmentedColormap = sauron_cmap_r

register_cmap(sauron_cmap)
register_cmap(sauron_cmap_r)
