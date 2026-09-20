"""Views onto an image: the shared base, the capability views, and their registry.

Two kinds of object in this package mean "an :class:`ImageData`, seen from a
particular angle":

- a **capability view** — ``ImageData.smooth``, ``.psf``, ``.compose``, ``.adaptive``
  — which exposes one family of operations (all of them derive from
  :class:`ImageOps`);
- a **result that is also an image** — :class:`~pynbodyext.plot.image.adaptive.AdaptiveMap`,
  whose painted map is an ``ImageData``.

Both need the image's geometry, units and labels, so that forwarding lives here,
once, in :class:`ImageDataView`.  Subclasses only add what is theirs.

New capability families do not require touching :class:`ImageData` (and certainly
not its base classes): define a subclass and register it::

    from pynbodyext.plot.image import ImageData, ImageOps


    @ImageData.register_ops("tessellation")
    class TessellationOps(ImageOps):
        def cells(self, size="1 kpc"):
            return self.derive(tessellate_function(self.data, size), "tessellate", {"size": size})


    image.tessellation.cells()  # the view is found through the registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ._arrays import value_limits

if TYPE_CHECKING:
    from .data import ImageData

__all__ = ["OPERATIONS", "ImageDataView", "ImageOp", "ImageOps", "register_ops"]

#: Registered capability views, keyed by the attribute they answer to.
OPERATIONS: dict[str, type[ImageOps]] = {}


def _describe(value: Any) -> str:
    """Compact rendering of one recorded operation parameter."""
    if isinstance(value, np.ndarray):
        return f"<array {value.shape}>"
    if hasattr(value, "shape") and hasattr(value, "extent") and hasattr(value, "data"):
        return f"<ImageData {value.shape}>"  # an image passed as an argument
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


def register_ops(name: str, view: type[ImageOps] | None = None, *, overwrite: bool = False) -> Any:
    """Register *view* as the capability answering to ``image.<name>``.

    Usable as a decorator (``@ImageData.register_ops("name")``) or as a call.

    Raises
    ------
    KeyError
        If *name* is taken and *overwrite* is false.
    TypeError
        If *view* is not an :class:`ImageOps` subclass.
    """

    def register(view_class: type[ImageOps]) -> type[ImageOps]:
        if not (isinstance(view_class, type) and issubclass(view_class, ImageOps)):
            raise TypeError(f"{view_class!r} must be an ImageOps subclass.")
        existing = OPERATIONS.get(name)
        if existing is not None and existing is not view_class and not overwrite:
            raise KeyError(f"A capability named {name!r} is already registered; pass overwrite=True to replace it.")
        OPERATIONS[name] = view_class
        return view_class

    return register(view) if view is not None else register


@dataclass(frozen=True)
class ImageDataView:
    """An :class:`ImageData` seen from one angle, forwarding its data and metadata.

    The view holds the image as :attr:`image` and re-exposes what every view is
    asked for — values, shape, geometry, units, labels — so subclasses (capability
    views, :class:`AdaptiveMap`) declare only the attributes that are their own.

    Attributes are read-only: writing to a view is not how anything here is
    meant to work.

    Examples
    --------
    >>> view = ImageDataView(image)  # doctest: +SKIP
    >>> (view.shape, view.extent) == (image.shape, image.extent)  # doctest: +SKIP
    True
    """

    image: ImageData

    @property
    def data(self) -> np.ndarray:
        """Values of the image."""
        return self.image.data

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the image."""
        return self.image.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions: always 2."""
        return self.image.ndim

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        """``(xmin, xmax, ymin, ymax)`` of the image."""
        return self.image.extent

    @property
    def x_edges(self) -> np.ndarray | None:
        """Bin edges along x."""
        return self.image.x_edges

    @property
    def y_edges(self) -> np.ndarray | None:
        """Bin edges along y."""
        return self.image.y_edges

    @property
    def x_centers(self) -> np.ndarray:
        """Centre of every column."""
        return self.image.x_centers

    @property
    def y_centers(self) -> np.ndarray:
        """Centre of every row."""
        return self.image.y_centers

    @property
    def x_uniform(self) -> bool:
        """Whether the columns are evenly spaced."""
        return self.image.x_uniform

    @property
    def y_uniform(self) -> bool:
        """Whether the rows are evenly spaced."""
        return self.image.y_uniform

    @property
    def uniform(self) -> bool:
        """Whether both axes are evenly spaced."""
        return self.image.uniform

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Size of one pixel as ``(dy, dx)``, in the units of the axes."""
        return self.image.pixel_size

    @property
    def x_units(self) -> Any:
        """Units of the x axis."""
        return self.image.x_units

    @property
    def y_units(self) -> Any:
        """Units of the y axis."""
        return self.image.y_units

    @property
    def x_label(self) -> str | None:
        """Name of the x axis."""
        return self.image.x_label

    @property
    def y_label(self) -> str | None:
        """Name of the y axis."""
        return self.image.y_label

    @property
    def label(self) -> str | None:
        """Name of the quantity the values represent."""
        return self.image.label

    @property
    def units(self) -> Any:
        """Units of the values."""
        return self.image.units


@dataclass(frozen=True)
class ImageOps(ImageDataView):
    """Base class for the capability views of an image.

    Subclasses implement one family of operations; on top of the image metadata
    forwarded by :class:`ImageDataView`, this adds the two helpers a family needs.

    Examples
    --------
    >>> ops = SomeOps(image)  # doctest: +SKIP
    >>> ops.shape == image.shape  # doctest: +SKIP
    True
    """

    def derive(self, data: Any, op_name: str, params: dict[str, Any] | None = None, **overrides: Any) -> ImageData:
        """Return a new image carrying *data*, recording *op_name* in ``.ops``.

        Shape-changing operations must pass their own ``x_edges``/``y_edges``.
        """
        return self.image._derived(data, op_name, params, **overrides)

    def limits(self) -> tuple[float, float]:
        """The value range this image would be drawn with."""
        return value_limits(self.image.data)

    def kernel_scale(self) -> tuple[float, float] | None:
        """Pixel size in axis units, or ``None`` when the grid has no single one."""
        try:
            return self.image.pixel_size
        except ValueError:  # unevenly spaced bins: kernel widths fall back to pixels
            return None
