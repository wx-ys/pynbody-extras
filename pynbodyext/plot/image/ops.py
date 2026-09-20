"""The shared parent of the image's capability views, and the registry that finds them.

``ImageData.smooth``, ``.psf``, ``.compose`` and ``.adaptive`` are *views*: small
objects that hold the image and expose one family of operations.  They all derive
from :class:`ImageOps`, which gives them — and any plugin added later — the same
surface: the image's geometry and metadata, :meth:`ImageOps.derive` to return a new
image with the operation recorded, and :meth:`ImageOps.limits` for colour scaling.

New families do not require touching :class:`ImageData` (and certainly not its base
classes): define a subclass and register it::

    from pynbodyext.plot.image import ImageData, ImageOps


    @ImageData.register_ops("tessellation")
    class TessellationOps(ImageOps):
        def cells(self, size="1 kpc"):
            return self.derive(tessellate_function(self.data, size), "tessellate", {"size": size})


    image.tessellation.cells()  # the view is found through the registry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._arrays import value_limits

if TYPE_CHECKING:
    import numpy as np

    from .data import ImageData

__all__ = ["OPERATIONS", "ImageOps", "register_ops"]

#: Registered capability views, keyed by the attribute they answer to.
OPERATIONS: dict[str, type[ImageOps]] = {}


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
class ImageOps:
    """Base class for the capability views of an image.

    Subclasses implement one family of operations; this class gives them the
    image's geometry and metadata, plus the two helpers a plugin needs.

    Examples
    --------
    >>> ops = SomeOps(image)  # doctest: +SKIP
    >>> ops.shape == image.shape  # doctest: +SKIP
    True
    """

    image: ImageData

    # ---- geometry and metadata, so a plugin never reaches into ImageData ----

    @property
    def data(self) -> np.ndarray:
        """Values of the image."""
        return self.image.data

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the image."""
        return self.image.shape

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

    @property
    def uniform(self) -> bool:
        """Whether both axes are evenly spaced."""
        return self.image.uniform

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Size of one pixel as ``(dy, dx)``, in the units of the axes."""
        return self.image.pixel_size

    # ---- what a plugin actually needs ----

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
