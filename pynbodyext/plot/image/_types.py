"""Type aliases for the image layer.

The vocabulary the annotations use, so a signature reads as what it accepts rather
than as ``Any``: a *map* can be an image, a binned array or a plain array; a *mask*
can be any of those or nothing; a *kernel width* is a number, a pynbody unit or a
``(y, x)`` pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

from numpy.typing import ArrayLike
from pynbody.units import UnitBase

from pynbodyext.util._type import UnitLike

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.contour import ContourSet
    from matplotlib.image import AxesImage

    from .data import ImageData

__all__ = ["Artist", "KernelWidth", "MapLike", "MaskLike", "UnitLike"]

#: A map as the layer accepts it: an image, a binned array, or an array.
MapLike: TypeAlias = Union["ImageData", ArrayLike]

#: A per-pixel mask: an array, an image, a binned array — or nothing at all.
MaskLike: TypeAlias = Union[ArrayLike, "ImageData", None]

#: A kernel width: a number (in pixels or in the units of the axes), a pynbody unit,
#: or a ``(y, x)`` pair.  Strings are *not* accepted — ``"1 kpc"`` has to be a
#: number or a unit, so that the conversion is never a guess.
KernelWidth: TypeAlias = float | UnitBase | tuple[float, float]

#: Anything the drawing methods return: an image, a mesh, a contour set or a bar.
Artist: TypeAlias = Union["AxesImage", "QuadMesh", "ContourSet", "Colorbar"]
