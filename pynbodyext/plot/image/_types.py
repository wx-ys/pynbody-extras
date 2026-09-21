"""Type aliases for the image layer, and the runtime half of its annotations.

The vocabulary the annotations use, so a signature reads as what it accepts rather
than as ``Any``: a *map* can be an image, a binned array or a plain array; a *mask*
can be any of those or nothing; a *kernel width* is a number, a pynbody unit or a
``(y, x)`` pair; an *artist* is anything a drawing method returns, which is also
what a colour bar can describe.

The annotations are strings deferred to check time (PEP 563), so a module that
writes ``MapLike`` does not have to import ``ImageData`` for Python to run — which
is what keeps the layer's import light: the drawing stack is imported when you
draw, and the calculator when you bin.  ``typing.get_type_hints``, and the
documentation, validation and plugin tools built on it, resolve those strings
against the globals of the module that wrote them, so a bare call raises
``NameError`` for the deferred names; :func:`resolve_type_hints` is the supported
way to resolve them, and it is what the layer's own tests use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Union

from numpy.typing import ArrayLike
from pynbody.units import UnitBase

from pynbodyext.util._type import UnitLike

if TYPE_CHECKING:
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.contour import ContourSet
    from matplotlib.image import AxesImage

    from .data import ImageData

__all__ = ["Artist", "KernelWidth", "MapLike", "MaskLike", "UnitLike", "resolve_type_hints"]

#: A map as the layer accepts it: an image, a binned array, or an array.
MapLike: TypeAlias = Union["ImageData", ArrayLike]

#: A per-pixel mask: an array, an image, a binned array — or nothing at all.
MaskLike: TypeAlias = Union[ArrayLike, "ImageData", None]

#: A kernel width: a number (in pixels or in the units of the axes), a pynbody unit,
#: or a ``(y, x)`` pair.  Strings are *not* accepted — ``"1 kpc"`` has to be a
#: number or a unit, so that the conversion is never a guess.
KernelWidth: TypeAlias = float | UnitBase | tuple[float, float]

#: Anything the drawing methods return, and so anything a colour bar can describe:
#: an image, a mesh, a contour set, a scatter collection or a bar.
Artist: TypeAlias = Union["AxesImage", "QuadMesh", "ContourSet", "PathCollection", "Colorbar"]


def annotation_namespace() -> dict[str, object]:
    """Every name the layer's annotations use, resolved to the object it means.

    Importing this imports what the annotations name — matplotlib's artist classes
    and the calculator's binned-result types — which is exactly what the layer
    defers until you draw or bin.  A consumer that wants the hints pays that cost
    here; see :func:`resolve_type_hints`.

    Returns
    -------
    dict of str to object
        The vocabulary, ready to hand to :func:`typing.get_type_hints`.
    """
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.contour import ContourSet
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.typing import ColorType
    from mpl_toolkits.axes_grid1.axes_divider import AxesDivider

    from pynbodyext.core.calculate.bins.arrays import BinsArray
    from pynbodyext.core.calculate.bins.axes import BinAxis
    from pynbodyext.core.calculate.bins.result import BinNDResult

    from .adaptive import AdaptiveMap, AdaptiveOps
    from .compose import ComposeOps, MapStyle
    from .data import ImageData
    from .display import DisplayOps
    from .noise import NoiseOps
    from .ops import ImageDataView, ImageOp, ImageOps, ProcessOps
    from .psf import PsfOps
    from .smooth import SmoothOps

    namespace: dict[str, object] = {
        "AdaptiveMap": AdaptiveMap,
        "AdaptiveOps": AdaptiveOps,
        "ArrayLike": ArrayLike,
        "Artist": Artist,
        "Axes": Axes,
        "AxesDivider": AxesDivider,
        "AxesImage": AxesImage,
        "BinAxis": BinAxis,
        "BinNDResult": BinNDResult,
        "BinsArray": BinsArray,
        "Callable": Callable,
        "ColorType": ColorType,
        "Colorbar": Colorbar,
        "Colormap": Colormap,
        "ComposeOps": ComposeOps,
        "ContourSet": ContourSet,
        "DisplayOps": DisplayOps,
        "ImageData": ImageData,
        "ImageDataView": ImageDataView,
        "ImageOp": ImageOp,
        "ImageOps": ImageOps,
        "KernelWidth": KernelWidth,
        "Line2D": Line2D,
        "MapLike": MapLike,
        "MapStyle": MapStyle,
        "MaskLike": MaskLike,
        "NoiseOps": NoiseOps,
        "Normalize": Normalize,
        "PathCollection": PathCollection,
        "ProcessOps": ProcessOps,
        "PsfOps": PsfOps,
        "QuadMesh": QuadMesh,
        "Sequence": Sequence,
        "SmoothOps": SmoothOps,
        "UnitLike": UnitLike,
    }
    return namespace


def resolve_type_hints(obj: object) -> dict[str, object]:
    """``typing.get_type_hints`` for anything this layer annotates.

    A bare ``typing.get_type_hints`` resolves an annotation against the globals of
    the module that wrote it, and these annotations deliberately name types those
    modules do not import: the image composes the views and the views import the
    image, and the calculator bridge names ``BinNDResult``/``BinsArray``/``BinAxis``.
    This resolves against the module's own globals *plus*
    :func:`annotation_namespace`, so every annotation of the layer and of its
    calculator bridge comes out as a real type.

    Parameters
    ----------
    obj : object
        Anything the layer annotates: a function, a method, a property's getter, a
        class, or the bridge's ``BinsArray.image``.

    Returns
    -------
    dict of str to object
        Annotation name to the type it means.

    Examples
    --------
    >>> from pynbodyext.plot.image.display import draw_image
    >>> resolve_type_hints(draw_image)["image"]
    <class 'pynbodyext.plot.image.data.ImageData'>
    """
    import sys
    import typing

    module = sys.modules.get(getattr(obj, "__module__", None) or "")
    globalns = dict(vars(module)) if module is not None else {}
    globalns.update(annotation_namespace())
    return typing.get_type_hints(obj, globalns=globalns, localns=globalns)
