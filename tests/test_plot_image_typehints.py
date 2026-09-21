"""The image layer's annotations must name real types, not ``Any``.

``Any`` in a signature tells a reader and a type checker nothing: it silences the
checker instead of describing the argument.  The layer has its own vocabulary for
what it accepts — an :class:`~pynbodyext.plot.image.data.ImageData`, a
``BinsArray``, a plain array (``MapLike``/``MaskLike``), a kernel width, the artist
a drawing method returns — so the annotations use it, and these tests keep them
from drifting back to ``Any``.

``Any`` stays where it is honest: inside a container (``dict[str, Any]``,
``Sequence[Any]``) and as the type of ``**kwargs`` forwarded to matplotlib.

The annotations are deferred (PEP 563) and several of the types they name are
imported lazily — matplotlib's drawing stack when you draw, the calculator's
binned-result types when you bin — so resolving them takes
:func:`~pynbodyext.plot.image._types.resolve_type_hints` rather than a bare
``typing.get_type_hints``.  These tests use it, which also proves the namespace it
resolves against is complete.
"""

from __future__ import annotations

import inspect
import re
import subprocess
import sys
import typing

import pytest
from matplotlib.collections import PathCollection

from pynbodyext.core.calculate.bins.arrays import BinsArray
from pynbodyext.core.calculate.bins.plot import BinPlotMixin
from pynbodyext.plot.image import ImageData, adaptive, cmaps, compose, data, display, noise, ops, psf, smooth
from pynbodyext.plot.image import add_colorbar
from pynbodyext.plot.image._types import resolve_type_hints
from pynbodyext.plot.image.adaptive import AdaptiveMap
from pynbodyext.plot.image.compose import MapStyle
from pynbodyext.plot.image.ops import ImageOp

#: Modules whose public functions and classes make up the image layer.
MODULES = (display, cmaps, compose, data, ops, adaptive, noise, psf, smooth)

#: Classes whose fields are user-facing arguments too.
ANNOTATED_CLASSES = (ImageData, ImageOp, MapStyle, AdaptiveMap)


def annotation_text(annotation: object) -> str:
    """The annotation as written, whitespace-normalised.

    Annotations here are strings (``from __future__ import annotations``), and
    resolving them into objects would need the ``TYPE_CHECKING`` imports at
    runtime; the written form is enough to tell a bare ``Any`` from a nested one.
    """
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return ""
    return re.sub(r"\s+", " ", str(annotation)).strip()


def bare_any(text: str) -> bool:
    """Whether *text* uses ``Any`` outside every subscript — i.e. a bare ``Any``."""
    depth = 0
    for index, character in enumerate(text):
        if character in "[(":
            depth += 1
        elif character in "])":
            depth = max(depth - 1, 0)
        elif character == "A" and depth == 0 and text[index : index + 3] == "Any":
            after = text[index + 3 : index + 4]
            before = text[index - 1 : index] if index else ""
            if not after.isalnum() and not before.isalnum():
                return True
    return False


def cls_member_callables(cls_name: str, cls: type) -> list[tuple[str, object]]:
    """Public methods, ``__call__`` and public properties of one class."""
    found: list[tuple[str, object]] = []
    for name, member in vars(cls).items():
        if not (name == "__call__" or not name.startswith("_")):
            continue
        if inspect.isfunction(member):
            found.append((f"{cls_name}.{name}", member))
        elif isinstance(member, property) and member.fget is not None:
            found.append((f"{cls_name}.{name}", member.fget))
    return found


def public_callables() -> list[tuple[str, object]]:
    """Every public function of the image layer, and every public method of its classes."""
    found: list[tuple[str, object]] = []
    for module in MODULES:
        for name, member in vars(module).items():
            if name.startswith("_") or getattr(member, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(member):
                found.append((f"{module.__name__}.{name}", member))
            elif inspect.isclass(member):
                found.extend(cls_member_callables(name, member))
    return found


#: Collected once, so a failure names the exact callable.
BRIDGE_CALLABLES: list[tuple[str, object]] = [
    ("BinsArray.image", BinsArray.image.fget),
    ("BinsArray.plot", BinsArray.plot),
    ("BinPlotMixin.plot", BinPlotMixin.plot),
    ("BinPlotMixin.imshow", BinPlotMixin.imshow),
]

CALLABLES = sorted(public_callables() + BRIDGE_CALLABLES, key=lambda pair: pair[0])

#: Class name -> its field annotations, as written.
FIELDS = [(cls.__name__, dict(vars(cls).get("__annotations__", {}))) for cls in ANNOTATED_CLASSES]


@pytest.mark.parametrize(("qualname", "callable"), CALLABLES, ids=[pair[0] for pair in CALLABLES])
def test_no_signature_uses_a_bare_any(qualname: str, callable: object) -> None:
    """``Any`` hides an argument; the layer has real names for what it takes."""
    signature = inspect.signature(callable)
    offenders = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
            continue  # ``**kwargs: Any`` goes to matplotlib and is honest
        text = annotation_text(parameter.annotation)
        if bare_any(text):
            offenders.append(f"{name}: {text}")
    returns = annotation_text(signature.return_annotation)
    if bare_any(returns):
        offenders.append(f"return: {returns}")
    assert offenders == [], f"{qualname} uses a bare Any for {', '.join(offenders)}"


@pytest.mark.parametrize(("qualname", "annotations"), FIELDS, ids=[pair[0] for pair in FIELDS])
def test_no_field_uses_a_bare_any(qualname: str, annotations: dict[str, object]) -> None:
    """A dataclass field is a user-facing argument too."""
    offenders = [f"{name}: {text}" for name, text in annotations.items() if bare_any(annotation_text(text))]
    assert offenders == [], f"{qualname} uses a bare Any for {', '.join(offenders)}"


def test_every_public_annotation_resolves() -> None:
    """Deferred annotations must resolve for the tools that read them.

    Documentation, validation and plugin code calls ``typing.get_type_hints``; a
    name that only exists for the type checker makes it fail with a ``NameError``.
    ``plot.image._types.resolve_type_hints`` knows the names the modules cannot
    import themselves, and this walks everything the layer and its bridge annotate.
    """
    unresolved = []
    for qualname, callable in CALLABLES:
        try:
            resolve_type_hints(callable)
        except Exception as exc:  # NameError, or a TypeError from a bad alias
            unresolved.append(f"{qualname} ({type(exc).__name__}: {exc})")
    assert unresolved == [], f"annotations do not resolve: {'; '.join(unresolved)}"


def test_the_artist_union_covers_every_drawing_method_artist() -> None:
    """The colour-bar API takes anything the drawing methods hand back.

    ``BinPlotMixin.plot(kind="scatter")`` returns a ``PathCollection`` — a
    ``ScalarMappable``, so a valid mappable — and it has to be accepted statically
    too, not only at runtime.
    """
    accepted = typing.get_args(resolve_type_hints(add_colorbar)["mappable"])
    assert PathCollection in accepted


def test_importing_the_layer_stays_off_the_drawing_stack_and_the_calculator() -> None:
    """Resolving every annotation eagerly would undo the lazy imports.

    ``import pynbodyext.plot.image`` is a cheap import on purpose: matplotlib's
    drawing stack arrives when you draw, and the calculator when you bin — about
    350 ms and 270 ms respectively.  A bare ``typing.get_type_hints`` on an
    annotation that names one of those raises ``NameError``; the supported route is
    :func:`~pynbodyext.plot.image._types.resolve_type_hints`, which pays for them
    only when asked.
    """
    code = (
        "import pynbodyext.plot.image, sys; "
        "assert not [m for m in sys.modules if m.startswith('pynbodyext.core.calculate')], 'calculator imported'; "
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# the two bridges a user calls by hand
# ---------------------------------------------------------------------------


def test_bins_array_image_property_is_an_ImageData() -> None:
    """``bins2d["vz.mean"].image`` is the image layer's own type."""
    annotation = inspect.signature(BinsArray.image.fget).return_annotation
    assert annotation_text(annotation) == "ImageData"


def test_from_bins_array_takes_a_BinsArray() -> None:
    """The array side of the bridge is a ``BinsArray``, not ``Any``."""
    assert annotation_text(inspect.signature(ImageData.from_bins_array).parameters["array"].annotation) == "BinsArray"


def test_bins_imshow_is_handled_by_the_image_layer() -> None:
    """The binned-result ``imshow`` returns the image module's artist."""
    returns = annotation_text(inspect.signature(BinPlotMixin.imshow).return_annotation)
    assert "AxesImage" in returns and "QuadMesh" in returns
