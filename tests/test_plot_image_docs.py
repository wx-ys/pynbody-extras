"""The user-facing API must stay documented where users actually read it.

The free functions are the implementation and carry the detailed reference; the
methods are what most people call, so they need their own summary, parameters,
returns and an example — a pointer at the function is not enough.  These tests keep
that from drifting back.
"""

from __future__ import annotations

import inspect

import pytest

from pynbodyext.plot.image.adaptive import AdaptiveOps
from pynbodyext.plot.image.compose import ComposeOps
from pynbodyext.plot.image.display import DisplayOps
from pynbodyext.plot.image.noise import NoiseOps
from pynbodyext.plot.image.ops import ImageOps
from pynbodyext.plot.image.psf import PsfOps
from pynbodyext.plot.image.smooth import SmoothOps

#: The classes users touch: every public method of theirs must be self-documenting.
USER_FACING = (DisplayOps, SmoothOps, PsfOps, NoiseOps, ComposeOps, AdaptiveOps, ImageOps)

#: Method -> the free function that does the work, which the docstring must name.
IMPLEMENTED_BY = {
    (DisplayOps, "normalize"): "normalize",
    (DisplayOps, "to_rgba"): "to_rgba",
    (DisplayOps, "draw"): "draw_image",
    (DisplayOps, "imshow"): "draw_imshow",
    (DisplayOps, "pcolormesh"): "draw_pcolormesh",
    (DisplayOps, "contour"): "draw_contour",
    (DisplayOps, "add_colorbar"): "add_colorbar",
    (SmoothOps, "gaussian"): "gaussian_smooth",
    (SmoothOps, "box"): "box_smooth",
    (SmoothOps, "median"): "median_filter",
    (SmoothOps, "downsample"): "downsample",
    (PsfOps, "convolve"): "convolve_psf",
    (PsfOps, "wiener"): "wiener_deconvolve",
    (PsfOps, "richardson_lucy"): "richardson_lucy",
    (PsfOps, "deconvolve"): "deconvolve_psf",
    (NoiseOps, "gaussian"): "add_noise",
    (NoiseOps, "poisson"): "add_poisson_noise",
    (ComposeOps, "__call__"): "compose_maps",
    (ComposeOps, "masks"): "create_map_mask",
    (ComposeOps, "imshow"): "imshow_compose",
    (AdaptiveOps, "bin"): "adaptive_bin_map",
    (ImageOps, "derive"): "ImageData",
    (ImageOps, "limits"): "normalize",
    (ImageOps, "kernel_scale"): "pixel_size",
}


def public_methods(cls: type) -> list[tuple[str, object]]:
    """Methods of *cls* that a user may call."""
    return [
        (name, member)
        for name, member in vars(cls).items()
        # ``__call__`` is the spelling for a whole family (``image.compose(other)``)
        if (name == "__call__" or not name.startswith("_")) and callable(member) and inspect.isfunction(member)
    ]


@pytest.mark.parametrize("cls", USER_FACING, ids=lambda cls: cls.__name__)
def test_every_method_documents_summary_parameters_returns_and_an_example(cls: type) -> None:
    missing = []
    for name, member in public_methods(cls):
        doc = inspect.getdoc(member) or ""
        lines = doc.splitlines()
        takes_arguments = [p for p in inspect.signature(member).parameters if p != "self"]
        lacks = []
        if len(lines) < 6 or not lines[0].strip():
            lacks.append("a summary")
        if takes_arguments and "Parameters" not in doc:
            lacks.append("Parameters")
        if "Returns" not in doc:
            lacks.append("Returns")
        if "Examples" not in doc:
            lacks.append("Examples")
        if lacks:
            missing.append(f"{name} (needs {', '.join(lacks)})")
    assert not missing, f"{cls.__name__} methods are under-documented: {'; '.join(missing)}"


@pytest.mark.parametrize("cls", USER_FACING, ids=lambda cls: cls.__name__)
def test_every_docstring_names_its_parameters(cls: type) -> None:
    """A parameter list is only useful if it lists the parameters."""
    undocumented = []
    for name, member in public_methods(cls):
        doc = inspect.getdoc(member) or ""
        for parameter, spec in inspect.signature(member).parameters.items():
            if parameter == "self" or spec.kind in (spec.VAR_KEYWORD, spec.VAR_POSITIONAL):
                continue  # ``*args``/``**kwargs`` are described as one block
            if f"{parameter} " not in doc and f"{parameter}:" not in doc and f"{parameter}," not in doc:
                undocumented.append(f"{name}({parameter})")
    assert not undocumented, f"{cls.__name__} does not describe: {', '.join(undocumented)}"


@pytest.mark.parametrize(
    ("cls", "method"),
    sorted(IMPLEMENTED_BY, key=lambda pair: (pair[0].__name__, pair[1])),
    ids=lambda pair: f"{pair[0].__name__}.{pair[1]}" if isinstance(pair, tuple) else str(pair),
)
def test_methods_name_the_function_that_implements_them(cls: type, method: str) -> None:
    doc = inspect.getdoc(getattr(cls, method)) or ""
    assert IMPLEMENTED_BY[(cls, method)] in doc, f"{cls.__name__}.{method} should point at {IMPLEMENTED_BY[(cls, method)]}"
