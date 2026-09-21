"""The ``pynbodyext.plot.image`` namespace and its optional dependency."""

from __future__ import annotations

import subprocess
import sys

import pytest

EXPECTED_NAMES = {
    "COLORBAR_LOCATIONS",
    "AdaptiveMap",
    "ImageData",
    "ImageDataView",
    "ImageOp",
    "ImageOps",
    "sauron_cmap",
    "sauron_cmap_r",
    "SAURON_RGB",
    "SAURON_POSITIONS",
    "MapStyle",
    "OPERATIONS",
    "STRETCHES",
    "add_colorbar",
    "add_noise",
    "add_poisson_noise",
    "as_image",
    "adaptive_bin_map",
    "adaptive_map_from_bins",
    "blend_images",
    "blend_stack",
    "box_smooth",
    "cmap_from_colors",
    "compose_maps",
    "convolve_psf",
    "create_map_mask",
    "deconvolve_psf",
    "downsample",
    "gaussian_psf",
    "gaussian_smooth",
    "get_cmap",
    "imshow_compose",
    "median_filter",
    "normalize",
    "normalize_psf",
    "register_cmap",
    "register_ops",
    "richardson_lucy",
    "to_rgba",
    "vel_cmap",
    "vel_cmap_r",
    "wiener_deconvolve",
}


def test_the_image_package_exports_its_public_api() -> None:
    from pynbodyext.plot import image

    assert set(image.__all__) == EXPECTED_NAMES
    for name in EXPECTED_NAMES:
        assert getattr(image, name) is not None


def test_the_plot_package_exposes_the_image_namespace() -> None:
    import pynbodyext.plot

    assert pynbodyext.plot.image.__name__ == "pynbodyext.plot.image"
    assert pynbodyext.plot.__all__ == ["image"]


def test_the_submodules_are_reachable() -> None:
    from pynbodyext.plot.image import adaptive, cmaps, compose, data, smooth, psf

    for module in (adaptive, cmaps, compose, data, smooth, psf):
        assert module.__name__.startswith("pynbodyext.plot.image.")


def test_importing_the_package_does_not_need_the_optional_powerbin() -> None:
    """``powerbin`` is optional: an install without it must still import."""
    code = (
        "import sys; sys.modules['powerbin'] = None; "
        "import pynbodyext.plot.image as image; "
        "assert callable(image.adaptive_bin_map)"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr


def test_adaptive_binning_reports_a_missing_powerbin(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from pynbodyext.plot.image import adaptive

    monkeypatch.setattr(adaptive, "POWERBIN_AVAILABLE", False)
    with pytest.raises(ImportError, match="powerbin"):
        adaptive.adaptive_bin_map(np.ones((4, 4)), np.ones((4, 4)), target_nbins=2)
