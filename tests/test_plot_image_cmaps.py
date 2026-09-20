"""Tests for ``pynbodyext.plot.image.cmaps``."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap

from pynbodyext.plot.image.cmaps import (
    SAURON_POSITIONS,
    SAURON_RGB,
    cmap_from_colors,
    get_cmap,
    register_cmap,
    sauron_cmap,
    sauron_cmap_r,
    to_rgba,
    vel_cmap,
    vel_cmap_r,
)


def test_sauron_cmap_is_a_256_entry_linear_segmented_map() -> None:
    assert isinstance(sauron_cmap, LinearSegmentedColormap)
    assert sauron_cmap.name == "sauron"
    assert sauron_cmap.N == 256
    assert vel_cmap is sauron_cmap
    assert sauron_cmap_r.name == "sauron_r"
    assert vel_cmap_r is sauron_cmap_r


def test_sauron_cmap_is_registered_with_matplotlib() -> None:
    registered = matplotlib.colormaps["sauron"]

    assert registered.name == sauron_cmap.name
    np.testing.assert_allclose(registered(np.linspace(0, 1, 9)), sauron_cmap(np.linspace(0, 1, 9)))
    np.testing.assert_allclose(
        matplotlib.colormaps["sauron_r"](np.linspace(0, 1, 9)), sauron_cmap_r(np.linspace(0, 1, 9))
    )


@pytest.mark.parametrize(("position", "rgb"), list(zip(SAURON_POSITIONS, SAURON_RGB, strict=True)))
def test_sauron_cmap_follows_the_published_table(position: float, rgb: np.ndarray) -> None:
    # The 256-entry look-up table samples the control points, so an interior knot
    # carries up to half a table step of error (0.5 is a corner in the red ramp).
    tolerance = 1e-9 if position in (0.0, 1.0) else 0.05
    np.testing.assert_allclose(sauron_cmap(position)[:3], rgb, atol=tolerance)


def test_sauron_table_is_the_published_one() -> None:
    assert len(SAURON_RGB) == len(SAURON_POSITIONS) == 11
    np.testing.assert_allclose(
        SAURON_POSITIONS, np.array([0, 42.5, 85, 105, 117.5, 127.5, 137.5, 150, 170, 212.5, 255]) / 255
    )
    # The table is symmetric about 1/2, which is where green sits (zero velocity).
    assert SAURON_POSITIONS[5] == 0.5
    np.testing.assert_allclose(np.array([1.0]) - SAURON_POSITIONS, SAURON_POSITIONS[::-1])
    np.testing.assert_allclose(SAURON_RGB[0], [0.0, 0.0, 0.0])  # black at the negative end
    np.testing.assert_allclose(SAURON_RGB[-1], [0.9, 0.9, 0.9])  # light grey, not white
    np.testing.assert_allclose(SAURON_RGB[5], [0.0, 0.9, 0.0])  # green on zero


def test_sauron_matches_the_reference_implementation() -> None:
    """The whole point of the table: our LUT is ``plotbin``'s, bit for bit."""
    plotbin = pytest.importorskip("plotbin.sauron_colormap")
    plotbin.register_sauron_colormap()
    sample = np.linspace(0.0, 1.0, 256)

    np.testing.assert_allclose(matplotlib.colormaps["sauron"](sample), sauron_cmap(sample), atol=0.0)
    np.testing.assert_allclose(matplotlib.colormaps["sauron_r"](sample), sauron_cmap_r(sample), atol=0.0)


def test_cmap_from_colors_builds_named_map() -> None:
    cmap = cmap_from_colors("black_to_white", ["#000000", "#FFFFFF"], N=8)

    assert isinstance(cmap, Colormap)
    assert cmap.name == "black_to_white"
    assert cmap.N == 8
    np.testing.assert_allclose(cmap(0.0), [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(cmap(1.0), [1.0, 1.0, 1.0, 1.0])


def test_cmap_from_colors_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="positions"):
        cmap_from_colors("bad", ["#000000", "#FFFFFF"], [0.0, 0.5, 1.0])


def test_register_cmap_is_idempotent() -> None:
    cmap = cmap_from_colors("idempotent_test_map", ["#000000", "#FFFFFF"])

    register_cmap(cmap)
    register_cmap(cmap)  # a second import of the module must not blow up

    assert matplotlib.colormaps["idempotent_test_map"].name == cmap.name


def test_get_cmap_accepts_name_or_object() -> None:
    assert get_cmap("sauron").name == sauron_cmap.name
    assert get_cmap(sauron_cmap) is sauron_cmap


def test_to_rgba_returns_unit_float_rgba() -> None:
    data = np.array([[0.0, 0.5], [1.0, np.nan]])

    rgba = to_rgba(data)

    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.float64
    assert 0.0 <= rgba.min() and rgba.max() <= 1.0
    np.testing.assert_allclose(rgba[0, 0], matplotlib.colors.to_rgba("#000000"))
    np.testing.assert_allclose(rgba[1, 0], [0.9, 0.9, 0.9, 1.0])  # SAURON's light-grey end


def test_to_rgba_leaves_non_finite_pixels_transparent() -> None:
    rgba = to_rgba(np.array([[1.0, np.nan]]))

    assert rgba[0, 1, 3] == 0.0


def test_to_rgba_honours_limits_and_stretch() -> None:
    data = np.array([[0.0, 10.0]])

    rgba = to_rgba(data, vmin=0.0, vmax=10.0, stretch="log")

    np.testing.assert_allclose(rgba[0, 0], matplotlib.colors.to_rgba("#000000"))
    np.testing.assert_allclose(rgba[0, 1], [0.9, 0.9, 0.9, 1.0])


def test_to_rgba_accepts_explicit_alpha() -> None:
    data = np.zeros((2, 2))

    rgba = to_rgba(data, alpha=np.array([[0.0, 0.25], [0.5, 1.0]]))

    np.testing.assert_allclose(rgba[..., 3], [[0.0, 0.25], [0.5, 1.0]])


def test_to_rgba_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        to_rgba(np.zeros(4))
