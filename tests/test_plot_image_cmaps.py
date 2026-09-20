"""Tests for ``pynbodyext.plot.image.cmaps``."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import Colormap, LinearSegmentedColormap

from pynbodyext.plot.image.cmaps import (
    velocity_cmap,
    VELOCITY_COLORS,
    VELOCITY_POSITIONS,
    cmap_from_colors,
    get_cmap,
    register_cmap,
    to_rgba,
    vel_cmap,
)


def test_velocity_cmap_is_a_256_entry_linear_segmented_map() -> None:
    assert isinstance(velocity_cmap, LinearSegmentedColormap)
    assert velocity_cmap.name == "velocity"
    assert velocity_cmap.N == 256
    assert vel_cmap is velocity_cmap


def test_velocity_cmap_is_registered_with_matplotlib() -> None:
    registered = matplotlib.colormaps["velocity"]

    assert registered.name == velocity_cmap.name
    np.testing.assert_allclose(registered(np.linspace(0, 1, 9)), velocity_cmap(np.linspace(0, 1, 9)))


@pytest.mark.parametrize(
    ("position", "color"),
    [
        (0.00, "#000000"),
        (0.18, "#0000FF"),
        (0.44, "#00FFFF"),
        (0.50, "#00FF00"),
        (0.56, "#FFFF00"),
        (0.84, "#FF0000"),
        (1.00, "#FFFFFF"),
    ],
)
def test_velocity_cmap_hits_every_colour_stop(position: float, color: str) -> None:
    expected = matplotlib.colors.to_rgba(color)
    # The stops are exact in the continuous definition; the 256-entry look-up
    # table samples them, so interior stops carry half a step of quantisation
    # error on the steepest segment (black/white ends are sampled exactly).
    tolerance = 1e-9 if position in (0.0, 1.0) else 0.04
    np.testing.assert_allclose(velocity_cmap(position), expected, atol=tolerance)


def test_velocity_cmap_stops_are_the_declared_ones() -> None:
    assert len(VELOCITY_COLORS) == len(VELOCITY_POSITIONS) == 7
    np.testing.assert_allclose(VELOCITY_POSITIONS, [0.00, 0.18, 0.44, 0.50, 0.56, 0.84, 1.00])
    assert VELOCITY_POSITIONS[3] == 0.5  # green sits on zero velocity


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
    assert get_cmap("velocity").name == velocity_cmap.name
    assert get_cmap(velocity_cmap) is velocity_cmap


def test_to_rgba_returns_unit_float_rgba() -> None:
    data = np.array([[0.0, 0.5], [1.0, np.nan]])

    rgba = to_rgba(data)

    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.float64
    assert 0.0 <= rgba.min() and rgba.max() <= 1.0
    np.testing.assert_allclose(rgba[0, 0], matplotlib.colors.to_rgba("#000000"))
    np.testing.assert_allclose(rgba[1, 0], matplotlib.colors.to_rgba("#FFFFFF"))


def test_to_rgba_leaves_non_finite_pixels_transparent() -> None:
    rgba = to_rgba(np.array([[1.0, np.nan]]))

    assert rgba[0, 1, 3] == 0.0


def test_to_rgba_honours_limits_and_stretch() -> None:
    data = np.array([[0.0, 10.0]])

    rgba = to_rgba(data, vmin=0.0, vmax=10.0, stretch="log")

    np.testing.assert_allclose(rgba[0, 0], matplotlib.colors.to_rgba("#000000"))
    np.testing.assert_allclose(rgba[0, 1], matplotlib.colors.to_rgba("#FFFFFF"))


def test_to_rgba_accepts_explicit_alpha() -> None:
    data = np.zeros((2, 2))

    rgba = to_rgba(data, alpha=np.array([[0.0, 0.25], [0.5, 1.0]]))

    np.testing.assert_allclose(rgba[..., 3], [[0.0, 0.25], [0.5, 1.0]])


def test_to_rgba_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        to_rgba(np.zeros(4))
