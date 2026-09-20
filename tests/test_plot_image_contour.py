"""Contour lines on an image: ``ImageData.contour`` and ``AdaptiveMap.contour``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.contour import ContourSet  # noqa: E402

from pynbodyext.plot.image import ImageData, add_colorbar  # noqa: E402


def blob(shape: tuple[int, int] = (30, 30), **kwargs: object) -> ImageData:
    """A signed bump, so contours have both positive and negative levels."""
    rows, cols = np.indices(shape, dtype=float)
    data = 10.0 * np.exp(-(((cols - 15.0) / 5.0) ** 2 + ((rows - 15.0) / 5.0) ** 2)) - 3.0
    defaults: dict[str, object] = {
        "extent": (0.0, float(shape[1]), 0.0, float(shape[0])),
        "label": "vz.mean",
        "units": "km/s",
    }
    defaults.update(kwargs)
    return ImageData(data, **defaults)


def test_contour_draws_the_requested_levels() -> None:
    fig, ax = plt.subplots()
    try:
        artist = blob().display.contour(ax=ax, levels=[-2.0, 0.0, 2.0])

        assert isinstance(artist, ContourSet)
        np.testing.assert_allclose(artist.levels, [-2.0, 0.0, 2.0])
        assert artist in ax.collections
    finally:
        plt.close(fig)


def test_contour_accepts_a_level_count() -> None:
    """An integer is a hint, so matplotlib picks the 'nice' levels."""
    fig, ax = plt.subplots()
    try:
        artist = blob().display.contour(ax=ax, levels=5)

        assert artist.levels.size >= 3
    finally:
        plt.close(fig)


def test_contour_overlays_an_existing_image() -> None:
    source = blob()
    fig, ax = plt.subplots()
    try:
        source.display.imshow(ax=ax, cmap="inferno")

        artist = source.display.contour(ax=ax, levels=[-2.0, 0.0, 2.0], colors="w", linewidths=0.8)

        assert len(ax.images) == 1  # the image is still there
        assert artist in ax.collections
        np.testing.assert_allclose(artist.levels, [-2.0, 0.0, 2.0])
    finally:
        plt.close(fig)


def test_filled_contours_use_contourf() -> None:
    fig, ax = plt.subplots()
    try:
        artist = blob().display.contour(ax=ax, levels=[-2.0, 0.0, 2.0, 4.0], filled=True)

        assert isinstance(artist, ContourSet)
        assert artist.filled is True
        np.testing.assert_allclose(artist.levels, [-2.0, 0.0, 2.0, 4.0])
    finally:
        plt.close(fig)


def test_contour_follows_the_bin_centres_of_an_uneven_grid() -> None:
    source = ImageData(np.arange(12.0).reshape(3, 4), x_edges=[0.0, 1.0, 3.0, 6.0, 10.0], y_edges=[0.0, 2.0, 5.0, 9.0])

    fig, ax = plt.subplots()
    try:
        source.display.contour(ax=ax, levels=3)

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert 0.5 <= xlim[0] and xlim[1] <= 8.0  # centres of the x bins
        assert 1.0 <= ylim[0] and ylim[1] <= 7.0  # centres of the y bins
    finally:
        plt.close(fig)


def test_contour_can_be_given_a_colour_bar() -> None:
    fig, ax = plt.subplots()
    try:
        artist = blob().display.contour(ax=ax, levels=4)

        bar = add_colorbar(artist, ax=ax)

        assert bar.ax.get_ylabel() == "vz.mean [km/s]"
    finally:
        plt.close(fig)


def test_contour_can_request_its_own_colour_bar() -> None:
    fig, ax = plt.subplots()
    try:
        blob().display.contour(ax=ax, levels=4, colorbar=True)

        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_contour_tolerates_empty_bins() -> None:
    source = blob()
    data = source.data.copy()
    data[0, 0] = np.nan

    fig, ax = plt.subplots()
    try:
        artist = source.with_data(data).display.contour(ax=ax, levels=3)

        assert isinstance(artist, ContourSet)
    finally:
        plt.close(fig)


def test_adaptive_map_can_contour_its_painted_map() -> None:
    pytest.importorskip("powerbin")
    rng = np.random.default_rng(0)
    value = blob()
    signal = ImageData(rng.uniform(1.0, 10.0, value.shape), extent=value.extent)

    binned = value.process.adaptive.bin(signal, target_nbins=5)

    fig, ax = plt.subplots()
    try:
        binned.display.imshow(ax=ax)
        artist = binned.display.contour(ax=ax, levels=3)

        assert isinstance(artist, ContourSet)
        assert artist in ax.collections
    finally:
        plt.close(fig)


def test_adaptive_map_contours_can_be_made_symmetric() -> None:
    pytest.importorskip("powerbin")
    rng = np.random.default_rng(2)
    value = blob()
    signal = ImageData(rng.uniform(1.0, 10.0, value.shape), extent=value.extent)
    binned = value.process.adaptive.bin(signal, target_nbins=5)

    fig, ax = plt.subplots()
    try:
        artist = binned.display.contour(ax=ax, symmetric=True, count=3)

        assert artist.levels.size == 7
        assert artist.levels[0] == pytest.approx(-artist.levels[-1])
        assert 0.0 in artist.levels
    finally:
        plt.close(fig)
