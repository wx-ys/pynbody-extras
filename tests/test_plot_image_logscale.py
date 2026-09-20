"""Logarithmic (and otherwise normed) display of maps that span decades.

Covers the case a density map runs into: the values span orders of magnitude, so
contour levels have to be geometric and the colour bar has to be logarithmic
rather than a floating linear scale.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

from pynbodyext.plot.image import ImageData  # noqa: E402
from pynbodyext.plot.image.display import add_colorbar  # noqa: E402


def density(shape: tuple[int, int] = (40, 40), **kwargs: object) -> ImageData:
    """A centrally concentrated, decades-spanning map (a density, in effect)."""
    rows, cols = np.indices(shape, dtype=float)
    radius = np.hypot(cols - shape[1] / 2, rows - shape[0] / 2) / (shape[0] / 4.0)
    data = 1e2 * np.exp(-(radius**2)) + 1e-4
    defaults: dict[str, object] = {
        "extent": (0.0, float(shape[1]), 0.0, float(shape[0])),
        "label": "mass.density",
        "units": "Msol kpc**-2",
    }
    defaults.update(kwargs)
    return ImageData(data, **defaults)


# ---------------------------------------------------------------------------
# contours
# ---------------------------------------------------------------------------


def test_contour_levels_are_geometric_on_a_log_scale() -> None:
    fig, ax = plt.subplots()
    try:
        artist = density().display.contour(ax=ax, log=True, levels=4)

        assert artist.levels.size == 5  # four intervals
        spacing = np.diff(np.log10(artist.levels))
        np.testing.assert_allclose(spacing, spacing[0])
        assert artist.levels[0] > 0
    finally:
        plt.close(fig)


def test_contour_levels_follow_a_log_norm_too() -> None:
    source = density()
    fig, ax = plt.subplots()
    try:
        from_norm = source.display.contour(ax=ax, norm=LogNorm(), levels=4)
        from_flag = source.display.contour(ax=ax, log=True, levels=4)

        np.testing.assert_allclose(from_norm.levels, from_flag.levels)
    finally:
        plt.close(fig)


def test_contour_understands_the_string_form_of_a_norm() -> None:
    """``norm="log"`` is matplotlib's own spelling, so it must behave like log=True."""
    source = density()
    fig, ax = plt.subplots()
    try:
        from_string = source.display.contour(ax=ax, norm="log", levels=4)
        from_flag = source.display.contour(ax=ax, log=True, levels=4)

        np.testing.assert_allclose(from_string.levels, from_flag.levels)
        assert np.all(np.diff(np.log10(from_string.levels)) > 0)
        assert isinstance(from_string.norm, LogNorm)
    finally:
        plt.close(fig)


def test_a_string_norm_is_refused_for_an_unknown_scale() -> None:
    with pytest.raises(ValueError, match="Unknown colour scale"):
        density().display.imshow(norm="gamma2")


def test_symmetric_refuses_a_string_log_norm_too() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        density().display.draw(symmetric=True, norm="log")


def test_contour_levels_stay_linear_without_log() -> None:
    fig, ax = plt.subplots()
    try:
        artist = density().display.contour(ax=ax, levels=4)

        # A linear locator over 1e-4 … 1e2 gives round numbers: evenly spaced, and
        # *not* a geometric run (the lowest level is zero, which log10 cannot take).
        spacing = np.diff(artist.levels)
        np.testing.assert_allclose(spacing, spacing[0])
        assert np.any(artist.levels <= 0) or not np.allclose(
            np.diff(np.log10(artist.levels)), np.diff(np.log10(artist.levels))[0]
        )
    finally:
        plt.close(fig)


def test_contour_log_ignores_non_positive_values() -> None:
    source = density()
    data = source.data.copy()
    data[0, 0] = 0.0
    data[0, 1] = -5.0

    fig, ax = plt.subplots()
    try:
        artist = source.with_data(data).display.contour(ax=ax, log=True, levels=3)

        assert artist.levels[0] > 0
        assert artist.levels[-1] <= np.nanmax(data)
    finally:
        plt.close(fig)


def test_contour_honours_explicit_levels_on_a_log_scale() -> None:
    fig, ax = plt.subplots()
    try:
        artist = density().display.contour(ax=ax, log=True, levels=[1e-3, 1e-1, 10.0])

        np.testing.assert_allclose(artist.levels, [1e-3, 1e-1, 10.0])
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# images
# ---------------------------------------------------------------------------


def test_imshow_can_use_a_log_norm() -> None:
    fig, ax = plt.subplots()
    try:
        artist = density().display.imshow(ax=ax, log=True)

        assert isinstance(artist.norm, LogNorm)
        assert artist.norm.vmin > 0 and artist.norm.vmax > artist.norm.vmin
        assert artist.norm.vmax == pytest.approx(1e2 + 1e-4, rel=1e-3)
    finally:
        plt.close(fig)


def test_imshow_accepts_the_string_form_and_labels_it_logarithmically() -> None:
    fig, ax = plt.subplots()
    try:
        artist = density().display.imshow(ax=ax, norm="log", colorbar=True)

        assert isinstance(artist.norm, LogNorm)
        assert artist.norm.vmin > 0 and artist.norm.vmax > artist.norm.vmin
        assert fig.axes[1].get_yscale() == "log"
    finally:
        plt.close(fig)


def test_to_rgba_and_map_style_accept_a_string_norm() -> None:
    from pynbodyext.plot.image import MapStyle, to_rgba

    values = np.array([[1e-3, 1e-1, 1e2]])

    by_name = to_rgba(values, norm="log")
    by_instance = to_rgba(values, norm=LogNorm(1e-3, 1e2))

    np.testing.assert_allclose(by_name, by_instance)
    style = MapStyle(cmap="inferno", norm="log")
    np.testing.assert_allclose(style.to_rgba(values), to_rgba(values, "inferno", norm=LogNorm(1e-3, 1e2)))
    assert isinstance(style.norm_for(values), LogNorm)


def test_draw_and_pcolormesh_support_log_too() -> None:
    source = density()
    uneven = source.with_data(source.data)
    fig, ax = plt.subplots()
    try:
        assert isinstance(source.display.draw(ax=ax, log=True).norm, LogNorm)
        assert isinstance(
            ImageData(
                source.data, x_edges=np.linspace(0, 40, 41) ** 1.2, y_edges=np.linspace(0, 40, 41)
            ).display.pcolormesh(ax=ax, log=True).norm,
            LogNorm,
        )
        assert uneven.shape == source.shape
    finally:
        plt.close(fig)


@pytest.mark.parametrize("loc", ["right", "left", "top", "bottom"])
def test_a_log_image_gets_a_log_colour_bar(loc: str) -> None:
    fig, ax = plt.subplots()
    try:
        source = density()
        source.display.imshow(ax=ax, log=True, colorbar=loc)

        bar = fig.axes[1]
        scale = bar.get_yscale() if loc in ("right", "left") else bar.get_xscale()
        assert scale == "log"
        label = bar.get_ylabel() if loc in ("right", "left") else bar.get_xlabel()
        assert "mass.density" in label
    finally:
        plt.close(fig)


def test_explicit_limits_can_be_given_for_the_log_scale() -> None:
    fig, ax = plt.subplots()
    try:
        artist = density().display.imshow(ax=ax, log=True, vmin=1e-3, vmax=10.0)

        assert (artist.norm.vmin, artist.norm.vmax) == (1e-3, 10.0)
    finally:
        plt.close(fig)


def test_log_needs_positive_values() -> None:
    source = ImageData(np.zeros((4, 4)))

    with pytest.raises(ValueError, match="positive"):
        source.display.imshow(log=True)


def test_log_and_symmetric_are_contradictory() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        density().display.draw(log=True, symmetric=True)


def test_norm_and_log_together_are_refused() -> None:
    with pytest.raises(ValueError, match="norm"):
        density().display.imshow(log=True, norm=Normalize(0, 1))


# ---------------------------------------------------------------------------
# colour bars built from an image
# ---------------------------------------------------------------------------


def test_add_colorbar_can_be_logarithmic() -> None:
    source = density()
    fig, ax = plt.subplots()
    try:
        ax.set_axis_off()

        bar = add_colorbar(source, ax=ax, log=True)

        assert isinstance(bar.mappable.norm, LogNorm)
        assert bar.ax.get_yscale() == "log"
        assert bar.ax.get_ylabel() == "mass.density [Msol kpc**-2]"
    finally:
        plt.close(fig)


def test_add_colorbar_accepts_the_string_form() -> None:
    source = density()
    fig, ax = plt.subplots()
    try:
        ax.set_axis_off()

        bar = add_colorbar(source, ax=ax, norm="log")

        assert isinstance(bar.mappable.norm, LogNorm)
        assert bar.ax.get_yscale() == "log"
    finally:
        plt.close(fig)


def test_add_colorbar_refuses_a_scale_the_artist_does_not_have() -> None:
    source = density()
    fig, ax = plt.subplots()
    try:
        linear = source.display.imshow(ax=ax)

        with pytest.raises(ValueError, match="artist"):
            add_colorbar(linear, ax=ax, log=True)

        # an artist that was drawn logarithmically carries the scale itself
        logarithmic = source.display.imshow(ax=ax, log=True)
        assert add_colorbar(logarithmic, ax=ax).ax.get_yscale() == "log"
    finally:
        plt.close(fig)


def test_image_display_add_colorbar_passes_the_log_flag() -> None:
    source = density()
    fig, ax = plt.subplots()
    try:
        source.display.imshow(ax=ax, log=True)

        bar = source.display.add_colorbar(ax=ax, log=True)

        assert isinstance(bar.mappable.norm, LogNorm)
    finally:
        plt.close(fig)
