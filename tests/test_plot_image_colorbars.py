"""Docked colour bars: ``add_colorbar`` and the ``colorbar=`` drawing options."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.image import AxesImage

from pynbodyext.plot.image import ImageData, add_colorbar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def image(shape: tuple[int, int] = (20, 20), **kwargs: object) -> ImageData:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 100.0, shape)
    defaults: dict[str, object] = {"extent": (0.0, float(shape[1]), 0.0, float(shape[0])), "label": "vz.mean"}
    defaults.update(kwargs)
    return ImageData(data, **defaults)


def test_add_colorbar_attaches_to_the_artist() -> None:
    fig, ax = plt.subplots()
    try:
        artist = image().imshow(ax=ax)

        bar = add_colorbar(artist)

        assert len(fig.axes) == 2
        assert bar.mappable is artist
        fig.canvas.draw()  # the divider places the bar when the figure is drawn
        # docked to the right: the bar's axes sits beside the image's axes
        assert bar.ax.get_position().x0 >= ax.get_position().x1
    finally:
        plt.close(fig)


@pytest.mark.parametrize("loc", ["right", "left", "top", "bottom"])
def test_add_colorbar_supports_every_side(loc: str) -> None:
    fig, ax = plt.subplots()
    try:
        artist = image().imshow(ax=ax)

        bar = add_colorbar(artist, ax=ax, loc=loc)

        fig.canvas.draw()
        positions = bar.ax.get_position()
        panel = ax.get_position()
        if loc == "right":
            assert positions.x0 >= panel.x1
        elif loc == "left":
            assert positions.x1 <= panel.x0
        elif loc == "top":
            assert positions.y0 >= panel.y1
        else:
            assert positions.y1 <= panel.y0
    finally:
        plt.close(fig)


def test_add_colorbar_accepts_an_image_that_was_drawn() -> None:
    """The complaint that started this: adaptive maps could not get a colour bar."""
    source = image(label="vz.mean", units="km/s")
    fig, ax = plt.subplots()
    try:
        artist = source.imshow(ax=ax, vmin=-200.0, vmax=200.0)

        bar = source.add_colorbar(ax=ax)

        assert bar.mappable is artist  # the drawn artist, so the scale matches exactly
        assert bar.ax.get_ylabel() == "vz.mean [km/s]"
    finally:
        plt.close(fig)


def test_add_colorbar_builds_a_scale_when_nothing_is_drawn() -> None:
    source = image(label="mass.sum")
    fig, ax = plt.subplots()
    try:
        ax.set_axis_off()

        bar = add_colorbar(source, ax=ax, cmap="inferno", vmin=0.0, vmax=10.0)

        assert bar.mappable.get_clim() == (0.0, 10.0)
        assert bar.mappable.get_cmap().name == "inferno"
    finally:
        plt.close(fig)


def test_adaptive_map_can_add_a_colorbar_after_imshow() -> None:
    pytest.importorskip("powerbin")
    rng = np.random.default_rng(1)
    value = ImageData(rng.normal(0.0, 100.0, (30, 30)), extent=(0.0, 30.0, 0.0, 30.0), label="vz.mean")
    signal = ImageData(rng.uniform(1.0, 10.0, (30, 30)), extent=(0.0, 30.0, 0.0, 30.0))
    binned = value.adaptive.bin(signal, target_nbins=5)

    fig, ax = plt.subplots()
    try:
        binned.imshow(ax=ax, symmetric=True)
        bar = binned.add_colorbar(ax=ax)

        assert bar.ax.get_ylabel() == "vz.mean"
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_adaptive_map_imshow_accepts_a_colorbar_request() -> None:
    pytest.importorskip("powerbin")
    value = image((20, 20))
    signal = ImageData(np.ones((20, 20)), extent=(0.0, 20.0, 0.0, 20.0))
    binned = value.adaptive.bin(signal, target_nbins=4)

    fig, ax = plt.subplots()
    try:
        binned.imshow(ax=ax, colorbar="bottom")

        assert len(fig.axes) == 2
        assert fig.axes[1].get_xlabel() == "vz.mean"
    finally:
        plt.close(fig)


def test_drawing_accepts_a_colorbar_location_and_kwargs() -> None:
    fig, ax = plt.subplots()
    try:
        image().imshow(ax=ax, colorbar="left", colorbar_kwargs={"size": "8%", "tick_label_size": 7})

        bar_ax = fig.axes[1]
        fig.canvas.draw()
        assert bar_ax.get_position().x1 <= ax.get_position().x0
        assert bar_ax.get_yticklabels()[0].get_fontsize() == 7
    finally:
        plt.close(fig)


def test_draw_passes_the_colorbar_options_through() -> None:
    fig, ax = plt.subplots()
    try:
        uneven = ImageData(
            np.zeros((4, 4)), x_edges=[0.0, 1.0, 3.0, 6.0, 10.0], y_edges=[0.0, 1.0, 2.0, 3.0, 4.0], label="vz.mean"
        )

        uneven.draw(ax=ax, colorbar=True)

        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_add_colorbar_rejects_an_unknown_location() -> None:
    with pytest.raises(ValueError, match="loc"):
        add_colorbar(image(), loc="centre")


def test_add_colorbar_needs_something_to_describe() -> None:
    with pytest.raises(TypeError, match="add_colorbar"):
        add_colorbar(42)


def test_add_colorbar_needs_axes_to_dock_to() -> None:
    plt.close("all")

    with pytest.raises(ValueError, match="No axes"):
        add_colorbar(image(), ax=None)


def test_imshow_compose_docks_both_colour_bars() -> None:
    from pynbodyext.plot.image import MapStyle, imshow_compose

    first = image()
    second = image()

    fig, ax = plt.subplots()
    try:
        artist = imshow_compose(
            first,
            second,
            ax=ax,
            style1=MapStyle(cmap="inferno"),
            style2=MapStyle(cmap="cividis"),
            label1="gas",
            label2="dark matter",
        )

        assert isinstance(artist, AxesImage)
        assert len(fig.axes) == 3
        left, right = fig.axes[1], fig.axes[2]
        fig.canvas.draw()
        assert left.get_position().x1 <= ax.get_position().x0
        assert right.get_position().x0 >= ax.get_position().x1
        assert (left.get_ylabel(), right.get_ylabel()) == ("gas", "dark matter")
    finally:
        plt.close(fig)
