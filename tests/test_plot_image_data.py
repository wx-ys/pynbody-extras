"""Tests for ``pynbodyext.plot.image.data`` (the ``BinND`` → image bridge)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pynbody
import pytest

from pynbodyext.core.calculate import Bin1D
from pynbodyext.plot.image.data import ImageData


def make_sim(n: int = 200) -> pynbody.SimSnap:
    rng = np.random.default_rng(1)
    sim = pynbody.new(dm=n)
    sim["x"] = rng.uniform(0.0, 10.0, n)
    sim["y"] = rng.uniform(0.0, 10.0, n)
    sim["mass"] = rng.uniform(0.5, 2.0, n)
    return sim


def make_bins(sim: pynbody.SimSnap, nbins: int = 10):
    return (
        Bin1D("x", vmin=0.0, vmax=10.0, nbins=nbins, alias="x")
        @ Bin1D("y", vmin=0.0, vmax=10.0, nbins=nbins, alias="y")
    )(sim)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_image_data_holds_data_and_metadata() -> None:
    image = ImageData(np.zeros((4, 5)), extent=[0, 1, 2, 3], label="mass.sum", units="Msol")

    assert image.data.shape == (4, 5)
    assert image.extent == (0.0, 1.0, 2.0, 3.0)
    assert image.label == "mass.sum"
    assert image.units == "Msol"


def test_image_data_rejects_non_2d_data() -> None:
    with pytest.raises(ValueError, match="2-D"):
        ImageData(np.zeros(5))


def test_image_data_rejects_a_bad_extent() -> None:
    with pytest.raises(ValueError, match="extent"):
        ImageData(np.zeros((2, 2)), extent=(0.0, 1.0, 2.0))


def test_image_data_is_usable_as_an_array() -> None:
    data = np.arange(6).reshape(2, 3)

    image = ImageData(data)

    np.testing.assert_array_equal(np.asarray(image), data)
    assert image.shape == (2, 3)
    assert image.ndim == 2


def test_with_data_replaces_values_and_keeps_metadata() -> None:
    image = ImageData(np.zeros((4, 4)), extent=(0, 1, 0, 1), label="vz", units="km/s")

    smoothed = image.with_data(np.ones((4, 4)))

    np.testing.assert_allclose(smoothed.data, 1.0)
    assert smoothed.extent == image.extent
    assert smoothed.label == "vz"
    assert smoothed.units == "km/s"
    assert image.data.sum() == 0.0  # the original is untouched


def test_pixel_size_comes_from_extent_and_shape() -> None:
    image = ImageData(np.zeros((10, 20)), extent=(0.0, 40.0, -10.0, 10.0))

    assert image.pixel_size == (2.0, 2.0)


def test_pixel_size_is_undefined_without_extent() -> None:
    with pytest.raises(ValueError, match="extent"):
        ImageData(np.zeros((4, 4))).pixel_size


# ---------------------------------------------------------------------------
# BinND bridge
# ---------------------------------------------------------------------------


def test_from_bins_reads_the_grid_extent_units_and_axes() -> None:
    sim = make_sim()
    bins = make_bins(sim)

    image = ImageData.from_bins(bins, "mass.sum")

    assert image.data.shape == bins.shape_bins
    np.testing.assert_allclose(image.data, np.asarray(bins["mass.sum"].grid))
    assert image.extent == tuple(bins.axes.extent)
    assert image.label == "mass.sum"
    assert image.x_axis == "x"
    assert image.y_axis == "y"
    assert image.units == bins["mass.sum"].units


def test_from_bins_accepts_a_label_override() -> None:
    bins = make_bins(make_sim())

    image = ImageData.from_bins(bins, "mass.sum", label="projected mass")

    assert image.label == "projected mass"


def test_from_bins_requires_two_axes() -> None:
    sim = make_sim()
    bins_1d = Bin1D("x", vmin=0.0, vmax=10.0, nbins=5, alias="x")(sim)

    with pytest.raises(ValueError, match="2-D"):
        ImageData.from_bins(bins_1d, "mass.sum")


def test_from_bins_composes_with_the_smoothers() -> None:
    from pynbodyext.plot.image.postprocess import gaussian_smooth

    bins = make_bins(make_sim())
    image = ImageData.from_bins(bins, "mass.sum")

    smoothed = image.with_data(gaussian_smooth(image.data, fwhm=2.0, pixel_scale=image.pixel_size))

    assert smoothed.data.shape == image.data.shape
    assert bool(np.isfinite(smoothed.data).any())


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------


def test_imshow_uses_the_stored_extent() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(np.zeros((4, 6)), extent=(0.0, 30.0, -5.0, 5.0), label="mass", units="Msol")

    fig, ax = plt.subplots()
    try:
        artist = image.imshow(ax=ax)
        assert tuple(artist.get_extent()) == (0.0, 30.0, -5.0, 5.0)
        assert ax.get_ylabel() == "mass [Msol]"
    finally:
        plt.close(fig)


def test_imshow_can_override_the_extent_and_create_axes() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(np.zeros((4, 6)), extent=(0.0, 30.0, -5.0, 5.0))

    artist = image.imshow(extent=(0.0, 1.0, 0.0, 1.0))
    try:
        assert tuple(artist.get_extent()) == (0.0, 1.0, 0.0, 1.0)
    finally:
        plt.close(artist.figure)
