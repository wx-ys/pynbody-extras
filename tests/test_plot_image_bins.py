"""The bridge between a 2-D ``BinNDResult`` and the image layer.

Drawing a binned result is the image layer's job: ``BinNDResult.imshow`` must hand
the grid over to :class:`ImageData`, which owns orientation, extent, axis labels
and the choice between ``imshow`` and ``pcolormesh``.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pynbody
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage

from pynbodyext.core.calculate import Bin1D
from pynbodyext.plot.image import ImageData


def make_sim(n: int = 400) -> pynbody.SimSnap:
    rng = np.random.default_rng(2)
    sim = pynbody.new(dm=n)
    sim["r"] = rng.uniform(1.0, 50.0, n)
    sim["K"] = rng.uniform(0.0, 5000.0, n)
    sim["mass"] = rng.uniform(0.5, 2.0, n)
    return sim


def make_bins(sim: pynbody.SimSnap, *, x_nbins: int = 4, y_nbins: int = 7, x_mode: str = "linear"):
    """A 2-D binning whose two axes deliberately differ in bin count and units."""
    x_axis = Bin1D("r", vmin=1.0, vmax=50.0, nbins=x_nbins, alias="X", mode=x_mode, units="kpc")
    y_axis = Bin1D("K", vmin=0.0, vmax=5000.0, nbins=y_nbins, alias="Y", units="K")
    return (x_axis @ y_axis)(sim)


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------


def test_from_bins_transposes_the_grid_into_image_convention() -> None:
    """A bin grid is (x, y); an image is (row=y, column=x)."""
    bins = make_bins(make_sim())

    image = ImageData.from_bins(bins, "mass.sum")

    assert bins.shape_bins == (4, 7)
    assert image.shape == (7, 4)
    np.testing.assert_allclose(image.data, np.asarray(bins["mass.sum"].grid).T)
    np.testing.assert_allclose(image.x_edges, np.asarray(bins.axes[0].edges))
    np.testing.assert_allclose(image.y_edges, np.asarray(bins.axes[1].edges))
    assert len(image.x_edges) == 5 and len(image.y_edges) == 8


def test_from_bins_keeps_values_on_the_right_pixels() -> None:
    sim = make_sim()
    # The x axis carries a strong gradient: after transposition it must run along
    # the columns of the image, not the rows.
    bins = (
        Bin1D("r", lows=[0.0, 5.0], highs=[5.0, 10.0], alias="X", units="kpc")
        @ Bin1D("K", lows=[0.0], highs=[5000.0], alias="Y", units="K")
    )(sim)

    image = ImageData.from_bins(bins, "count")

    assert image.shape == (1, 2)
    counts = np.asarray(bins["count"].grid)
    np.testing.assert_allclose(image.data, counts.T)
    assert image.data.shape == (1, 2)


def test_from_bins_marks_non_uniform_bins() -> None:
    image = ImageData.from_bins(make_bins(make_sim(), x_mode="log"), "mass.sum")

    assert not image.x_uniform  # logarithmic along x
    assert image.y_uniform
    assert image.data.shape == (7, 4)


def test_from_bins_needs_two_axes() -> None:
    bins_1d = Bin1D("r", vmin=1.0, vmax=50.0, nbins=5, alias="R", units="kpc")(make_sim())

    with pytest.raises(ValueError, match="2-D"):
        ImageData.from_bins(bins_1d, "mass.sum")


# ---------------------------------------------------------------------------
# BinNDResult.imshow delegates to the image layer
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _agg_backend() -> None:
    matplotlib.use("Agg")


def test_binnd_imshow_draws_the_binned_grid() -> None:
    import matplotlib.pyplot as plt

    sim = make_sim()
    bins = make_bins(sim)

    fig, ax = plt.subplots()
    try:
        artist = bins.imshow("mass.sum", ax=ax)

        assert isinstance(artist, AxesImage)
        assert artist.get_array().shape == bins.shape_bins[::-1]
        np.testing.assert_allclose(artist.get_array(), np.asarray(bins["mass.sum"].grid).T)
        assert tuple(artist.get_extent()) == tuple(bins.axes.extent)
    finally:
        plt.close(fig)


def test_binnd_imshow_labels_the_axes_with_units() -> None:
    import matplotlib.pyplot as plt

    bins = make_bins(make_sim())

    fig, ax = plt.subplots()
    try:
        bins.imshow("mass.sum", ax=ax)

        assert ax.get_xlabel() == "r [kpc]"
        assert ax.get_ylabel() == "K [K]"
    finally:
        plt.close(fig)


def test_binnd_imshow_keeps_the_aspect_and_figsize_options() -> None:
    import matplotlib.pyplot as plt

    bins = make_bins(make_sim())

    fig, ax = plt.subplots()
    try:
        bins.imshow("mass.sum", ax=ax, aspect="auto", cmap="inferno")

        assert ax.get_aspect() == "auto"
        assert ax.images[0].get_cmap().name == "inferno"
    finally:
        plt.close(fig)

    artist = bins.imshow("mass.sum", figsize=(3.0, 3.0))
    plt.close(artist.figure)


def test_binnd_imshow_switches_to_pcolormesh_for_uneven_bins() -> None:
    """Logarithmic bins cannot be drawn with imshow, and the bridge knows it."""
    import matplotlib.pyplot as plt

    bins = make_bins(make_sim(), x_nbins=6, x_mode="log")

    fig, ax = plt.subplots()
    try:
        artist = bins.imshow("mass.sum", ax=ax)

        assert isinstance(artist, QuadMesh)
        np.testing.assert_allclose(ax.get_xlim(), (1.0, 50.0))
    finally:
        plt.close(fig)


def test_binnd_imshow_rejects_one_dimensional_results() -> None:
    bins_1d = Bin1D("r", vmin=1.0, vmax=50.0, nbins=5, alias="R")(make_sim())

    with pytest.raises(ValueError, match="2 bin axes|2-D"):
        bins_1d.imshow("count")


def test_binnd_imshow_accepts_a_colorbar_request() -> None:
    import matplotlib.pyplot as plt

    bins = make_bins(make_sim())

    fig, ax = plt.subplots()
    try:
        bins.imshow("mass.sum", ax=ax, colorbar=True)

        assert len(fig.axes) == 2
        assert "mass.sum" in fig.axes[1].get_ylabel()
    finally:
        plt.close(fig)
