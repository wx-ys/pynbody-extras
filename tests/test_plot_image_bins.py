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
# binned arrays are (x, y); images are (row=y, column=x)
# ---------------------------------------------------------------------------


def test_as_image_transposes_a_binned_array() -> None:
    from pynbodyext.plot.image import as_image

    bins = make_bins(make_sim())
    query_result = bins["mass.sum"]

    image = as_image(query_result)

    assert image.shape == bins.shape_bins[::-1]
    np.testing.assert_allclose(image.data, np.asarray(query_result.grid).T)
    np.testing.assert_allclose(image.x_edges, np.asarray(bins.axes[0].edges))
    np.testing.assert_allclose(image.y_edges, np.asarray(bins.axes[1].edges))
    assert image.x_units == "kpc"
    assert image.units == query_result.units


def test_as_image_passes_an_image_through_unchanged() -> None:
    from pynbodyext.plot.image import as_image

    source = ImageData(np.zeros((4, 4)))

    assert as_image(source) is source


def test_as_image_wraps_a_plain_array_without_geometry() -> None:
    from pynbodyext.plot.image import as_image

    image = as_image(np.zeros((4, 4)))

    assert image.extent is None
    assert image.x_edges is None


def test_as_image_rejects_nonsense() -> None:
    from pynbodyext.plot.image import as_image

    with pytest.raises(TypeError, match="as_image"):
        as_image("not an image")


def test_adaptive_bin_accepts_a_binned_array_without_transposing() -> None:
    """The reported footgun: ``bins.s["count"]`` must work as it reads."""
    pytest.importorskip("powerbin")
    bins = make_bins(make_sim())
    velocity = ImageData.from_bins(bins, "vz.mean")

    from_query = velocity.process.adaptive.bin(bins["mass.sum"], target_nbins=6)
    from_image = velocity.process.adaptive.bin(ImageData.from_bins(bins, "mass.sum"), target_nbins=6)

    np.testing.assert_allclose(from_query.value, from_image.value, equal_nan=True)
    np.testing.assert_allclose(from_query.bin_capacity, from_image.bin_capacity)
    assert from_query.n_bins == from_image.n_bins


def test_a_raw_binned_grid_is_rejected_with_an_orientation_hint() -> None:
    """A raw (x, y) array cannot be told apart from an image: say so clearly."""
    pytest.importorskip("powerbin")
    bins = make_bins(make_sim())
    velocity = ImageData.from_bins(bins, "vz.mean")

    with pytest.raises(ValueError, match="transpos"):
        velocity.process.adaptive.bin(np.asarray(bins["mass.sum"]), target_nbins=6)


def test_free_adaptive_binning_also_accepts_binned_arrays() -> None:
    pytest.importorskip("powerbin")
    from pynbodyext.plot.image import adaptive_bin_map

    bins = make_bins(make_sim())

    via_arrays = adaptive_bin_map(bins["vz.mean"], bins["mass.sum"], target_nbins=6, verbose=0)
    via_images = adaptive_bin_map(
        ImageData.from_bins(bins, "vz.mean"), ImageData.from_bins(bins, "mass.sum"), target_nbins=6, verbose=0
    )

    np.testing.assert_allclose(via_arrays.value, via_images.value, equal_nan=True)


def test_compose_accepts_a_binned_array_for_the_other_map() -> None:
    bins = make_bins(make_sim())
    velocity = ImageData.from_bins(bins, "vz.mean")

    composed = velocity.process.compose(bins["mass.sum"])
    expected = velocity.process.compose(ImageData.from_bins(bins, "mass.sum"))

    np.testing.assert_allclose(composed, expected)


def test_with_data_hints_at_a_transposed_grid() -> None:
    bins = make_bins(make_sim())
    velocity = ImageData.from_bins(bins, "vz.mean")

    with pytest.raises(ValueError, match="transpos"):
        velocity.with_data(np.asarray(bins["mass.sum"]))


def test_masks_can_be_taken_from_an_image() -> None:
    pytest.importorskip("powerbin")
    bins = make_bins(make_sim())
    velocity = ImageData.from_bins(bins, "vz.mean")
    mask = ImageData.from_bins(bins, "mass.sum").data > 0.0  # already image-oriented

    blurred = velocity.process.psf.convolve(fwhm=2.0, mask=mask)

    assert blurred.shape == velocity.shape
    assert bool(np.isfinite(blurred.data).any())


def test_a_mask_cut_from_a_binned_array_gets_the_orientation_hint() -> None:
    """Comparing a BinsArray drops its type, so only the hint can save the user."""
    bins = make_bins(make_sim())
    velocity = ImageData.from_bins(bins, "vz.mean")
    mask = bins["mass.sum"] > 0.0  # a plain (x, y) array by now

    with pytest.raises(ValueError, match="transpos"):
        velocity.process.smooth.box(size=3, mask=mask)


# ---------------------------------------------------------------------------
# the bridge, discoverable from the binned array itself
# ---------------------------------------------------------------------------


def test_a_binned_array_can_hand_over_its_image() -> None:
    bins = make_bins(make_sim())

    image = bins["mass.sum"].image

    assert isinstance(image, ImageData)
    assert image.shape == bins.shape_bins[::-1]
    np.testing.assert_allclose(image.data, np.asarray(bins["mass.sum"].grid).T)
    assert image.x_units == "kpc"
    assert image.units == bins["mass.sum"].units


def test_the_binned_image_can_be_drawn_directly() -> None:
    import matplotlib.pyplot as plt

    bins = make_bins(make_sim())

    artist = bins["mass.sum"].image.display.draw(colorbar=True)

    assert len(artist.figure.axes) == 2
    plt.close(artist.figure)


def test_a_one_dimensional_binned_array_has_no_image() -> None:
    bins_1d = Bin1D("r", vmin=1.0, vmax=50.0, nbins=5, alias="R")(make_sim())

    with pytest.raises(ValueError, match="2-D"):
        _ = bins_1d["count"].image


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
