"""Tests for ``pynbodyext.plot.image.data`` (the ``BinND`` → image bridge).

Covers both ends of the range: the simple map (same units on both axes, uniform
linear bins) and the general one (per-axis units, labels, and bin edges that need
not be evenly spaced).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pynbody
import pytest
from matplotlib.collections import QuadMesh

from pynbodyext.core.calculate import Bin1D
from pynbodyext.plot.image.data import ImageData


def make_sim(n: int = 300) -> pynbody.SimSnap:
    rng = np.random.default_rng(1)
    sim = pynbody.new(dm=n)
    sim["r"] = rng.uniform(1.0, 50.0, n)
    sim["K"] = rng.uniform(0.0, 5000.0, n)
    sim["mass"] = rng.uniform(0.5, 2.0, n)
    return sim


def make_bins(sim: pynbody.SimSnap, *, x_mode: str = "linear", nbins: int = 6):
    """2-D bins whose axes carry their own units (kpc and K)."""
    x_axis = Bin1D("r", vmin=1.0, vmax=50.0, nbins=nbins, alias="X", mode=x_mode, units="kpc")
    y_axis = Bin1D("K", vmin=0.0, vmax=5000.0, nbins=nbins, alias="Y", units="K")
    return (x_axis @ y_axis)(sim)


# ---------------------------------------------------------------------------
# the simple case: same units both ways, uniform linear bins
# ---------------------------------------------------------------------------


def test_simple_case_uses_one_extent_and_shared_units() -> None:
    image = ImageData(
        np.zeros((4, 5)),
        extent=(0.0, 5.0, -2.0, 2.0),
        x_units="kpc",
        y_units="kpc",
        x_label="x",
        y_label="y",
        label="mass.sum",
        units="1e10 Msol",
    )

    assert image.shape == (4, 5)
    assert image.extent == (0.0, 5.0, -2.0, 2.0)
    np.testing.assert_allclose(image.x_edges, [0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(image.y_edges, [-2, -1, 0, 1, 2])
    assert image.x_units == image.y_units == "kpc"
    assert image.label == "mass.sum"
    assert image.units == "1e10 Msol"


def test_default_axis_metadata_is_left_empty() -> None:
    image = ImageData(np.zeros((3, 3)))

    assert image.extent is None
    assert image.x_edges is None and image.y_edges is None
    assert image.x_units is None and image.y_units is None
    assert image.x_label is None and image.y_label is None
    assert image.uniform
    np.testing.assert_allclose(image.x_centers, [0.5, 1.5, 2.5])
    assert image.pixel_size == (1.0, 1.0)


# ---------------------------------------------------------------------------
# per-axis units, labels and non-uniform bin edges
# ---------------------------------------------------------------------------


def test_axes_may_carry_different_units_and_labels() -> None:
    image = ImageData(
        np.zeros((3, 4)),
        x_edges=[0.0, 1.0, 2.0, 3.0, 4.0],
        y_edges=[0.0, 100.0, 200.0, 300.0],
        x_units="kpc",
        y_units="K",
        x_label="radius",
        y_label="temperature",
    )

    assert image.x_units == "kpc"
    assert image.y_units == "K"
    assert image.x_label == "radius"
    assert image.y_label == "temperature"
    assert image.extent == (0.0, 4.0, 0.0, 300.0)


def test_non_uniform_edges_are_kept_and_centred_correctly() -> None:
    edges = [0.0, 1.0, 3.0, 10.0]

    image = ImageData(np.zeros((2, 3)), x_edges=edges, y_edges=[0.0, 5.0, 10.0])

    np.testing.assert_allclose(image.x_edges, edges)
    np.testing.assert_allclose(image.x_centers, [0.5, 2.0, 6.5])
    assert not image.x_uniform
    assert image.y_uniform
    assert not image.uniform
    assert image.extent == (0.0, 10.0, 0.0, 10.0)


def test_extent_is_derived_from_edges_when_only_edges_are_given() -> None:
    image = ImageData(np.zeros((2, 2)), x_edges=[1.0, 2.0, 4.0], y_edges=[-1.0, 0.0, 1.0])

    assert image.extent == (1.0, 4.0, -1.0, 1.0)


def test_extent_and_edges_must_agree() -> None:
    with pytest.raises(ValueError, match="extent"):
        ImageData(np.zeros((2, 2)), extent=(0.0, 1.0, 0.0, 1.0), x_edges=[0.0, 1.0, 2.0], y_edges=[0.0, 1.0, 2.0])


def test_edges_must_match_the_image_shape() -> None:
    with pytest.raises(ValueError, match="x_edges"):
        ImageData(np.zeros((2, 4)), x_edges=[0.0, 1.0, 2.0], y_edges=[0.0, 1.0, 2.0])


def test_edges_must_increase() -> None:
    with pytest.raises(ValueError, match="increas"):
        ImageData(np.zeros((2, 3)), x_edges=[0.0, 2.0, 1.0, 3.0], y_edges=[0.0, 1.0, 2.0])


def test_image_data_rejects_non_2d_data() -> None:
    with pytest.raises(ValueError, match="2-D"):
        ImageData(np.zeros(5))


def test_image_data_rejects_a_bad_extent() -> None:
    with pytest.raises(ValueError, match="extent"):
        ImageData(np.zeros((2, 2)), extent=(0.0, 1.0, 2.0))


# ---------------------------------------------------------------------------
# array behaviour, pixel size and copying
# ---------------------------------------------------------------------------


def test_image_data_is_usable_as_an_array() -> None:
    data = np.arange(6).reshape(2, 3)

    image = ImageData(data)

    np.testing.assert_array_equal(np.asarray(image), data)
    assert image.shape == (2, 3)
    assert image.ndim == 2


def test_pixel_size_covers_both_axes_independently() -> None:
    image = ImageData(np.zeros((5, 10)), extent=(0.0, 40.0, 0.0, 10.0))

    assert image.pixel_size == (2.0, 4.0)


def test_pixel_size_defaults_to_pixels_without_edges() -> None:
    assert ImageData(np.zeros((4, 4))).pixel_size == (1.0, 1.0)


def test_pixel_size_is_undefined_for_non_uniform_bins() -> None:
    image = ImageData(np.zeros((4, 4)), x_edges=[0.0, 1.0, 3.0, 6.0, 10.0], y_edges=[0.0, 1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="uniform"):
        _ = image.pixel_size


def test_with_data_replaces_values_and_keeps_metadata() -> None:
    image = ImageData(
        np.zeros((4, 4)),
        extent=(0, 1, 0, 1),
        x_units="kpc",
        y_units="kpc",
        x_label="x",
        y_label="y",
        label="vz",
        units="km/s",
    )

    smoothed = image.with_data(np.ones((4, 4)))

    np.testing.assert_allclose(smoothed.data, 1.0)
    assert smoothed.extent == image.extent
    assert smoothed.x_units == "kpc" and smoothed.y_units == "kpc"
    assert smoothed.x_label == "x" and smoothed.y_label == "y"
    assert smoothed.label == "vz"
    assert smoothed.units == "km/s"
    assert image.data.sum() == 0.0  # the original is untouched


def test_with_data_can_override_axis_metadata() -> None:
    image = ImageData(np.zeros((4, 4)), extent=(0, 1, 0, 1))

    relabelled = image.with_data(image.data, y_label="temperature", y_units="K")

    assert relabelled.y_label == "temperature"
    assert relabelled.y_units == "K"
    assert relabelled.x_label is None


# ---------------------------------------------------------------------------
# BinND bridge
# ---------------------------------------------------------------------------


def test_from_bins_reads_edges_units_labels_and_value_metadata() -> None:
    bins = make_bins(make_sim())

    image = ImageData.from_bins(bins, "mass.sum")

    # A bin grid is (x, y); an image is (row=y, column=x).
    assert image.data.shape == bins.shape_bins[::-1]
    np.testing.assert_allclose(image.data, np.asarray(bins["mass.sum"].grid).T)
    np.testing.assert_allclose(image.x_edges, np.asarray(bins.axes[0].edges))
    np.testing.assert_allclose(image.y_edges, np.asarray(bins.axes[1].edges))
    assert image.extent == tuple(bins.axes.extent)
    assert image.x_label == "r"  # the axis property, not just the alias
    assert image.y_label == "K"
    assert image.x_units == "kpc"
    assert image.y_units == "K"
    assert image.label == "mass.sum"
    assert image.units == bins["mass.sum"].units
    assert image.uniform  # linear bins in both directions


def test_from_bins_keeps_non_uniform_bins() -> None:
    bins = make_bins(make_sim(), x_mode="log")

    image = ImageData.from_bins(bins, "mass.sum")

    widths = np.diff(image.x_edges)
    assert not np.allclose(widths, widths[0])  # logarithmic bins
    assert not image.x_uniform
    assert image.y_uniform


def test_from_bins_accepts_label_and_unit_overrides() -> None:
    bins = make_bins(make_sim())

    image = ImageData.from_bins(
        bins, "mass.sum", label="surface density", units="Msol kpc**-2", x_label="R", y_units="deg"
    )

    assert image.label == "surface density"
    assert image.units == "Msol kpc**-2"
    assert image.x_label == "R"
    assert image.y_units == "deg"


def test_from_bins_requires_two_axes() -> None:
    bins_1d = Bin1D("r", vmin=1.0, vmax=50.0, nbins=5, alias="R", units="kpc")(make_sim())

    with pytest.raises(ValueError, match="2-D"):
        ImageData.from_bins(bins_1d, "mass.sum")


def test_from_bins_rejects_axes_with_gaps() -> None:
    sim = make_sim()
    gapped = Bin1D("r", lows=[0, 5, 10], highs=[1, 6, 11], alias="X", units="kpc")
    bins = (gapped @ Bin1D("K", vmin=0.0, vmax=5000.0, nbins=3, alias="Y", units="K"))(sim)

    with pytest.raises(ValueError, match="contiguous"):
        ImageData.from_bins(bins, "mass.sum")


def test_from_bins_composes_with_the_smoothers() -> None:
    from pynbodyext.plot.image.postprocess import gaussian_smooth

    image = ImageData.from_bins(make_bins(make_sim()), "mass.sum")

    smoothed = image.with_data(gaussian_smooth(image.data, fwhm=2.0, pixel_scale=image.pixel_size))

    assert smoothed.data.shape == image.data.shape
    assert bool(np.isfinite(smoothed.data).any())


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------


def test_imshow_uses_the_stored_extent_and_axis_labels() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(
        np.zeros((4, 6)), extent=(0.0, 30.0, -5.0, 5.0), x_label="radius", y_label="height", x_units="kpc", y_units="pc"
    )

    fig, ax = plt.subplots()
    try:
        artist = image.display.imshow(ax=ax)
        assert tuple(artist.get_extent()) == (0.0, 30.0, -5.0, 5.0)
        assert ax.get_xlabel() == "radius [kpc]"
        assert ax.get_ylabel() == "height [pc]"
    finally:
        plt.close(fig)


def test_imshow_can_override_the_extent_and_create_axes() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(np.zeros((4, 6)), extent=(0.0, 30.0, -5.0, 5.0))

    artist = image.display.imshow(extent=(0.0, 1.0, 0.0, 1.0))
    try:
        assert tuple(artist.get_extent()) == (0.0, 1.0, 0.0, 1.0)
    finally:
        plt.close(artist.figure)


def test_colorbar_is_opt_in_and_uses_the_value_metadata() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(np.zeros((4, 6)), extent=(0, 1, 0, 1), label="mass.sum", units="1e10 Msol")

    fig, ax = plt.subplots()
    try:
        image.display.imshow(ax=ax)
        assert len(fig.axes) == 1
        image.display.imshow(ax=ax, colorbar=True)
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "mass.sum [1e10 Msol]"
    finally:
        plt.close(fig)


def test_imshow_refuses_non_uniform_bins() -> None:
    image = ImageData(np.zeros((4, 4)), x_edges=[0.0, 1.0, 3.0, 6.0, 10.0], y_edges=[0.0, 1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="pcolormesh"):
        image.display.imshow()


def test_pcolormesh_draws_non_uniform_bins() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(
        np.arange(12.0).reshape(3, 4),
        x_edges=[0.0, 1.0, 3.0, 6.0, 10.0],
        y_edges=[0.0, 2.0, 5.0, 9.0],
        x_units="kpc",
        y_units="kpc",
    )

    fig, ax = plt.subplots()
    try:
        artist = image.display.pcolormesh(ax=ax)
        assert isinstance(artist, QuadMesh)
        assert ax.get_xlim() == (0.0, 10.0)
        assert ax.get_ylim() == (0.0, 9.0)
        assert ax.get_xlabel() == "[kpc]"
    finally:
        plt.close(fig)


def test_pcolormesh_also_works_for_uniform_bins() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = ImageData(np.zeros((4, 6)), extent=(0.0, 30.0, -5.0, 5.0))

    fig, ax = plt.subplots()
    try:
        artist = image.display.pcolormesh(ax=ax)
        assert isinstance(artist, QuadMesh)
        assert ax.get_xlim() == (0.0, 30.0)
    finally:
        plt.close(fig)
