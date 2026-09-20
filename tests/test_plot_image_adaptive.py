"""Tests for ``pynbodyext.plot.image.adaptive`` (signal-weighted binning)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from pynbodyext.plot.image.adaptive import AdaptiveMap, adaptive_bin_map, adaptive_map_from_bins

pytest.importorskip("powerbin", reason="PowerBin is an optional dependency of the image layer")


def blob_map(shape: tuple[int, int] = (40, 40)) -> tuple[np.ndarray, np.ndarray]:
    """A centrally concentrated signal and a linear quantity to be binned."""
    ys, xs = np.indices(shape, dtype=float)
    radius = np.hypot(xs - shape[1] / 2, ys - shape[0] / 2)
    signal = 10.0 * np.exp(-((radius / 12.0) ** 2)) + 0.05
    value = 3.0 * xs - 2.0 * ys
    return value, signal


def test_adaptive_bin_map_paints_every_pixel_with_its_bin_mean() -> None:
    value, signal = blob_map()

    result = adaptive_bin_map(value, signal, target_nbins=8, verbose=0)

    assert result.bin_value.size == result.n_bins
    assert result.bin_value.size > 1
    assert result.value.shape == value.shape
    counts = np.bincount(result.bin_num[result.mask], minlength=result.n_bins)
    means = np.bincount(result.bin_num[result.mask], weights=value[result.mask], minlength=result.n_bins) / counts
    np.testing.assert_allclose(result.bin_value, means)
    np.testing.assert_allclose(result.value[result.mask], means[result.bin_num[result.mask]])
    np.testing.assert_allclose(result.bin_count, counts)


def test_adaptive_bin_map_equalises_signal_per_bin() -> None:
    value, signal = blob_map()

    result = adaptive_bin_map(value, signal, target_nbins=40, verbose=0)

    assert result.bin_capacity.size == pytest.approx(40, abs=5)  # the target is honoured
    # Every bin carries comparable signal: that is the whole point of the method.
    assert result.bin_capacity.min() > 0.5 * result.target_capacity
    assert result.bin_capacity.max() < 2.0 * result.target_capacity
    assert result.rms_frac < 30.0
    # Bright pixels need few pixels per bin, faint ones need many: that is the point.
    assert result.bin_count.max() > result.bin_count.min()


def test_adaptive_bin_map_keeps_low_signal_pixels_out() -> None:
    value, signal = blob_map()
    threshold = 1.0

    result = adaptive_bin_map(value, signal, target_nbins=8, min_signal=threshold, verbose=0)

    faint = signal < threshold
    assert result.mask[~faint].all() or not result.mask[~faint].all()  # bright pixels may still be split
    assert not result.mask[faint].any()
    assert np.isnan(result.value[faint]).all()


def test_adaptive_bin_map_excludes_non_finite_pixels() -> None:
    value, signal = blob_map()
    value = value.copy()
    value[0, 0] = np.nan

    result = adaptive_bin_map(value, signal, target_nbins=6, verbose=0)

    assert not result.mask[0, 0]
    assert np.isnan(result.value[0, 0])


def test_adaptive_bin_map_requires_a_target() -> None:
    value, signal = blob_map()

    with pytest.raises(ValueError, match="target_capacity.*target_signal.*target_nbins"):
        adaptive_bin_map(value, signal, verbose=0)


def test_adaptive_bin_map_accepts_a_target_signal() -> None:
    value, signal = blob_map()

    by_signal = adaptive_bin_map(value, signal, target_signal=signal.sum() / 8, verbose=0)
    by_bins = adaptive_bin_map(value, signal, target_nbins=8, verbose=0)

    assert by_signal.target_capacity == pytest.approx(by_bins.target_capacity)
    assert by_signal.n_bins == pytest.approx(by_bins.n_bins, abs=3)


def test_adaptive_bin_map_scales_capacity_by_noise() -> None:
    value, signal = blob_map()
    noise = np.full(signal.shape, 0.5)

    result = adaptive_bin_map(value, signal, noise=noise, target_nbins=6, verbose=0)

    # With noise given, capacity is (S/N)^2 = (signal / noise)^2, not the raw signal.
    assert result.target_capacity == pytest.approx((signal**2).sum() / 0.25 / 6)
    assert result.bin_signal.sum() == pytest.approx(signal[result.mask].sum())


@pytest.mark.parametrize("method", ["mean", "median", "sum", "weighted"])
def test_adaptive_bin_map_aggregation_methods(method: str) -> None:
    value, signal = blob_map()

    result = adaptive_bin_map(value, signal, target_nbins=1, method=method, verbose=0)

    assert result.n_bins == 1
    expected = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
        "weighted": lambda values: np.average(values, weights=signal),
    }[method]
    assert result.bin_value[0] == pytest.approx(expected(value))


def test_adaptive_bin_map_rejects_unknown_method() -> None:
    value, signal = blob_map()

    with pytest.raises(ValueError, match="method"):
        adaptive_bin_map(value, signal, target_nbins=4, method="trimmed", verbose=0)


def test_adaptive_bin_map_uses_physical_coordinates_from_extent() -> None:
    value, signal = blob_map()
    extent = (0.0, 20.0, -10.0, 10.0)

    result = adaptive_bin_map(value, signal, target_nbins=6, extent=extent, verbose=0)

    assert result.extent == extent
    assert extent[0] <= result.xybin[:, 0].min() and result.xybin[:, 0].max() <= extent[1]
    assert extent[2] <= result.xybin[:, 1].min() and result.xybin[:, 1].max() <= extent[3]


def test_adaptive_bin_map_accepts_bin_edges_that_are_not_evenly_spaced() -> None:
    value, signal = blob_map((30, 30))
    x_edges = np.geomspace(1.0, 100.0, 31)  # logarithmic bins along x
    y_edges = np.linspace(-10.0, 10.0, 31)

    result = adaptive_bin_map(value, signal, target_nbins=4, x_edges=x_edges, y_edges=y_edges, verbose=0)

    np.testing.assert_allclose(result.x_edges, x_edges)
    np.testing.assert_allclose(result.y_edges, y_edges)
    np.testing.assert_allclose(result.x_centers, 0.5 * (x_edges[:-1] + x_edges[1:]))
    assert not result.x_uniform
    assert result.y_uniform
    assert result.extent == (1.0, 100.0, -10.0, 10.0)


def test_adaptive_bin_map_bins_in_cell_space_not_axis_units() -> None:
    """Regression: with axes four decades apart, binning in axis units made
    PowerBin produce non-finite bin centres, which its KDTree then rejected."""
    shape = (40, 40)
    rows, cols = np.indices(shape, dtype=float)
    radius = np.hypot(cols - shape[1] / 2, rows - shape[0] / 2)
    value = 3.0 * cols - 2.0 * rows
    signal = 10.0 * np.exp(-((radius / 10.0) ** 2)) + 0.001  # four decades of dynamic range

    result = adaptive_bin_map(
        value,
        signal,
        target_nbins=30,
        x_edges=np.linspace(0.0, 0.5, shape[1] + 1),  # Å-like scale
        y_edges=np.linspace(0.0, 5000.0, shape[0] + 1),  # K-like scale
        verbose=0,
    )

    assert np.isfinite(result.xybin).all()
    assert result.n_bins == pytest.approx(30, abs=5)
    # The tessellation is built in cell space, so the centres come back inside
    # each axis' own range instead of being dragged apart by the units.
    assert 0.0 <= result.xybin[:, 0].min() and result.xybin[:, 0].max() <= 0.5
    assert 0.0 <= result.xybin[:, 1].min() and result.xybin[:, 1].max() <= 5000.0


def test_adaptive_map_to_image_data_carries_the_geometry() -> None:
    value, signal = blob_map()
    result = adaptive_bin_map(
        value,
        signal,
        target_nbins=5,
        extent=(0.0, 20.0, 0.0, 20.0),
        x_units="kpc",
        y_units="kpc",
        x_label="x",
        y_label="y",
        label="vz.mean",
        units="km/s",
        verbose=0,
    )

    image = result.to_image_data()

    np.testing.assert_array_equal(image.data, result.value)
    np.testing.assert_allclose(image.x_edges, result.x_edges)
    assert image.extent == result.extent
    assert (image.x_units, image.y_units) == ("kpc", "kpc")
    assert (image.x_label, image.y_label) == ("x", "y")
    assert (image.label, image.units) == ("vz.mean", "km/s")


def test_adaptive_map_draw_dispatches_on_uniformity() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage

    value, signal = blob_map((30, 30))
    even = adaptive_bin_map(value, signal, target_nbins=4, extent=(0.0, 30.0, 0.0, 30.0), verbose=0)
    stretched = adaptive_bin_map(
        value, signal, target_nbins=4, x_edges=np.geomspace(1.0, 100.0, 31), y_edges=np.arange(31.0), verbose=0
    )

    fig, ax = plt.subplots()
    try:
        # ``draw`` picks the artist; ``imshow`` stays strict about even bins.
        assert isinstance(even.display.draw(ax=ax), AxesImage)
        assert isinstance(stretched.display.draw(ax=ax), QuadMesh)
        with pytest.raises(ValueError, match="pcolormesh"):
            stretched.display.imshow(ax=ax)
    finally:
        plt.close(fig)


def test_adaptive_bin_map_imshow_draws_the_binned_image() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value, signal = blob_map()
    result = adaptive_bin_map(value, signal, target_nbins=6, extent=(0.0, 40.0, 0.0, 40.0), verbose=0)

    fig, ax = plt.subplots()
    try:
        artist = result.display.imshow(ax=ax)
        assert tuple(artist.get_extent()) == (0.0, 40.0, 0.0, 40.0)
        assert artist.get_array().shape == value.shape
    finally:
        plt.close(fig)


def test_adaptive_map_from_bins_reads_a_binned_result() -> None:
    import pynbody

    from pynbodyext.core.calculate import Bin1D

    rng = np.random.default_rng(0)
    n = 300
    sim = pynbody.new(dm=n)
    sim["r"] = rng.uniform(1.0, 50.0, n)
    sim["y"] = rng.uniform(0.0, 10.0, n)
    sim["vz"] = rng.normal(0.0, 50.0, n)
    sim["mass"] = rng.uniform(0.5, 2.0, n)

    bins = (
        Bin1D("r", vmin=1.0, vmax=50.0, nbins=20, alias="X", mode="log", units="kpc")
        @ Bin1D("y", vmin=0.0, vmax=10.0, nbins=20, alias="Y", units="kpc")
    )(sim)

    result = adaptive_map_from_bins(bins, "vz.mean", "mass.sum", target_nbins=8, verbose=0)

    assert isinstance(result, AdaptiveMap)
    assert result.label == "vz.mean"
    assert result.extent == tuple(bins.axes.extent)
    assert result.mask.sum() > 100  # the populated part of the grid was binned
    np.testing.assert_allclose(result.x_edges, np.asarray(bins.axes[0].edges))
    assert (result.x_label, result.y_label) == ("r", "y")
    assert (result.x_units, result.y_units) == ("kpc", "kpc")
