"""Tests for ``pynbodyext.plot.image.postprocess``.

Covers NaN-aware smoothing, unit-aware kernel sizes, display stretches, and
block downsampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.plot.image.postprocess import box_smooth, downsample, gaussian_smooth, median_filter, normalize


def impulse(shape: tuple[int, int] = (15, 15), value: float = 1.0) -> np.ndarray:
    data = np.zeros(shape, dtype=float)
    data[shape[0] // 2, shape[1] // 2] = value
    return data


# ---------------------------------------------------------------------------
# gaussian_smooth
# ---------------------------------------------------------------------------


def test_gaussian_smooth_conserves_total_signal() -> None:
    smoothed = gaussian_smooth(impulse(), sigma=1.0)
    assert smoothed.sum() == pytest.approx(1.0, rel=1e-6)


def test_gaussian_smooth_spreads_signal_over_neighbours() -> None:
    smoothed = gaussian_smooth(impulse(), sigma=1.0)
    assert smoothed[7, 7] > smoothed[7, 8] > smoothed[7, 9] > 0


def test_gaussian_smooth_fwhm_and_sigma_agree() -> None:
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))  # fwhm of a sigma=1 kernel
    np.testing.assert_allclose(gaussian_smooth(impulse(), fwhm=fwhm), gaussian_smooth(impulse(), sigma=1.0))


def test_gaussian_smooth_requires_exactly_one_width() -> None:
    with pytest.raises(ValueError, match="sigma.*fwhm|fwhm.*sigma"):
        gaussian_smooth(impulse())
    with pytest.raises(ValueError, match="sigma.*fwhm|fwhm.*sigma"):
        gaussian_smooth(impulse(), sigma=1.0, fwhm=1.0)


def test_gaussian_smooth_is_nan_aware() -> None:
    data = np.ones((9, 9))
    data[4, 4] = np.nan

    smoothed = gaussian_smooth(data, sigma=1.5)

    assert np.isnan(smoothed[4, 4])
    np.testing.assert_allclose(smoothed[~np.isnan(smoothed)], 1.0, rtol=1e-9)


def test_gaussian_smooth_leaves_non_finite_input_non_finite() -> None:
    clean = impulse()
    data = clean.copy()
    data[0, 0] = np.inf

    smoothed = gaussian_smooth(data, sigma=1.0)

    assert not np.isfinite(smoothed[0, 0])
    # A bad corner must not leak into the rest of the image.
    np.testing.assert_allclose(smoothed[7, 7], gaussian_smooth(clean, sigma=1.0)[7, 7])


def test_gaussian_smooth_honours_mask() -> None:
    data = np.ones((11, 11))

    smoothed = gaussian_smooth(data, sigma=1.0, mask=np.zeros((11, 11), dtype=bool))

    assert np.isnan(smoothed).all()


def test_gaussian_smooth_uses_pixel_scale_for_kernel_width() -> None:
    data = impulse()
    np.testing.assert_allclose(
        gaussian_smooth(data, fwhm=0.5, pixel_scale=(0.25, 0.25)), gaussian_smooth(data, fwhm=2.0)
    )


def test_gaussian_smooth_accepts_per_axis_width() -> None:
    anisotropic = gaussian_smooth(impulse(), sigma=(0.0, 2.0))
    isotropic = gaussian_smooth(impulse(), sigma=2.0)

    assert np.isclose(anisotropic.sum(), 1.0)
    assert not np.allclose(anisotropic, isotropic)


def test_gaussian_smooth_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        gaussian_smooth(np.zeros(5), sigma=1.0)


# ---------------------------------------------------------------------------
# box_smooth / median_filter
# ---------------------------------------------------------------------------


def test_box_smooth_averages_neighbours() -> None:
    data = np.zeros((5, 5))
    data[2, 2] = 9.0

    smoothed = box_smooth(data, size=3)

    np.testing.assert_allclose(smoothed[2, 2], 1.0)


def test_box_smooth_is_nan_aware() -> None:
    data = np.ones((7, 7))
    data[3, 3] = np.nan

    smoothed = box_smooth(data, size=3)

    assert np.isnan(smoothed[3, 3])
    np.testing.assert_allclose(smoothed[0, 0], 1.0)


def test_median_filter_removes_hot_pixel() -> None:
    data = np.zeros((5, 5))
    data[2, 2] = 100.0

    filtered = median_filter(data, size=3)

    np.testing.assert_allclose(filtered, 0.0)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_linear_maps_range_to_unit_interval() -> None:
    values = np.array([[0.0, 5.0], [10.0, 2.5]])

    normalized = normalize(values)

    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized[1, 1] == pytest.approx(0.25)


def test_normalize_honours_explicit_limits() -> None:
    normalized = normalize(np.array([0.0, 5.0, 10.0]), vmin=0.0, vmax=20.0)
    np.testing.assert_allclose(normalized, [0.0, 0.25, 0.5])


def test_normalize_clips_to_unit_interval() -> None:
    normalized = normalize(np.array([-10.0, 0.0, 20.0]), vmin=0.0, vmax=10.0)
    np.testing.assert_allclose(normalized, [0.0, 0.0, 1.0])


def test_normalize_uses_percentiles_for_robust_limits() -> None:
    values = np.concatenate([np.linspace(0.0, 1.0, 100), [1000.0]])

    full = normalize(values, percentiles=(0, 100))
    robust = normalize(values, percentiles=(0, 99))

    assert full[-1] == pytest.approx(1.0)
    assert full[50] < 0.01  # the single outlier sets the whole scale
    assert robust[-1] == 1.0
    assert robust[50] == pytest.approx(0.5, abs=0.02)


def test_normalize_keeps_non_finite_values_non_finite() -> None:
    values = np.array([[0.0, np.nan], [np.inf, 1.0]])

    normalized = normalize(values)

    assert np.isnan(normalized[0, 1])
    assert np.isnan(normalized[1, 0])


def test_normalize_returns_zeros_when_limits_are_equal() -> None:
    np.testing.assert_allclose(normalize(np.full((2, 2), 3.0)), 0.0)


@pytest.mark.parametrize("stretch", ["sqrt", "log", "asinh", "hist"])
def test_stretches_are_monotonic_and_bounded(stretch: str) -> None:
    values = np.linspace(0.0, 100.0, 101)

    normalized = normalize(values, stretch=stretch)

    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)
    assert np.all(np.diff(normalized) >= 0)


def test_log_stretch_boosts_low_end() -> None:
    values = np.array([0.0, 10.0, 100.0])

    linear = normalize(values)
    log = normalize(values, stretch="log")

    assert log[1] > linear[1]


def test_normalize_rejects_unknown_stretch() -> None:
    with pytest.raises(ValueError, match="stretch"):
        normalize(np.zeros((2, 2)), stretch="gamma")


# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


def test_downsample_block_mean() -> None:
    data = np.arange(16.0).reshape(4, 4)

    reduced = downsample(data, factor=2)

    assert reduced.shape == (2, 2)
    np.testing.assert_allclose(reduced, [[2.5, 4.5], [10.5, 12.5]])


def test_downsample_block_sum_and_median() -> None:
    data = np.arange(16.0).reshape(4, 4)

    np.testing.assert_allclose(downsample(data, factor=2, func="sum"), [[10.0, 18.0], [42.0, 50.0]])
    np.testing.assert_allclose(downsample(data, factor=4, func="median"), [[7.5]])


def test_downsample_is_nan_aware() -> None:
    data = np.ones((4, 4))
    data[0, 0] = np.nan

    reduced = downsample(data, factor=2)

    np.testing.assert_allclose(reduced, 1.0)


def test_downsample_rejects_incomplete_blocks() -> None:
    with pytest.raises(ValueError, match="divisible"):
        downsample(np.zeros((5, 4)), factor=2)


def test_downsample_rejects_unknown_func() -> None:
    with pytest.raises(ValueError, match="func"):
        downsample(np.zeros((4, 4)), factor=2, func="mode")
