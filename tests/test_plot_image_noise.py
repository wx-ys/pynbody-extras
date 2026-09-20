"""Noise: ``add_noise`` (Gaussian) and ``add_poisson_noise`` (counting)."""

from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.plot.image import ImageData, add_noise, add_poisson_noise


def flat(value: float = 100.0, shape: tuple[int, int] = (64, 64), **kwargs: object) -> ImageData:
    defaults: dict[str, object] = {
        "extent": (0.0, float(shape[1]), 0.0, float(shape[0])),
        "label": "counts",
        "units": "counts",
    }
    defaults.update(kwargs)
    return ImageData(np.full(shape, value), **defaults)


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------


def test_gaussian_noise_has_the_requested_sigma() -> None:
    source = flat(100.0)

    noisy = add_noise(source.data, sigma=5.0, rng=0)

    residual = noisy - source.data
    assert residual.std() == pytest.approx(5.0, rel=0.05)
    assert residual.mean() == pytest.approx(0.0, abs=0.5)


def test_a_target_signal_to_noise_sets_the_sigma_per_pixel() -> None:
    source = flat(100.0)

    noisy = add_noise(source.data, snr=20.0, rng=0)

    residual = noisy - source.data
    assert residual.std() == pytest.approx(5.0, rel=0.05)  # 100 / 20


def test_a_sigma_map_is_used_pixel_by_pixel() -> None:
    data = np.zeros((8, 8))
    sigma = np.full((8, 8), 3.0)
    sigma[:, 4:] = 30.0

    noisy = add_noise(data, sigma=sigma, rng=1)

    assert noisy[:, :4].std() < noisy[:, 4:].std() / 5.0


def test_noise_is_deterministic_for_a_seed() -> None:
    source = flat()

    first = add_noise(source.data, sigma=1.0, rng=7)
    second = add_noise(source.data, sigma=1.0, rng=7)
    third = add_noise(source.data, sigma=1.0, rng=8)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_noise_leaves_non_finite_pixels_alone() -> None:
    source = flat()
    data = source.data.copy()
    data[0, 0] = np.nan

    noisy = add_noise(data, sigma=5.0, rng=0)

    assert np.isnan(noisy[0, 0])


def test_noise_respects_a_mask() -> None:
    source = flat()
    mask = np.zeros(source.shape, dtype=bool)
    mask[:, :10] = True

    noisy = add_noise(source.data, sigma=5.0, mask=mask, rng=0)

    np.testing.assert_allclose(noisy[:, 10:], source.data[:, 10:])  # untouched
    assert noisy[:, :10].std() > 0.0


def test_gaussian_noise_needs_one_of_sigma_or_snr() -> None:
    with pytest.raises(ValueError, match="sigma.*snr|snr.*sigma"):
        add_noise(np.ones((4, 4)))
    with pytest.raises(ValueError, match="sigma.*snr|snr.*sigma"):
        add_noise(np.ones((4, 4)), sigma=1.0, snr=10.0)


# ---------------------------------------------------------------------------
# Poisson noise
# ---------------------------------------------------------------------------


def test_poisson_noise_scales_with_the_exposure() -> None:
    source = flat(100.0)

    shallow = add_poisson_noise(source.data, exposure=0.1, rng=0)  # 10 counts
    deep = add_poisson_noise(source.data, exposure=10.0, rng=0)  # 1000 counts

    assert (shallow - source.data).std() == pytest.approx(np.sqrt(100.0 / 0.1), rel=0.15)
    assert (deep - source.data).std() < (shallow - source.data).std() / 5.0


def test_poisson_noise_subtracts_the_background_again() -> None:
    source = flat(10.0)

    noisy = add_poisson_noise(source.data, exposure=2.0, background=5.0, rng=3)

    assert (noisy - source.data).mean() == pytest.approx(0.0, abs=1.5)
    assert (noisy - source.data).std() == pytest.approx(np.sqrt(15.0 * 2.0) / 2.0, rel=0.2)


def test_poisson_noise_is_deterministic_and_masked() -> None:
    source = flat(20.0)
    mask = np.zeros(source.shape, dtype=bool)
    mask[0, 0] = True

    first = add_poisson_noise(source.data, exposure=1.0, rng=5)
    second = add_poisson_noise(source.data, exposure=1.0, rng=5)
    masked = add_poisson_noise(source.data, exposure=1.0, mask=mask, rng=5)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(masked[1:, 1:], source.data[1:, 1:])


def test_poisson_noise_rejects_a_negative_expectation() -> None:
    with pytest.raises(ValueError, match="negative"):
        add_poisson_noise(np.full((4, 4), -3.0), exposure=1.0)


def test_poisson_noise_needs_a_positive_exposure() -> None:
    with pytest.raises(ValueError, match="exposure"):
        add_poisson_noise(np.ones((4, 4)), exposure=0.0)


# ---------------------------------------------------------------------------
# the capability view
# ---------------------------------------------------------------------------


def test_the_noise_family_is_reachable_from_the_image() -> None:
    source = flat(50.0)

    gaussian = source.noise.gaussian(sigma=1.0, rng=0)
    poisson = source.noise.poisson(exposure=1.0, rng=0)

    assert isinstance(gaussian, ImageData)
    assert gaussian.shape == source.shape
    assert gaussian.extent == source.extent
    assert gaussian.label == "counts"
    assert [op.name for op in gaussian.ops] == ["add_noise"]
    assert [op.name for op in poisson.ops] == ["add_poisson_noise"]
    assert gaussian.ops[0].params["sigma"] == 1.0
    assert poisson.ops[0].params["exposure"] == 1.0


def test_the_noise_family_is_registered_and_exposed() -> None:
    from pynbodyext.plot.image import OPERATIONS

    assert "noise" in OPERATIONS
    assert "add_noise" in __import__("pynbodyext.plot.image", fromlist=["image"]).__all__
