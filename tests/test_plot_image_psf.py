"""Tests for ``pynbodyext.plot.image.psf`` (observational effects)."""

from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.plot.image.psf import (
    convolve_psf,
    deconvolve_psf,
    gaussian_psf,
    normalize_psf,
    richardson_lucy,
    wiener_deconvolve,
)


def impulse(shape: tuple[int, int] = (31, 31)) -> np.ndarray:
    data = np.zeros(shape, dtype=float)
    data[shape[0] // 2, shape[1] // 2] = 1.0
    return data


# ---------------------------------------------------------------------------
# gaussian_psf / normalize_psf
# ---------------------------------------------------------------------------


def test_gaussian_psf_is_normalised_and_centred() -> None:
    psf = gaussian_psf(fwhm=4.0)

    assert psf.ndim == 2
    assert psf.shape[0] == psf.shape[1]
    assert psf.shape[0] % 2 == 1  # odd, so the kernel has a true centre
    assert psf.sum() == pytest.approx(1.0)

    centre = psf.shape[0] // 2
    assert np.unravel_index(np.argmax(psf), psf.shape) == (centre, centre)
    sigma = 4.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert psf[centre, centre] == pytest.approx(1.0 / (2.0 * np.pi * sigma**2), rel=1e-3)


def test_gaussian_psf_is_symmetric() -> None:
    psf = gaussian_psf(fwhm=3.0)

    np.testing.assert_allclose(psf, psf[::-1, :])
    np.testing.assert_allclose(psf, psf[:, ::-1])


def test_gaussian_psf_fwhm_and_sigma_agree() -> None:
    fwhm = 3.0
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    np.testing.assert_allclose(gaussian_psf(fwhm=fwhm), gaussian_psf(sigma=sigma))


def test_gaussian_psf_requires_exactly_one_width() -> None:
    with pytest.raises(ValueError, match="sigma.*fwhm|fwhm.*sigma"):
        gaussian_psf()
    with pytest.raises(ValueError, match="sigma.*fwhm|fwhm.*sigma"):
        gaussian_psf(fwhm=2.0, sigma=1.0)


def test_gaussian_psf_ellipticity_rotates_the_major_axis() -> None:
    flat = gaussian_psf(fwhm=4.0, e=0.5, theta=0.0)
    upright = gaussian_psf(fwhm=4.0, e=0.5, theta=90.0)

    assert flat.shape[1] > flat.shape[0]  # major axis along x
    assert upright.shape == flat.shape[::-1]  # rotated to lie along y
    np.testing.assert_allclose(flat.T, upright)
    assert not np.allclose(flat, gaussian_psf(fwhm=4.0, size=flat.shape))  # e actually stretches it


def test_gaussian_psf_rejects_impossible_ellipticity() -> None:
    with pytest.raises(ValueError, match="e"):
        gaussian_psf(fwhm=2.0, e=1.0)
    with pytest.raises(ValueError, match="e"):
        gaussian_psf(fwhm=2.0, e=-0.1)


def test_gaussian_psf_uses_pixel_scale() -> None:
    np.testing.assert_allclose(gaussian_psf(fwhm=0.5, pixel_scale=0.25), gaussian_psf(fwhm=2.0))


def test_gaussian_psf_honours_explicit_size() -> None:
    assert gaussian_psf(fwhm=2.0, size=7).shape == (7, 7)
    assert gaussian_psf(fwhm=2.0, size=(5, 9)).shape == (5, 9)


def test_gaussian_psf_clips_to_the_image_shape() -> None:
    psf = gaussian_psf(fwhm=10.0, shape=(11, 13))

    assert psf.shape[0] <= 11
    assert psf.shape[1] <= 13


def test_normalize_psf_scales_to_unit_sum() -> None:
    psf = normalize_psf(np.ones((5, 5)))

    assert psf.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(psf, 1.0 / 25.0)


def test_normalize_psf_rejects_zero_sum() -> None:
    with pytest.raises(ValueError, match="zero"):
        normalize_psf(np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# convolve_psf
# ---------------------------------------------------------------------------


def test_convolve_psf_turns_an_impulse_into_the_kernel() -> None:
    psf = gaussian_psf(fwhm=3.0, size=9)

    blurred = convolve_psf(impulse((41, 41)), psf)

    assert blurred.sum() == pytest.approx(1.0)
    centre = 20
    assert np.unravel_index(np.argmax(blurred), blurred.shape) == (centre, centre)
    assert blurred[centre, centre] == pytest.approx(psf[4, 4])
    assert blurred[centre - 2, centre] == pytest.approx(psf[2, 4])


def test_convolve_psf_preserves_flux_of_a_flat_field() -> None:
    data = np.full((21, 21), 2.0)

    blurred = convolve_psf(data, gaussian_psf(fwhm=2.0, size=9))

    np.testing.assert_allclose(blurred, 2.0)


def test_convolve_psf_accepts_a_width_instead_of_a_kernel() -> None:
    data = impulse((31, 31))

    from_width = convolve_psf(data, fwhm=3.0)
    from_kernel = convolve_psf(data, gaussian_psf(fwhm=3.0, shape=data.shape))

    np.testing.assert_allclose(from_width, from_kernel)


def test_convolve_psf_requires_a_kernel() -> None:
    with pytest.raises(ValueError, match="psf.*fwhm|fwhm.*psf"):
        convolve_psf(impulse((11, 11)))
    with pytest.raises(ValueError, match="psf.*fwhm|fwhm.*psf"):
        convolve_psf(impulse((11, 11)), gaussian_psf(fwhm=2.0), fwhm=3.0)


def test_convolve_psf_is_nan_aware() -> None:
    data = np.ones((21, 21))
    data[10, 10] = np.nan

    blurred = convolve_psf(data, gaussian_psf(fwhm=2.0, size=9))

    assert np.isnan(blurred[10, 10])
    np.testing.assert_allclose(blurred[np.isfinite(blurred)], 1.0)


def test_convolve_psf_normalises_away_a_masked_region() -> None:
    data = np.ones((21, 21))
    mask = np.ones((21, 21), dtype=bool)
    mask[:, :10] = False  # left half is not observed

    blurred = convolve_psf(data, gaussian_psf(fwhm=2.0, size=9), mask=mask)

    np.testing.assert_allclose(blurred[:, 10:], 1.0)
    assert np.isnan(blurred[:, :10]).all()


# ---------------------------------------------------------------------------
# deconvolution
# ---------------------------------------------------------------------------


def blurred_source(psf: np.ndarray, *, shape: tuple[int, int] = (41, 41), value: float = 100.0) -> np.ndarray:
    source = impulse(shape) * value
    return convolve_psf(source, psf, mode="same")


def test_wiener_deconvolve_sharpens_a_blurred_source() -> None:
    psf = gaussian_psf(fwhm=4.0, size=11)
    blurred = blurred_source(psf)

    restored = wiener_deconvolve(blurred, psf, balance=1e-6)

    assert restored.max() > 3.0 * blurred.max()  # the blur is undone, not merely rescaled
    assert restored.max() > 0.3 * 100.0  # ...recovering a good part of the source amplitude
    assert np.unravel_index(np.argmax(restored), restored.shape) == (20, 20)


def test_richardson_lucy_sharpens_a_blurred_source() -> None:
    psf = gaussian_psf(fwhm=3.0, size=9)
    blurred = blurred_source(psf)

    restored = richardson_lucy(blurred, psf, iterations=30)

    assert restored.max() > 5.0 * blurred.max()
    assert np.unravel_index(np.argmax(restored), restored.shape) == (20, 20)
    assert restored.min() >= 0.0


def test_deconvolve_psf_dispatches_on_method() -> None:
    psf = gaussian_psf(fwhm=3.0, size=9)
    blurred = blurred_source(psf)

    wiener = deconvolve_psf(blurred, psf, method="wiener", balance=1e-4)
    lucy = deconvolve_psf(blurred, psf, method="richardson_lucy", iterations=5)

    assert wiener.shape == blurred.shape
    assert lucy.shape == blurred.shape
    assert not np.allclose(wiener, lucy)


def test_deconvolve_psf_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        deconvolve_psf(impulse((11, 11)), gaussian_psf(fwhm=2.0), method="magic")


def test_deconvolution_keeps_non_finite_pixels_non_finite() -> None:
    psf = gaussian_psf(fwhm=3.0, size=9)
    blurred = blurred_source(psf)
    blurred[0, 0] = np.nan

    restored = wiener_deconvolve(blurred, psf)

    assert np.isnan(restored[0, 0])
