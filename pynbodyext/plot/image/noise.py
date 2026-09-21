"""Observation noise: what a map looks like once it is counted, not computed.

Two models, both seeded so a figure can be reproduced:

* :func:`add_noise` — Gaussian white noise, either with a known ``sigma`` (scalar
  or a per-pixel map) or with a target ``snr`` per pixel (``sigma = |data| / snr``,
  the "this observation reaches S/N = 20" way of saying it).
* :func:`add_poisson_noise` — counting noise: the data are treated as expected
  counts, an ``exposure`` says how many counts per unit signal were collected
  (bigger = deeper = quieter), and an optional uniform ``background`` is added
  before counting and subtracted again afterwards.

Both leave non-finite pixels alone and honour a ``mask`` (``True`` = observed), and
both accept anything the rest of the layer accepts as a map — a plain array, an
:class:`~pynbodyext.plot.image.data.ImageData`, or a binned array::

    image.process.noise.gaussian(snr=20, rng=0)
    image.process.noise.poisson(exposure=0.5, background=2.0, rng=1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._arrays import aligned_values, as_2d, validity_mask
from .ops import ImageOps

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ._types import KernelWidth, MapLike, MaskLike
    from .data import ImageData

__all__ = ["NoiseOps", "add_noise", "add_poisson_noise"]


def _generator(rng: int | np.random.Generator | None) -> np.random.Generator:
    """A generator from a seed, an existing generator, or nothing at all."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _broadcast(value: MapLike, shape: tuple[int, int], *, name: str) -> np.ndarray:
    """A scalar or an array matching *shape*."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(shape, float(array))
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape} or a scalar.")
    return array


def add_noise(
    data: ArrayLike,
    *,
    sigma: KernelWidth | None = None,
    snr: float | ArrayLike | None = None,
    mask: MaskLike = None,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian white noise to a map.

    Parameters
    ----------
    data : array_like
        The map; non-finite pixels are left alone.
    sigma : float or array_like, optional
        Noise standard deviation, per pixel when an array is given.  Exactly one of
        *sigma* and *snr* must be given.
    snr : float or array_like, optional
        Target signal-to-noise per pixel, so ``sigma = |data| / snr``.
    mask : array_like of bool, optional
        Pixels to perturb; ``False`` pixels are returned unchanged.  An image-like
        mask is oriented for you.
    rng : int, numpy.random.Generator, optional
        Seed or generator, for a reproducible figure.

    Returns
    -------
    numpy.ndarray
        The noisy map.

    Examples
    --------
    >>> noisy = add_noise(flux, snr=20, rng=0)  # doctest: +SKIP
    """
    array = as_2d(data)
    if (sigma is None) == (snr is None):
        raise ValueError("Pass exactly one of 'sigma' or 'snr'.")
    if sigma is None:
        noise_sigma = np.abs(array) / _broadcast(snr, array.shape, name="snr")
    else:
        noise_sigma = _broadcast(sigma, array.shape, name="sigma")
    touched = validity_mask(array, None if mask is None else aligned_values(mask))
    if sigma is None:
        touched &= np.isfinite(noise_sigma) & (noise_sigma > 0.0)
    else:
        touched &= np.isfinite(noise_sigma)
    drawn = _generator(rng).normal(0.0, 1.0, array.shape) * noise_sigma
    return np.where(touched, array + drawn, array)


def add_poisson_noise(
    data: ArrayLike,
    *,
    exposure: float = 1.0,
    background: float = 0.0,
    mask: MaskLike = None,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Add counting noise to a map of expected counts.

    The expectation per pixel is ``(data + background) * exposure`` counts; the
    draw is divided by *exposure* and the background subtracted again, so the result
    is in the same units as the input with variance ``(data + background) /
    exposure``.  Larger *exposure* therefore means a deeper, quieter observation.

    Parameters
    ----------
    data : array_like
        Expected signal per pixel; must not be negative once the background is
        added.  Non-finite pixels are left alone.
    exposure : float, default: 1.0
        Counts collected per unit signal; must be positive.
    background : float, default: 0.0
        Uniform background counted along with the signal, in the same units.
    mask : array_like of bool, optional
        Pixels to perturb; ``False`` pixels are returned unchanged.
    rng : int, numpy.random.Generator, optional
        Seed or generator, for a reproducible figure.

    Returns
    -------
    numpy.ndarray
        The noisy map.

    Examples
    --------
    >>> noisy = add_poisson_noise(counts, exposure=0.1, rng=0)  # doctest: +SKIP
    """
    array = as_2d(data)
    if exposure <= 0.0:
        raise ValueError(f"exposure must be positive, got {exposure!r}.")
    expectation = array + float(background)
    touched = validity_mask(array, None if mask is None else aligned_values(mask)) & (expectation >= 0.0)
    if mask is None and np.any(np.isfinite(expectation) & (expectation < 0.0)):
        raise ValueError("add_poisson_noise needs a non-negative expectation; lower the background.")
    counted = _generator(rng).poisson(np.where(touched, expectation, 0.0) * exposure)
    return np.where(touched, counted / exposure - float(background), array)


@dataclass(frozen=True)
class NoiseOps(ImageOps):
    """The noise family of an image: ``image.process.noise.gaussian(snr=20)``."""

    def gaussian(
        self,
        *,
        sigma: KernelWidth | None = None,
        snr: float | ArrayLike | None = None,
        mask: MaskLike = None,
        rng: int | np.random.Generator | None = None,
    ) -> ImageData:
        """Add Gaussian white noise, from a known sigma or a target S/N.

        Parameters
        ----------
        sigma : float or array_like, optional
            Noise standard deviation — a scalar, or a per-pixel map.  Give exactly
            one of *sigma* and *snr*.
        snr : float or array_like, optional
            Target signal-to-noise per pixel, so ``sigma = |data| / snr`` — the "this
            observation reaches S/N = 20" spelling.
        mask : array_like of bool, optional
            Pixels to perturb; ``False`` pixels come back unchanged, and an
            image-like mask is oriented for you.
        rng : int or numpy.random.Generator, optional
            Seed or generator, so a noisy figure can be reproduced — the seed is
            recorded in ``.ops``.

        Returns
        -------
        ImageData
            The noisy image, geometry and labels intact, with ``add_noise(...)``
            appended to ``.ops``.  Non-finite pixels are left alone.

        Examples
        --------
        >>> noisy = model.process.noise.gaussian(snr=20, rng=0)  # doctest: +SKIP
        >>> noisy.ops[-1]  # doctest: +SKIP
        add_noise(snr=20.0, rng=0)

        See Also
        --------
        poisson :
            Counting noise, for a map of expected counts.
        :func:`~pynbodyext.plot.image.noise.add_noise` :
            The array-level form.
        """
        noisy = add_noise(self.data, sigma=sigma, snr=snr, mask=mask, rng=rng)
        return self.derive(noisy, "add_noise", {"sigma": sigma, "snr": snr, "mask": mask, "rng": _seed_of(rng)})

    def poisson(
        self,
        *,
        exposure: float = 1.0,
        background: float = 0.0,
        mask: MaskLike = None,
        rng: int | np.random.Generator | None = None,
    ) -> ImageData:
        """Add counting (Poisson) noise to a map of expected counts.

        Parameters
        ----------
        exposure : float, default: 1.0
            Counts collected per unit signal.  Larger means deeper and quieter: the
            variance of the result is ``(data + background) / exposure``.
        background : float, default: 0.0
            Uniform background counted along with the signal, in the same units, and
            subtracted again afterwards.
        mask : array_like of bool, optional
            Pixels to perturb; ``False`` pixels come back unchanged.
        rng : int or numpy.random.Generator, optional
            Seed or generator, recorded in ``.ops``.

        Returns
        -------
        ImageData
            The noisy image in the input's units, with ``add_poisson_noise(...)``
            appended to ``.ops``.  Non-finite pixels are left alone, and a negative
            expectation is refused.

        Examples
        --------
        >>> shallow = counts.process.noise.poisson(exposure=0.05, background=2.0, rng=1)  # doctest: +SKIP

        See Also
        --------
        :func:`~pynbodyext.plot.image.noise.add_poisson_noise` :
            The array-level form.
        """
        noisy = add_poisson_noise(self.data, exposure=exposure, background=background, mask=mask, rng=rng)
        return self.derive(
            noisy,
            "add_poisson_noise",
            {"exposure": exposure, "background": background, "mask": mask, "rng": _seed_of(rng)},
        )


def _seed_of(rng: int | np.random.Generator | None) -> int | None:
    """What to record about the generator: the seed when there is one."""
    if rng is None or isinstance(rng, np.random.Generator):
        return None
    return rng
