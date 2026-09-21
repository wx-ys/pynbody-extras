"""Observational effects: apply a point-spread function, or invert one.

Two directions are supported, both on plain 2-D arrays:

* :func:`convolve_psf` turns a model map into what an observation would see, by
  convolving it with :func:`gaussian_psf` or with any measured PSF image.
* :func:`wiener_deconvolve` and :func:`richardson_lucy` go the other way, undoing
  a known blur.  Deconvolution amplifies noise, so the forward direction is the
  one to use when comparing to data; the inverse is a visualisation tool.

Non-finite pixels (empty bins, masked regions) are handled the same way as in
:mod:`~pynbodyext.plot.image.smooth`: they neither contribute to nor receive
signal, and stay non-finite in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from ._arrays import aligned_values, as_2d, as_pair, masked_filter, resolve_sigma, validity_mask
from .ops import ImageOps

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ._types import KernelWidth, MaskLike
    from .data import ImageData

__all__ = [
    "PsfOps",
    "convolve_psf",
    "deconvolve_psf",
    "gaussian_psf",
    "normalize_psf",
    "richardson_lucy",
    "wiener_deconvolve",
]

_METHODS = ("auto", "fft", "direct")


def normalize_psf(psf: ArrayLike) -> np.ndarray:
    """Scale a kernel so that it sums to one, preserving total flux.

    Raises
    ------
    ValueError
        If the kernel sums to zero or is not finite.
    """
    array = np.asarray(psf, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"psf must be a 2-D kernel, got shape {array.shape}.")
    total = array.sum()
    if not np.isfinite(total) or total == 0.0:
        raise ValueError(f"psf sums to {total}, cannot normalize a zero-sum kernel.")
    return array / total


def gaussian_psf(
    fwhm: KernelWidth | None = None,
    *,
    sigma: KernelWidth | None = None,
    size: int | tuple[int, int] | None = None,
    shape: tuple[int, int] | None = None,
    e: float = 0.0,
    theta: float = 0.0,
    pixel_scale: KernelWidth | None = None,
    normalize: bool = True,
    truncate: float = 4.0,
) -> np.ndarray:
    """Build a Gaussian point-spread function kernel.

    Parameters
    ----------
    fwhm : float, optional
        Full width at half maximum, in pixels unless *pixel_scale* is given.
        Exactly one of *fwhm* and *sigma* must be supplied.
    sigma : float, optional
        Standard deviation of the Gaussian.
    size : int or (int, int), optional
        Kernel shape.  Defaults to ``2 * ceil(truncate * sigma) + 1`` in each
        direction, clipped to *shape* when that is given.
    shape : (int, int), optional
        Shape of the image the kernel will be applied to; the kernel is clipped
        to it so that it never exceeds the image.
    e : float, default: 0.0
        Ellipticity in ``[0, 1)``: ``e = 0`` is circular, and larger values
        lengthen the major axis while keeping the enclosed area fixed.
    theta : float, default: 0.0
        Position angle of the major axis, measured in degrees from the ``+x``
        direction (the image's second axis) and increasing counter-clockwise.
    pixel_scale : float or (float, float), optional
        Physical size of one pixel; when given, *fwhm*/*sigma* are interpreted
        in those units.
    normalize : bool, default: True
        Whether to scale the kernel to unit sum.
    truncate : float, default: 4.0
        Kernel radius in units of sigma, used when *size* is not given.

    Returns
    -------
    numpy.ndarray
        2-D kernel with an odd number of pixels along each direction.

    Examples
    --------
    >>> gaussian_psf(fwhm=3.0, size=9).sum()  # doctest: +SKIP
    1.0
    """
    width = resolve_sigma(sigma, fwhm, pixel_scale)
    if not 0.0 <= e < 1.0:
        raise ValueError(f"Ellipticity e must be in [0, 1), got {e!r}.")
    if isinstance(width, tuple):
        if e != 0.0:
            raise ValueError("Ellipticity applies to a circular width; pass a scalar 'sigma' or 'fwhm'.")
        sigma_y, sigma_x = width
    else:
        sigma_x = width / np.sqrt(1.0 - e)
        sigma_y = width * np.sqrt(1.0 - e)
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("PSF width must be positive.")
    size_y, size_x = _kernel_size(size, shape, sigma_x, sigma_y, theta, truncate)
    half_y, half_x = size_y // 2, size_x // 2
    offsets_y, offsets_x = np.mgrid[-half_y : half_y + 1, -half_x : half_x + 1]
    angle = np.radians(theta)
    along_major = offsets_x * np.cos(angle) + offsets_y * np.sin(angle)
    along_minor = -offsets_x * np.sin(angle) + offsets_y * np.cos(angle)
    kernel = np.exp(-0.5 * ((along_major / sigma_x) ** 2 + (along_minor / sigma_y) ** 2))
    return normalize_psf(kernel) if normalize else kernel


def _kernel_size(
    size: int | tuple[int, int] | None,
    shape: tuple[int, int] | None,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    truncate: float,
) -> tuple[int, int]:
    """Resolve the kernel shape from an explicit size, an image shape and sigma."""
    if size is not None:
        pair = as_pair(size, name="size")
        size_y, size_x = (pair, pair) if isinstance(pair, int) else pair
    else:
        # Project the (possibly rotated) ellipse onto each image axis, so that a
        # kernel rotated by 90 degrees gets a transposed canvas.
        angle = np.radians(theta)
        spread_x = np.hypot(sigma_x * np.cos(angle), sigma_y * np.sin(angle))
        spread_y = np.hypot(sigma_x * np.sin(angle), sigma_y * np.cos(angle))
        size_y = 2 * int(np.ceil(truncate * spread_y)) + 1
        size_x = 2 * int(np.ceil(truncate * spread_x)) + 1
        if shape is not None:
            size_y = min(size_y, shape[0] if shape[0] % 2 else shape[0] - 1)
            size_x = min(size_x, shape[1] if shape[1] % 2 else shape[1] - 1)
    if size_y < 1 or size_x < 1:
        raise ValueError(f"Kernel size {(size_y, size_x)} must be positive.")
    return size_y, size_x


def convolve_psf(
    image: ArrayLike,
    psf: ArrayLike | None = None,
    *,
    fwhm: KernelWidth | None = None,
    sigma: KernelWidth | None = None,
    mode: str = "same",
    method: str = "auto",
    mask: MaskLike = None,
    normalize: bool = True,
    pixel_scale: KernelWidth | None = None,
    **psf_kwargs: Any,
) -> np.ndarray:
    """Convolve an image with a point-spread function.

    Parameters
    ----------
    image : array_like
        2-D image to blur; non-finite pixels are left out of the convolution.
    psf : array_like, optional
        Kernel to convolve with.  Give either *psf* or ``fwhm``/``sigma``.
    fwhm, sigma : float, optional
        Width of a Gaussian kernel built on the fly (see :func:`gaussian_psf`).
    mode : str, default: "same"
        Output size, as in ``scipy.signal.fftconvolve``.
    method : {"auto", "fft", "direct"}, default: "auto"
        Convolution implementation; ``"auto"`` picks FFT for kernels larger than
        64 pixels and a direct convolution otherwise.
    mask : array_like of bool, optional
        Per-pixel validity mask; ``False`` and non-finite pixels neither
        contribute nor receive signal.
    normalize : bool, default: True
        Scale the kernel to unit sum first, so that flux is preserved.
    pixel_scale : float or (float, float), optional
        Physical size of one pixel, for a Gaussian kernel.
    **psf_kwargs
        Forwarded to :func:`gaussian_psf` (e.g. ``e``, ``theta``, ``size``).

    Returns
    -------
    numpy.ndarray
        Blurred image, with non-finite pixels where the input had none.

    Examples
    --------
    >>> observed = convolve_psf(model_image, fwhm=0.3, pixel_scale=0.05)  # doctest: +SKIP
    """
    if (psf is None) == (fwhm is None and sigma is None):
        raise ValueError("Pass either 'psf' or one of 'fwhm'/'sigma' (not both).")
    array = as_2d(image)
    if psf is None:
        kernel = gaussian_psf(fwhm=fwhm, sigma=sigma, shape=array.shape, pixel_scale=pixel_scale, **psf_kwargs)
        if not normalize:
            kernel = kernel * kernel.sum()
    else:
        kernel = normalize_psf(psf) if normalize else np.asarray(psf, dtype=float)
    valid = validity_mask(array, mask)
    return masked_filter(array, valid, lambda values: _convolve(values, kernel, mode=mode, method=method))


def _convolve(values: np.ndarray, kernel: np.ndarray, *, mode: str, method: str) -> np.ndarray:
    """Convolve *values* with *kernel* using the requested implementation."""
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(_METHODS)}.")
    if method == "auto":
        method = "fft" if kernel.size > 64 else "direct"
    if method == "fft":
        return signal.fftconvolve(values, kernel, mode=mode)
    return signal.convolve(values, kernel, mode=mode)


def _embedded_kernel(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place *kernel* on a zero canvas of *shape* with its centre at index (0, 0)."""
    kernel_y, kernel_x = kernel.shape
    if kernel_y > shape[0] or kernel_x > shape[1]:
        raise ValueError(f"psf shape {kernel.shape} must not exceed the image shape {shape}.")
    canvas = np.zeros(shape, dtype=float)
    canvas[:kernel_y, :kernel_x] = kernel
    return np.roll(canvas, (-(kernel_y // 2), -(kernel_x // 2)), axis=(0, 1))


def wiener_deconvolve(image: ArrayLike, psf: ArrayLike, *, balance: float = 1e-2, mask: MaskLike = None) -> np.ndarray:
    """Undo a blur with a Wiener filter in the Fourier domain.

    Parameters
    ----------
    image : array_like
        Blurred 2-D image.
    psf : array_like
        The kernel the image was blurred with.
    balance : float, default: 1e-2
        Noise-to-signal regularisation.  Smaller values sharpen harder and
        amplify noise more; this is the knob to trade the two against each other.
    mask : array_like of bool, optional
        Per-pixel validity mask.

    Returns
    -------
    numpy.ndarray
        Deconvolved image, non-finite where the input was non-finite.
    """
    array = as_2d(image)
    kernel = normalize_psf(psf)
    valid = validity_mask(array, mask)
    observed = np.where(valid, array, 0.0)
    transfer = np.fft.rfft2(_embedded_kernel(kernel, array.shape))
    power = np.abs(transfer) ** 2
    regularisation = max(float(balance), 0.0) * power.max()
    restored = np.fft.irfft2(np.conj(transfer) / (power + regularisation) * np.fft.rfft2(observed), s=array.shape)
    return np.where(valid, restored, np.nan)


def richardson_lucy(
    image: ArrayLike, psf: ArrayLike, *, iterations: int = 10, epsilon: float = 1e-12, mask: MaskLike = None
) -> np.ndarray:
    """Undo a blur with the Richardson-Lucy algorithm.

    This is an iterative, positivity-preserving maximum-likelihood restoration.
    Compared with :func:`wiener_deconvolve` it produces no ringing but needs many
    iterations to sharpen, and it converges to a spiky image if run for too long.

    Parameters
    ----------
    image : array_like
        Blurred 2-D image.
    psf : array_like
        The kernel the image was blurred with.
    iterations : int, default: 10
        Number of iterations.
    epsilon : float, default: 1e-12
        Floor used when dividing by the re-blurred estimate.
    mask : array_like of bool, optional
        Per-pixel validity mask.

    Returns
    -------
    numpy.ndarray
        Deconvolved image, non-finite where the input was non-finite.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations!r}.")
    array = as_2d(image)
    kernel = normalize_psf(psf)
    valid = validity_mask(array, mask)
    observed = np.where(valid, array, 0.0)
    estimate = np.clip(observed, 0.0, None)
    mirrored = kernel[::-1, ::-1]
    for _ in range(iterations):
        blurred = signal.fftconvolve(estimate, kernel, mode="same")
        ratio = np.divide(observed, blurred, out=np.zeros_like(observed), where=blurred > epsilon)
        estimate = np.clip(estimate * signal.fftconvolve(ratio, mirrored, mode="same"), 0.0, None)
    return np.where(valid, estimate, np.nan)


def deconvolve_psf(image: ArrayLike, psf: ArrayLike, *, method: str = "wiener", **kwargs: Any) -> np.ndarray:
    """Deconvolve *image* with *psf* using the requested algorithm.

    Parameters
    ----------
    image, psf : array_like
        Blurred image and the kernel it was blurred with.
    method : {"wiener", "richardson_lucy"}, default: "wiener"
        Algorithm; extra keyword arguments go to the chosen function.

    Returns
    -------
    numpy.ndarray
        Deconvolved image.
    """
    if method == "wiener":
        return wiener_deconvolve(image, psf, **kwargs)
    if method == "richardson_lucy":
        return richardson_lucy(image, psf, **kwargs)
    raise ValueError(f"Unknown method {method!r}; choose 'wiener' or 'richardson_lucy'.")


@dataclass(frozen=True)
class PsfOps(ImageOps):
    """The observational family of an image: ``image.process.psf.convolve(fwhm=3)``.

    Each method returns a new :class:`~pynbodyext.plot.image.data.ImageData`.  As
    with smoothing, a Gaussian width is given in the units of the axes whenever
    the grid is evenly spaced, and in pixels otherwise.
    """

    def convolve(
        self,
        psf: ArrayLike | None = None,
        *,
        fwhm: KernelWidth | None = None,
        sigma: KernelWidth | None = None,
        mode: str = "same",
        method: str = "auto",
        mask: MaskLike = None,
        normalize: bool = True,
        **psf_kwargs: Any,
    ) -> ImageData:
        """Blur the image with a PSF, forward-modelling an observation.

        Parameters
        ----------
        psf : array_like, optional
            The kernel to convolve with — from :func:`gaussian_psf` or a measured
            PSF.  Give either *psf* or ``fwhm``/``sigma``.
        fwhm, sigma : float, optional
            Width of a Gaussian kernel built on the fly, in the units of the axes when
            the grid is evenly spaced, and in pixels otherwise.  Give one of the two.
        mode : str, default: "same"
            Output size, as in ``scipy.signal.fftconvolve``.
        method : {"auto", "fft", "direct"}, default: "auto"
            Convolution implementation; ``"auto"`` uses FFT for kernels larger than 64
            pixels and a direct convolution otherwise.
        mask : array_like of bool, optional
            Pixels to include; the rest neither contribute nor receive signal, and an
            image-like mask is oriented for you.
        normalize : bool, default: True
            Scale the kernel to unit sum first, so that flux is preserved.
        **psf_kwargs
            Forwarded to :func:`gaussian_psf` when a Gaussian kernel is built, e.g.
            ``e``, ``theta``, ``size`` or ``pixel_scale``.

        Returns
        -------
        ImageData
            The blurred image, with the same geometry as this one.

        Examples
        --------
        What a telescope would see, before anything is added to it:

        >>> observed = model.process.psf.convolve(fwhm=3.0)  # doctest: +SKIP
        >>> observed = model.process.psf.convolve(measured_psf, mode="same")  # doctest: +SKIP

        See Also
        --------
        wiener, richardson_lucy :
            Going the other way, to undo a known blur.
        :func:`~pynbodyext.plot.image.psf.convolve_psf` :
            The array-level form.
        """
        blurred = convolve_psf(
            self.image.data,
            psf,
            fwhm=fwhm,
            sigma=sigma,
            mode=mode,
            method=method,
            mask=None if mask is None else aligned_values(mask),
            normalize=normalize,
            pixel_scale=self.kernel_scale(),
            **psf_kwargs,
        )
        return self.image._derived(
            blurred, "convolve_psf", {"fwhm": fwhm, "sigma": sigma, "mode": mode, "method": method, "mask": mask}
        )

    def wiener(self, psf: ArrayLike, *, balance: float = 1e-2, mask: MaskLike = None) -> ImageData:
        """Undo a blur with a Wiener filter, in the Fourier domain.

        Fast and steady, at the cost of ringing around sharp features; the result is
        for looking at, not for measuring.

        Parameters
        ----------
        psf : array_like
            The kernel the image was blurred with (from :func:`gaussian_psf` or a
            measured PSF).
        balance : float, default: 1e-2
            Noise-to-signal regularisation.  Smaller sharpens harder and amplifies
            noise more — this is the knob to trade the two.
        mask : array_like of bool, optional
            Pixels to include, oriented for you when image-like.

        Returns
        -------
        ImageData
            The restored image, with ``wiener_deconvolve(...)`` appended to ``.ops``.

        Examples
        --------
        >>> sharpened = observed.process.psf.wiener(psf, balance=1e-6)  # doctest: +SKIP

        See Also
        --------
        richardson_lucy, deconvolve :
            The iterative alternative, and the dispatcher.
        :func:`~pynbodyext.plot.image.psf.wiener_deconvolve` :
            The array-level form.
        """
        restored = wiener_deconvolve(
            self.image.data, psf, balance=balance, mask=None if mask is None else aligned_values(mask)
        )
        return self.image._derived(restored, "wiener_deconvolve", {"balance": balance, "mask": mask})

    def richardson_lucy(
        self, psf: ArrayLike, *, iterations: int = 10, epsilon: float = 1e-12, mask: MaskLike = None
    ) -> ImageData:
        """Undo a blur iteratively (Richardson-Lucy).

        Positivity-preserving and ring-free, but it needs many iterations to sharpen
        and turns spiky if run for too long.

        Parameters
        ----------
        psf : array_like
            The kernel the image was blurred with.
        iterations : int, default: 10
            Number of iterations.  A few dozen is a useful range; more will keep
            concentrating the signal onto individual pixels.
        epsilon : float, default: 1e-12
            Floor used when dividing by the re-blurred estimate.
        mask : array_like of bool, optional
            Pixels to include, oriented for you when image-like.

        Returns
        -------
        ImageData
            The restored image, with ``richardson_lucy(...)`` appended to ``.ops``.

        Examples
        --------
        >>> sharpened = observed.process.psf.richardson_lucy(psf, iterations=30)  # doctest: +SKIP

        See Also
        --------
        :func:`~pynbodyext.plot.image.psf.richardson_lucy` : the array-level form.
        """
        restored = richardson_lucy(
            self.image.data,
            psf,
            iterations=iterations,
            epsilon=epsilon,
            mask=None if mask is None else aligned_values(mask),
        )
        return self.image._derived(restored, "richardson_lucy", {"iterations": iterations, "mask": mask})

    def deconvolve(self, psf: ArrayLike, *, method: str = "wiener", **kwargs: Any) -> ImageData:
        """Undo a blur, choosing the algorithm by name.

        Parameters
        ----------
        psf : array_like
            The kernel the image was blurred with.
        method : {"wiener", "richardson_lucy"}, default: "wiener"
            The algorithm; remaining keyword arguments go to it (``balance`` for
            Wiener, ``iterations`` for Richardson-Lucy).

        Returns
        -------
        ImageData
            The restored image, with ``deconvolve_psf(...)`` appended to ``.ops``.

        Examples
        --------
        >>> sharpened = observed.process.psf.deconvolve(psf, method="richardson_lucy", iterations=20)  # doctest: +SKIP

        See Also
        --------
        wiener, richardson_lucy :
            The two algorithms, called directly.
        :func:`~pynbodyext.plot.image.psf.deconvolve_psf` :
            The array-level form.
        """
        restored = deconvolve_psf(self.image.data, psf, method=method, **kwargs)
        return self.image._derived(restored, "deconvolve_psf", {"method": method, **kwargs})
