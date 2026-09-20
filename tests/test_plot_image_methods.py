"""The method API of :class:`ImageData`: families, provenance and chaining.

The free functions stay the implementation (and have their own tests); these
tests pin the behaviour of the thin methods built on top of them.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from pynbodyext.plot.image import AdaptiveMap, ImageData, ImageOp, MapStyle, gaussian_smooth, median_filter
from pynbodyext.plot.image.psf import convolve_psf, gaussian_psf
from pynbodyext.plot.image.postprocess import box_smooth, downsample, normalize


def image(shape: tuple[int, int] = (8, 8), **kwargs: object) -> ImageData:
    """A small image with a unit scale, i.e. pixel_size == 1."""
    data = np.arange(float(shape[0] * shape[1])).reshape(shape)
    defaults: dict[str, object] = {"extent": (0.0, float(shape[1]), 0.0, float(shape[0])), "label": "vz.mean"}
    defaults.update(kwargs)
    return ImageData(data, **defaults)


# ---------------------------------------------------------------------------
# smoothing family
# ---------------------------------------------------------------------------


def test_smooth_gaussian_matches_the_free_function() -> None:
    source = image()

    via_method = source.postprocess.smooth.gaussian(fwhm=2.0)

    np.testing.assert_allclose(via_method.data, gaussian_smooth(source.data, fwhm=2.0, pixel_scale=source.pixel_size))


def test_smooth_methods_keep_the_geometry_and_metadata() -> None:
    source = image(x_units="kpc", y_units="kpc", units="km/s")

    for smoothed in (source.postprocess.smooth.gaussian(fwhm=2.0), source.postprocess.smooth.box(size=3), source.postprocess.smooth.median(size=3)):
        assert smoothed.shape == source.shape
        assert smoothed.extent == source.extent
        assert smoothed.label == "vz.mean"
        assert (smoothed.x_units, smoothed.y_units) == ("kpc", "kpc")
        assert smoothed.units == "km/s"


def test_smooth_methods_record_what_they_did() -> None:
    smoothed = image().postprocess.smooth.gaussian(fwhm=2.0)

    assert [op.name for op in smoothed.ops] == ["gaussian_smooth"]
    assert smoothed.ops[0].params["fwhm"] == 2.0


def test_box_and_median_methods_match_their_functions() -> None:
    source = image()

    np.testing.assert_allclose(source.postprocess.smooth.box(size=3).data, box_smooth(source.data, 3))
    np.testing.assert_allclose(source.postprocess.smooth.median(size=3).data, median_filter(source.data, 3))


def test_downsample_shrinks_the_image_and_its_edges() -> None:
    source = image((8, 8), extent=(0.0, 8.0, 0.0, 8.0))

    reduced = source.postprocess.smooth.downsample(factor=2)

    assert reduced.shape == (4, 4)
    np.testing.assert_allclose(reduced.data, downsample(source.data, 2))
    np.testing.assert_allclose(reduced.x_edges, [0.0, 2.0, 4.0, 6.0, 8.0])
    np.testing.assert_allclose(reduced.y_edges, [0.0, 2.0, 4.0, 6.0, 8.0])
    assert reduced.extent == source.extent
    assert reduced.ops[-1].name == "downsample"


def test_kernel_widths_are_in_axis_units_on_a_uniform_grid() -> None:
    source = image((4, 4), extent=(0.0, 8.0, 0.0, 8.0))  # pixel_size == 2

    unit_based = source.postprocess.smooth.gaussian(fwhm=4.0)

    np.testing.assert_allclose(unit_based.data, gaussian_smooth(source.data, fwhm=4.0, pixel_scale=(2.0, 2.0)))
    assert not np.allclose(unit_based.data, gaussian_smooth(source.data, fwhm=4.0))


# ---------------------------------------------------------------------------
# PSF family
# ---------------------------------------------------------------------------


def test_psf_convolve_matches_the_free_function() -> None:
    source = image()

    via_method = source.postprocess.psf.convolve(fwhm=2.0)

    np.testing.assert_allclose(via_method.data, convolve_psf(source.data, fwhm=2.0, pixel_scale=source.pixel_size))
    assert [op.name for op in via_method.ops] == ["convolve_psf"]


def test_psf_deconvolution_methods_return_images() -> None:
    kernel = gaussian_psf(fwhm=2.0, size=5)
    blurred = image().postprocess.psf.convolve(kernel)

    for restored in (
        blurred.postprocess.psf.deconvolve(kernel),
        blurred.postprocess.psf.wiener(kernel),
        blurred.postprocess.psf.richardson_lucy(kernel, iterations=3),
    ):
        assert restored.shape == blurred.shape
        assert restored.extent == blurred.extent
        assert restored.ops[-1].name in {"deconvolve_psf", "wiener_deconvolve", "richardson_lucy"}


# ---------------------------------------------------------------------------
# stretching, colouring, compositing
# ---------------------------------------------------------------------------


def test_normalize_returns_an_image_but_to_rgba_returns_colours() -> None:
    source = image()

    stretched = source.display.normalize(stretch="log")
    coloured = source.display.to_rgba(cmap="viridis")

    assert isinstance(stretched, ImageData)
    np.testing.assert_allclose(stretched.data, normalize(source.data, stretch="log"))
    assert stretched.ops[-1].name == "normalize"
    assert isinstance(coloured, np.ndarray)
    assert coloured.shape == (*source.shape, 4)


# ---------------------------------------------------------------------------
# adaptive binning
# ---------------------------------------------------------------------------


def test_adaptive_bin_returns_a_map_carrying_this_image_geometry() -> None:
    pytest.importorskip("powerbin")
    value = image((30, 30), extent=(0.0, 30.0, 0.0, 30.0), x_units="kpc", y_units="kpc", label="vz.mean", units="km/s")
    signal = value.with_data(np.linspace(1.0, 10.0, 900).reshape(30, 30), label="mass.sum")

    binned = value.postprocess.adaptive.bin(signal, target_nbins=6)

    assert isinstance(binned, AdaptiveMap)
    assert isinstance(binned.image, ImageData)
    assert binned.label == "vz.mean"
    assert binned.units == "km/s"
    assert (binned.x_units, binned.y_units) == ("kpc", "kpc")
    assert binned.image.extent == value.extent
    np.testing.assert_allclose(binned.image.x_edges, value.x_edges)
    assert binned.n_bins > 1
    assert binned.mask.any()
    # The painted map is an ordinary image, so it keeps the machinery.
    assert binned.to_image_data() is binned.image
    assert binned.image.postprocess.smooth.box(size=3).shape == binned.image.shape


def test_adaptive_bin_accepts_plain_arrays_and_statistics() -> None:
    pytest.importorskip("powerbin")
    value = image((20, 20))
    signal = np.ones((20, 20))

    binned = value.postprocess.adaptive.bin(signal, target_capacity=200.0, method="median")

    assert binned.method == "median"
    assert binned.target_capacity == 200.0


# ---------------------------------------------------------------------------
# provenance and immutability
# ---------------------------------------------------------------------------


def test_chaining_records_every_step_in_order() -> None:
    chained = (
        image()
        .postprocess.smooth.gaussian(fwhm=2.0)
        .postprocess.psf.convolve(fwhm=1.0)
        .display.normalize(stretch="sqrt")
    )

    assert [op.name for op in chained.ops] == ["gaussian_smooth", "convolve_psf", "normalize"]
    assert "gaussian_smooth" in repr(chained)
    assert repr(chained).startswith("ImageData(")


def test_methods_do_not_touch_the_original() -> None:
    source = image()
    before = source.data.copy()

    source.postprocess.smooth.gaussian(fwhm=2.0)
    source.postprocess.smooth.downsample(factor=2)
    source.display.normalize()

    np.testing.assert_array_equal(source.data, before)
    assert source.ops == ()


def test_with_data_keeps_the_provenance_but_plain_construction_does_not() -> None:
    smoothed = image().postprocess.smooth.gaussian(fwhm=2.0)

    replaced = smoothed.with_data(np.zeros(smoothed.shape))
    fresh = ImageData(np.zeros(smoothed.shape))

    assert [op.name for op in replaced.ops] == ["gaussian_smooth"]
    assert fresh.ops == ()


def test_image_op_renders_compact_parameters() -> None:
    op = ImageOp("gaussian_smooth", {"fwhm": 2.0, "mask": np.ones((3, 3), dtype=bool)})

    assert repr(op) == "gaussian_smooth(fwhm=2.0, mask=<array (3, 3)>)"


def test_compose_and_masks_are_reachable_from_the_image() -> None:
    first = image((4, 4))
    second = first.with_data(np.ones((4, 4)))

    masks = first.postprocess.compose.masks(line_angle=0.0, width=0.0)
    composed = first.postprocess.compose(second, style=MapStyle(cmap="gray"), other_style=MapStyle(cmap="viridis"))

    assert masks[0].shape == first.shape
    assert composed.shape == (*first.shape, 4)


def test_draw_is_the_one_call_that_always_works() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage

    even = image((4, 4))
    uneven = ImageData(np.zeros((4, 4)), x_edges=[0.0, 1.0, 3.0, 6.0, 10.0], y_edges=[0.0, 1.0, 2.0, 3.0, 4.0])

    fig, ax = plt.subplots()
    try:
        assert isinstance(even.display.draw(ax=ax), AxesImage)
        assert isinstance(uneven.display.draw(ax=ax), QuadMesh)
    finally:
        plt.close(fig)
