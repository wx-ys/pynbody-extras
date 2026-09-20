"""The capability views: ``ImageOps``, the registry, and adding a new family."""

from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.plot.image import OPERATIONS, ImageData, ImageOps, register_ops
from pynbodyext.plot.image._arrays import value_limits
from pynbodyext.plot.image.adaptive import AdaptiveMap
from pynbodyext.plot.image.ops import ImageDataView

#: Everything a view re-exposes from the image it is a view of.
FORWARDED = (
    "data",
    "shape",
    "ndim",
    "extent",
    "x_edges",
    "y_edges",
    "x_centers",
    "y_centers",
    "x_uniform",
    "y_uniform",
    "uniform",
    "pixel_size",
    "x_units",
    "y_units",
    "x_label",
    "y_label",
    "label",
    "units",
)


def image(shape: tuple[int, int] = (6, 6), **kwargs: object) -> ImageData:
    data = np.arange(float(shape[0] * shape[1])).reshape(shape)
    defaults: dict[str, object] = {
        "extent": (0.0, float(shape[1]), 0.0, float(shape[0])),
        "x_units": "kpc",
        "y_units": "kpc",
        "label": "vz.mean",
        "units": "km/s",
    }
    defaults.update(kwargs)
    return ImageData(data, **defaults)


# ---------------------------------------------------------------------------
# one parent, several views
# ---------------------------------------------------------------------------


def test_one_base_forwards_the_image_to_every_view() -> None:
    """Views (and the binned map) share ``ImageDataView`` instead of copying it."""
    assert issubclass(ImageOps, ImageDataView)
    assert issubclass(AdaptiveMap, ImageDataView)


@pytest.mark.parametrize("name", FORWARDED)
def test_forwarding_is_declared_once(name: str) -> None:
    assert isinstance(getattr(ImageDataView, name), property), f"{name} should live on ImageDataView"
    assert name not in ImageOps.__dict__, f"ImageOps must not re-declare {name}"
    assert name not in AdaptiveMap.__dict__, f"AdaptiveMap must not re-declare {name}"


def test_views_forward_the_image_itself() -> None:
    source = image()

    for view in (source.smooth, source.psf, source.compose, source.adaptive):
        assert view.image is source
        assert view.data is source.data
        assert view.shape == source.shape
        assert view.extent == source.extent
        assert (view.x_units, view.y_units) == ("kpc", "kpc")
        assert (view.x_label, view.y_label) == (source.x_label, source.y_label)
        assert (view.label, view.units) == ("vz.mean", "km/s")
        assert view.uniform is source.uniform
        assert view.pixel_size == source.pixel_size
        np.testing.assert_array_equal(view.x_centers, source.x_centers)


def test_views_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    source = image()

    with pytest.raises(FrozenInstanceError):
        source.smooth.image = image()  # type: ignore[misc]


def test_image_data_does_not_inherit_its_capabilities() -> None:
    """Capabilities are composed as views, so a new family needs no new base."""
    assert ImageData.__bases__ == (object,)


def test_every_capability_is_a_view_sharing_the_parent() -> None:
    source = image()

    views = (source.smooth, source.psf, source.compose, source.adaptive)

    assert all(isinstance(view, ImageOps) for view in views)
    for view in views:
        assert view.image is source
        assert view.data is source.data
        assert view.shape == source.shape
        assert view.extent == source.extent
        assert (view.x_units, view.y_units) == ("kpc", "kpc")
        assert view.label == "vz.mean"


def test_views_expose_the_helpers_a_plugin_needs() -> None:
    source = image((4, 4), extent=(0.0, 8.0, 0.0, 8.0))
    view = source.smooth

    assert view.limits() == value_limits(source.data)
    assert view.kernel_scale() == (2.0, 2.0)
    uneven = ImageData(np.zeros((4, 4)), x_edges=[0.0, 1.0, 3.0, 6.0, 10.0], y_edges=[0.0, 1.0, 2.0, 3.0, 4.0])
    assert uneven.smooth.kernel_scale() is None


def test_a_view_can_add_the_colour_bar_of_its_image() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = image(label="vz.mean")
    fig, ax = plt.subplots()
    try:
        artist = source.display.imshow(ax=ax)

        bar = source.display.add_colorbar(ax=ax)

        assert bar.mappable is artist
        assert bar.ax.get_ylabel() == "vz.mean [km/s]"
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# a binned map is a view too
# ---------------------------------------------------------------------------


def test_adaptive_map_is_a_view_of_its_painted_image() -> None:
    pytest.importorskip("powerbin")
    source = image((20, 20))
    signal = ImageData(np.ones((20, 20)), extent=(0.0, 20.0, 0.0, 20.0), label="mass.sum")

    binned = source.adaptive.bin(signal, target_nbins=4)

    assert isinstance(binned, ImageDataView)
    assert not isinstance(binned, ImageOps)  # it is a result, not a capability
    assert binned.value is binned.image.data
    assert binned.data is binned.image.data
    assert (binned.shape, binned.extent) == (binned.image.shape, binned.image.extent)
    assert (binned.x_units, binned.label) == (binned.image.x_units, binned.image.label)
    np.testing.assert_array_equal(binned.x_edges, binned.image.x_edges)


def test_derive_returns_a_new_image_and_records_the_operation() -> None:
    source = image()
    view = source.psf

    derived = view.derive(source.data * 2.0, "double", {"factor": 2.0})

    assert derived is not source
    np.testing.assert_allclose(derived.data, source.data * 2.0)
    assert derived.ops[-1].name == "double"
    assert derived.ops[-1].params == {"factor": 2.0}
    assert source.ops == ()


# ---------------------------------------------------------------------------
# adding a family
# ---------------------------------------------------------------------------


class ToyOps(ImageOps):
    """A stand-in for a future family (tessellation, spectra, …)."""

    def double(self) -> ImageData:
        return self.derive(self.data * 2.0, "double", {})


@pytest.fixture
def toy_family() -> object:
    """Register the toy family for one test and clean up afterwards."""
    OPERATIONS.pop("toy", None)
    yield ToyOps
    OPERATIONS.pop("toy", None)


def test_a_new_family_is_reachable_without_touching_image_data(toy_family: object) -> None:
    ImageData.register_ops("toy", ToyOps)
    source = image()

    doubled = source.toy.double()

    assert isinstance(source.toy, ImageOps)
    np.testing.assert_allclose(doubled.data, source.data * 2.0)
    assert "toy" in ImageData.operations()
    assert ImageData.operations()["toy"] is ToyOps


def test_register_ops_works_as_a_decorator(toy_family: object) -> None:
    @ImageData.register_ops("toy")
    class Registered(ImageOps):
        pass

    assert ImageData.operations()["toy"] is Registered
    assert isinstance(image().toy, Registered)


def test_register_ops_rejects_a_conflicting_name() -> None:
    original = OPERATIONS["smooth"]
    try:
        with pytest.raises(KeyError, match="already registered"):
            ImageData.register_ops("smooth", ToyOps)

        ImageData.register_ops("smooth", ToyOps, overwrite=True)

        assert OPERATIONS["smooth"] is ToyOps
    finally:
        OPERATIONS["smooth"] = original


def test_register_ops_rejects_something_that_is_not_a_view() -> None:
    with pytest.raises(TypeError, match="ImageOps"):
        register_ops("toy", dict)  # type: ignore[arg-type]


def test_builtin_families_are_registered_for_discovery() -> None:
    assert {"smooth", "psf", "compose", "adaptive"} <= set(OPERATIONS)


def test_unknown_attributes_still_raise(toy_family: object) -> None:
    with pytest.raises(AttributeError, match="no attribute"):
        image().nonexistent
