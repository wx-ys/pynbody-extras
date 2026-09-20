"""The capability views: ``ImageOps``, the registry, and adding a new family."""

from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.plot.image import OPERATIONS, ImageData, ImageOps, register_ops
from pynbodyext.plot.image._arrays import value_limits


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
