"""Behavioural tests for the ``WrapBox`` transform.

``WrapBox`` was only ever covered through display tests, so the wrapping itself —
the conventions, the offset dtype, the inverse — had no regression net.  Two real
bugs were living in that gap: an explicit ``boxsize`` argument was ignored, and the
family-array hook wrapped with an ``int8`` offset without promoting it, silently
producing wrong coordinates for positions far outside the box.
"""

from __future__ import annotations

import warnings

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray

from pynbodyext.transforms import WrapBox
from pynbodyext.transforms.wrap import WrapTransformation, normalize_convention


def make_sim(*, n: int = 8, box: float | None = 10.0, span: float = 3.0, units: str | None = None, seed: int = 0):
    """A small snapshot with random positions, optionally with a boxsize property."""
    sim = pynbody.new(n)
    pos = np.random.default_rng(seed).uniform(-span, span, size=(n, 3))
    sim["pos"] = SimArray(pos, units) if units else pos
    sim["mass"] = np.ones(n)
    if box is not None:
        sim.properties["boxsize"] = box
    return sim


def positions(sim) -> np.ndarray:
    return np.asarray(sim["pos"], dtype=float)


def wrap(sim, **kwargs) -> WrapTransformation:
    """Run ``WrapBox`` on *sim* and return the transformation handle it built."""
    handle = WrapBox(**kwargs).run(sim).value
    assert isinstance(handle, WrapTransformation)
    return handle


@pytest.mark.parametrize(("convention", "low", "high"), [("center", -5.0, 5.0), ("upper", 0.0, 10.0)])
def test_convention_wraps_into_its_documented_range(convention: str, low: float, high: float) -> None:
    sim = make_sim(box=10.0, span=30.0)
    wrap(sim, convention=convention)
    pos = positions(sim)
    assert pos.min() >= low - 1e-9
    assert pos.max() < high


def test_center_matches_the_modular_formula() -> None:
    """An independent check of the wrapping math, far outside the box."""
    sim = make_sim(box=10.0, span=400.0)
    original = positions(sim)
    wrap(sim, convention="center")  # offsets fit in int8, so no promotion
    assert np.allclose(positions(sim), ((original + 5.0) % 10.0) - 5.0, atol=1e-9)


def test_minirange_picks_the_narrower_convention_per_axis() -> None:
    """Axis 0 is split across the seam (center wins), axis 1 is not (upper wins)."""
    sim = pynbody.new(2)
    sim["pos"] = np.array([[0.5, 4.5, 0.0], [9.5, 5.5, 0.0]])
    sim["mass"] = np.ones(2)
    sim.properties["boxsize"] = 10.0

    wrap(sim, convention="minirange")

    assert positions(sim).tolist() == [[0.5, 4.5, 0.0], [-0.5, 5.5, 0.0]]


def test_explicit_boxsize_is_used_without_a_snapshot_property() -> None:
    """Regression: the explicit ``boxsize`` was ignored, so nothing was wrapped."""
    sim = make_sim(box=None, span=30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a "skipping wrap" warning would fail here
        wrap(sim, boxsize=10.0, convention="center")
    pos = positions(sim)
    assert pos.min() >= -5.0 - 1e-9
    assert pos.max() < 5.0


def test_explicit_boxsize_wins_over_the_snapshot_property() -> None:
    sim = make_sim(box=1000.0, span=30.0)
    wrap(sim, boxsize=10.0, convention="center")
    pos = positions(sim)
    assert pos.min() >= -5.0 - 1e-9
    assert pos.max() < 5.0


def test_revert_restores_the_positions() -> None:
    sim = make_sim(box=10.0, span=30.0)
    original = positions(sim).copy()
    handle = wrap(sim, convention="minirange")
    assert not np.allclose(positions(sim), original)
    handle.revert()
    assert np.allclose(positions(sim), original)


def test_revert_survives_a_removed_boxsize_property() -> None:
    """Regression: the inverse used to crash with a ``TypeError``."""
    sim = make_sim(box=10.0, span=30.0)
    original = positions(sim).copy()
    handle = wrap(sim, convention="center")
    del sim.properties["boxsize"]
    handle.revert()
    assert np.allclose(positions(sim), original)


def test_array_path_promotes_the_offset_dtype_like_the_snapshot_path() -> None:
    """Regression: the family-array hook cast to int8 and overflowed silently."""
    sim = make_sim(box=10.0, span=1.0)
    handle = wrap(sim, convention="center")

    sim["pos"] += 5000.0
    original = positions(sim)
    with pytest.warns(UserWarning, match="auto-promote k dtype"):
        handle.apply_transformation_to_array("pos", None)

    assert np.allclose(positions(sim), ((original + 5.0) % 10.0) - 5.0, atol=1e-9)


def test_array_path_ignores_arrays_other_than_pos() -> None:
    sim = make_sim(box=10.0, span=30.0)
    handle = wrap(sim, convention="center")
    rho = np.array(sim["mass"])
    handle.apply_transformation_to_array("mass", None)
    assert np.array_equal(np.array(sim["mass"]), rho)


def test_missing_boxsize_skips_with_a_warning() -> None:
    sim = make_sim(box=None)
    original = positions(sim).copy()
    with pytest.warns(UserWarning, match="no boxsize"):
        WrapBox().run(sim)
    assert np.allclose(positions(sim), original)


def test_non_positive_boxsize_skips_with_a_warning() -> None:
    sim = make_sim(box=0.0)
    original = positions(sim).copy()
    with pytest.warns(UserWarning, match="must be positive"):
        WrapBox().run(sim)
    assert np.allclose(positions(sim), original)


def test_unit_boxsize_is_converted_to_position_units() -> None:
    sim = make_sim(box=None, span=2000.0, units="kpc")
    wrap(sim, boxsize=pynbody.units.Unit("1 Mpc"), convention="center")
    pos = positions(sim)
    assert pos.min() >= -500.0 - 1e-6
    assert pos.max() < 500.0


def test_empty_snapshot_is_a_no_op() -> None:
    sim = make_sim(n=0, box=10.0)
    handle = wrap(sim, convention="minirange")
    assert positions(sim).shape == (0, 3)
    assert handle._k_offsets is not None and handle._k_offsets.shape == (0, 3)


def test_default_convention_is_minirange() -> None:
    assert WrapBox().convention == "minirange"
    assert normalize_convention("CENTER") == "center"


def test_invalid_convention_is_rejected_when_the_node_is_built() -> None:
    """Fail at construction rather than after a run has started."""
    with pytest.raises(ValueError, match="Unknown wrapping convention"):
        WrapBox(convention="middle")
    with pytest.raises(ValueError, match="Unknown wrapping convention"):
        WrapTransformation(make_sim(), boxsize=10.0, convention="middle")
