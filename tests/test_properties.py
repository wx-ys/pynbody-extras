"""Behaviour of the built-in properties in ``pynbodyext.properties``.

They sit at the top of a pipeline, so their awkward inputs deserve a stated
answer rather than whatever NumPy happens to raise from inside them.
"""

from __future__ import annotations

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray

from pynbodyext.filters import Sphere
from pynbodyext.properties import ParamContain, ParamSum
from pynbodyext.properties.base import RadiusAtSurfaceDensity


def make_galaxy(n: int = 200) -> pynbody.SimSnap:
    """A small spherically symmetric snapshot with ``pos``, ``mass`` and ``r``."""
    rng = np.random.default_rng(0)
    radius = rng.uniform(0.0, 30.0, n)
    cos, phi = rng.uniform(-1.0, 1.0, n), rng.uniform(0.0, 2.0 * np.pi, n)
    sin = np.sqrt(1.0 - cos**2)
    sim = pynbody.new(dm=n)
    sim["pos"] = SimArray(np.c_[radius * sin * np.cos(phi), radius * sin * np.sin(phi), radius * cos], "kpc")
    sim["mass"] = SimArray(rng.uniform(0.5, 2.0, n), "Msol")
    sim["r"] = SimArray(radius, "kpc")
    return sim


def empty_selection(sim: pynbody.SimSnap) -> object:
    """How an empty particle set is reached in practice: a filter that matches none."""
    return sim[Sphere("0.0001 kpc")]


def test_param_contain_says_which_calculator_got_an_empty_snapshot() -> None:
    """The lifecycle names the input; what follows is the calculation's own failure.

    ``cumulative[-1]`` on an empty array is NumPy's report, and it is left to say
    so — the warning above it is what makes the report readable.
    """
    with pytest.warns(UserWarning, match="ParamContain received a snapshot with no particles"):
        with pytest.raises(IndexError):
            ParamContain(0.5)(empty_selection(make_galaxy()))


def test_the_lifecycle_warns_when_a_calculator_gets_no_particles() -> None:
    """Every node gets the same look at its input, without writing anything.

    ``ParamSum`` answers an empty set (zero) rather than failing, so for it the
    warning is the whole diagnosis — the answer would otherwise look like a value.
    """
    with pytest.warns(UserWarning, match="ParamSum received a snapshot with no particles"):
        total = ParamSum("mass")(empty_selection(make_galaxy()))

    assert float(total) == 0.0


def test_a_snapshot_with_particles_is_quiet() -> None:
    import warnings as warnings_module

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error")
        ParamContain(0.5)(make_galaxy())
        ParamSum("mass")(make_galaxy())


def test_param_contain_distinguishes_no_particles_from_no_weight() -> None:
    """Two different reasons there is no radius to report, said differently."""
    sim = make_galaxy()
    sim["mass"] = SimArray(np.zeros(len(sim)), "Msol")

    with pytest.raises(ValueError, match="Non-positive total"):
        ParamContain(0.5)(sim)


def test_radius_at_surface_density_says_which_calculator_got_an_empty_snapshot() -> None:
    with pytest.warns(UserWarning, match="RadiusAtSurfaceDensity received a snapshot with no particles"):
        with pytest.raises(IndexError):
            RadiusAtSurfaceDensity(1.0)(empty_selection(make_galaxy()))


def test_the_properties_still_answer_a_populated_snapshot() -> None:
    """The guard must not change what a snapshot with particles gets."""
    half_mass = ParamContain(0.5)(make_galaxy())

    assert 0.0 < float(half_mass) < 30.0
