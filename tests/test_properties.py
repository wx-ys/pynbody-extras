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
from pynbodyext.properties import ParamContain
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


def test_param_contain_says_the_snapshot_it_got_has_no_particles() -> None:
    """An empty selection is a state to name, not an arithmetic accident."""
    with pytest.raises(ValueError, match="has none") as caught:
        ParamContain(0.5)(empty_selection(make_galaxy()))

    assert "filter matched nothing" in str(caught.value), "the usual cause should be spelled out"


def test_param_contain_distinguishes_no_particles_from_no_weight() -> None:
    """Two different reasons there is no radius to report, said differently."""
    sim = make_galaxy()
    sim["mass"] = SimArray(np.zeros(len(sim)), "Msol")

    with pytest.raises(ValueError, match="Non-positive total"):
        ParamContain(0.5)(sim)


def test_radius_at_surface_density_says_the_snapshot_it_got_has_no_particles() -> None:
    with pytest.raises(ValueError, match="has none"):
        RadiusAtSurfaceDensity(1.0)(empty_selection(make_galaxy()))


def test_the_properties_still_answer_a_populated_snapshot() -> None:
    """The guard must not change what a snapshot with particles gets."""
    half_mass = ParamContain(0.5)(make_galaxy())

    assert 0.0 < float(half_mass) < 30.0
