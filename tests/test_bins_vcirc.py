"""Tests for the ``BinNDResult["vcirc"]`` derived circular-velocity profile.

``vcirc`` is a radial-only derived property: available on 1-D bins whose axis is
``r`` or ``rxy``.  It evaluates the gravitational acceleration at points on the
mid-plane circle of each bin radius and returns ``sqrt(|a| * R)`` in ``km s**-1``.
"""

from __future__ import annotations

import numpy as np
import pynbody
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.core.calculate import Bin1D, BinsArray


def point_mass_sim() -> pynbody.SimSnap:
    """A single central point mass (clean analytic |a| = G M / R**2)."""
    sim = pynbody.new(dm=1)
    sim["pos"] = SimArray(np.array([[0.0, 0.0, 0.0]]), units="kpc")
    sim["mass"] = SimArray(np.array([1e10]), units="Msol")
    sim["smooth"] = SimArray(np.array([0.0]), units="kpc")
    return sim


def test_vcirc_available_for_radial_axes_only():
    """``vcirc`` appears only on 1-D r/rxy profiles, not other axes."""
    sim = point_mass_sim()
    bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=2)(sim)

    assert "vcirc" in bins.queries.names(), "vcirc should be available on an rxy profile"
    assert isinstance(bins["vcirc"], BinsArray)

    sim2 = pynbody.new(dm=6)
    sim2["x"] = SimArray(np.linspace(-5, 5, 6), units="kpc")
    sim2["mass"] = SimArray(np.arange(6.0), units="Msol")
    sim2["smooth"] = SimArray(np.zeros(6), units="kpc")
    x_bins = Bin1D("x", vmin="-5 kpc", vmax="5 kpc", nbins=3)(sim2)

    assert "vcirc" not in x_bins.queries.names(), "vcirc should not be available on an x profile"


def test_vcirc_matches_analytic_point_mass():
    """For a central point mass, vcirc(R) = sqrt(G M / R) (km s**-1)."""
    sim = point_mass_sim()
    bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=2)(sim)
    vcirc = bins["vcirc"]

    assert vcirc.units is not None, "vcirc must carry units"
    assert "km" in str(vcirc.units), f"expected km units, got {vcirc.units!r}"

    G = float(units.G.in_units("kpc Msol**-1 km**2 s**-2"))
    R = np.asarray(bins.centers)  # kpc
    M = 1e10  # Msol
    expected = np.sqrt(G * M / R)  # km s**-1
    np.testing.assert_allclose(np.asarray(vcirc), expected, rtol=1e-6)


def test_vcirc_is_finite_and_nonnegative():
    sim = point_mass_sim()
    bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=4)(sim)
    vcirc = np.asarray(bins["vcirc"])
    assert np.all(np.isfinite(vcirc))
    assert np.all(vcirc >= 0.0)
