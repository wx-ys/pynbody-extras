"""Bins unit-preservation tests for BinNDResult.

Two regressions where a :class:`BinsArray` result silently lost its physical
units are pinned here:

1. A **sub-snapshot** axis field (``halo["rxy"]``) is an ``IndexedSimArray`` —
   a distinct class that is *not* a ``SimArray`` subclass but carries ``.units``.
   The materializer used to only recognise ``SimArray``, so the bin edges were
   built from a plain ndarray and the derived ``measure`` / ``mass.density``
   became dimensionless.
2. A genuinely **dimensionless** ``measure`` must not strip a unit-ed
   numerator's units during the density division (pynbody's
   ``SimArray / NoUnit`` degrades the quotient to ``NoUnit``).
"""

from __future__ import annotations

import numpy as np
import pynbody
from pynbody.array import SimArray

from pynbodyext.core.calculate import Bin1D, BinsArray


def make_unit_galaxy() -> pynbody.SimSnap:
    """A small galaxy snapshot whose ``rxy`` and ``mass`` carry real units."""
    sim = pynbody.new(dm=6)
    sim["x"] = SimArray(np.linspace(-5, 5, 6), units="kpc")
    sim["y"] = SimArray(np.linspace(-3, 3, 6), units="kpc")
    sim["rxy"] = np.sqrt(sim["x"] ** 2 + sim["y"] ** 2)
    sim["mass"] = SimArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), units="Msol")
    return sim


def test_subresult_axis_measure_and_density_preserve_units():
    """A sub-snapshot axis field must not drop its physical units.

    ``central_galaxy["rxy"]`` on a halo sub-snap is an :class:`IndexedSimArray`
    carrying kpc units.  The bin edges must inherit those units so the derived
    ``measure`` carries kpc² and ``mass.density`` carries ``Msol / length²``.
    Previously ``_coerce_edges_like`` only recognised ``SimArray`` (not
    ``IndexedSimArray``), so the edges silently became a plain ndarray and the
    measure/density lost their units.
    """
    sim = make_unit_galaxy()
    sub = sim[pynbody.filt.LowPass("x", 5.0)]

    bins = Bin1D(prop="rxy", vmin="0 kpc", vmax="30 kpc", nbins=3)(sub)

    measure = bins["measure"]
    assert isinstance(measure, BinsArray)
    assert measure.units is not None, "measure lost its physical units on a sub-snapshot"
    assert "kpc" in str(measure.units), f"expected length units, got {measure.units!r}"

    density = bins["mass.density"]
    assert density.units is not None, "density lost its units on a sub-snapshot"
    assert "Msol" in str(density.units), f"expected mass/length units, got {density.units!r}"


def test_density_preserves_numerator_units_when_measure_is_dimensionless():
    """A dimensionless (NoUnit) measure must not strip the numerator's units.

    When a bin axis carries no physical units (``r`` is a plain array), the
    derived ``measure`` is dimensionless.  A unit-ed mass field binned on such
    an axis should still yield ``mass.sum / 1`` — i.e. ``Msol`` — not
    ``NoUnit()``.  pynbody's ``SimArray / NoUnit`` degrades the quotient to
    ``NoUnit``, so the density division must drop the denominator's wrapper and
    divide by a bare array to keep the mass units.
    """
    sim = pynbody.new(dm=6)
    sim["r"] = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)  # dimensionless axis
    sim["mass"] = SimArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), units="Msol")

    bins = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)

    density = bins["mass.density"]
    assert density.units is not None, "density lost its units on a dimensionless axis"
    assert "Msol" in str(density.units), f"expected mass units, got {density.units!r}"
