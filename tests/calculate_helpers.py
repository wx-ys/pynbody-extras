"""Shared calculator test builders.

Tracked helper module (not collected by pytest) so test files that need the
standard small snapshot / pipeline no longer depend on a locally-untracked
module.  Kept import-light: only numpy and pynbodyext.
"""

from __future__ import annotations

import numpy as np
import pynbody

from pynbodyext.core.calculate import FilterBase, Param, Pipeline, PropertyBase


def make_sim() -> pynbody.SimSnap:
    """Return a 6-particle snapshot with ``mass``, ``temp`` and ``r`` fields."""
    sim = pynbody.new(6)
    sim["mass"] = np.arange(6.0)
    sim["temp"] = np.arange(6.0) * 10.0
    sim["r"] = np.arange(6.0)
    return sim


@PropertyBase.dataclass
class MassSum(PropertyBase[float]):
    """Sum of the ``mass`` field."""

    def calculate(self, sim, params=None) -> float:
        return float(sim["mass"].sum())


@FilterBase.dataclass
class RBelow(FilterBase):
    """A boolean mask selecting particles with ``r`` below ``radius``."""

    radius: Param[float] = Param(field_name="r")

    def calculate(self, sim, params=None) -> bool:
        return sim["r"] < self.radius


@PropertyBase.dataclass
class TempMean(PropertyBase[float]):
    """Mean of the ``temp`` field."""

    def calculate(self, sim, params=None) -> float:
        return float(sim["temp"].mean())


def make_pipeline(root_name: str = "p") -> Pipeline:
    """Return a two-output pipeline (``m`` = mass sum, ``t`` = temp mean)."""
    shared = RBelow(5.0)
    return Pipeline(
        {"m": MassSum().filter(shared), "t": TempMean().filter(shared)},
        name=root_name,
    )
