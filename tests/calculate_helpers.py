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

    def calculate(self, sim, params=None) -> np.typing.NDArray[np.bool_]:
        return sim["r"] < self.radius


@PropertyBase.dataclass
class TempMean(PropertyBase[float]):
    """Mean of the ``temp`` field."""

    def calculate(self, sim, params=None) -> float:
        return float(sim["temp"].mean())


class OpaqueProperty(PropertyBase[float]):
    """A hand-written node with no declared identity (the ``generic`` signature kind).

    Neither a dataclass nor a ``signature_payload`` override, so its signature can
    only tell it apart from other nodes by the object itself: not constructible,
    and not stable across sessions.
    """

    def calculate(self, sim, params=None) -> float:
        return 0.0


def make_pipeline(root_name: str = "p") -> Pipeline:
    """Return a two-output pipeline (``m`` = mass sum, ``t`` = temp mean)."""
    shared = RBelow(5.0)
    return Pipeline(
        {"m": MassSum().filter(shared), "t": TempMean().filter(shared)},
        name=root_name,
    )


def kind_samples() -> list[object]:
    """One calculator per node kind the signature encoder can emit.

    Shared by the node-kind dispatch test and the signature round-trip test, so a
    new kind is added here once and both learn about it.
    """
    from pynbodyext.core.calculate import CombinedCalculator, LambdaProperty, TransformChain
    from pynbodyext.core.calculate.nodes.expr import as_property
    from pynbodyext.filters import FamilyFilter, Sphere
    from pynbodyext.transforms import ShiftPosTo, WrapBox

    return [
        MassSum(),  # dataclass
        MassSum() * 2,  # op_property
        MassSum() ** 0,  # constant_property
        ~(Sphere("30 kpc") & FamilyFilter("star")),  # filter_op
        MassSum().filter(Sphere("30 kpc")),  # scoped
        LambdaProperty(lambda sim: 1.0),  # lambda_property
        as_property(WrapBox()),  # calculator_value_property
        TransformChain(WrapBox(), ShiftPosTo("ssc")),  # transform_chain
        CombinedCalculator(MassSum(), MassSum()),  # combined
        make_pipeline(),  # pipeline
        OpaqueProperty(),  # generic
    ]
