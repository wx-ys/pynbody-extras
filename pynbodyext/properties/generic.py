"""Generic property calculators backed by the new calculator framework.

This module provides small reusable property nodes for common halo and galaxy
measurements, such as centers, angular-momentum vectors, virial radii, spin
parameters, and pattern speeds.

All local classes inherit from the new :class:`pynbodyext.calculate.PropertyBase`.
The :class:`KappaRot` property is re-exported from :mod:`pynbodyext.core.calculate`
so this module stays aligned with the new calculator implementation.
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pynbody.analysis.halo import hybrid_center, shrink_sphere_center, virial_radius
from pynbody.array import SimArray

from pynbodyext.calculate import PropertyBase

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap


__all__ = [
    "CenPos",
    "CenVel",
    "AngMomVec",
    "KappaRot",
    "KappaRotMean",
    "VirialRadius",
    "SpinParam",
    "PatternSpeed",
]

@PropertyBase.dataclass
class CenPos(PropertyBase[SimArray | np.ndarray]):
    """Center position property"""

    mode : Literal["ssc", "com", "pot", "hyb"] = "ssc"

    def __post_init__(self) -> None:
        if self.mode not in ("ssc", "com", "pot", "hyb"):
            raise ValueError(f"Invalid mode: {self.mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")

    def calculate(self, sim: SimSnap, params: Any = None) -> SimArray | np.ndarray:
        if params.mode == "com":
            cen = cast("SimArray", sim.mean_by_mass("pos"))
        elif params.mode == "pot":
            i = sim["phi"].argmin()
            cen = cast("SimArray", sim["pos"][i].copy())
        elif params.mode == "ssc":
            cen = cast("SimArray", shrink_sphere_center(sim))
        elif params.mode == "hyb":
            cen = cast("SimArray", hybrid_center(sim, r = "5 kpc"))
        else:
            raise ValueError(f"Invalid mode: {params.mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")
        if isinstance(cen, SimArray):
            cen.sim = sim
        return cen

@PropertyBase.dataclass
class CenVel(PropertyBase[SimArray]):
    """Center velocity property"""

    mode : Literal["com"] = "com"

    def __post_init__(self) -> None:
        if self.mode != "com":
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'com'.")

    def calculate(self, sim: SimSnap, params: Any = None) -> SimArray:
        if params.mode == "com":
            cen = cast("SimArray", sim.mean_by_mass("vel"))
        else:
            raise ValueError(f"Invalid mode: {params.mode}. Expected one of ['com'].")
        if isinstance(cen, SimArray):
            cen.sim = sim
        return cen


@PropertyBase.dataclass
class AngMomVec(PropertyBase[SimArray]):
    """Angular momentum vector property"""

    def calculate(self, sim: SimSnap, params: Any = None) -> SimArray:
        mass = sim["mass"].reshape((len(sim["mass"]), 1))
        pos = sim["pos"]
        vel = sim["vel"]

        cross = np.cross(pos, vel)
        angmom = (mass * cross).sum(axis=0)

        angmom.units = sim["mass"].units * sim["pos"].units * sim["vel"].units
        return angmom

@PropertyBase.dataclass
class KappaRot(PropertyBase[float]):
    """
    Calculate the fraction of kinetic energy in ordered rotation.

    Notes
    -----
    The kappa_rot parameter is defined as in eq. (1) of Sales et al. (2010) [1]_.

    References
    ----------
    .. [1] Sales, L. V., et al. 2010, MNRAS, 409, 1541
    """

    def calculate(self, sim: SimSnap, params: Any = None) -> float:
        Krot = np.sum(0.5 * sim["mass"] * (sim["vcxy"] ** 2))
        K = np.sum(sim["mass"] * sim["ke"])
        return float(Krot / K)

@PropertyBase.dataclass
class KappaRotMean(PropertyBase[float]):
    """
    Calculate the mean of the ratio of rotational kinetic energy to total kinetic energy per particle.

    Notes
    -----
    This is the mean of (0.5 * m * vcxy^2) / (m * ke) over given particles.
    """
    def calculate(self, sim: SimSnap, params: Any = None) -> float:
        krot = 0.5 * sim["vcxy"] ** 2
        ke = sim["ke"]
        ratio = krot / ke
        return float(np.mean(ratio))

@PropertyBase.dataclass
class VirialRadius(PropertyBase[float]):
    """Virial radius property"""
    overdensity: float = 178.
    rho_def: Literal["critical", "matter"] = "critical"

    def __post_init__(self) -> None:
        if self.rho_def not in ("critical", "matter"):
            raise ValueError(f"Invalid rho_def: {self.rho_def}. Expected one of ['critical', 'matter'].")

    def calculate(self, sim: SimSnap, params: Any = None) -> float:
        return virial_radius(sim, overden=params.overdensity, rho_def=params.rho_def)

@PropertyBase.dataclass
class SpinParam(PropertyBase[float]):
    """The spin parameter is defined as in eq. (5) of Bullock et al. (2001) [1]_.

    Notes
    -----
    This calculator assumes the halo has been centered on the coordinate
    origin and its bulk velocity is zero.

    References
    ----------
    .. [1] Bullock, J. S., et al. 2001, MNRAS, 321, 559
    """

    def calculate(self, sim: SimSnap, params: Any = None) -> float:
        """Return the spin parameter lambda' of a centered halo.

        Returns
        -------
        float
            The dimensionless spin parameter lambda' of the halo.
        """
        from pynbody.analysis.angmom import spin_parameter

        spin = spin_parameter(sim)
        return spin

@PropertyBase.dataclass
class PatternSpeed(PropertyBase[SimArray]):
    """Calculate pattern speed in Z direction, assuming disk in x-y plane.

    Notes
    -----
    The calculation is based on eq. 46 of Pfenniger & Romero-Gómez (2023) [1]_.

    References
    ----------
    .. [1] Pfenniger, D., & Romero-Gómez, M. 2023, A&A, 673, A36
    """

    def calculate(self, sim: SimSnap, params: Any = None) -> SimArray:

        Ixx = (sim["mass"]*sim["x"]*sim["x"]).sum()
        Iyy = (sim["mass"]*sim["y"]*sim["y"]).sum()
        Ixy = (sim["mass"]*sim["x"]*sim["y"]).sum()
        # I_plus = 1/2*(Ixx+Iyy)
        I_minus = 1/2*(Ixx-Iyy)
        d_Ixy = (sim["mass"]*(sim["x"]*sim["vy"]+sim["y"]*sim["vx"])).sum()
        d_I_minus = (sim["mass"]*(sim["x"]*sim["vx"]-sim["y"]*sim["vy"])).sum()

        omega_Iz = 1/2*(I_minus*d_Ixy-d_I_minus*Ixy)/(I_minus*I_minus+Ixy*Ixy)
        omega_Iz.sim = sim.ancestor
        return omega_Iz
