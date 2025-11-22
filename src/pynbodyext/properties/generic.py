


from typing import Literal

import numpy as np
from pynbody.analysis.angmom import ang_mom_vec
from pynbody.analysis.halo import hybrid_center, shrink_sphere_center, virial_radius
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from .base import PropertyBase

__all__ = ["CenPos","CenVel","AngMomVec", "KappaRot", "VirialRadius", "SpinParam"]




class CenPos(PropertyBase[SimArray | np.ndarray]):
    """Center position property"""

    def __init__(self, mode: Literal["ssc", "com", "pot", "hyb"] = "ssc" ):
        self.mode = mode

    def instance_signature(self):
        return (self.__class__.__name__, self.mode)

    def calculate(self, sim: SimSnap) -> SimArray | np.ndarray:
        if self.mode == "com":
            cen = sim.mean_by_mass("pos")
        elif self.mode == "pot":
            i = sim["phi"].argmin()
            cen = sim["pos"][i].copy()
        elif self.mode == "ssc":
            cen = shrink_sphere_center(sim)
        elif self.mode == "hyb":
            cen = hybrid_center(sim)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected one of ['ssc', 'com', 'pot', 'hyb'].")
        if isinstance(cen, SimArray):
            cen.sim = sim
        return cen

class CenVel(PropertyBase[SimArray]):
    """Center velocity property"""

    def __init__(self, mode: Literal["com"] = "com" ):
        self.mode = mode

    def instance_signature(self):
        return (self.__class__.__name__, self.mode)

    def calculate(self, sim: SimSnap) -> SimArray:
        if self.mode == "com":
            cen = sim.mean_by_mass("vel")
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected one of ['com'].")
        if isinstance(cen, SimArray):
            cen.sim = sim
        return cen


class AngMomVec(PropertyBase[SimArray]):
    """Angular momentum vector property"""
    def calculate(self, sim: SimSnap) -> SimArray:
        return ang_mom_vec(sim)


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
    def calculate(self, sim: SimSnap) -> float:
        Krot = np.sum(0.5 * sim["mass"] * (sim["vcxy"] ** 2))
        K = np.sum(sim["mass"] * sim["ke"])
        return float(Krot / K)

class VirialRadius(PropertyBase[float]):
    """Virial radius property"""

    def __init__(self, overdensity: float = 178, rho_def: Literal["critical", "matter"] = "critical"):
        self.overdensity = overdensity
        self.rho_def = rho_def

    def instance_signature(self):
        return (self.__class__.__name__, self.overdensity, self.rho_def)

    def calculate(self, sim: SimSnap) -> float:
        return virial_radius(sim, overden=self.overdensity, rho_def=self.rho_def)

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

    def calculate(self, sim: SimSnap) -> float:
        """Return the spin parameter lambda' of a centered halo.

        Returns
        -------
        float
            The dimensionless spin parameter lambda' of the halo.
        """
        from pynbody.analysis.angmom import spin_parameter

        spin = spin_parameter(sim)
        return spin
