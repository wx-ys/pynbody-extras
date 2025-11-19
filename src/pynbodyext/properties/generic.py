


import numpy as np
from pynbody.analysis.angmom import ang_mom_vec
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from .base import PropertyBase

__all__ = ["AngMomVec", "KappaRot"]
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





