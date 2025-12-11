


from typing import Literal

import numpy as np
from pynbody.analysis.halo import hybrid_center, shrink_sphere_center, virial_radius
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from .base import PropertyBase

__all__ = ["CenPos","CenVel","AngMomVec", "KappaRot", "KappaRotMean", "VirialRadius", "SpinParam", "PatternSpeed"]

def ang_mom_vec(snap):
    """
    Calculates the angular momentum vector of the specified snapshot.

    Parameters
    ----------
    snap : SimSnap
        The snapshot to analyze

    Returns
    -------
    array.SimArray
        The angular momentum vector of the snapshot
    """
    mass = snap["mass"].reshape((len(snap["mass"]), 1))
    pos = snap["pos"]
    vel = snap["vel"]

    # Try to import dask only if needed
    use_dask = False
    try:
        import dask.array as da
        if isinstance(pos, da.Array) and isinstance(vel, da.Array):
            use_dask = True
    except ImportError:
        use_dask = False

    if use_dask:
        cross = da.map_blocks(np.cross, pos, vel, dtype=pos.dtype)
        angmom = (mass * cross).sum(axis=0)
    else:
        cross = np.cross(pos, vel)
        angmom = (mass * cross).sum(axis=0)

    angmom.units = snap["mass"].units * snap["pos"].units * snap["vel"].units
    return angmom


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
            cen = hybrid_center(sim, r = "5 kpc")
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


class KappaRotMean(PropertyBase[float]):
    """
    Calculate the mean of the ratio of rotational kinetic energy to total kinetic energy per particle.

    Notes
    -----
    This is the mean of (0.5 * m * vcxy^2) / (m * ke) over given particles.
    """
    def calculate(self, sim: SimSnap) -> float:
        krot = 0.5 * sim["vcxy"] ** 2
        ke = sim["ke"]
        ratio = krot / ke
        return float(np.mean(ratio))

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


class PatternSpeed(PropertyBase[SimArray]):
    """Calculate pattern speed in Z direction, assuming disk in x-y plane.

    Notes
    -----
    The calculation is based on eq. 46 of Pfenniger & Romero-Gómez (2023) [1]_.

    References
    ----------
    .. [1] Pfenniger, D., & Romero-Gómez, M. 2023, A&A, 673, A36
    """

    def calculate(self, sim: SimSnap) -> SimArray:

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
