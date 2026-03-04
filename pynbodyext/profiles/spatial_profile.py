

from typing import Any, Literal

from pynbody.snapshot import SimSnap

from pynbodyext.log import logger
from pynbodyext.util._type import BinsAlgorithmFunc, RegistBinAlgorithmString, SimNpPrArray, SimNpPrArrayFunc

from .bins import BinsSet
from .profile import Profile

__all__ = ["SpatialProfile", "RadialProfile"]

class SpatialProfile(Profile):
    pass

class RadialProfile(SpatialProfile):
    def __init__(self,
        sim: SimSnap,
        *,
        ndim: Literal[2, 3] = 3,
        weight: str | SimNpPrArray | SimNpPrArrayFunc | None = None,
        bins_type: RegistBinAlgorithmString | BinsAlgorithmFunc = "lin",
        nbins: int | SimNpPrArray = 100,
        bin_min: float | None = None,
        bin_max: float | None = None,
        bins_set: BinsSet | None = None,
        **kwargs: Any):
        if ndim == 2:
            bins_by = "rxy"
            bins_area = "annulus"
        elif ndim == 3:
            bins_by = "r"
            bins_area = "spherical_shell"
        else:
            raise ValueError("ndim must be 2 or 3")

        super().__init__(
            sim,
            weight = weight,
            bins_by = bins_by,
            bins_area = bins_area,
            bins_type = bins_type,
            nbins = nbins,
            bin_min = bin_min,
            bin_max = bin_max,
            bins_set = bins_set,
            **kwargs
        )

@SpatialProfile.profile_property
def density(pro: SpatialProfile) -> SimNpPrArray:
    return pro["mass"]["sum"] / pro["binsize"]

@SpatialProfile.profile_property
def mass_enc(pro: SpatialProfile) -> SimNpPrArray:
    return pro["mass"]["sum"].cumsum()

@SpatialProfile.profile_property
def beta(pro: SpatialProfile) -> SimNpPrArray:
    """ This parameter quantifies the system's degree of radial anisotropy.
    If all orbits are circular, beta = -infinity; if all orbits are radial, beta = 1.
    beta > 0, are radially biased; beta < 0, are tangentially biased.
    as defined in Binney & Tremaine 2008, eq. (4.61).

    .. math::
        \\beta = 1 - (\\sigma_{\\theta}^2 + \\sigma_{\\phi}^2) / (2 \\sigma_r^2) = 1 - (\\sigma_{\\theta}^2 + \\sigma_{\\phi}^2) / (2 \\sigma_r^2)

    """
    if pro.bins.bins_by not in ["r",]:
        logger.warning("Beta parameter is useful for spherical systems. Consider using RadialProfile with ndim=3")
    # we can also calculate using (refer to pynbody's calculation of velocity dispersion):
    # 1.5 - (pro['vx']["disp"] ** 2 + pro['vy']["disp"] ** 2 + pro['vz']["disp"] ** 2) / pro['vr']["disp"] ** 2 / 2
    return 1 - (pro["vphi"]["rms"]**2 + pro["vtheta"]["rms"]**2)/(2*pro["vr"]["rms"]**2)
