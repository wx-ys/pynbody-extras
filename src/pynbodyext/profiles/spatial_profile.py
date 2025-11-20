

from typing import Any, Literal

from pynbody.snapshot import SimSnap

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
