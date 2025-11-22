
from typing import Any

from pynbody.snapshot import SimSnap

from pynbodyext.util._type import BinsAlgorithmFunc, RegistBinAlgorithmString, SimNpPrArray, SimNpPrArrayFunc

from .bins import BinsSet
from .profile import Profile

__all__ = ["TimeProfile","StarAgeProfile"]

class TimeProfile(Profile):
    pass


class StarAgeProfile(TimeProfile):

    def __init__(self,
        sim: SimSnap,
        *,
        weight: str | SimNpPrArray | SimNpPrArrayFunc | None = "mass",
        bins_type: RegistBinAlgorithmString | BinsAlgorithmFunc = "lin",
        nbins: int | SimNpPrArray = 100,
        bin_min: float | None = None,
        bin_max: float | None = None,
        bins_set: BinsSet | None = None,
        **kwargs: Any):
        if len(sim.s) <= 0:
            raise ValueError("Simulation snapshot contains no star particles")

        super().__init__(
            sim.s,
            weight = weight,
            bins_by = "age",
            bins_area = "length",
            bins_type = bins_type,
            nbins = nbins,
            bin_min = bin_min,
            bin_max = bin_max,
            bins_set = bins_set,
            **kwargs
        )
@StarAgeProfile.profile_property
def sfr(pro: StarAgeProfile) -> SimNpPrArray:
    return pro["mass"]["sum"] / pro["binsize"]
