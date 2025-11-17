"""
Base class for profile builders.

This module defines :class:`~pynbodyext.profiles.base.ProfileBuilderBase`, an abstract
generic base for components that construct a :class:`~pynbodyext.profiles.profile.ProfileBase`
(or subclass) from a :class:`~pynbody.snapshot.SimSnap`.

Typical usage
-------------
Subclass :class:`~pynbodyext.profiles.base.ProfileBuilderBase` and implement ``__call__``
to return a concrete :class:`~pynbodyext.profiles.profile.Profile` (or any
:class:`~pynbodyext.profiles.profile.ProfileBase` subclass).

>>> from pynbodyext.profiles.profile import Profile
>>> class RadialProfileBuilder(ProfileBuilderBase[Profile]):
...     def __init__(self, nbins: int = 50):
...         self._nbins = nbins
...     def __call__(self, sim) -> Profile:
...         return Profile(sim, nbins=self._nbins, bins_type="lin", bins_by="r")

"""

from collections.abc import Callable
from typing import Any, Generic, Literal, TypeVar

from pynbody.snapshot import SimSnap
from pynbody.units import UnitBase

from pynbodyext.calculate import CalculatorBase

from .bins import (
    BinsSet,
    RegistBinAlgorithmString,
)
from .profile import ProfileBase
from .spatial_profile import (
    AllArray,
    BinsAlgorithmFunc,
    RadialProfile,
)

__all__ = ["ProfileBuilderBase", "RadialProfileBuilder"]

TProf = TypeVar("TProf", bound=ProfileBase, covariant=True)


class ProfileBuilderBase(CalculatorBase[TProf], Generic[TProf]):
    """
    Abstract base: build a :class:`~pynbodyext.profiles.profile.ProfileBase` from a
    :class:`~pynbody.snapshot.SimSnap`.

    Contract
    --------
    - Input: :class:`~pynbody.snapshot.SimSnap`
    - Output: :class:`~pynbodyext.profiles.profile.ProfileBase` (or subclass)
    - Implementations should avoid mutating ``sim`` and return a fresh profile instance.
    """

    def __call__(self, sim: SimSnap) -> TProf:
        """
        Create and return a concrete :class:`~pynbodyext.profiles.profile.ProfileBase`.

        Parameters
        ----------
        sim : :class:`~pynbody.snapshot.SimSnap`
            Simulation snapshot used to construct the profile.

        Returns
        -------
        :class:`~pynbodyext.profiles.profile.ProfileBase`
            The constructed profile (concrete subclass).
        """
        raise NotImplementedError


class RadialProfileBuilder(ProfileBuilderBase[RadialProfile]):
    def __init__(
        self,
        ndim: Literal[2,3] = 3,
        weight: AllArray | str | Callable[[SimSnap], AllArray] | None = None,
        bins_type: RegistBinAlgorithmString | BinsAlgorithmFunc = "lin",
        nbins: AllArray | int = 100,
        bin_min: str | float | UnitBase | Callable[[SimSnap], float] | None = None,
        bin_max: str | float | UnitBase | Callable[[SimSnap], float] | None = None,
        bins_set: BinsSet | None = None,
        **kwargs: Any
    ):
        if ndim not in [2,3]:
            raise ValueError("ndim must be either 2 or 3")
        self.ndim = ndim
        self.weight = weight
        self.bins_type = bins_type
        self.nbins = nbins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bins_set = bins_set
        self.kwargs = kwargs

    def __call__(self, sim: SimSnap) -> RadialProfile:
        if self.bin_min is not None:
            bin_min = self.bin_min(sim) if callable(self.bin_min) else self.bin_min
            bin_min = self._in_sim_units(bin_min, "pos", sim)
        else:
            bin_min = None
        if self.bin_max is not None:
            bin_max = self.bin_max(sim) if callable(self.bin_max) else self.bin_max
            bin_max = self._in_sim_units(bin_max, "pos", sim)
        else:
            bin_max = None

        return RadialProfile(
            sim,
            ndim=self.ndim,
            weight=self.weight,
            bins_type=self.bins_type,
            nbins=self.nbins,
            bin_min=bin_min,
            bin_max=bin_max,
            bins_set=self.bins_set,
            **self.kwargs
        )
