"""
Generic calculation interface for pynbody snapshots.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeAlias, TypeVar  #Protocol need python version >3.8

import numpy as np
from numpy.typing import NDArray
from pynbody import filt as _pyn_filt, units
from pynbody.array import SimArray
from pynbody.family import Family
from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation

try:
    from typing import Self  # type: ignore  # python >= 3.11
except ImportError:
    from typing_extensions import Self


SingleElementSimArray: TypeAlias = SimArray          # size == 1
SingleElementNDArray: TypeAlias = NDArray[Any]       # size == 1

ReturnT = TypeVar("ReturnT",covariant=True)

FilterLike: TypeAlias = _pyn_filt.Filter | np.ndarray | Family | slice | int | np.int32 | np.int64
TransformLike: TypeAlias = Callable[[SimSnap], Transformation]

class SimCallable(Protocol[ReturnT]):
    def __call__(self, sim: SimSnap, *args: Any, **kwargs: Any) -> ReturnT: ...

class CalculatorBase(SimCallable[ReturnT], Generic[ReturnT], ABC):
    """An abstract base class for performing calculations on particle data."""

    _filter: FilterLike | None
    _transformation: TransformLike | None
    _revert_transformation: bool

    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        # initialize per-instance filter and transformation settings
        self = super().__new__(cls)
        self._filter = None
        self._transformation = None
        self._revert_transformation = True
        return self

    def __call__(self, sim: SimSnap) -> ReturnT:
        """Executes the calculation on a given simulation snapshot."""
        trans_obj = None
        if self._transformation is not None:
            trans_obj = self._transformation(sim)
        if self._filter is not None:
            sim = sim[self._filter]
        try:
            result = self.calculate(sim)
        finally:
            if trans_obj is not None and self._revert_transformation:
                trans_obj.revert()
        return result

    @abstractmethod
    def calculate(self, sim: SimSnap, *args: Any, **kwargs: Any) -> ReturnT:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate"
        )

    def with_filter(self: Self, filt: FilterLike) -> Self:
        """Return a new CalculatorBase instance with the given filter applied."""
        self._filter = filt
        return self

    def __getitem__(self, key: FilterLike) -> Self:
        return self.with_filter(key)

    def with_transformation(self: Self, transformation: TransformLike, revert: bool = True) -> Self:
        """Return a new CalculatorBase instance with the given transformation applied."""
        self._transformation = transformation
        self._revert_transformation = revert
        return self

    def _in_sim_units(
        self,
        value: str | units.UnitBase | float | int | SingleElementSimArray | SingleElementNDArray,
        sim_parameter: str,
        sim: SimSnap,
    ) -> float:
        """
        Convert a value into the simulation's native units for a given array.

        This converts `value` - which may be a string with units (e.g., "10 kpc"),
        a pynbody Unit object, or a raw number assumed to already be in native units -
        to a float expressed in the units of `sim[sim_parameter]`.

        Parameters
        ----------
        value : str or pynbody.units.UnitBase or float or int or single-element SimArray or NDArray
            The value to be converted.
            - str: parsed by `pynbody.units.Unit`, e.g., "10 kpc", "200 km s**-1".
            - UnitBase: converted using `.in_units(target_units, **context)`.
            - number: treated as already in the target native units.
        sim_parameter : str
            The name of the simulation array (e.g., 'r', 'rxy', 'mass') whose units
            define the conversion target.
        sim : pynbody.snapshot.SimSnap
            The simulation snapshot.

        Returns
        -------
        float
            The numeric value in the native units of `sim[sim_parameter]`.
        """
        if isinstance(value, str):
            value = units.Unit(value)
        if isinstance(value, units.UnitBase):
            value = float(value.in_units(sim[sim_parameter].units,
                                          **sim[sim_parameter].conversion_context()))
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 1:
                if isinstance(value, SimArray):
                    value = value.in_units(sim[sim_parameter].units,
                                       **sim[sim_parameter].conversion_context()).item()
                else:
                    value = value.item()
            else:
                raise TypeError(
                    "value must be a single-element array if it is an ndarray, got array with shape "
                    f"{value.shape}"
                )
        if isinstance(value, (int, float)):
            return float(value)

        raise TypeError(
            "value must be str, pynbody.units.UnitBase (or unit-like), or a number, or a single-element SimArray or NDArray, got "
            f"{type(value)}"
        )


    def __repr__(self):
        if hasattr(self,"name"):
            return f"<Calculator {self.name}()>"
        return f"<Calculator {self.__class__.__name__}()>"
