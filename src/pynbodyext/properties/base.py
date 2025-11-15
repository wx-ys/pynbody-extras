

from collections.abc import Sequence
from typing import Generic, TypeVar

import numpy as np
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import CalculatorBase

__all__ = ["PropertyBase", "ParamSum", "ParameterContain"]


TProp = TypeVar("TProp", bound=SimArray | float | np.ndarray, covariant=True)


class PropertyBase(CalculatorBase[TProp],Generic[TProp]):
    """ Simsnap -> TProp """

    def __call__(self, sim: SimSnap) -> TProp:
        raise NotImplementedError



class ParamSum(PropertyBase[SimArray]):
    """Calculate sum of a parameter for selected particles."""

    def __init__(self, parameter: str):
        """
        Parameters
        ----------
        parameter : str
            Parameter to sum up (e.g., 'mass', 'sfr')
        """
        self.parameter = parameter

    def __call__(self, sim : SimSnap) -> SimArray:
        """
        Calculate the sum of the specified parameter for the given simulation snapshot.
        """
        return sim[self.parameter].sum()


class ParameterContain(PropertyBase[SimArray]):
    """
    Calculates the value of a key at which a certain fraction of a
    cumulative parameter is contained.

    For example, it can calculate the radius that contains half of the total mass
    (i.e., the half-mass radius).

    The calculation uses linear interpolation between points for improved accuracy.
    """
    def __init__(self, cal_key: str = "r", frac: float | Sequence[float] = 0.5, parameter: str = "mass"):
        """
        Initializes the calculator.

        Sort particles by cal_key, and then cumsum parameter,
        return cal_key where the cumsum of parameter equals frac * sum of parameter

        Parameters
        ----------
        cal_key : str, default 'r'
            Key to sort particles by ('r' for 3D radius, 'rxy' for projected radius)
        frac : float or Sequence of floats, default 0.5
            Fraction of the total parameter to include (must be between 0 and 1)
        parameter : str, default 'mass'
            Parameter to sum up (e.g., 'mass', 'sfr')

        """
        self.cal_key = cal_key
        self.parameter = parameter
        self.frac = frac
        # Normalize and validate frac now (type-safe for mypy)
        if isinstance(frac, (int, float, np.floating)):
            fval = float(frac)
            if not (0 < fval < 1):
                raise ValueError(f"Fraction must be between 0 and 1, got {frac}")
            self._frac_array: np.ndarray = np.array([fval], dtype=float)
            self._frac_is_scalar: bool = True
        elif isinstance(frac, Sequence):
            arr = np.asarray(frac, dtype=float)
            if arr.ndim != 1:
                raise ValueError("frac must be a 1D sequence of floats")
            if not np.all((arr > 0) & (arr < 1)):
                raise ValueError(f"Each fraction must be between 0 and 1, got {arr}")
            self._frac_array = arr
            self._frac_is_scalar = False
        else:
            raise TypeError("frac must be a float or a sequence of floats")

    def __call__(self, sim: SimSnap) -> SimArray:
        """
        Parameters
        ----------
        sim : SimSnap
            Input snapshot.

        Returns
        -------
        SimArray
            Value(s) of `cal_key` at the fractional cumulative points.
            For multiple fractions, returns an array (length = len(frac)).

        Algorithm
        ---------
        1. Sort particles by `cal_key`.
        2. Compute cumulative sum of `parameter`.
        3. Normalize cumulative to (0, 1).
        4. Interpolate `cal_key` values at each requested fraction `frac`.

        Raises
        ------
        ValueError
            If any fraction is not in (0,1).
        """

        # Get the parameter and cal_key arrays
        parameter_array = sim[self.parameter]
        cal_key_array = sim[self.cal_key]

        # Sort the arrays
        indices = np.argsort(cal_key_array)
        cal_key_sorted = cal_key_array[indices]
        parameter_sorted = parameter_array[indices]

        # Compute the cumulative sum and normalize to (0,1)
        parameter_cumsum = parameter_sorted.cumsum()

        # normalize to (0,1)
        denom = float(parameter_cumsum[-1] - parameter_cumsum[0])
        if denom <= 0.0:
            raise ValueError(
                f"Non-positive total '{self.parameter}' encountered; cannot normalize cumulative."
                )
        parameter_cumsum = (parameter_cumsum - parameter_cumsum[0]) / denom

        # Interpolate using the normalized array version of frac
        results_arr = np.interp(self._frac_array, parameter_cumsum, cal_key_sorted)
        results = float(results_arr[0]) if self._frac_is_scalar else results_arr

        cal_key_crit = SimArray(results)
        cal_key_crit.units = cal_key_sorted.units
        cal_key_crit.sim = sim.ancestor

        return cal_key_crit
