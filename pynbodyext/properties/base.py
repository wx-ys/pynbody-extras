"""
Compatibility property module backed by the new calculator framework.

This module preserves the historical import path::

    from pynbodyext.properties.base import PropertyBase, ParamSum, ParameterContain

while delegating the core property and expression implementation to
:mod:`pynbodyext.core.calculate`.

The following names are re-exported directly from the new framework:

- :class:`PropertyBase`
- :class:`ConstantProperty`
- :class:`LambdaProperty`
- :class:`OpProperty`
- :class:`ParamSum`
- :class:`ParamContain`
- :class:`KappaRot`

Legacy compatibility alias:

- :class:`ParameterContain` -> :class:`ParamContain`

The concrete density-style properties that do not yet exist in
:mod:`pynbodyext.core.calculate` remain implemented locally:

- :class:`VolumeDensity`
- :class:`SurfaceDensity`
- :class:`RadiusAtSurfaceDensity`
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pynbody.array import SimArray

from pynbodyext.calculate import PropertyBase
from pynbodyext.filters.filt import Annulus, BandPass, ValueLike, ValueLikeFunc, VolumeFilter

__all__ = [
    "PropertyBase",
    "ParamSum",
    "ParamContain",
    "VolumeDensity",
    "SurfaceDensity",
    "RadiusAtSurfaceDensity",
]


def _normalize_frac(frac: Any) -> tuple[np.ndarray, bool]:
    frac_array = np.asarray([frac] if isinstance(frac, (int, float, np.floating)) else frac, dtype=float)
    if frac_array.ndim != 1:
        raise ValueError("frac must be a scalar or 1D sequence.")
    if not np.all((frac_array > 0) & (frac_array < 1)):
        raise ValueError(f"frac values must be between 0 and 1, got {frac_array}.")
    return frac_array, frac_array.size == 1 and isinstance(frac, (int, float, np.floating))
class ParamContain(PropertyBase[SimArray]):
    """Containment radius for one or more cumulative fractions.

    Parameters
    ----------
    cal_key : str, default: "r"
        Field to sort by, commonly radius.
    frac : float or sequence or dynamic value, default: 0.5
        Cumulative fraction or fractions to locate.  Values must be between 0
        and 1.  Calculator-valued fractions are resolved at run time.
    parameter : str, default: "mass"
        Weight field used for the cumulative sum.

    Returns
    -------
    pynbody.array.SimArray
        Radius or radii at the requested containment fractions.
    """

    dynamic_param_specs = {"frac": None}

    def __init__(
        self,
        cal_key: str = "r",
        frac: Any = 0.5,
        parameter: str = "mass",
    ) -> None:
        super().__init__()
        self.cal_key = cal_key
        self.parameter = parameter
        self.frac = frac

    def instance_signature(self) -> tuple[Any, ...]:
        return ("ParamContain", self.cal_key, self.parameter)


    def calculate_with_params(self, sim, params = None):

        frac = params["frac"]
        frac_array, frac_is_scalar = _normalize_frac(frac)
        key = sim[self.cal_key]
        weight = sim[self.parameter]
        order = np.argsort(np.asarray(key))
        key_sorted = np.asarray(key)[order]
        weight_sorted = np.asarray(weight)[order]
        cumulative = np.cumsum(weight_sorted)
        denom = float(cumulative[-1] - cumulative[0])
        if denom <= 0.0:
            raise ValueError(f"Non-positive total {self.parameter!r}; cannot compute containment radius.")
        cumulative = (cumulative - cumulative[0]) / denom
        values = np.interp(frac_array, cumulative, key_sorted)
        result: float | np.ndarray = float(values[0]) if frac_is_scalar else values
        contained = SimArray(result)
        if hasattr(key, "units"):
            contained.units = key.units
        if hasattr(sim, "ancestor"):
            contained.sim = sim.ancestor
        return contained


class ParamSum(PropertyBase[Any]):
    """Sum a simulation field.

    Parameters
    ----------
    parameter : str
        Field name to sum.
    """

    def __init__(self, parameter: str) -> None:
        super().__init__()
        self.parameter = parameter

    def instance_signature(self) -> tuple[Any, ...]:
        return ("ParamSum", self.parameter)

    def calculate(self, sim: Any) -> Any:
        return sim[self.parameter].sum()

class VolumeDensity(PropertyBase[SimArray]):
    """Mean volume density inside a volume filter or radius range."""
    dynamic_param_specs = {"rmax": "pos", "rmin": "pos"}

    def __init__(
        self,
        rmax: VolumeFilter | ValueLike | ValueLikeFunc,
        parameter: str = "mass",
        rmin: ValueLike | ValueLikeFunc = 0.0,
    ) -> None:
        super().__init__()
        self.rmax = rmax
        self.rmin = rmin
        self.parameter = parameter

    def instance_signature(self) -> tuple[Any, ...]:
        return (
            self.__class__.__name__,
            self.parameter,
        )

    def calculate_with_params(self, sim, params = None):
        selector = Annulus(params["rmin"], params["rmax"])

        param_sum = sim[selector][self.parameter].sum()
        volume = selector.volume(sim)

        den = param_sum / volume
        if isinstance(volume, float):
            den.units = sim[self.parameter].units / sim["pos"].units**3
        den.sim = sim.ancestor
        den.in_units(sim[self.parameter].units / sim["pos"].units**3)
        return den

class SurfaceDensity(PropertyBase[SimArray]):
    """Mean surface density inside an annulus."""
    dynamic_param_specs = {"rmax": "pos", "rmin": "pos"}

    def __init__(
        self,
        rmax: ValueLike | ValueLikeFunc,
        parameter: str = "mass",
        rmin: ValueLike | ValueLikeFunc = 0.0,
    ) -> None:
        super().__init__()
        self.rmax = rmax
        self.rmin = rmin
        self.parameter = parameter

    def instance_signature(self) -> tuple[Any, ...]:
        return (
            self.__class__.__name__,
            self.parameter,
        )

    def calculate_with_params(self, sim, params = None):
        selector = BandPass("rxy", params["rmin"], params["rmax"])
        param_sum = sim[selector][self.parameter].sum()
        area = np.pi * (params["rmax"]**2 - params["rmin"]**2)
        den = param_sum / area
        den.units = sim[self.parameter].units / sim["pos"].units**2
        den.sim = sim.ancestor
        return den


class RadiusAtSurfaceDensity(PropertyBase[SimArray]):
    """Radius where the surface density reaches a target value.

    In ``mode='shell'`` the surface density is measured in the shell
    ``[r - eps/2, r + eps/2]``.

    In ``mode='total'`` the surface density is defined as::

        Sigma(<r) = M(<r) / (pi r^2)

    A 1D bisection in radius is used to solve ``Sigma(r) = target``.
    """

    dynamic_param_specs = {"target": None}

    def __init__(
        self,
        target: ValueLike | ValueLikeFunc,
        parameter: str = "mass",
        mode: str = "shell",
        r_key: str = "rxy",
        eps: float = 0.01,
    ) -> None:
        super().__init__()
        if mode not in ("shell", "total"):
            raise ValueError("mode must be 'shell' or 'total'")
        self.target = target
        self.parameter = parameter
        self.mode = mode
        self.r_key = r_key
        self.eps = eps

    def instance_signature(self) -> tuple[Any, ...]:
        return (
            self.__class__.__name__,
            self.parameter,
            self.mode,
            self.r_key,
            self.eps,
        )

    def _target_value(self, sim, params = None):
        surf_units = sim[self.parameter].units / sim["pos"].units**2
        raw_target = params["target"]
        return self._in_sim_units(raw_target, self.parameter, sim, target_units=surf_units)

    @staticmethod
    def _sigma_at_radius(
        r_val: float,
        r_sorted: np.ndarray,
        m_cum: np.ndarray,
        eps: float,
        mode: str,
    ) -> float:
        if r_val <= 0:
            return 0.0

        if mode == "total":
            hi = np.searchsorted(r_sorted, r_val, side="right")
            if hi <= 0:
                return 0.0
            m_inside = m_cum[hi - 1]
            area = np.pi * (r_val**2)
            return 0.0 if area <= 0 else float(m_inside / area)

        rin = max(r_val - 0.5 * eps, 0.0)
        rout = r_val + 0.5 * eps
        if rout <= 0:
            return 0.0

        lo = np.searchsorted(r_sorted, rin, side="left")
        hi = np.searchsorted(r_sorted, rout, side="right")

        if hi <= 0 or hi <= lo:
            return 0.0

        m_shell = m_cum[hi - 1] - (m_cum[lo - 1] if lo > 0 else 0.0)
        area_shell = np.pi * (rout**2 - rin**2)
        return 0.0 if area_shell <= 0 else float(m_shell / area_shell)

    def calculate_with_params(self, sim, params = None):
        r_arr = sim[self.r_key]
        m_arr = sim[self.parameter]

        pos_units = sim["pos"].units
        mass_units = sim[self.parameter].units

        r_vals = np.asarray(r_arr.in_units(pos_units))
        m_vals = np.asarray(m_arr.in_units(mass_units))

        idx = np.argsort(r_vals)
        r_sorted = r_vals[idx]
        m_sorted = m_vals[idx]
        m_cum = np.cumsum(m_sorted)

        r_min = float(r_sorted[0])
        r_max = float(r_sorted[-1])
        if r_max <= r_min:
            raise ValueError("Degenerate radius distribution")

        target = self._target_value(sim, params)

        sample_grid = np.linspace(max(r_min, self.eps), r_max, 256)
        sigma_grid = np.array(
            [
                self._sigma_at_radius(r, r_sorted, m_cum, self.eps, self.mode)
                for r in sample_grid
            ],
            dtype=float,
        )

        diff = sigma_grid - target
        sign_changes = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
        if len(sign_changes) == 0:
            raise ValueError("Could not bracket target surface density")

        left = float(sample_grid[sign_changes[0]])
        right = float(sample_grid[sign_changes[0] + 1])

        for _ in range(80):
            mid = 0.5 * (left + right)
            sigma_mid = self._sigma_at_radius(mid, r_sorted, m_cum, self.eps, self.mode)
            if abs(sigma_mid - target) < 1e-10:
                left = right = mid
                break
            sigma_left = self._sigma_at_radius(left, r_sorted, m_cum, self.eps, self.mode)
            if (sigma_left - target) * (sigma_mid - target) <= 0:
                right = mid
            else:
                left = mid

        radius = SimArray(0.5 * (left + right))
        radius.units = pos_units
        radius.sim = sim.ancestor
        return radius
