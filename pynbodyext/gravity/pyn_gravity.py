

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from .base import Gravity


def calculate_potential(
    sim: SimSnap,
    positions: NDArray[np.float64] | SimArray | None = None,
    eps: float = 0.0,
    method: Literal["direct", "tree"] = "tree",
    threads: int = 0,
    **kwargs: Any,
) -> SimArray:
    """
    Compute gravitational potentials at specified positions using direct summation.

    Parameters
    ----------
    sim : SimSnap
        The simulation snapshot containing particle data.
    positions : NDArray[np.float64] | SimArray
        An array of shape (N, 3) specifying the positions where potentials are computed.
    eps : float, optional
        Softening length to avoid singularities, by default 0.0.
    threads : int, optional
        Number of threads to use for computation, by default 0 (uses all available).

    Returns
    -------
    SimArray
        An array of gravitational potentials at the specified positions.

    Examples
    --------
    >>> import pynbody
    >>> from pynbodyext.gravity import calculate_potential
    >>> sim = pynbody.load("path_to_simulation_file")
    >>> # Compute potentials for all particles
    >>> potentials = calculate_potential(sim, method="tree", threads=4)
    """
    leaf_capacity = kwargs.get("leaf_capacity", 8)
    multipole_order = kwargs.get("multipole_order", 3)
    grav_helper = Gravity(sim["pos"], sim["mass"],leaf_capacity=leaf_capacity, multipole_order=multipole_order)

    if isinstance(positions, SimArray):
        positions = positions.in_units(sim["pos"].units)

    if method == "direct":
        pot = grav_helper.direct_potentials(positions, eps, threads)
    elif method == "tree":
        theta = kwargs.get("theta", 0.7)
        pot = grav_helper.tree_potentials(positions,theta, eps, threads)
    else:
        raise ValueError(f"Unknown method: {method}")

    res = SimArray(pot, units.G * sim["mass"].units / sim["pos"].units)
    res.sim = sim
    return res.in_units("km**2 s**-2")

def calculate_acceleration(
    sim: SimSnap,
    positions: NDArray[np.float64] | SimArray | None = None,
    eps: float = 0.0,
    method: Literal["direct", "tree"] = "tree",
    threads: int = 0,
    **kwargs: Any,
) -> SimArray:
    """
    Compute gravitational accelerations at specified positions using direct summation.

    Parameters
    ----------
    sim : SimSnap
        The simulation snapshot containing particle data.
    positions : NDArray[np.float64] | SimArray
        An array of shape (N, 3) specifying the positions where accelerations are computed.
    eps : float, optional
        Softening length to avoid singularities, by default 0.0.
    threads : int, optional
        Number of threads to use for computation, by default 0 (uses all available).

    Returns
    -------
    SimArray
        An array of gravitational accelerations at the specified positions.

    Examples
    --------
    >>> import pynbody
    >>> from pynbodyext.gravity import calculate_acceleration
    >>> sim = pynbody.load("path_to_simulation_file")
    >>> # Compute accelerations for all particles
    >>> accelerations = calculate_acceleration(sim, method="tree", threads=4)
    """
    leaf_capacity = kwargs.get("leaf_capacity", 8)
    multipole_order = kwargs.get("multipole_order", 3)
    grav_helper = Gravity(sim["pos"], sim["mass"],leaf_capacity=leaf_capacity, multipole_order=multipole_order)

    if isinstance(positions, SimArray):
        positions = positions.in_units(sim["pos"].units)

    if method == "direct":
        acc = grav_helper.direct_accelerations(positions, eps, threads)
    elif method == "tree":
        theta = kwargs.get("theta", 0.7)
        acc = grav_helper.tree_accelerations(positions, theta, eps, threads)
    else:
        raise ValueError(f"Unknown method: {method}")

    res = SimArray(acc, units.G * sim["mass"].units / sim["pos"].units**2)
    res.sim = sim
    return res.in_units("km s**-2")



