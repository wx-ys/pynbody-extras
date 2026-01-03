

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pynbody import units
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from .base import Gravity, KernelKind


def _coerce_softening(
    sim: SimSnap,
    softening: NDArray[np.float64] | SimArray | float | None,
) -> NDArray[np.float64] | float | None:
    if softening is None:
        return None
    if isinstance(softening, SimArray):
        # Convert to same length units as positions, then drop units for Rust backend.
        soft = softening.in_units(sim["pos"].units)
        arr = np.asarray(soft, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr)
        return arr
    if isinstance(softening, (float, int)):
        return float(softening)
    return np.asarray(softening, dtype=np.float64)

def calculate_potential(
    sim: SimSnap,
    positions: NDArray[np.float64] | SimArray | None = None,
    softening: NDArray[np.float64] | SimArray | float | None = None,
    method: Literal["direct", "tree"] = "tree",
    threads: int = 0,
    *,
    kernel: KernelKind = KernelKind.No,
    **kwargs: Any,
) -> SimArray:
    """
    Compute gravitational potentials for a pynbody snapshot.

    Parameters
    ----------
    sim : SimSnap
        Simulation snapshot containing ``pos`` and ``mass``.
    positions : NDArray[np.float64] | SimArray | None, optional
        Target positions (M, 3). If None, computes at particle positions.
        If a SimArray is provided, it is converted to ``sim["pos"].units``.
    softening : float | array (N,) | SimArray | None, optional
        Softening length(s). Scalars apply a uniform softening; arrays are
        per-particle softenings. If a SimArray is provided, it is converted
        to ``sim["pos"].units`` before passing to the backend.
    method : {"direct", "tree"}, optional
        Evaluation method.
    threads : int, optional
        Number of threads (0 uses all available).
    kernel : KernelKind, optional
        Softening kernel kind used by the backend (default: Newtonian).

    Other Parameters
    ----------------
    theta : float, optional
        Tree opening angle (tree method only).
    leaf_capacity : int, optional
        Octree leaf capacity (tree method only).
    multipole_order : int, optional
        Multipole expansion order (tree method only).

    Returns
    -------
    SimArray
        Potentials with units of ``G * mass / length`` (returned in ``km^2 s^-2``).

    Examples
    --------
    Potentials for all particles (TreeBH):

    >>> pot = calculate_potential(sim, method="tree", theta=0.7)

    Direct summation (small N, debugging / validation):

    >>> pot_d = calculate_potential(sim, method="direct", threads=8)

    Potentials at custom target points (in the same units as sim["pos"]):

    >>> targets = np.array([[0.0, 0.0, 0.0],
    ...                     [10.0, 0.0, 0.0]], dtype=np.float64)
    >>> pot_t = calculate_potential(sim, positions=targets, method="tree", theta=0.6)

    With softening + kernel:

    >>> pot_soft = calculate_potential(sim, softening=0.01, kernel=KernelKind.Plummer, method="tree")
    """
    leaf_capacity = kwargs.get("leaf_capacity", 8)
    multipole_order = kwargs.get("multipole_order", 3)

    soft = _coerce_softening(sim, softening)

    grav_helper = Gravity(
        sim["pos"],
        sim["mass"],
        softening=soft,
        kernel=kernel,
        leaf_capacity=leaf_capacity,
        multipole_order=multipole_order,
    )

    if isinstance(positions, SimArray):
        positions = positions.in_units(sim["pos"].units)

    if method == "direct":
        pot = grav_helper.direct_potentials(positions, threads)
    elif method == "tree":
        theta = kwargs.get("theta", 0.7)
        pot = grav_helper.tree_potentials(positions,theta, threads)
    else:
        raise ValueError(f"Unknown method: {method}")

    res = SimArray(pot, units.G * sim["mass"].units / sim["pos"].units)
    res.sim = sim
    return res.in_units("km**2 s**-2")

def calculate_acceleration(
    sim: SimSnap,
    positions: NDArray[np.float64] | SimArray | None = None,
    softening: NDArray[np.float64] | SimArray | float | None = None,
    method: Literal["direct", "tree"] = "tree",
    threads: int = 0,
    *,
    kernel: KernelKind = KernelKind.No,
    **kwargs: Any,
) -> SimArray:
    """
    Compute gravitational accelerations for a pynbody snapshot.

    Parameters
    ----------
    sim : SimSnap
        Simulation snapshot containing ``pos`` and ``mass``.
    positions : NDArray[np.float64] | SimArray | None, optional
        Target positions (M, 3). If None, computes at particle positions.
        If a SimArray is provided, it is converted to ``sim["pos"].units``.
    softening : float | array (N,) | SimArray | None, optional
        Softening length(s). Scalars apply a uniform softening; arrays are
        per-particle softenings. If a SimArray is provided, it is converted
        to ``sim["pos"].units`` before passing to the backend.
    method : {"direct", "tree"}, optional
        Evaluation method.
    threads : int, optional
        Number of threads (0 uses all available).
    kernel : KernelKind, optional
        Softening kernel kind used by the backend (default: Newtonian).

    Other Parameters
    ----------------
    theta : float, optional
        Tree opening angle (tree method only).
    leaf_capacity : int, optional
        Octree leaf capacity (tree method only).
    multipole_order : int, optional
        Multipole expansion order (tree method only).

    Returns
    -------
    SimArray
        Accelerations with units of ``G * mass / length^2`` (returned in ``km s^-2``).

    Examples
    --------
    Accelerations for all particles (TreeBH):

    >>> acc = calculate_acceleration(sim, method="tree", theta=0.7)

    Direct summation:

    >>> acc_d = calculate_acceleration(sim, method="direct", threads=8)

    Accelerations at custom target points:

    >>> targets = sim["pos"][:1024]  # SimArray is OK
    >>> acc_t = calculate_acceleration(sim, positions=targets, method="tree", theta=0.6)

    With softening + kernel:

    >>> acc_soft = calculate_acceleration(sim, softening=0.02, kernel=KernelKind.Spline, method="tree")
    """
    leaf_capacity = kwargs.get("leaf_capacity", 8)
    multipole_order = kwargs.get("multipole_order", 3)

    soft = _coerce_softening(sim, softening)

    grav_helper = Gravity(
        sim["pos"],
        sim["mass"],
        softening=soft,
        kernel=kernel,
        leaf_capacity=leaf_capacity,
        multipole_order=multipole_order,
    )

    if isinstance(positions, SimArray):
        positions = positions.in_units(sim["pos"].units)

    if method == "direct":
        acc = grav_helper.direct_accelerations(positions, threads)
    elif method == "tree":
        theta = kwargs.get("theta", 0.7)
        acc = grav_helper.tree_accelerations(positions, theta, threads)
    else:
        raise ValueError(f"Unknown method: {method}")

    res = SimArray(acc, units.G * sim["mass"].units / sim["pos"].units**2)
    res.sim = sim
    return res.in_units("km s**-2")



