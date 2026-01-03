"""High-performance gravity solvers (TreeBH + direct summation).

This module provides a thin, stable Python API over a Rust backend
(`pynbodyext._rust`) implementing:

- direct summation (potentials and accelerations),
- a Barnes–Hut TreeBH solver backed by an Octree,
- optional gravitational softening with selectable softening kernel.

Softening and kernels
---------------------
Softening is controlled by ``softening``:

- ``None``: no softening (pure Newtonian),
- scalar ``float``: same softening for all particles,
- array ``(N,)``: per-particle softening.

The softening kernel is selected via :class:`KernelKind`:

- ``KernelKind.No``: Newtonian (no kernel),
- ``KernelKind.Plummer``: Plummer-like softening,
- ``KernelKind.Spline``: spline softening.

Quickstart
----------
Compute accelerations for a particle set (TreeBH):

>>> import numpy as np
>>> from pynbodyext.gravity import Gravity, KernelKind
>>> pos = np.random.rand(10000, 3)
>>> mass = np.ones(10000)
>>> g = Gravity(pos, mass, softening=0.01, kernel=KernelKind.Spline)
>>> acc = g.tree_accelerations(theta=0.7)

Evaluate at arbitrary target points (direct summation):

>>> targets = np.array([[0.5, 0.5, 0.5],
...                     [0.1, 0.2, 0.3]], dtype=np.float64)
>>> pot = g.direct_potentials(positions=targets, threads=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

try:
    from pynbodyext._rust import (
        Octree as _Octree,
        direct_accelerations_at_points_py as _direct_accelerations_at_points_py,
        direct_accelerations_py as _direct_accelerations_py,
        direct_potentials_at_points_py as _direct_potentials_at_points_py,
        direct_potentials_py as _direct_potentials_py,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pynbodyext.gravity requires the Rust extension module `pynbodyext._rust` to be built. "
        "Install pynbodyext with Rust enabled or build the Rust bindings with maturin."
    ) from exc

from pynbodyext.log import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Gravity"]

class KernelKind(Enum):
    """Softening kernel type used by direct and TreeBH solvers.

    Notes
    -----
    The actual kernel implementation is provided by the Rust backend.
    """
    No = None
    Plummer = 0
    Spline = 1

@dataclass(eq=True, frozen=True)
class TreeOptions:
    """Options controlling Octree construction and multipole accuracy.

    Parameters
    ----------
    leaf_capacity : int, optional
        Maximum number of particles stored in a leaf node. Smaller values
        produce a finer tree (more nodes) and higher accuracy for a given
        opening criterion.
    multipole_order : int, optional
        Order of the multipole expansion used for far-field approximations.
        1 = monopole, 2 = dipole, 3 = quadrupole, ... (implementation max 5).
    kernel : KernelKind, optional
        Softening kernel kind used by the solver.
    """
    leaf_capacity: int = 8
    multipole_order: int = 3
    kernel: KernelKind = KernelKind.No


def _build_tree(positions: NDArray[np.float64], masses: NDArray[np.float64], softening: NDArray[np.float64] | None, options: TreeOptions) -> _Octree:
    """Build an Octree for the given particle positions and masses.

    Parameters
    ----------
    positions : ndarray of shape (N, 3), float64
        Particle positions in simulation units.
    masses : ndarray of shape (N,), float64
        Particle masses in simulation units.
    options : TreeOptions
        Options for the tree construction.

    Returns
    -------
    tree : _Octree
        The constructed Octree.
    """
    tree = _Octree(
        positions,
        masses,
        options.leaf_capacity,
        options.multipole_order,
        softening,
        options.kernel.value,
    )
    return tree



class Gravity:
    """TreeBH and direct-summation gravity helper for a fixed particle set.

    Parameters
    ----------
    positions : ndarray of shape (N, 3), float64
        Particle positions.
    masses : ndarray of shape (N,), float64
        Particle masses.
    softening : None | float | ndarray of shape (N,), optional
        Softening length(s). If a float is given, it is broadcast to all
        particles. If an array is given, it must be per-particle (shape (N,)).
    kernel : KernelKind, optional
        Default kernel used by direct and TreeBH methods unless overridden
        per-call (see method ``kernel=...`` arguments).
    leaf_capacity : int, optional
        Default Octree leaf capacity used for the cached tree.
    multipole_order : int, optional
        Default multipole order used for the cached tree.

    Notes
    -----
    - The cached Octree (property :pyattr:`tree`) is built lazily on first access.
    - Calls to :meth:`tree_potentials` / :meth:`tree_accelerations` may request
      a tree with different options; such trees are built on-demand.

    Examples
    --------
    Minimal (Newtonian):

    >>> g = Gravity(pos, mass)

    With softening and kernel:

    >>> g = Gravity(pos, mass, softening=0.02, kernel=KernelKind.Plummer)

    Per-particle softening:

    >>> eps = np.full(len(mass), 0.01, dtype=np.float64)
    >>> eps[:100] = 0.05
    >>> g = Gravity(pos, mass, softening=eps, kernel=KernelKind.Spline)
    """
    def __init__(
        self,
        positions: NDArray[np.float64],
        masses: NDArray[np.float64],
        softening: NDArray[np.float64] | float | None = None,
        kernel: KernelKind = KernelKind.No,
        leaf_capacity: int = 8,
        multipole_order: int = 3
    ) -> None:

        pos, mass = map(np.asarray, (positions, masses))
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must be a float64 array of shape (N, 3)")
        if mass.shape != (pos.shape[0],):
            raise ValueError("masses must be a float64 array of shape (N,)")
        soft_arr: NDArray[np.float64] | None
        if softening is None:
            soft_arr = None
        elif np.isscalar(softening):
            soft_arr = np.full((pos.shape[0],), float(softening), dtype=np.float64)
        else:
            soft_arr = np.asarray(softening, dtype=np.float64)
            if soft_arr.shape != (pos.shape[0],):
                raise ValueError("softening must be a float64 array of shape (N,)")

        pos = pos.astype(np.float64)
        mass = mass.astype(np.float64)
        self.pos = pos
        self.mass = mass
        self.softening = soft_arr

        self.tree_options = TreeOptions(
            leaf_capacity,
            multipole_order,
            kernel=KernelKind(kernel),
        )
        self._tree: None | _Octree = None # Lazy initialization


    def get_tree(
        self,
        leaf_capacity: int = 8,
        multipole_order: int = 3,
        kernel: KernelKind = KernelKind.No
    ) -> _Octree:
        """Return (and build if needed) the Octree for the requested options."""
        options = TreeOptions(leaf_capacity, multipole_order, kernel=KernelKind(kernel))
        if options == self.tree_options:
            return self.tree
        logger.debug(
            "Building new Octree with leaf_capacity=%d, multipole_order=%d",
            leaf_capacity,
            multipole_order
        )
        return _build_tree(self.pos, self.mass, self.softening, options)

    @property
    def tree(self) -> _Octree:
        """Octree built for the instance's stored TreeOptions.

        Builds the Octree on first access using the instance's TreeOptions.
        """
        if self._tree is None:
            self._tree = _build_tree(self.pos, self.mass, self.softening, self.tree_options)
        return self._tree

    def direct_potentials(
        self,
        positions: NDArray[np.float64] | None = None,
        threads: int = 0,
        kernel: KernelKind | None = None
    ) -> NDArray[np.float64]:
        """Compute gravitational potentials using direct summation.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions. If None, evaluate at the internal particle positions.
        threads : int, optional
            Number of threads (0 uses all available).
        kernel : KernelKind or None, optional
            Override kernel for this call. If None, uses the instance default.

        Returns
        -------
        ndarray (M,)
            Potentials at the target positions (float64).

        Notes
        -----
        Softening comes from ``self.softening`` (scalar-broadcasted to per-particle
        in the constructor) and is applied by the Rust backend.

        Examples
        --------
        Potentials at particle positions:

        >>> pot = g.direct_potentials()

        Potentials at custom target points:

        >>> targets = np.array([[0.0, 0.0, 0.0],
        ...                     [1.0, 0.0, 0.0]], dtype=np.float64)
        >>> pot_t = g.direct_potentials(positions=targets, threads=4)

        Temporarily override kernel without rebuilding the instance:

        >>> pot_plummer = g.direct_potentials(kernel=KernelKind.Plummer)
        """
        k = self.tree_options.kernel if kernel is None else KernelKind(kernel)
        if positions is None:
            return _direct_potentials_py(self.pos, self.mass, threads, self.softening, k.value)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return _direct_potentials_at_points_py(self.pos, pos, self.mass, threads, self.softening, k.value)

    def direct_accelerations(
        self,
        positions: NDArray[np.float64] | None = None,
        threads: int = 0,
        kernel: KernelKind | None = None,
    ) -> NDArray[np.float64]:
        """Compute gravitational accelerations using direct summation.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions. If None, evaluate at the internal particle positions.
        threads : int, optional
            Number of threads (0 uses all available).
        kernel : KernelKind or None, optional
            Override kernel for this call. If None, uses the instance default.

        Returns
        -------
        ndarray (M, 3)
            Accelerations at the target positions (float64).

        Examples
        --------
        Accelerations at particle positions:

        >>> acc = g.direct_accelerations()

        Accelerations at custom target points:

        >>> targets = np.random.rand(128, 3).astype(np.float64)
        >>> acc_t = g.direct_accelerations(positions=targets, threads=8)

        Override kernel for just this call:

        >>> acc_spline = g.direct_accelerations(kernel=KernelKind.Spline)
        """
        k = self.tree_options.kernel if kernel is None else KernelKind(kernel)
        if positions is None:
            return _direct_accelerations_py(self.pos, self.mass,threads, self.softening, k.value)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return _direct_accelerations_at_points_py(self.pos, pos, self.mass, threads, self.softening, k.value)



    def tree_potentials(
        self,
        positions: NDArray[np.float64] | None = None,
        theta: float = 0.7,
        threads: int = 0,
        leaf_capacity: int = 8,
        multipole_order: int = 3,
        kernel: KernelKind | None = None,
        ) -> NDArray[np.float64]:
        """Compute potentials using the TreeBH (Barnes–Hut) solver.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions. If None, evaluate at the internal particle positions.
        theta : float, optional
            Opening angle; smaller is more accurate (typical 0.3–0.8).
        threads : int, optional
            Number of threads (0 uses all available).
        leaf_capacity : int, optional
            Octree leaf capacity for this call.
        multipole_order : int, optional
            Multipole expansion order for this call.
        kernel : KernelKind or None, optional
            Override kernel for this call.

        Returns
        -------
        ndarray (M,)
            Potentials (float64).

        Examples
        --------
        Typical TreeBH call:

        >>> pot = g.tree_potentials(theta=0.7, threads=0)

        Higher accuracy (smaller theta) and higher multipole order:

        >>> pot_hi = g.tree_potentials(theta=0.4, multipole_order=5)

        Evaluate at target points (no need to create another Gravity):

        >>> targets = np.array([[0.2, 0.3, 0.4]], dtype=np.float64)
        >>> pot_t = g.tree_potentials(positions=targets, theta=0.6)

        Use a different kernel for this call (builds a tree with that kernel):

        >>> pot_plum = g.tree_potentials(kernel=KernelKind.Plummer)
        """
        k = self.tree_options.kernel if kernel is None else KernelKind(kernel)
        tree = self.get_tree(
            leaf_capacity=leaf_capacity,
            multipole_order=multipole_order,
            kernel=k
        )
        if positions is None:
            return tree.compute_potentials(theta, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return tree.potentials_at_points(pos,theta, threads)

    def tree_accelerations(
        self,
        positions: NDArray[np.float64] | None = None,
        theta: float = 0.7,
        threads: int = 0,
        leaf_capacity: int = 8,
        multipole_order: int = 3,
        kernel: KernelKind | None = None,
        ) -> NDArray[np.float64]:
        """Compute accelerations using the TreeBH (Barnes–Hut) solver.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions. If None, evaluate at the internal particle positions.
        theta : float, optional
            Opening angle; smaller is more accurate.
        threads : int, optional
            Number of threads (0 uses all available).
        leaf_capacity : int, optional
            Octree leaf capacity for this call.
        multipole_order : int, optional
            Multipole expansion order for this call.
        kernel : KernelKind or None, optional
            Override kernel for this call.

        Returns
        -------
        ndarray (M, 3)
            Accelerations (float64).

        Examples
        --------
        Typical TreeBH call:

        >>> acc = g.tree_accelerations(theta=0.7)

        Faster (less accurate) evaluation:

        >>> acc_fast = g.tree_accelerations(theta=0.9)

        Evaluate at custom points:

        >>> targets = np.random.rand(256, 3).astype(np.float64)
        >>> acc_t = g.tree_accelerations(positions=targets, theta=0.7, threads=4)

        Override kernel for this call:

        >>> acc_spline = g.tree_accelerations(kernel=KernelKind.Spline)
        """
        k = self.tree_options.kernel if kernel is None else KernelKind(kernel)
        tree = self.get_tree(leaf_capacity=leaf_capacity, multipole_order=multipole_order, kernel=k)
        if positions is None:
            return tree.compute_accelerations(theta, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return tree.accelerations_at_points(pos, theta, threads)



