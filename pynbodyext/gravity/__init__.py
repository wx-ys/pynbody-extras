"""High-performance gravity solvers.

This module currently provides a TreeBH solver implemented in Rust,
backed by an Octree. The low-level implementation lives in the
:mod:`gravity` extension module built from the ``crates/gravity``
crate; this wrapper exposes a stable, documented API.

Examples
--------
>>> import numpy as np
>>> from pynbodyext.gravity import tree_force
>>> pos = np.random.rand(1000, 3)
>>> mass = np.ones(1000)
>>> acc = tree_force(pos, mass, theta=0.7, eps=0.0)
"""

from __future__ import annotations

from dataclasses import dataclass
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

@dataclass(eq=True, frozen=True)
class TreeOptions:
    """Options for the TreeBH gravity solver.

    Parameters
    ----------
    leaf_capacity : int, optional
        Maximum number of particles per leaf node.
    multipole_order : int, optional
        Order of multipole expansion (1 = monopole, 2 = dipole, 3 = quadrupole,..., max 5).
    """
    leaf_capacity: int = 8
    multipole_order: int = 3


def _build_tree(positions: NDArray[np.float64], masses: NDArray[np.float64], options: TreeOptions) -> _Octree:
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
    )
    return tree



class Gravity:
    def __init__(self, positions: NDArray[np.float64], masses: NDArray[np.float64], leaf_capacity: int = 8, multipole_order: int = 3):

        pos, mass = map(np.asarray, (positions, masses))
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must be a float64 array of shape (N, 3)")
        if mass.shape != (pos.shape[0],):
            raise ValueError("masses must be a float64 array of shape (N,)")
        pos = pos.astype(np.float64)
        mass = mass.astype(np.float64)
        self.pos = pos
        self.mass = mass

        self.tree_options = TreeOptions(leaf_capacity, multipole_order)
        self._tree: None | _Octree = None # Lazy initialization


    def get_tree(self, leaf_capacity: int = 8, multipole_order: int = 3) -> _Octree:
        """Get the Octree used for TreeBH gravity calculations, building it if necessary.

        Parameters
        ----------
        leaf_capacity : int, optional
            Maximum number of particles per leaf node.
        multipole_order : int, optional
            Order of multipole expansion (1 = monopole, 2 = dipole, 3 = quadrupole,..., max 5).

        Returns
        -------
        tree : _Octree
            The Octree used for TreeBH gravity calculations.
        """
        options = TreeOptions(leaf_capacity, multipole_order)
        if options == self.tree_options:
            return self.tree
        logger.debug("Building new Octree with leaf_capacity=%d, multipole_order=%d", leaf_capacity, multipole_order)
        return _build_tree(self.pos, self.mass, options)

    @property
    def tree(self) -> _Octree:
        """The Octree used for TreeBH gravity calculations."""
        if self._tree is None:
            self._tree = _build_tree(self.pos, self.mass, self.tree_options)
        return self._tree

    def direct_potentials(self, positions: NDArray[np.float64] | None = None, eps: float = 0.0, threads: int = 0) -> NDArray[np.float64]:
        """
        Compute gravitational potentials using direct summation.

        Parameters
        ----------
        positions : ndarray of shape (N, 3), float64 or None
            Particle positions in simulation units. If None, use the positions
            of internal particles.
        eps : float, optional
            Gravitational softening length. This is applied as
            ``1 / sqrt(r^2 + eps^2)`` inside the solver.
        threads : int, optional
            Number of threads to use. If 0, uses all available cores.

        Returns
        -------
        pot : ndarray of shape (M,), float64
            Gravitational potentials at target position.
        """
        if positions is None:
            return _direct_potentials_py(self.pos, self.mass, eps, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return _direct_potentials_at_points_py(self.pos, pos, self.mass, eps, threads)

    def direct_accelerations(self, positions: NDArray[np.float64] | None = None, eps: float = 0.0, threads: int = 0) -> NDArray[np.float64]:
        """
        Compute gravitational accelerations using direct summation.

        Parameters
        ----------
        positions : ndarray of shape (N, 3), float64 or None
            Particle positions in simulation units. If None, use the positions
            of internal particles.
        eps : float, optional
            Gravitational softening length. This is applied as
            ``1 / sqrt(r^2 + eps^2)^3`` inside the solver.
        threads : int, optional
            Number of threads to use. If 0, uses all available cores.

        Returns
        -------
        acc : ndarray of shape (M, 3), float64
            Gravitational accelerations at target position.
        """
        if positions is None:
            return _direct_accelerations_py(self.pos, self.mass, eps, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return _direct_accelerations_at_points_py(self.pos, pos, self.mass, eps, threads)



    def tree_potentials(
        self,
        positions: NDArray[np.float64] | None = None,
        theta: float = 0.7,
        eps: float = 0.0,
        threads: int = 0,
        leaf_capacity: int = 8,
        multipole_order: int = 3,
        ) -> NDArray[np.float64]:
        """
        Compute gravitational potentials using the TreeBH solver.

        Parameters
        ----------
        positions : ndarray of shape (N, 3), float64 or None
            Particle positions in simulation units. If None, use the positions
            of initial particles.
        theta : float, optional
            Barnes-Hut opening angle. Smaller values are more accurate
            but slower. Typical values are in the range 0.3-0.8.
        eps : float, optional
            Gravitational softening length. This is applied as
            ``1 / sqrt(r^2 + eps^2)`` inside the solver.
        threads : int, optional
            Number of threads to use. If 0, uses all available cores.
        leaf_capacity : int, optional
            Maximum number of particles per leaf node.
        multipole_order : int, optional
            Order of multipole expansion (1 = monopole, 2 = dipole, 3 = quadrupole,..., max 5).

        Returns
        -------
        pot : ndarray of shape (N,), float64
            Gravitational potentials at target position.
        """
        tree = self.get_tree(leaf_capacity=leaf_capacity, multipole_order=multipole_order)
        if positions is None:
            return tree.compute_potentials(theta, eps, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return tree.potentials_at_points(pos,theta, eps, threads)

    def tree_accelerations(
        self,
        positions: NDArray[np.float64] | None = None,
        theta: float = 0.7,
        eps: float = 0.0,
        threads: int = 0,
        leaf_capacity: int = 8,
        multipole_order: int = 3,
        ) -> NDArray[np.float64]:
        """
        Compute gravitational accelerations using the TreeBH solver.

        Parameters
        ----------
        positions : ndarray of shape (N, 3), float64
            Particle positions in simulation units.
        theta : float, optional
            Barnes-Hut opening angle. Smaller values are more accurate
            but slower. Typical values are in the range 0.3-0.8.
        eps : float, optional
            Gravitational softening length. This is applied as
            ``1 / sqrt(r^2 + eps^2)^3`` inside the solver.
        threads : int, optional
            Number of threads to use. If 0, uses all available cores.
        leaf_capacity : int, optional
            Maximum number of particles per leaf node.
        multipole_order : int, optional
            Order of multipole expansion (1 = monopole, 2 = dipole, 3 = quadrupole,..., max 5).

        Returns
        -------
        acc : ndarray of shape (N, 3), float64
            Gravitational accelerations at target position.
        """
        tree = self.get_tree(leaf_capacity=leaf_capacity, multipole_order=multipole_order)
        if positions is None:
            return tree.compute_accelerations(theta, eps, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return tree.accelerations_at_points(pos, theta, eps, threads)



