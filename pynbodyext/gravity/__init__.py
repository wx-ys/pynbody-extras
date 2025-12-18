"""High-performance gravity solvers (TreeBH + direct summation).

This module exposes a small, stable Python wrapper around a Rust-implemented
Octree and TreeBH solver (provided by the `pynbodyext._rust` extension).
It provides:

- direct summation routines for potentials and accelerations,
- a TreeBH solver backed by an Octree with configurable multipole order
  and leaf capacity.

Example
-------
>>> import numpy as np
>>> from pynbodyext.gravity import Gravity
>>> pos = np.random.rand(10000, 3)
>>> mass = np.ones(10000)
>>> g = Gravity(pos, mass)
>>> acc = g.tree_accelerations(theta=0.7)
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
    """TreeBH and direct-summation gravity helper for a fixed particle set.

    Instantiate with the particle positions and masses to build (lazily)
    an Octree for TreeBH evaluations and to provide direct-summation helpers.

    Example
    -------
    >>> g = Gravity(positions, masses, leaf_capacity=8, multipole_order=3)
    >>> pot_direct = g.direct_potentials()
    >>> acc_tree = g.tree_accelerations(theta=0.7)

    The instance stores numpy.float64 arrays of positions (N, 3) and masses (N,)
    and builds the underlying Octree on first use or when requested via
    get_tree(...) with different TreeOptions.
    """
    def __init__(self, positions: NDArray[np.float64], masses: NDArray[np.float64], leaf_capacity: int = 8, multipole_order: int = 3):
        """
        Initialize the Gravity helper with particle positions and masses.

        Parameters
        ----------
        positions : ndarray of shape (N, 3), float64
            Particle positions in simulation units.
        masses : ndarray of shape (N,), float64
            Particle masses in simulation units.
        leaf_capacity : int, optional
            Maximum number of particles per leaf node.
        multipole_order : int, optional
            Order of multipole expansion (1 = monopole, 2 = dipole, 3 = quadrupole,..., max 5).
        """
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
        """Return (and build if needed) the Octree for the requested options.

        If the requested leaf_capacity and multipole_order match the stored
        options, the existing tree is returned; otherwise a new Octree is
        constructed and returned (the stored tree is not replaced).

        Parameters
        ----------
        leaf_capacity : int, optional
            Maximum particles per leaf node.
        multipole_order : int, optional
            Multipole expansion order.

        Returns
        -------
        _Octree
            The Octree configured with the requested options.
        """
        options = TreeOptions(leaf_capacity, multipole_order)
        if options == self.tree_options:
            return self.tree
        logger.debug("Building new Octree with leaf_capacity=%d, multipole_order=%d", leaf_capacity, multipole_order)
        return _build_tree(self.pos, self.mass, options)

    @property
    def tree(self) -> _Octree:
        """Octree built for the instance's stored TreeOptions.

        Builds the Octree on first access using the instance's TreeOptions.
        """
        if self._tree is None:
            self._tree = _build_tree(self.pos, self.mass, self.tree_options)
        return self._tree

    def direct_potentials(self, positions: NDArray[np.float64] | None = None, eps: float = 0.0, threads: int = 0) -> NDArray[np.float64]:
        """
        Compute gravitational potentials using direct summation.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions where potentials are evaluated. If None, evaluate
            at the internal particle positions.
        eps : float, optional
            Softening length used as 1/sqrt(r^2 + eps^2) in the kernel.
        threads : int, optional
            Number of threads to use (0 means use all available cores).

        Returns
        -------
        ndarray (M,)
            Potentials at the target positions (dtype float64).
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
        positions : ndarray (M, 3) or None, optional
            Target positions where accelerations are evaluated. If None, evaluate
            at the internal particle positions.
        eps : float, optional
            Softening length used so the kernel scales as 1/(r^2 + eps^2)^(3/2).
        threads : int, optional
            Number of threads to use (0 means use all available cores).

        Returns
        -------
        ndarray (M, 3)
            Accelerations at the target positions (dtype float64).
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
        Compute potentials using the TreeBH (Barnes-Hut) solver.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions where potentials are evaluated. If None, evaluate
            at the internal particle positions.
        theta : float, optional
            Barnes-Hut opening angle; smaller = more accurate (typical 0.3-0.8).
        eps : float, optional
            Softening length applied as in direct_potentials.
        threads : int, optional
            Number of threads to use (0 means use all available cores).
        leaf_capacity : int, optional
            Max particles per leaf node for the Octree.
        multipole_order : int, optional
            Multipole expansion order.

        Returns
        -------
        ndarray (M,)
            Potentials at the target positions (dtype float64).
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
        Compute accelerations using the TreeBH (Barnes-Hut) solver.

        Parameters
        ----------
        positions : ndarray (M, 3) or None, optional
            Target positions where accelerations are evaluated. If None, evaluate
            at the internal particle positions.
        theta : float, optional
            Barnes-Hut opening angle; smaller = more accurate.
        eps : float, optional
            Softening length used as in direct_accelerations.
        threads : int, optional
            Number of threads to use (0 means use all available cores).
        leaf_capacity : int, optional
            Max particles per leaf node for the Octree.
        multipole_order : int, optional
            Multipole expansion order.

        Returns
        -------
        ndarray (M, 3)
            Accelerations at the target positions (dtype float64).
        """
        tree = self.get_tree(leaf_capacity=leaf_capacity, multipole_order=multipole_order)
        if positions is None:
            return tree.compute_accelerations(theta, eps, threads)
        pos = np.asarray(positions, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, "positions must be of shape (N, 3)"
        return tree.accelerations_at_points(pos, theta, eps, threads)



