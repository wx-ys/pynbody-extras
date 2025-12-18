"""
pynbodyext.gravity — public exports for gravity helpers.

This package exposes the core Gravity helper (TreeBH + direct-summation)
and two high-level convenience functions that operate on pynbody Snapshots:

- Gravity: low-level helper that accepts numpy arrays of positions and
  masses and provides direct and tree (Barnes-Hut) gravity routines.
- calculate_potential(sim, ...): evaluate scalar gravitational potential
  for positions associated with a pynbody SimSnap; returns a SimArray.
- calculate_acceleration(sim, ...): evaluate vector gravitational
  accelerations for positions associated with a pynbody SimSnap; returns a SimArray.

Example
-------
>>> from pynbodyext.gravity import calculate_acceleration
>>> a = calculate_acceleration(sim, method="tree", theta=0.7)
"""


from .base import Gravity
from .pyn_gravity import calculate_acceleration, calculate_potential

__all__ = ["Gravity", "calculate_potential", "calculate_acceleration"]
