


import numpy as np
from pynbody.analysis.angmom import calc_faceon_matrix
from pynbody.snapshot.simsnap import SimSnap
from pynbody.transformation import Rotation, Transformation

from pynbodyext.util._type import get_signature_safe

from .base import TransformBase

__all__ = ["AlignAngMomVec"]


class AlignVec(TransformBase[Rotation]):
    """
    Generic transformation to align a vector (e.g., angular momentum, velocity) with the z-axis.
    Subclasses must implement `get_vec(self, sim: SimSnap) -> np.ndarray`.
    """
    def __init__(self, up: np.ndarray | None = None, move_all: bool = True):
        """
        Parameters
        ----------
        up : array_like or None, optional
            Desired 'up' vector (direction for positive y-axis after rotation). If `None`,
            a safe `up` will be computed inside `calculate` based on the snapshot's angular
            momentum vector.
        move_all : bool, optional
            Whether to apply the transformation to all particles in the ancestor snapshot.
        """
        self.up = up
        self.move_all = move_all

    def instance_signature(self):
        up_sig = get_signature_safe(self.up, fallback_to_id=True)
        move_all_sig = get_signature_safe(self.move_all, fallback_to_id=True)
        return (self.__class__.__name__, up_sig, move_all_sig)

    def get_vec(self, sim: SimSnap) -> np.ndarray:
        """
        Subclasses must override this to return the vector to align.
        """
        raise NotImplementedError("Subclasses must implement get_vec")

    def calculate(self, sim: SimSnap, previous: Transformation | None = None) -> Rotation:
        """
        Rotate the simulation snapshot to align the specified vector with the z-axis.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot to be transformed.

        Returns
        -------
        Rotation
            The rotation object describing the transformation applied to the simulation snapshot.
        """
        vec = self.get_vec(sim)
        if self.up is None:
            safe_up = self._safe_up(vec, up=None)
        else:
            safe_up = self.up
        trans = calc_faceon_matrix(vec, up=safe_up)
        target = self.get_target(sim, previous)
        rota = target.rotate(trans, description=self.__class__.__name__)
        return rota

    @staticmethod
    def _safe_up(ang: np.ndarray, up: np.ndarray | None = None, parallel_tol: float = 1e-6) -> np.ndarray:
        """
        Return a safe 'up' vector that is not (nearly) parallel to `ang`.

        Parameters
        ----------
        ang : array_like
            The angular momentum vector.
        up : array_like or None
            Preferred up vector. If None, a default of [0,1,0] is used (but possibly replaced
            if parallel); if provided and nearly parallel to `ang`, a safe axis is chosen.
        parallel_tol : float
            Tolerance for considering vectors parallel (default 1e-6).

        Returns
        -------
        numpy.ndarray
            A unit vector usable as `up` that is not (nearly) parallel to `ang`.
        """
        ang = np.asarray(ang, dtype=float)
        if np.isnan(ang).any() or np.linalg.norm(ang) == 0:
            raise ValueError(f"Angular momentum vector is zero or NaN {ang}")

        angn = ang / np.linalg.norm(ang)

        if up is None:
            up_arr = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            up_arr = np.asarray(up, dtype=float)
            if np.linalg.norm(up_arr) == 0 or np.isnan(up_arr).any():
                up_arr = np.array([0.0, 1.0, 0.0], dtype=float)

        upn = up_arr / np.linalg.norm(up_arr)

        # If up and ang are nearly parallel, pick coordinate axis least aligned with ang
        if abs(np.dot(angn, upn)) > 1.0 - parallel_tol:
            axes = np.eye(3, dtype=float)
            dots = np.abs(axes @ angn)
            upn = axes[np.argmin(dots)]

        return upn


class AlignAngMomVec(TransformBase[Rotation]):
    """
    Aligns the angular momentum vector with the z-axis.
    """

    def get_vec(self, sim: SimSnap) -> np.ndarray:
        from pynbodyext.properties.generic import AngMomVec
        return AngMomVec()(sim)

