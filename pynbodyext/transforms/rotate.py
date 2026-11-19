import numpy as np
from pynbody.analysis.angmom import calc_faceon_matrix
from pynbody.transformation import Rotation

from pynbodyext.calculate import Param, TransformBase

__all__ = ["AlignVec"]


@TransformBase.dataclass
class AlignVec(TransformBase[Rotation]):
    """
    Generic transformation to align a vector (e.g., angular momentum) with the z-axis.

    Parameters
    ----------
    vector : array-like or Callable
        The vector to align with the z-axis. Can be a Param that depends on the simulation
    up : array-like, optional
        for y-axis alignment. If None, a safe default is chosen to avoid parallelism with the vector.
    move_all : bool, default: True
        Whether to move all particles or only a subset.

    """

    signature_name = "pynbodyext.transforms.AlignVec"
    frame_operation = "rotation"

    vector: Param[np.ndarray]
    up: np.ndarray | None = None

    def prepare_params(self, sim, values):
        # Reject undefined axes while the caller's frame is still known.
        direction = self._unit_direction(values["vector"])
        if self.up is not None:
            up = self._unit_direction(self.up)
            if np.linalg.norm(np.cross(up, direction)) == 0:
                raise ValueError("The up vector is parallel to the alignment direction")
        return super().prepare_params(sim, values)

    @staticmethod
    def _unit_direction(vector: np.ndarray) -> np.ndarray:
        ang = np.asarray(vector, dtype=float)
        if ang.shape != (3,) or np.isinf(ang).any():
            raise ValueError("Alignment requires a finite three-component direction")
        norm = np.linalg.norm(ang)
        if np.isnan(ang).any() or norm == 0:
            raise ValueError(f"Angular momentum vector is zero or NaN {ang}")
        if not np.isfinite(norm):
            raise ValueError("Alignment requires a finite direction norm")
        return ang / norm

    def build_handle(self, sim, target, params=None):
        """Apply the transform and return a handle."""
        vec = self._unit_direction(params.vector)
        safe_up = self._unit_direction(params.up) if params.up is not None else self._safe_up(vec)

        trans = calc_faceon_matrix(vec, up=safe_up)
        return target.rotate(trans, description=self.__class__.__name__)

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
        angn = AlignVec._unit_direction(ang)

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
