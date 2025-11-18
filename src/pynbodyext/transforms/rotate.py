

from collections.abc import Sequence

from pynbody.analysis.angmom import calc_faceon_matrix
from pynbody.snapshot.simsnap import SimSnap
from pynbody.transformation import Rotation

from pynbodyext.properties.generic import AngMomVec

from .base import TransformBase


class AlignAngMomVec(TransformBase):
    """
    A transformation class to rotate a simulation snapshot such that the angular momentum vector
    aligns with the z-axis. Optionally, an additional orientation vector can be specified to define
    the direction of the positive y-axis post-transformation.
    """
    def __init__(self, up: Sequence[float] | None = None):
        """
        Parameters
        ----------
        up : array_like, optional
            An additional orientation vector. The components of this vector perpendicular to the
            angular momentum vector define the direction to transform to 'up', i.e., to the positive
            y-axis post-transformation. Defaults to [0, 1., 0].
        """
        if up is None:
            up = [0, 1., 0]
        self.up = up

    def __call__(self, sim: SimSnap) -> Rotation:
        """
        Rotate the simulation snapshot to align the angular momentum vector with the z-axis.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot to be transformed.

        Returns
        -------
        Rotation
            The rotation object describing the transformation applied to the simulation snapshot.
        """
        ang = AngMomVec()(sim)
        trans = calc_faceon_matrix(ang, up=self.up)
        rota = sim.rotate(trans,description="AlignAngMomVec")
        return rota
