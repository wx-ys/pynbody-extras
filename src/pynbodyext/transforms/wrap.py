

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike
from pynbody import transformation, units
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from .base import TransformBase

__all__ = ["WrapBox"]

class WrapTransformation(transformation.Transformation):
    """A pynbody Transformation to wrap particle positions into a periodic box.

    Memory behavior:
    - Instead of storing a full copy of `pos` to undo, we store per-axis integer
      offsets `k` such that: wrapped_pos = original_pos - k * L. Typically, using
      int16 reduces memory dramatically vs. a float64 copy of pos.
    """
    def __init__(
        self,
        f: SimSnap,
        boxsize: float | units.UnitBase | None=None,
        convention: Literal["center", "upper"] = "center",
        k_dtype: DTypeLike = np.int8):
        """
        Parameters
        ----------
        f : pynbody.snapshot.SimSnap
            The simulation snapshot to which this transformation will be applied.
        boxsize : float or pynbody.units.UnitBase, optional
            The size of the periodic box. If not specified, the box size is taken
            from the snapshot's properties.
        convention : str, optional
            The wrapping convention.
            - 'center': wraps particles to the range [-boxsize/2, boxsize/2). (Default)
            - 'upper': wraps particles to the range [0, boxsize).
        k_dtype : numpy dtype, optional
            Integer dtype for the offset counters (default: int16).
            Use int8 if you are certain |k|<=127 for all particles.
        """
        self.boxsize = boxsize
        self.convention = convention
        self._k_dtype = k_dtype
        self._k_offsets: np.ndarray | None = None  # shape (N, 3), ints
        description = convention + "_wrap"
        super().__init__(f, description=description)

    def _resolve_boxsize_float(self, f: SimSnap | None) -> float | None:
        if f is None:
            return None
        L = self.boxsize
        if L is None:
            try:
                L = f.ancestor.properties["boxsize"]
            except (AttributeError, KeyError):
                L = None
        if L is None:
            return None
        if isinstance(L, units.UnitBase):
            L = L.ratio(f["pos"].units, **f.conversion_context())
        return float(L)

    def _select_k_dtype(self, max_abs: float) -> np.dtype[Any]:
        """Pick the smallest signed integer dtype that can hold max_abs."""
        if max_abs <= np.iinfo(np.int8).max:
            return np.dtype(np.int8)
        if max_abs <= np.iinfo(np.int16).max:
            return np.dtype(np.int16)
        if max_abs <= np.iinfo(np.int32).max:
            return np.dtype(np.int32)
        return np.dtype(np.int64)

    def _apply_to_snapshot(self, f: SimSnap) -> None:
        """Wraps positions in-place and records integer offsets per axis."""
        # Use raw numpy views for ufuncs like floor that expect unitless arrays.
        x = f["x"].view(np.ndarray)
        y = f["y"].view(np.ndarray)
        z = f["z"].view(np.ndarray)

        L = self._resolve_boxsize_float(f)
        if L is None:
            warnings.warn(
                "wrap: no boxsize specified and snapshot has no 'boxsize' property; skipping wrap",
                stacklevel=2,
            )
            return
        if self.convention == "center":
            lower = -0.5 * L
        elif self.convention == "upper":
            lower = 0.0
        else:
            raise ValueError("Unknown wrapping convention")

        # Compute integer offsets k so that: wrapped = pos - k*L in each axis.
        # k = floor((pos - lower)/L)
        kx_f = np.floor((x - lower) / L)
        ky_f = np.floor((y - lower) / L)
        kz_f = np.floor((z - lower) / L)

        max_abs = float(
            max(
                np.max(np.abs(kx_f)),
                np.max(np.abs(ky_f)),
                np.max(np.abs(kz_f)),
            )
        )

        # Choose minimal safe dtype as a dtype instance
        new_k_dtype = self._select_k_dtype(max_abs)

        if new_k_dtype != self._k_dtype:
            warnings.warn(
                f"wrap: auto-promote k dtype from {self._k_dtype} to {new_k_dtype} "
                f"(max |k| = {max_abs:.0f})", stacklevel=2
            )
            self._k_dtype = new_k_dtype

        kx = kx_f.astype(self._k_dtype, copy=False)
        ky = ky_f.astype(self._k_dtype, copy=False)
        kz = kz_f.astype(self._k_dtype, copy=False)

        # Apply wrapping: pos := pos - k*L, all in-place
        x -= kx * L
        y -= ky * L
        z -= kz * L

        # Save offsets for unapply
        self._k_offsets = np.column_stack((kx, ky, kz))

    def _unapply_to_snapshot(self, f: SimSnap) -> None:
        """Reverts positions using stored integer offsets k (pos += k*L)."""
        if self._k_offsets is None:
            # Fall back: nothing recorded. No-op to avoid surprising errors.
            return

        x = f["x"].view(np.ndarray)
        y = f["y"].view(np.ndarray)
        z = f["z"].view(np.ndarray)

        L = self._resolve_boxsize_float(f)

        kx = self._k_offsets[:, 0]
        ky = self._k_offsets[:, 1]
        kz = self._k_offsets[:, 2]

        x += kx * L
        y += ky * L
        z += kz * L

        # Free memory
        self._k_offsets = None

    def _apply_to_array(self, array: SimArray) -> None:
        """Applies wrapping to a standalone 'pos' array view.

        Note: This path does not record offsets (array-level unapply is not used
        by pynbody for snapshot transformations). We operate in-place on the array.
        """
        if array.name != "pos":
            return

        f = array.sim
        L = self._resolve_boxsize_float(f)
        if L is None:
            warnings.warn(
                "wrap: no boxsize specified and snapshot has no 'boxsize' property; skipping wrap",
                stacklevel=2,
            )
            return
        if self.convention == "center":
            lower = -0.5 * L
        elif self.convention == "upper":
            lower = 0.0
        else:
            raise ValueError("Unknown wrapping convention")

        # array is an (N,3) SimArray; switch to raw ndarray view for ufuncs
        A = array.view(np.ndarray)

        # Compute k per axis and wrap in-place
        for coord in (0, 1, 2):
            v = A[:, coord]
            k = np.floor((v - lower) / L).astype(self._k_dtype, copy=False)
            v -= k * L


class WrapBox(TransformBase[WrapTransformation]):
    """Wraps particle positions to lie within a periodic box.

    This transform can be applied temporarily using a `with` statement.
    """

    def __init__(
        self,
        boxsize: float | units.UnitBase | None = None,
        convention: Literal["center", "upper"]="center"):
        """

        Parameters
        ----------
        boxsize : float or pynbody.units.UnitBase, optional
            The size of the periodic box. If not specified, the box size is taken
            from the snapshot's properties.
        convention : str, optional
            The wrapping convention.
            - 'center': wraps particles to the range [-boxsize/2, boxsize/2). (Default)
            - 'upper': wraps particles to the range [0, boxsize).
        """
        if convention not in ("center", "upper"):
            raise ValueError("Unknown wrapping convention, must be 'center' or 'upper'")

        self.boxsize = boxsize
        self.convention = convention

    def __call__(self, sim: SimSnap) -> WrapTransformation:
        """
        Wraps particle positions to lie within a periodic box.

        This creates a transformation that can be applied temporarily using a
        `with` statement.

        Parameters
        ----------
        sim : pynbody.snapshot.SimSnap
            The simulation snapshot to wrap.

        Returns
        -------
        WrapTransformation
            A transformation object suitable for use in a `with` block.
        """
        boxsize = self._in_sim_units(self.boxsize, "pos", sim) if self.boxsize is not None else None
        return WrapTransformation(sim, boxsize=boxsize, convention=self.convention)
