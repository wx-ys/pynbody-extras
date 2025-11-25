

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike
from pynbody import transformation, units
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from pynbodyext.util._type import get_signature_safe

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
        f: SimSnap | transformation.Transformation,
        boxsize: float | units.UnitBase | None=None,
        convention: Literal["center", "upper", "minirange"] = "minirange",
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
            - 'minirange': per axis, chooses between 'center' and 'upper'
              to minimise the coordinate range after wrapping.
        k_dtype : numpy dtype, optional
            Integer dtype for the offset counters (default: int8).
            Use a larger type (e.g. int16/int32) if needed.
        """
        if convention not in ("center", "upper", "minirange"):
            raise ValueError(
                "Unknown wrapping convention, must be 'center', 'upper' or 'minirange'"
            )
        self.boxsize = boxsize
        self.convention = convention
        self._k_dtype = k_dtype
        self._k_offsets: np.ndarray | None = None  # shape (N, 3), ints
        description = convention + "_wrap"
        super().__init__(f, description=description)

    def _resolve_boxsize_float(self, f: SimSnap | None) -> float | None:
        if f is None:
            return None
        try:
            L = f.ancestor.properties["boxsize"]
        except (AttributeError, KeyError):
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
    def _compute_k_and_wrapped(self, v: np.ndarray, L: float, lower: float) -> tuple[np.ndarray, np.ndarray]:
        """Given positions v, box size L and lower bound, return (k, wrapped_v)."""
        k_f = np.floor((v - lower) / L)
        k = k_f.astype(self._k_dtype, copy=False)
        wrapped = v - k * L
        return k, wrapped

    def _promote_and_cast_k(
        self, *k_f_list: np.ndarray
    ) -> list[np.ndarray]:
        max_abs = 0.0
        for k_f in k_f_list:
            if k_f.size:
                cur = float(np.max(np.abs(k_f)))
                max_abs = max(max_abs, cur)

        new_k_dtype = self._select_k_dtype(max_abs)
        if new_k_dtype != self._k_dtype:
            warnings.warn(
                f"wrap: auto-promote k dtype from {self._k_dtype} to {new_k_dtype} "
                f"(max |k| = {max_abs:.0f})",
                stacklevel=2,
            )
            self._k_dtype = new_k_dtype

        return [k_f.astype(self._k_dtype, copy=False) for k_f in k_f_list]

    def _compute_kf_for_axes(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, L: float, lower: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kx_f = np.floor((x - lower) / L)
        ky_f = np.floor((y - lower) / L)
        kz_f = np.floor((z - lower) / L)
        return kx_f, ky_f, kz_f

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

        if self.convention in ("center", "upper"):
            lower = -0.5 * L if self.convention == "center" else 0.0

            # Compute integer offsets k so that: wrapped = pos - k*L in each axis.
            # k = floor((pos - lower)/L)
            kx_f, ky_f, kz_f = self._compute_kf_for_axes(x, y, z, L, lower)
            kx, ky, kz = self._promote_and_cast_k(kx_f, ky_f, kz_f)

            # Apply wrapping: pos := pos - k*L, all in-place
            x -= kx * L
            y -= ky * L
            z -= kz * L
        elif self.convention == "minirange":

            lower_center = -0.5 * L
            lower_upper = 0.0

             # compute candidate k_f for both center & upper so dtype promotion can consider all axes
            kx_c_f, ky_c_f, kz_c_f = self._compute_kf_for_axes(x, y, z, L, lower_center)
            kx_u_f, ky_u_f, kz_u_f = self._compute_kf_for_axes(x, y, z, L, lower_upper)

            ( kx_c, ky_c, kz_c, kx_u, ky_u, kz_u,
            ) = self._promote_and_cast_k(
                kx_c_f, ky_c_f, kz_c_f,
                kx_u_f, ky_u_f, kz_u_f,
            )

            axes_v   = (x,    y,    z)
            axes_k_c = (kx_c, ky_c, kz_c)
            axes_k_u = (kx_u, ky_u, kz_u)
            chosen_k = []

            for v, kc, ku, _lower_c, _lower_u in zip(
                axes_v,
                axes_k_c,
                axes_k_u,
                (lower_center,)*3,
                (lower_upper,)*3, strict=False,
            ):
                if v.size == 0:
                    chosen_k.append(np.zeros_like(v, dtype=self._k_dtype))
                    continue

                wrapped_c = v - kc * L
                wrapped_u = v - ku * L

                range_c = float(wrapped_c.max() - wrapped_c.min()) if wrapped_c.size else 0.0
                range_u = float(wrapped_u.max() - wrapped_u.min()) if wrapped_u.size else 0.0

                if range_c <= range_u:
                    v[:] = wrapped_c
                    chosen_k.append(kc)
                else:
                    v[:] = wrapped_u
                    chosen_k.append(ku)

            kx, ky, kz = chosen_k

        else:
            raise ValueError("Unknown wrapping convention")
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

        # array is an (N,3) SimArray; switch to raw ndarray view for ufuncs
        A = array.view(np.ndarray)

        if self.convention in ("center", "upper"):
            lower = -0.5 * L if self.convention == "center" else 0.0
            for coord in (0, 1, 2):
                v = A[:, coord]
                k = np.floor((v - lower) / L).astype(self._k_dtype, copy=False)
                v -= k * L

        elif self.convention == "minirange":
            lower_center = -0.5 * L
            lower_upper = 0.0

            for coord in (0, 1, 2):
                v = A[:, coord]
                if v.size == 0:
                    continue

                k_c, wrapped_c = self._compute_k_and_wrapped(v, L, lower_center)
                k_u, wrapped_u = self._compute_k_and_wrapped(v, L, lower_upper)

                range_c = float(wrapped_c.max() - wrapped_c.min()) if wrapped_c.size else 0.0
                range_u = float(wrapped_u.max() - wrapped_u.min()) if wrapped_u.size else 0.0

                if range_c <= range_u:
                    v[:] = wrapped_c
                else:
                    v[:] = wrapped_u
        else:
            raise ValueError("Unknown wrapping convention")


class WrapBox(TransformBase[WrapTransformation]):
    """Wraps particle positions to lie within a periodic box.

    This transform can be applied temporarily using a `with` statement.
    """

    def __init__(
        self,
        boxsize: float | units.UnitBase | None = None,
        convention: Literal["center", "upper", "minirange"]="minirange",
        move_all: bool = True):
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
            - 'minirange': per axis, chooses between 'center' and 'upper'
              to minimise the coordinate range after wrapping.
        move_all: bool, default True
            Whether to perform the wrapping on ancestors (all particles).
        """
        if convention not in ("center", "upper", "minirange"):
            raise ValueError(
                "Unknown wrapping convention, must be 'center', 'upper' or 'minirange'"
            )
        self.boxsize = boxsize
        self.convention = convention
        self.move_all = move_all

    def instance_signature(self):
        boxsize_sig = get_signature_safe(self.boxsize, fallback_to_id=True)
        convention_sig = get_signature_safe(self.convention, fallback_to_id=True)
        move_all_sig = get_signature_safe(self.move_all, fallback_to_id=True)
        return (self.__class__.__name__, boxsize_sig, convention_sig, move_all_sig)

    def calculate(self, sim: SimSnap, previous: transformation.Transformation | None = None) -> WrapTransformation:
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
        target_sim = self.get_target(sim, previous)
        return WrapTransformation(target_sim, boxsize=boxsize, convention=self.convention)
