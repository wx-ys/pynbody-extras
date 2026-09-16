"""Periodic-box wrapping transform.

:class:`WrapBox` wraps particle positions back into the simulation box before a
calculator reads them, and undoes the wrap when the scoped transform ends.

Positions are wrapped one axis at a time and only the per-axis integer offsets are
kept, so undoing costs a few bytes per particle instead of three floats; the offset
dtype is promoted, with a warning, when it does not fit the range.

The box size comes from the ``boxsize`` argument when it is given, otherwise from
the snapshot's ``boxsize`` property.  When neither is available the wrap is skipped
with a warning rather than guessed at.  ``pynbody`` gives a transformation two
apply hooks, and both go through the same wrapping core: ``_apply_to_snapshot``
(records offsets, undoable) and ``_apply_to_array`` (family views, apply-only).
"""

import warnings
from collections.abc import Mapping
from typing import Any, Literal, TypeGuard, cast

import numpy as np
from numpy.typing import DTypeLike
from pynbody import transformation, units
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import Param, TransformBase
from pynbodyext.log import logger

__all__ = ["WrapBox"]

#: The wrapping convention: ``"center"`` -> ``[-L/2, L/2)``, ``"upper"`` -> ``[0, L)``,
#: ``"minirange"`` -> per axis whichever of the two gives the smaller range.
Convention = Literal["center", "upper", "minirange"]

#: Conventions accepted by :class:`WrapBox` and the transformation.
CONVENTIONS: tuple[Convention, ...] = ("center", "upper", "minirange")
DEFAULT_CONVENTION: Convention = "minirange"

_NO_BOXSIZE = "wrap: no boxsize specified and snapshot has no 'boxsize' property; skipping wrap"
_NON_POSITIVE_BOXSIZE = "wrap: boxsize must be positive, got {}; skipping wrap"
_CANNOT_UNDO = "wrap: cannot undo, the boxsize is unknown; leaving the positions wrapped"


def normalize_convention(convention: str) -> Convention:
    """Return the canonical lower-case convention, or raise ``ValueError``."""
    normalized = str(convention).lower()
    if normalized not in CONVENTIONS:
        raise ValueError(f"Unknown wrapping convention {convention!r}, must be one of {CONVENTIONS}")
    return cast("Convention", normalized)


def _boxsize_in_units(boxsize: Any, units_now: units.UnitBase, conversion: Mapping[str, Any] | None = None) -> float:
    """Express *boxsize* as a number in *units_now*.

    A plain number is taken to be already in *units_now*; a unit-aware value is
    converted.  That conversion is what lets an undo survive the positions being
    re-expressed in different units between the wrap and the revert.
    """
    if isinstance(boxsize, units.UnitBase):
        return float(boxsize.ratio(units_now, **(conversion or {})))
    if isinstance(boxsize, SimArray):
        return float(boxsize.in_units(units_now, **(conversion or {})))
    return float(boxsize)


class WrapTransformation(transformation.Transformation):
    """Wrap particle positions into a periodic box.

    Undoing stores per-axis integer offsets ``k`` (``wrapped = original - k * L``)
    rather than a copy of ``pos``: an ``int8`` offset covers the usual case at a
    fraction of the memory of a float64 copy, and the dtype is promoted
    automatically, with a warning, when the offsets need more range.
    """

    def __init__(
        self,
        f: SimSnap | transformation.Transformation,
        boxsize: float | units.UnitBase | None = None,
        convention: Convention = DEFAULT_CONVENTION,
        k_dtype: DTypeLike = np.int8,
    ) -> None:
        """Wrap the snapshot (or chain onto a transformation) straight away.

        Parameters
        ----------
        f : pynbody.snapshot.SimSnap or pynbody.transformation.Transformation
            The snapshot to wrap, or the transformation to chain this one onto.
        boxsize : float or pynbody.units.UnitBase, optional
            Size of the periodic box, in position units. When omitted, the box size
            is taken from ``f.ancestor.properties["boxsize"]``; when neither is
            available the wrap is skipped with a warning.
        convention : {"minirange", "center", "upper"}, default "minirange"
            ``"center"`` wraps into ``[-boxsize/2, boxsize/2)``, ``"upper"`` into
            ``[0, boxsize)``, and ``"minirange"`` picks per axis whichever of those
            two gives the smaller coordinate range.
        k_dtype : numpy dtype, default numpy.int8
            Integer dtype for the offset counters; a larger type is selected
            automatically when the offsets do not fit.
        """
        convention_l = normalize_convention(convention)
        self.boxsize = boxsize
        self.convention = convention_l
        self._k_dtype = k_dtype
        self._k_offsets: np.ndarray | None = None  # shape (N, 3), ints
        self._boxsize_used: float | None = None  # the L the offsets were taken with
        description = f"Wrap{convention_l.capitalize()}"
        super().__init__(f, description=description)

    # ── box size ─────────────────────────────────────────────────────────────

    def _snapshot_boxsize(self, f: SimSnap) -> Any | None:
        """Return ``f.ancestor.properties["boxsize"]`` when the snapshot has one."""
        try:
            return f.ancestor.properties["boxsize"]
        except (AttributeError, KeyError):
            return None

    def _boxsize_for(self, f: SimSnap | None, units_now: units.UnitBase) -> float | None:
        """Return the configured box size as a number in *units_now*, or ``None``."""
        if f is None:
            return None
        boxsize = self.boxsize if self.boxsize is not None else self._snapshot_boxsize(f)
        if boxsize is None:
            return None
        return _boxsize_in_units(boxsize, units_now, f.conversion_context())

    @staticmethod
    def _check_boxsize(boxsize: float | None) -> TypeGuard[float]:
        """Warn (and report) whether *boxsize* can be wrapped with."""
        if boxsize is None:
            warnings.warn(_NO_BOXSIZE, stacklevel=3)
            logger.warning(_NO_BOXSIZE)
            return False
        if boxsize <= 0:
            warnings.warn(_NON_POSITIVE_BOXSIZE.format(boxsize), stacklevel=3)
            logger.warning(_NON_POSITIVE_BOXSIZE.format(boxsize))
            return False
        return True

    # ── offset dtype ─────────────────────────────────────────────────────────

    @staticmethod
    def _select_k_dtype(max_abs: float) -> np.dtype[Any]:
        """Pick the smallest signed integer dtype that can hold max_abs."""
        if max_abs <= np.iinfo(np.int8).max:
            return np.dtype(np.int8)
        if max_abs <= np.iinfo(np.int16).max:
            return np.dtype(np.int16)
        if max_abs <= np.iinfo(np.int32).max:
            return np.dtype(np.int32)
        return np.dtype(np.int64)

    def _promote_and_cast_k(self, *k_f_list: np.ndarray) -> list[np.ndarray]:
        max_abs = 0.0
        for k_f in k_f_list:
            if k_f.size:
                cur = float(np.max(np.abs(k_f)))
                max_abs = max(max_abs, cur)

        new_k_dtype = self._select_k_dtype(max_abs)
        if new_k_dtype != self._k_dtype:
            warnings.warn(
                f"wrap: auto-promote k dtype from {self._k_dtype} to {new_k_dtype} (max |k| = {max_abs:.0f})",
                stacklevel=2,
            )
            self._k_dtype = new_k_dtype

        return [k_f.astype(self._k_dtype, copy=False) for k_f in k_f_list]

    # ── wrapping ─────────────────────────────────────────────────────────────

    def _wrap_axes(self, axes: tuple[np.ndarray, np.ndarray, np.ndarray], L: float) -> list[np.ndarray]:
        """Wrap the coordinate arrays in place; return the integer offset per axis.

        Both hooks below go through here, so the snapshot path and the array path
        cannot disagree about the convention or about the offset dtype.
        """
        if self.convention == "minirange":
            # Compute both candidates, promote once over all of them (so every axis
            # is cast to the same dtype), then take the narrower wrapping per axis.
            center_f = [np.floor((v + 0.5 * L) / L) for v in axes]
            upper_f = [np.floor(v / L) for v in axes]
            cast = self._promote_and_cast_k(*center_f, *upper_f)
            center_k, upper_k = cast[: len(axes)], cast[len(axes) :]
            offsets: list[np.ndarray] = []
            for axis, (v, k_center, k_upper) in enumerate(zip(axes, center_k, upper_k, strict=True)):
                if v.size == 0:
                    offsets.append(np.zeros_like(v, dtype=self._k_dtype))
                    continue
                wrapped_center = v - k_center * L
                wrapped_upper = v - k_upper * L
                span_center = float(wrapped_center.max() - wrapped_center.min())
                span_upper = float(wrapped_upper.max() - wrapped_upper.min())
                logger.debug(
                    "wrap[minirange]: axis %d chose %s (span center=%.6g, upper=%.6g)",
                    axis,
                    "center" if span_center <= span_upper else "upper",
                    span_center,
                    span_upper,
                )
                if span_center <= span_upper:
                    v[:] = wrapped_center
                    offsets.append(k_center)
                else:
                    v[:] = wrapped_upper
                    offsets.append(k_upper)
            return offsets

        lower = -0.5 * L if self.convention == "center" else 0.0
        offsets = self._promote_and_cast_k(*(np.floor((v - lower) / L) for v in axes))
        for axis, offset in zip(axes, offsets, strict=True):
            np.subtract(axis, offset * L, out=axis)  # in place, without rebinding the name
        return offsets

    # ── pynbody hooks ────────────────────────────────────────────────────────

    def _apply_to_snapshot(self, f: SimSnap) -> None:
        """Wrap positions in place and record the integer offsets for a later undo."""
        L = self._boxsize_for(f, f["pos"].units)
        logger.debug("wrap: resolved boxsize L=%s", L)
        if not self._check_boxsize(L):
            return
        self._k_offsets = np.column_stack(self._wrap_axes((f["x"], f["y"], f["z"]), L))
        # Keep the size *with its units*: positions may be re-expressed in other units
        # before the undo, and then the offsets have to be scaled along with them.
        self._boxsize_used = SimArray(L, f["pos"].units)

    def _unapply_to_snapshot(self, f: SimSnap) -> None:
        """Undo the wrap with the recorded offsets (``pos += k * L``)."""
        if self._k_offsets is None:
            return  # nothing recorded: never wrapped, or already undone

        # Prefer the box size the offsets were taken with: it is the exact inverse even
        # if the snapshot's property has changed or disappeared since.  It is converted
        # into the units the positions are in *now*, so a unit change is undone too.
        L = (
            None
            if self._boxsize_used is None
            else _boxsize_in_units(self._boxsize_used, f["pos"].units, f.conversion_context())
        )
        if L is None:
            warnings.warn(_CANNOT_UNDO, stacklevel=2)
            logger.warning(_CANNOT_UNDO)
            return

        f["x"] += self._k_offsets[:, 0] * L
        f["y"] += self._k_offsets[:, 1] * L
        f["z"] += self._k_offsets[:, 2] * L

        self._k_offsets = None
        self._boxsize_used = None

    def _apply_to_array(self, array: SimArray) -> None:
        """Wrap a standalone ``pos`` array view in place.

        pynbody calls this for family views, whose rows are a subset of the snapshot,
        so no offsets are recorded here: this path is apply-only (pynbody never asks
        a transformation to undo an array).
        """
        if array.name != "pos":
            return

        L = self._boxsize_for(array.sim, array.units)
        if not self._check_boxsize(L):
            return

        pos = array.view(np.ndarray)  # raw view: floor() wants unitless arrays
        self._wrap_axes((pos[:, 0], pos[:, 1], pos[:, 2]), L)


@TransformBase.dataclass
class WrapBox(TransformBase[WrapTransformation]):
    """Wrap particle positions into a periodic box.

    Parameters
    ----------
    boxsize : float or pynbody.units.UnitBase, optional
        Size of the periodic box, in position units. When omitted, the box size is
        taken from the snapshot's ``boxsize`` property; if that is missing too, the
        wrap is skipped with a warning.
    convention : {"minirange", "center", "upper"}, default "minirange"
        ``"center"`` wraps into ``[-boxsize/2, boxsize/2)``, ``"upper"`` into
        ``[0, boxsize)``, and ``"minirange"`` picks per axis whichever of those two
        gives the smaller coordinate range.
    move_all : bool, default True
        Whether to wrap the whole snapshot (the ancestor) rather than the current view.
    """

    boxsize: Param[float | units.UnitBase | None] = Param(default=None, field_name="pos")
    convention: Convention = DEFAULT_CONVENTION

    def __post_init__(self) -> None:
        # Reject a typo when the node is built, not after a run has started.
        normalize_convention(self.convention)

    def build_handle(self, sim: Any, target: Any, params: Any = None) -> WrapTransformation:
        return WrapTransformation(target, boxsize=params.boxsize, convention=self.convention)
