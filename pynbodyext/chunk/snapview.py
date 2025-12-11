"""
Utilities for creating lightweight "view" and "mini" snapshots from an existing
`pynbody.snapshot.SimSnap` instance.

- `SimSnapView` provides a read-only facade that delegates metadata and
  snapshot-level queries to an existing snapshot without exposing particle
  access. It is useful when constructing composite snapshots (e.g., chunked
  views) that want to reuse SimSnap API calls but not duplicate particle
  buffers.

- `MiniSimSnap` is a minimal `SubSnap` that contains one particle (if present)
  from each family in the ancestor snapshot. It is used to probe array metadata
  (shape, dtype, units, family) without loading full particle data.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from pynbody.family import Family, get_family
from pynbody.snapshot import SimSnap
from pynbody.snapshot.subsnap import FamilySubSnap, IndexedSubSnap, SubSnap
from pynbody.units import UnitBase

if TYPE_CHECKING:
    from pynbody.array import SimArray

__all__ = ["SimSnapView", "MiniSimSnap"]


class SimSnapViewBase(SimSnap):
    """
    Base class for snapshot "views".

    This class stores references to snapshot-level metadata taken from a provided
    `base` snapshot (transformations, family slice mapping, and particle count).
    It intentionally does not duplicate or expose particle arrays.

    Parameters
    ----------
    base : SimSnap
        The snapshot instance that backs this view. Metadata will be referenced
        from `base`.
    """
    def __init__(self, base: SimSnap):
        super().__init__()
        self._transformations = base._transformations
        self._family_slice = base._family_slice
        self._num_particles = len(base)
        self.properties = base.properties


class SimSnapView(SimSnapViewBase): # should use ExposedBaseSnapshotMixin ?
    """
    Read-only view of an existing `SimSnap`.

    This class delegates metadata and high-level queries to `base` and is
    intended for situations where one wants the SimSnap API surface (keys,
    loadable/derivable keys, unit handling) but without loading or mutating
    particle data. It is commonly used internally to avoid duplicating SimSnap
    methods for composite snapshots such as chunked snapshots.

    Attributes
    ----------
    base : SimSnap
        The underlying snapshot object that backs this view.

    Examples
    --------
    >>> view = SimSnapView(existing_snapshot)
    >>> view.keys()
    >>> view.physical_units()
    """
    base: SimSnap
    def __init__(self, simsnap: SimSnap):
        self.base = simsnap
        SimSnapViewBase.__init__(self, simsnap)


    def physical_units(self, distance="kpc", velocity="km s^-1", mass="Msol", persistent=True, convert_parent=True):

        self.base.physical_units(distance, velocity, mass, persistent, convert_parent)
        return super().physical_units(distance, velocity, mass, persistent, convert_parent)

    def _get_dims(self, dims=None):
        return self.base._get_dims(dims)

    def families(self):
        return self.base.families()

    @property
    def _filename(self):
        return f"{self.base._filename}:View"

    def family_keys(self, fam=None):
        return self.base.family_keys(fam)

    def keys(self):
        return self.base.keys()

    def loadable_keys(self, fam=None):
        return self.base.loadable_keys(fam)

    def derivable_keys(self):
        return self.base.derivable_keys()


@dataclass(slots=True)
class ArrayInfo:
    """ Metadata container describing a SimArray """
    shape: tuple[int, ...]
    dtype: np.dtype
    units: UnitBase
    name: str | None
    family: Family | None


class MiniSimSnap(SubSnap):
    """
    Minimal SubSnap containing at most one particle from each family.

    `MiniSimSnap` is a lightweight `SubSnap` constructed from an existing
    `SimSnap` ancestor that includes the first particle (if present) from every
    family. It is intended for safely probing array metadata (shape, dtype,
    units, family) without loading the full snapshot.

    Typical usage:
    - Create a MiniSimSnap via `MiniSimSnap(simsnap)`.
    - Use `get_array_info(key, family)` to inspect array metadata for a
      particular key and family without reading large arrays into memory.
    """

    base: SimSnap
    def __init__(self, simsnap: SimSnap):
        """
        Build a `MiniSimSnap` from `simsnap`.

        Parameters
        ----------
        simsnap : SimSnap
            Snapshot used to derive the minimal representative snapshot.
        """
        base = self.create_minisnap(simsnap)
        super().__init__(base,slice(0, len(base)))
        self._descriptor = ""

    @property
    def _filename(self):
        return str(self.base._filename)

    def __repr__(self):
        class_name = f"Mini{self.base.__class__.__name__}"
        filename = self.base._filename
        if filename != "":
            return f'<{class_name} "{filename}" len={len(self.base)}>'
        else:
            return f"<{class_name} len={len(self.base)}>"

    @classmethod
    def create_minisnap(cls, simsnap: SimSnap) -> SimSnap:
        """
        Construct a minimal snapshot that contains a single representative particle
        from each family present in the ancestor.

        The function finds the first index of each family in the ancestor's
        `_family_slice` mapping, selects those indices, and returns a loaded
        copy of the resulting indexed `SubSnap`.

        Parameters
        ----------
        simsnap : SimSnap
            Snapshot whose ancestor will be inspected for families.

        Returns
        -------
        SimSnap
            A loaded snapshot containing at most one particle per family.
        """
        ancestor = simsnap.ancestor
        idx = []
        for fam in ancestor.families():
            fam_slice = ancestor._family_slice[fam]
            if fam_slice.stop > fam_slice.start:
                idx.append(fam_slice.start)
        sel = cast("IndexedSubSnap", ancestor[idx])
        return sel.load_copy()

    def get_array_info(self, key: str, family: str | Family | None = None) -> ArrayInfo:
        """
        Return array metadata for `key` and optional `family` without loading
        the full dataset.

        Parameters
        ----------
        key : str
            Array key name (e.g. 'pos', 'vel', 'mass', or custom property).
        family : str | Family | None, optional
            Family to query. If `None`, the base snapshot is used (global key).

        Returns
        -------
        ArrayInfo
            A dataclass instance containing `shape`, `dtype`, `units`, `name`,
            and `family` as discovered on the sample snapshot.

        Notes
        -----
        This method accesses `self.base` (the mini snapshot) and reads the
        target array from that snapshot; because the mini snapshot contains at
        most one particle per family, the returned `shape` will reflect that
        minimal sampling (useful for metadata only).
        """
        sample_snap: SimSnap
        assert isinstance(key, str), "key must be a string"
        if family is None:
            sample_snap = self.base
        else:
            sample_snap = cast("FamilySubSnap", self.base[get_family(family)])
        arr: SimArray = sample_snap[key]
        shape: tuple[int, ...] = arr.shape
        dtype: np.dtype = arr.dtype
        units = arr.units
        name: str | None = arr.name
        family_: Family | None = arr.family
        return ArrayInfo(shape, dtype, units, name, family_)
