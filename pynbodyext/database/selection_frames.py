"""Replay witnessed native operations on an independent view of the saved root."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import SimArray

from pynbodyext.core.calculate.store.identity import sim_identity_key

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.selection_observations import SelectionContext

    from .database import Database


def load_in_frame(
    database: Database, input_id: int, context: SelectionContext, *, location_id: int | None = None
) -> Any:
    """Reload the exact particle identities and reproduce their recorded selection frame."""
    if not context.complete:
        reasons = [
            step.reason or step.kind
            for step in context.steps
            if step.reason or step.kind not in {"rotation", "shift_pos", "shift_vel"}
        ]
        raise ValueError("cannot replay this selection frame: " + ", ".join(reasons))
    desc = database._description(input_id)
    if (
        sim_identity_key(desc.root_signature) != sim_identity_key(context.root_signature)
        or desc.root_count != context.root_count
        or desc.unit_basis != context.unit_basis
    ):
        raise ValueError("selection frame does not match the saved root revision, count or unit convention")
    selected, _ = database._load_description(desc, location_id=location_id, convert_units=False)
    source = selected.ancestor
    if tuple((str(family), len(source[family])) for family in source.families()) != context.family_layout:
        raise ValueError("loaded root family layout differs from the recorded selection frame")

    # A custom loader can return the same cached root to several callers.
    # This pynbody snapshot keeps its own arrays and preserves root-relative indexing.
    # A full slice also gives HDF-backed snapshots a string view name, which
    # pynbody's copy-on-access constructor requires for its independent copy.
    root = source[:].get_copy_on_access_simsnap()
    if context.unit_basis:
        distance, velocity, mass = context.unit_basis
        root.physical_units(distance=distance, velocity=velocity, mass=mass)
    for field, units in (("pos", context.position_units), ("vel", context.velocity_units)):
        if units is not None:
            array = root[field]
            if str(array.units) != units:
                array.convert_units(units)

    for step in context.steps:
        target = root
        if step.target_indices is not None:
            if any(type(i) is not int or not 0 <= i < len(root) for i in step.target_indices):
                raise ValueError("coordinate operation target is outside the saved root")
            target = root[np.asarray(step.target_indices, dtype=np.intp)]
        if step.kind == "rotation":
            target.rotate(np.asarray(step.values, dtype=float))
        else:
            field = "pos" if step.kind == "shift_pos" else "vel"
            displacement = np.asarray(step.values, dtype=float)
            if displacement.shape != (3,) or not np.isfinite(displacement).all():
                raise ValueError("invalid recorded coordinate displacement")
            if step.units is None:
                raise ValueError("recorded coordinate displacement is missing its units")
            if step.units != str(target[field].units):
                displacement = SimArray(displacement, step.units).in_units(
                    target[field].units, **root.conversion_context()
                )
            if step.kind == "shift_pos":
                target.translate(displacement)
            else:
                target.offset_velocity(displacement)
    if desc.indices is None:
        return root
    return root[np.asarray(desc.indices, dtype=np.intp)]
