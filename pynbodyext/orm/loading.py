"""Load an explicitly selected local root without keeping an ORM transaction open."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pynbody.snapshot import SimSnap

from pynbodyext.core.calculate.runtime.logical_identity import ConfiguredSnapshotIdentity

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.logical_identity import LogicalSnapshotIdentity

    from .models import DataLocation, DataObject

SnapshotLoader = Callable[[str], SimSnap]


@dataclass(frozen=True)
class LoadedInput:
    """A loaded root or explicitly replayed view and its calculator provider.

    ``sim_identity`` follows the existing view/transformation rules. Metadata
    objects are detached; immediate input hierarchy and locations are loaded.
    Keep the loader's interpretation equivalent when reusing its loader_id.
    For load_selection, input names the view while location belongs to its root.
    """

    sim: SimSnap
    input: DataObject
    location: DataLocation
    sim_identity: ConfiguredSnapshotIdentity


def validate_local_location(uri: str) -> None:
    """Allow absolute local file names or multipart prefixes, without filesystem IO."""
    if type(uri) is not str or not uri.strip() or "://" in uri or "\0" in uri or not os.path.isabs(uri):
        raise ValueError("loading requires an absolute local snapshot path or prefix")


def load_snapshot(
    data: DataObject, location: DataLocation, logical: LogicalSnapshotIdentity, loader: SnapshotLoader
) -> LoadedInput:
    """Invoke a trusted binding and validate its root and logical identity."""
    sim = loader(location.uri)
    if not isinstance(sim, SimSnap):
        raise TypeError("snapshot loader must return a pynbody SimSnap")
    if sim.ancestor is not sim:
        raise ValueError("snapshot loader must return a root snapshot, not a view")
    filename = os.fspath(sim.filename)
    validate_local_location(filename)
    provider = ConfiguredSnapshotIdentity({filename: logical})
    if provider(sim) != logical.signature():
        raise ValueError("loaded snapshot has an external transformation or invalid logical identity")
    return LoadedInput(sim=sim, input=data, location=location, sim_identity=provider)
