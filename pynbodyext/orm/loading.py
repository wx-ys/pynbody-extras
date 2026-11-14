"""Validate identities and load explicit local roots without open ORM transactions."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pynbody.snapshot import SimSnap

from pynbodyext.core.calculate.runtime.logical_identity import ConfiguredSnapshotIdentity, LogicalSnapshotIdentity
from pynbodyext.core.calculate.runtime.sim_identity import SimIdentityProvider, snapshot_identity
from pynbodyext.core.calculate.store.identity import sim_identity_key

if TYPE_CHECKING:
    from .models import DataLocation, DataObject

SnapshotLoader = Callable[[str], SimSnap]


@dataclass(frozen=True)
class InputIdentity:
    """Parsed root/view address; path inputs do not invent a logical hierarchy."""

    signature: tuple[Any, ...]
    logical: LogicalSnapshotIdentity | None
    kind: str
    root_signature: tuple[Any, ...]

    @property
    def loader_id(self) -> str:
        return "default" if self.logical is None else self.logical.loader_id

    @property
    def revision(self) -> str | None:
        return None if self.logical is None else self.logical.revision


def parse_input_identity(value: LogicalSnapshotIdentity | tuple[Any, ...]) -> InputIdentity:
    """Validate a logical or normalized absolute-path snapshot/view identity.

    Parsing never loads files or guesses scientific Simulation/Timestep IDs.
    Halo numbers preserve built-in scalar types; view digests use the existing
    snapshot_identity contract. Repository path registration is explicitly opted in.
    """
    signature = value.signature() if isinstance(value, LogicalSnapshotIdentity) else value
    if type(signature) is not tuple or len(signature) < 2 or signature[0] != "sim":
        raise ValueError("ORM inputs require a stable path or configured logical snapshot identity")
    logical = None
    if signature[1] == "pynbodyext.logical-snapshot/v1":
        if len(signature) < 7:
            raise ValueError("incomplete configured logical snapshot identity")
        logical = LogicalSnapshotIdentity(*signature[2:7])
        root = logical.signature()
    else:
        path = signature[1]
        validate_local_location(path)
        if os.path.abspath(path) != path:
            raise ValueError("path identity requires a normalized absolute snapshot path")
        root = signature[:2]
    suffix = signature[len(root):]
    kind = "snapshot"
    if suffix:
        if len(suffix) == 2 and suffix[0] == "subset":
            kind = "subset"
        elif len(suffix) == 3 and suffix[0] == "halo" and type(suffix[1]) in (str, int, float, bool):
            kind = "halo"
        else:
            raise ValueError("unsupported snapshot view suffix")
        if not isinstance(suffix[-1], str) or re.fullmatch(r"[0-9a-f]{16}", suffix[-1]) is None:
            raise ValueError("view requires the existing particle-index digest")
    sim_identity_key(signature)
    return InputIdentity(signature, logical, kind, root)


@dataclass(frozen=True)
class LoadedInput:
    """A loaded root/view and provider, with detached immediate ORM relations.

    Providers preserve root/view/transformation semantics. For load_selection,
    input names the view while location belongs to its root. Path inputs use
    snapshot_identity; configured inputs use ConfiguredSnapshotIdentity.
    """

    sim: SimSnap
    input: DataObject
    location: DataLocation
    sim_identity: SimIdentityProvider


def validate_local_location(uri: str) -> None:
    """Allow absolute local file names or multipart prefixes, without filesystem IO."""
    if type(uri) is not str or not uri.strip() or "://" in uri or "\0" in uri or not os.path.isabs(uri):
        raise ValueError("loading requires an absolute local snapshot path or prefix")


def load_snapshot(
    data: DataObject, location: DataLocation, logical: LogicalSnapshotIdentity | None, loader: SnapshotLoader
) -> LoadedInput:
    """Invoke a trusted binding and validate the exact registered root identity."""
    sim = loader(location.uri)
    if not isinstance(sim, SimSnap):
        raise TypeError("snapshot loader must return a pynbody SimSnap")
    if sim.ancestor is not sim:
        raise ValueError("snapshot loader must return a root snapshot, not a view")
    filename = os.fspath(sim.filename)
    validate_local_location(filename)
    provider: SimIdentityProvider = (
        snapshot_identity if logical is None else ConfiguredSnapshotIdentity({filename: logical})
    )
    expected = data.signature if logical is None else logical.signature()
    if sim_identity_key(provider(sim)) != sim_identity_key(expected):
        raise ValueError("loaded snapshot has an external transformation or invalid registered identity")
    return LoadedInput(sim=sim, input=data, location=location, sim_identity=provider)
