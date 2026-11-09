"""Explicit logical snapshot identities, with locations kept out of cache keys.

Configuration declares data equivalence; it does not verify file contents. When
revision is omitted, refresh affected calculations with ``reuse=False`` after
changing data or particle order. Loading interpretation changes need a new
``loader_id``. Neither configuration nor lookup reads snapshot files.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeVar

from .sim_identity import (
    _has_transformations,
    _is_file_name,
    _snapshot_view_identity,
    id_based_sim_identity,
    snapshot_identity,
)

__all__ = ["LogicalSnapshotIdentity", "ConfiguredSnapshotIdentity"]

if TYPE_CHECKING:
    from collections.abc import Mapping

_CONFIG_SCHEMA = "pynbodyext.snapshot-identity-config/v1"
_IDENTITY_FIELDS = {"project_id", "simulation_id", "timestep_id", "revision", "loader_id"}
_Path = TypeVar("_Path", bound=str | os.PathLike[str])


def _nonempty_string(value: Any, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class LogicalSnapshotIdentity:
    """User-maintained snapshot identity, independent of storage location.

    IDs are literal strings. ``revision=None`` intentionally permits reuse after
    data changes until explicit refresh. ``loader_id`` labels loading options;
    it is not automatically derived from pynbody configuration or plugins.
    """

    project_id: str
    simulation_id: str
    timestep_id: str
    revision: str | None = None
    loader_id: str = "default"

    def __post_init__(self) -> None:
        for name in ("project_id", "simulation_id", "timestep_id", "loader_id"):
            _nonempty_string(getattr(self, name), name)
        if self.revision is not None:
            _nonempty_string(self.revision, "revision")

    def signature(self) -> tuple[Any, ...]:
        """Stable typed identity accepted by the engine, store and SampleTable."""
        return (
            "sim",
            "pynbodyext.logical-snapshot/v1",
            self.project_id,
            self.simulation_id,
            self.timestep_id,
            self.revision,
            self.loader_id,
        )


def _location_path(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path)
    _nonempty_string(text, "path")
    if "://" in text or not _is_file_name(text):
        raise ValueError("path must name a local snapshot file or prefix")
    return os.path.abspath(text)


def _bind(bindings: dict[str, LogicalSnapshotIdentity], path: str, identity: LogicalSnapshotIdentity) -> None:
    if not isinstance(identity, LogicalSnapshotIdentity):
        raise TypeError("path bindings require LogicalSnapshotIdentity values")
    if path in bindings and bindings[path] != identity:
        raise ValueError(f"conflicting logical identities for path {path!r}")
    bindings[path] = identity


class ConfiguredSnapshotIdentity:
    """Callable identity provider configured by local snapshot paths/prefixes.

    Pass this object as ``sim_identity`` to Calculator.run, EvalEngine or
    SampleTable. Multiple locations may declare the same logical identity.
    The mapping is copied; construct a new provider to change configuration.
    Unknown snapshots use the existing path/session fallback. External
    transformations remain session-only, including for configured snapshots.
    """

    def __init__(self, locations: Mapping[_Path, LogicalSnapshotIdentity]) -> None:
        bindings: dict[str, LogicalSnapshotIdentity] = {}
        for path, identity in locations.items():
            _bind(bindings, _location_path(path), identity)
        self._bindings = MappingProxyType(bindings)

    def __call__(self, sim: Any) -> tuple[Any, ...]:
        ancestor = getattr(sim, "ancestor", sim)
        if _has_transformations(sim) or _has_transformations(ancestor):
            return id_based_sim_identity(sim)
        filename: Any = getattr(ancestor, "filename", None)
        if _is_file_name(filename):
            identity = self._bindings.get(os.path.abspath(os.fspath(filename)))
            if identity is not None:
                return _snapshot_view_identity(sim, ancestor, identity.signature())
        return snapshot_identity(sim)

    def to_json(self) -> str:
        """Serialize explicit absolute locations and logical fields, without IO.

        Locations are machine-specific configuration. Change their bindings for
        another machine; logical fields and cache keys remain unchanged.
        """
        return json.dumps(
            {
                "schema": _CONFIG_SCHEMA,
                "entries": [
                    {"path": path, "identity": asdict(identity)} for path, identity in sorted(self._bindings.items())
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def from_json(text: str) -> ConfiguredSnapshotIdentity:
        """Load a validated configuration; relative locations and conflicts fail."""
        data = json.loads(text)
        if not isinstance(data, dict) or set(data) != {"schema", "entries"} or data["schema"] != _CONFIG_SCHEMA:
            raise ValueError("unsupported snapshot-identity configuration schema or fields")
        if not isinstance(data["entries"], list):
            raise ValueError("configuration entries must be a list")
        bindings: dict[str, LogicalSnapshotIdentity] = {}
        for entry in data["entries"]:
            if not isinstance(entry, dict) or set(entry) != {"path", "identity"}:
                raise ValueError("configuration entries require path and identity fields")
            path = entry["path"]
            _nonempty_string(path, "path")
            if not os.path.isabs(path):
                raise ValueError("configuration paths must be absolute")
            identity = entry["identity"]
            if not isinstance(identity, dict) or set(identity) != _IDENTITY_FIELDS:
                raise ValueError("invalid logical identity fields")
            _bind(bindings, _location_path(path), LogicalSnapshotIdentity(**identity))
        return ConfiguredSnapshotIdentity(bindings)
