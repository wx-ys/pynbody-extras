"""Catalogue metadata and exact, replayable particle memberships.

Only the explicitly qualified HaloNumberCatalogue adapter is supported initially.
Its already loaded membership vector is copied; no halo is materialized during
registration. Existing result-store identities remain exact particle identities,
so different catalogue namespaces may intentionally share the same computation.
"""

from __future__ import annotations

import hashlib
import json
import os
import weakref
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.halo import HaloCatalogue
from pynbody.halo.number_array import HaloNumberCatalogue

from pynbodyext.core.calculate.runtime.logical_identity import ConfiguredSnapshotIdentity
from pynbodyext.core.calculate.runtime.sim_identity import particle_index_digest, snapshot_identity
from pynbodyext.orm.catalogues import typed_object_key
from pynbodyext.orm.loading import parse_input_identity

from .models import CatalogueBinding, ObjectSelection
from .objects import Input, Results

if TYPE_CHECKING:
    from .database import Database

_SCHEMA = "pynbodyext.database.halo-number/v1"


def _number(key: str) -> Any:
    """Decode only the scalar shape emitted by the owning typed ID encoder."""
    try:
        payload = json.loads(key)
        part = payload[1][1][0]
        value = float.fromhex(part[1]) if part[0] == "float" else part[1]
        if typed_object_key(value) != key:
            raise ValueError("noncanonical catalogue ID")
        return value
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise ValueError("invalid stored catalogue ID") from exc


def _adapter(database: Database, catalogue_id: int) -> dict[str, Any]:
    database._check()
    with database._repo.session() as session:
        binding = session.get(CatalogueBinding, catalogue_id)
        if binding is None:
            raise NotImplementedError("catalogue has no registered particle replay adapter")
        data = json.loads(binding.adapter_json)
    if (
        type(data) is not dict
        or set(data) != {"schema", "array", "ignore", "memberships", "keys"}
        or data["schema"] != _SCHEMA
        or type(data["array"]) is not str
        or type(data["memberships"]) is not list
        or any(type(number) is not int for number in data["memberships"])
        or type(data["keys"]) is not list
        or any(type(number) is not int for number in data["keys"])
        or (data["ignore"] is not None and type(data["ignore"]) is not int)
    ):
        raise ValueError("invalid catalogue replay adapter")
    expected = sorted(set(data["memberships"]) - {data["ignore"]})
    if sorted(data["keys"]) != expected or len(set(data["keys"])) != len(data["keys"]):
        raise ValueError("catalogue IDs conflict with captured membership")
    return data


def _capture_catalogue(catalogue: HaloNumberCatalogue) -> str:
    """Copy the qualified adapter state without materializing any halo."""
    root = catalogue.base
    memberships = np.asarray(root[catalogue._array])
    if memberships.ndim != 1 or len(memberships) != len(root) or memberships.dtype.kind not in "iu":
        raise ValueError("catalogue adapter requires a one-dimensional integer membership array")
    keys = [int(number) for number in catalogue.keys()]
    ignore = None if catalogue._ignore is None else int(catalogue._ignore)
    expected = sorted({int(number) for number in memberships} - {ignore})
    if sorted(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError("catalogue membership changed since its creation; construct a new catalogue and revision")
    return json.dumps(
        {
            "schema": _SCHEMA,
            "array": catalogue._array,
            "ignore": ignore,
            "memberships": [int(number) for number in memberships],
            "keys": keys,
        },
        sort_keys=True,
    )


def register_catalogue(
    database: Database, snapshot: Input, catalogue: Any, *, name: str, revision: str | None = None
) -> Catalogue:
    """Register HaloNumberCatalogue metadata without iterating or loading halos.

    Capture its already loaded integer number array and ignore sentinel so an
    unmaterialized object can still be reloaded after the catalogue is gone.
    Other finders require their own qualified adapter, not guessed constructor
    defaults. Reusing a namespace with changed membership requires a new revision.
    """
    database._check()
    if not isinstance(snapshot, Input) or snapshot._database is not database or snapshot._row.parent_id is not None:
        raise ValueError("catalogue root must be a snapshot from this database")
    if not isinstance(catalogue, HaloCatalogue):
        raise TypeError("catalogue must be a supported pynbody HaloCatalogue")
    if type(catalogue) is not HaloNumberCatalogue:
        raise NotImplementedError(f"no qualified replay adapter for catalogue {type(catalogue).__name__}")
    root = catalogue.base
    if root is None or root.ancestor is not root:
        raise ValueError("catalogue must belong to an available root snapshot")
    description = database._description(snapshot.input_id)
    if os.path.abspath(os.fspath(root.filename)) not in {location.uri for location in snapshot._row.locations}:
        raise ValueError("catalogue root location is not registered for this snapshot")
    logical = parse_input_identity(description.signature).logical
    provider = ConfiguredSnapshotIdentity({root.filename: logical}) if logical else snapshot_identity
    if provider(root) != description.signature or len(root) != description.root_count:
        raise ValueError("catalogue root does not match registered snapshot")
    text = _capture_catalogue(catalogue)
    keys = json.loads(text)["keys"]
    row = database._repo.register_catalogue(
        snapshot.input_id, name, finder="pynbody.halo.number_array.HaloNumberCatalogue", revision=revision
    )
    with database._repo.session() as session, session.begin():
        binding = session.get(CatalogueBinding, row.id)
        if binding is None:
            session.add(CatalogueBinding(catalogue_id=row.id, adapter_json=text))
        elif binding.adapter_json != text:
            raise ValueError("catalogue membership changed; register a new catalogue revision")
    for number in keys:
        database._repo.register_catalogue_object(row.id, number)
    database._catalogue_live[row.id] = weakref.ref(catalogue)
    return Catalogue(database, row.id)


class Catalogues:
    """Registered catalogue namespaces belonging to one root snapshot."""

    def __init__(self, snapshot: Input) -> None:
        self._snapshot = snapshot

    def _rows(self) -> list[Any]:
        db = self._snapshot._database
        db._check()
        return db._repo.catalogues(self._snapshot.input_id)

    def __iter__(self) -> Any:
        return iter([Catalogue(self._snapshot._database, row.id) for row in self._rows()])

    def __len__(self) -> int:
        return len(self._rows())

    def __getitem__(self, key: str) -> Catalogue:
        rows = self._rows()
        exact = [row for row in rows if row.identity_key == key]
        matches = exact or [row for row in rows if row.catalogue_id == key]
        if not matches:
            raise KeyError(key)
        if len(matches) != 1:
            raise ValueError(f"ambiguous catalogue {key!r}; choose one of {[row.identity_key for row in matches]}")
        return Catalogue(self._snapshot._database, matches[0].id)

    def __repr__(self) -> str:
        rows = self._rows()
        return f"<{len(rows)} registered catalogues>\n" + "\n".join(
            f"{row.catalogue_id} | {row.finder} | revision={row.revision!r} | {row.identity_key}" for row in rows[:20]
        )


class Catalogue:
    """A scientific catalogue namespace; membership is loaded only on request."""

    def __init__(self, database: Database, catalogue_id: int) -> None:
        self._database = database
        self.catalogue_id = catalogue_id

    @property
    def _row(self) -> Any:
        self._database._check()
        return self._database._repo.get_catalogue(self.catalogue_id)

    @property
    def key(self) -> str:
        return self._row.identity_key

    @property
    def name(self) -> str:
        return self._row.catalogue_id

    @property
    def halos(self) -> Halos:
        return Halos(self)

    @property
    def subhalos(self) -> Halos:
        return Halos(self, kind="subhalo")

    def __repr__(self) -> str:
        row = self._row
        return f"<catalogue {row.catalogue_id!r} revision={row.revision!r} key={row.identity_key!r}>"


class Halos:
    """Lookup by original typed halo ID, never by catalogue row position."""

    def __init__(self, catalogue: Catalogue, *, kind: str = "halo", host_id: int | None = None) -> None:
        self._catalogue = catalogue
        self._kind = kind
        self._host_id = host_id

    def _rows(self) -> list[Any]:
        db = self._catalogue._database
        db._check()
        return db._repo.catalogue_objects(self._catalogue.catalogue_id, kind=self._kind, host_id=self._host_id)

    def __len__(self) -> int:
        return len(self._rows())

    def __iter__(self) -> Any:
        return iter(
            [Halo(self._catalogue, row.id, _number(row.typed_object_key), kind=self._kind) for row in self._rows()]
        )

    def __getitem__(self, number: Any) -> Halo:
        key = typed_object_key(number)
        for row in self._rows():
            if row.typed_object_key == key:
                return Halo(self._catalogue, row.id, _number(row.typed_object_key), kind=self._kind)
        raise KeyError(number)

    def __repr__(self) -> str:
        rows = self._rows()
        return f"<{len(rows)} registered {self._kind}s>\n" + "\n".join(
            f"{self._kind} {_number(row.typed_object_key)!r}" for row in rows[:20]
        )


class _HaloResults:
    """An unknown membership has no saved results, and querying never loads it."""

    def __init__(self, halo: Halo) -> None:
        self._halo = halo

    def __getitem__(self, key: Any) -> Any:
        source = self._halo._bound()
        if source is None:
            raise KeyError(key)
        return Results(source)[key]

    def __len__(self) -> int:
        source = self._halo._bound()
        return 0 if source is None else len(Results(source))

    def __repr__(self) -> str:
        source = self._halo._bound()
        return "<0 results; membership not yet materialized>" if source is None else repr(Results(source))


class Halo(Input):
    """A catalogue object which acquires an exact Input on first materialization."""

    def __init__(self, catalogue: Catalogue, object_id: int, number: Any, *, kind: str = "halo") -> None:
        self._database = catalogue._database
        self.catalogue = catalogue
        self.object_id = object_id
        self.number = number
        self.kind = kind

    @property
    def key(self) -> str:
        return f"{self.catalogue.key}/{self.kind}/{typed_object_key(self.number)}"

    @property
    def name(self) -> str:
        return f"{self.kind} {self.number}"

    @property
    def input_id(self) -> int:
        bound = self._bound()
        if bound is None:
            raise ValueError("halo has no exact input until its membership is materialized")
        return bound.input_id

    @property
    def subhalos(self) -> Halos:
        return Halos(self.catalogue, kind="subhalo", host_id=self.object_id)

    @property
    def results(self) -> _HaloResults:
        return _HaloResults(self)

    def _bound(self) -> Input | None:
        db = self._database
        db._check()
        with db._repo.session() as session:
            binding = session.get(ObjectSelection, self.object_id)
            input_id = None if binding is None else binding.input_id
        if input_id is None:
            return None
        desc = db._description(input_id)
        root = self.catalogue._row.root_input
        if (
            desc.root_signature != root.signature
            or parse_input_identity(desc.signature).kind not in ("halo", "subset")
            or (
                self.kind == "halo"
                and parse_input_identity(desc.signature).kind == "halo"
                and typed_object_key(desc.signature[-2]) != typed_object_key(self.number)
            )
        ):
            raise ValueError("catalogue object binding conflicts with its exact input")
        return Input(db, input_id)

    def _materialized(self) -> Input:
        bound = self._bound()
        if bound is not None:
            return bound
        if self.kind != "halo":
            raise NotImplementedError(
                "subhalo has no explicit particle selection; number-array adapters do not infer one"
            )
        db = self._database
        row = self.catalogue._row
        adapter = _adapter(db, row.id)
        root_desc = db._description(row.root_input_id)
        if root_desc.root_count != len(adapter["memberships"]):
            raise ValueError("catalogue membership root count changed; register a new revision")
        # Compare exact Python integers: coercing mixed uint64 IDs through a
        # float64 array can silently combine two distinct haloes.
        indices = tuple(i for i, number in enumerate(adapter["memberships"]) if number == self.number)
        if not indices:
            raise ValueError("registered halo is missing from captured membership")
        signature = root_desc.root_signature + ("halo", self.number, particle_index_digest(indices))
        data = db._record_input(replace(root_desc, signature=signature, indices=indices))
        _bind_object(db, self.object_id, data.id)
        return Input(db, data.id)

    def load(self, *, location_id: int | None = None) -> Any:
        return self._materialized().load(location_id=location_id)

    def __repr__(self) -> str:
        self._database._check()
        return f"<{self.kind} {self.number!r} catalogue={self.catalogue.name!r} key={self.key!r}>"


def _bind_object(database: Database, object_id: int, input_id: int) -> None:
    with database._repo.session() as session, session.begin():
        binding = session.get(ObjectSelection, object_id)
        if binding is None:
            session.add(ObjectSelection(object_id=object_id, input_id=input_id))
        elif binding.input_id != input_id:
            raise ValueError("catalogue membership changed; register a new revision")


def associate_halo(database: Database, raw_halo: Any, saved_input_id: int) -> bool:
    """Link a raw Halo only to explicitly registered, matching live catalogues.

    Return False when the catalogue is unknown; the exact view remains available
    through snapshot.selections. No scientific catalogue is inferred from a number.
    """
    catalogue = getattr(raw_halo, "_halo_catalogue", None)
    number = getattr(raw_halo, "_halo_number", None)
    if catalogue is None or number is None:
        return False
    number = number.item() if isinstance(number, np.generic) else number
    return associate_captured_halo(database, catalogue, number, saved_input_id)


def associate_captured_halo(database: Database, catalogue: Any, number: Any, saved_input_id: int) -> bool:
    """Associate a run-time catalogue reference and number after deferred save."""
    database._check()
    desc = database._description(saved_input_id)
    matched = False
    for catalogue_id, reference in list(database._catalogue_live.items()):
        if reference() is not catalogue:
            continue
        handle = Catalogue(database, catalogue_id)
        halo = handle.halos[number]
        adapter = _adapter(database, catalogue_id)
        expected = tuple(i for i, value in enumerate(adapter["memberships"]) if value == number)
        if (
            desc.root_signature != handle._row.root_input.signature
            or desc.indices != expected
            or parse_input_identity(desc.signature).kind != "halo"
            or typed_object_key(desc.signature[-2]) != typed_object_key(number)
        ):
            raise ValueError("saved halo differs from registered catalogue membership; use a new revision")
        _bind_object(database, halo.object_id, saved_input_id)
        matched = True
    return matched


def associate_saved_catalogue(database: Database, catalogue: Any, halo_number: Any, input_id: int) -> bool:
    """Associate a captured live catalogue, automatically describing known adapters.

    An automatic namespace uses the class display name and a content-derived
    revision of the copied membership description. Unknown adapters return False;
    callers can keep the exact view and report that catalogue metadata is absent.
    """
    if catalogue is None:
        return False
    halo_number = halo_number.item() if isinstance(halo_number, np.generic) else halo_number
    if associate_captured_halo(database, catalogue, halo_number, input_id):
        return True
    if type(catalogue) is not HaloNumberCatalogue:
        return False
    text = _capture_catalogue(catalogue)
    revision = hashlib.sha256(text.encode("utf-8")).hexdigest()
    data = database._repo.get_input(input_id)
    if data.parent_id is None:
        raise ValueError("catalogue association requires a saved halo view")
    register_catalogue(
        database, Input(database, data.parent_id), catalogue, name="HaloNumberCatalogue", revision=revision
    )
    return associate_captured_halo(database, catalogue, halo_number, input_id)
