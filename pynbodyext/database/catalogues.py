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
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.halo import HaloCatalogue
from pynbody.halo.number_array import HaloNumberCatalogue
from sqlalchemy import func, select, tuple_
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import aliased

from pynbodyext.core.calculate.runtime.logical_identity import ConfiguredSnapshotIdentity
from pynbodyext.core.calculate.runtime.persistence import capture_input
from pynbodyext.core.calculate.runtime.sim_identity import particle_index_digest, snapshot_identity
from pynbodyext.core.calculate.store.identity import sim_identity_key
from pynbodyext.orm.catalogues import typed_object_key
from pynbodyext.orm.loading import parse_input_identity
from pynbodyext.orm.models import Catalogue as CatalogueRow, CatalogueObject, DataObject

from .models import CatalogueBinding, ObjectSelection
from .objects import Input, Results

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

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


def catalogue_locator(database_id: str, catalogue_key: str, kind: str, typed_key: str) -> tuple[str, ...]:
    """Address an original object ID within the exact snapshot and catalogue revision."""
    return (database_id, "catalogue", catalogue_key, kind, typed_key)


def host_locators(
    session: Session, database_id: str, locators: set[tuple[str, ...]]
) -> dict[tuple[str, ...], tuple[str, ...]]:
    """Resolve known subhalo hosts in the same read transaction as the result versions."""
    from ._query_batch import matching_rows

    keys = [(loc[2], loc[4]) for loc in locators if len(loc) == 5 and loc[1] == "catalogue" and loc[3] == "subhalo"]
    host = aliased(CatalogueObject)
    address = tuple_(CatalogueRow.identity_key, CatalogueObject.typed_object_key)
    query = (
        select(CatalogueRow.identity_key, CatalogueObject.typed_object_key, host.typed_object_key)
        .select_from(CatalogueObject)
        .join(CatalogueRow, CatalogueRow.id == CatalogueObject.catalogue_fk)
        .join(host, host.id == CatalogueObject.host_id)
        .where(CatalogueObject.kind == "subhalo")
    )
    return {
        catalogue_locator(database_id, cat, "subhalo", sub): catalogue_locator(database_id, cat, "halo", parent)
        for cat, sub, parent in matching_rows(session, query, address, keys, width=2)
    }


@dataclass(frozen=True)
class CatalogueEntry:
    number: Any
    object_locator: tuple[str, ...]
    host_locator: tuple[str, ...] | None
    membership_known: bool
    input_key: str | None


class Halos:
    """Lookup by original typed halo ID, never by catalogue row position."""

    def __init__(self, catalogue: Catalogue, *, kind: str = "halo", host_id: int | None = None) -> None:
        self._catalogue = catalogue
        self._kind = kind
        self._host_id = host_id

    def _conditions(self) -> list[Any]:
        conditions = [CatalogueObject.catalogue_fk == self._catalogue.catalogue_id, CatalogueObject.kind == self._kind]
        if self._host_id is not None:
            conditions.append(CatalogueObject.host_id == self._host_id)
        return conditions

    def __len__(self) -> int:
        db = self._catalogue._database
        db._check()
        with db._repo.session() as session:
            return int(session.scalar(select(func.count()).select_from(CatalogueObject).where(*self._conditions())))

    def __iter__(self) -> Any:
        db = self._catalogue._database
        db._check()
        rows = db._repo.catalogue_objects(self._catalogue.catalogue_id, kind=self._kind, host_id=self._host_id)
        return (Halo(self._catalogue, row.id, _number(row.typed_object_key), kind=self._kind) for row in rows)

    def __getitem__(self, number: Any) -> Halo:
        typed_object_key(number)
        db = self._catalogue._database
        db._check()
        rows = db._repo.catalogue_objects(
            self._catalogue.catalogue_id, kind=self._kind, host_id=self._host_id, object_key=number
        )
        if not rows:
            raise KeyError(number)
        return Halo(self._catalogue, rows[0].id, number, kind=self._kind)

    def register(self, number: Any, *, particles: Any = None) -> Halo:
        """Declare a catalogue object and optionally bind its exact particle view.

        Use `halo.subhalos.register(number, particles=view)` for a known host.
        Omitting particles registers the catalogue entry with unknown membership.
        """
        db = self._catalogue._database
        db._check_write()
        typed_object_key(number)
        root = self._catalogue._row.root_input
        desc = None
        if particles is not None:
            if isinstance(particles, Input):
                if particles._database is not db:
                    raise ValueError("particle input belongs to another database")
                desc = (
                    particles._exact_description()
                    if isinstance(particles, Halo)
                    else db._description(particles.input_id)
                )
            else:
                logical = parse_input_identity(root.signature).logical
                root_locations = db._repo.get_input(root.id).locations
                provider = (
                    ConfiguredSnapshotIdentity({loc.uri: logical for loc in root_locations})
                    if logical
                    else snapshot_identity
                )
                desc = capture_input(particles, provider)
            if desc.indices is None or desc.root_signature != root.signature:
                raise ValueError("particles must be an exact selection of this catalogue's root snapshot")
            if self._kind == "halo" and parse_input_identity(desc.signature).kind == "halo":
                if typed_object_key(desc.signature[-2]) != typed_object_key(number):
                    raise ValueError("particle halo number differs from the catalogue number")
        if self._kind == "halo":
            with db._repo.session() as session:
                has_adapter = session.get(CatalogueBinding, self._catalogue.catalogue_id) is not None
            if has_adapter:
                adapter = _adapter(db, self._catalogue.catalogue_id)
                if type(number) is not int or number not in adapter["keys"]:
                    raise ValueError("halo number is absent from the registered catalogue revision")
                if desc is not None:
                    expected = tuple(i for i, value in enumerate(adapter["memberships"]) if value == number)
                    if desc.indices != expected:
                        raise ValueError("particle indices conflict with the registered catalogue revision")
                    # Preserve the adapter's existing calculation address when binding a plain particle slice.
                    signature = root.signature + ("halo", number, particle_index_digest(expected))
                    desc = replace(desc, signature=signature)
        row = db._repo.register_catalogue_object(
            self._catalogue.catalogue_id, number, kind=self._kind, host_id=self._host_id
        )
        if desc is not None:
            _bind_object(db, row.id, db._record_input(desc).id)
        return Halo(self._catalogue, row.id, number, kind=self._kind)

    def _inventory(
        self, *, limit: int | None = None, offset: int = 0, resolve_inputs: bool = False
    ) -> list[CatalogueEntry]:
        if type(offset) is not int or offset < 0 or (limit is not None and (type(limit) is not int or limit < 0)):
            raise ValueError("limit and offset must be nonnegative integers")
        cat = self._catalogue
        db = cat._database
        namespace, catalogue = db.database_id, cat._row
        host = aliased(CatalogueObject)
        query = (
            select(
                CatalogueObject.typed_object_key,
                host.typed_object_key,
                DataObject.identity_key,
                CatalogueBinding.catalogue_id,
            )
            .select_from(CatalogueObject)
            .outerjoin(host, host.id == CatalogueObject.host_id)
            .outerjoin(ObjectSelection, ObjectSelection.object_id == CatalogueObject.id)
            .outerjoin(DataObject, DataObject.id == ObjectSelection.input_id)
            .outerjoin(CatalogueBinding, CatalogueBinding.catalogue_id == CatalogueObject.catalogue_fk)
            .where(*self._conditions())
            .order_by(CatalogueObject.id)
            .limit(limit)
            .offset(offset)
        )
        with db._repo.session() as session:
            rows = session.execute(query).all()
        adapter_keys: dict[Any, str] = {}
        if resolve_inputs and self._kind == "halo":
            missing: dict[int, list[int]] = {
                _number(key): [] for key, _, exact, adapter_id in rows if exact is None and adapter_id is not None
            }
            if missing:
                for i, number in enumerate(_adapter(db, cat.catalogue_id)["memberships"]):
                    if number in missing:
                        missing[number].append(i)
                for number, indices in missing.items():
                    if indices:
                        signature = catalogue.root_input.signature + ("halo", number, particle_index_digest(indices))
                        adapter_keys[number] = sim_identity_key(signature)
        result = []
        for key, host_key, exact, adapter_id in rows:
            number = _number(key)
            result.append(
                CatalogueEntry(
                    number,
                    catalogue_locator(namespace, catalogue.identity_key, self._kind, key),
                    catalogue_locator(namespace, catalogue.identity_key, "halo", host_key) if host_key else None,
                    exact is not None or (self._kind == "halo" and adapter_id is not None),
                    exact or adapter_keys.get(number),
                )
            )
        return result

    def to_dataframe(self, *, limit: int | None = None, offset: int = 0) -> Any:
        """Read one stable page of catalogue metadata, including objects with unknown particles."""
        import pandas as pd

        rows = self._inventory(limit=limit, offset=offset)
        return pd.DataFrame(
            {
                "number": pd.Series([row.number for row in rows], dtype=object),
                "object_locator": [row.object_locator for row in rows],
                "host_locator": [row.host_locator for row in rows],
                "membership_known": [row.membership_known for row in rows],
                "input_key": [row.input_key for row in rows],
            }
        )

    def __repr__(self) -> str:
        db = self._catalogue._database
        db._check()
        query = (
            select(CatalogueObject.typed_object_key).where(*self._conditions()).order_by(CatalogueObject.id).limit(20)
        )
        with db._repo.session() as session:
            labels = [f"{self._kind} {_number(key)!r}" for key in session.scalars(query)]
        return f"<{len(self)} registered {self._kind}s>\n" + "\n".join(labels)


class _HaloResults:
    """An unknown membership has no saved results, and querying never loads it."""

    def __init__(self, halo: Halo) -> None:
        self._halo = halo

    def where(self, **predicates: Any) -> Any:
        """Discover saved outputs for this scientific object without a particle load."""
        return self._halo._database.results.where(objects=[self._halo], **predicates)

    def __getitem__(self, key: Any) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, key: Any) -> Any:
        """Read an exact saved result without materializing this object."""
        source = self._halo._bound()
        if source is not None:
            return Results(source).get(key)
        if self._halo.kind != "halo":
            return None
        try:
            desc = self._halo._exact_description()
        except NotImplementedError:
            return None
        db = self._halo._database
        source = db._registered_input(desc.signature)
        return None if source is None else Results(source).get(key)

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
    def object_locator(self) -> tuple[str, ...]:
        return catalogue_locator(
            self._database.database_id, self.catalogue.key, self.kind, typed_object_key(self.number)
        )

    @property
    def host(self) -> Halo | None:
        """The parent recorded for this subhalo in this exact catalogue revision."""
        self._database._check()
        host = self._database._repo.get_catalogue_object(self.object_id).host
        return None if host is None else Halo(self.catalogue, host.id, _number(host.typed_object_key))

    @property
    def host_locator(self) -> tuple[str, ...] | None:
        host = self.host
        return None if host is None else host.object_locator

    def _study_address(self) -> tuple[str, str, tuple[str, ...]]:
        key = sim_identity_key(self._exact_description().signature)
        return (
            self._database.database_id,
            key,
            ("catalogue", self.catalogue.key, self.kind, typed_object_key(self.number)),
        )

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

    def _exact_description(self) -> Any:
        """Resolve captured membership without loading or registering particles."""
        bound = self._bound()
        if bound is not None:
            return self._database._description(bound.input_id)
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
        return replace(root_desc, signature=signature, indices=indices)

    def _materialized(self) -> Input:
        bound = self._bound()
        if bound is not None:
            return bound
        db = self._database
        data = db._record_input(self._exact_description())
        _bind_object(db, self.object_id, data.id)
        return Input(db, data.id)

    def load(self, *, location_id: int | None = None) -> Any:
        return self._materialized().load(location_id=location_id)

    def __repr__(self) -> str:
        self._database._check()
        return f"<{self.kind} {self.number!r} catalogue={self.catalogue.name!r} key={self.key!r}>"


def _bind_object(database: Database, object_id: int, input_id: int) -> None:
    with database._repo.session() as session, session.begin():
        session.execute(insert(ObjectSelection).values(object_id=object_id, input_id=input_id).on_conflict_do_nothing())
        binding = session.get(ObjectSelection, object_id)
        if binding is None or binding.input_id != input_id:
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
