"""Metadata discovery for pinned scientific outputs and their observed particle uses."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from sqlalchemy import inspect, select
from sqlalchemy.dialects.sqlite import insert

from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.core.calculate.store.base import RecordRef
from pynbodyext.core.calculate.store.identity import sim_identity_key
from pynbodyext.core.calculate.study.table import SavedResultRef
from pynbodyext.orm.models import DataObject, Recipe

from ._query_batch import matching_rows
from .member_models import OutputObservation, SelectionOrigin, SelectionUse
from .models import RecipeOutput, SaveOperation
from .objects import Input, input_locator
from .recipe_queries import matches_calculator, matches_scope, matches_transforms, output_declarations, signature_of

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .database import Database
    from .members import Member, Origin

Path = tuple[str, ...]
Item = TypeVar("Item")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def index_recipe(session: Session, signature: CalculatorSignature) -> None:
    """Publish the recipe index atomically with the confirmed result metadata."""
    if session.get(RecipeOutput, (signature.hash, "[]")) is not None:
        return
    rows = [
        {
            "recipe_hash": signature.hash,
            "path_json": _json(path),
            "ordinal": index,
            "leaf": leaf,
            "declaration_json": _json(declaration),
        }
        for index, (path, leaf, declaration) in enumerate(output_declarations(signature))
    ]
    session.execute(insert(RecipeOutput).on_conflict_do_nothing(), rows)


class _Matches(Sequence[Item], Generic[Item]):
    def __init__(self, items: Iterable[Item]) -> None:
        self._items = tuple(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self._items)

    @overload
    def __getitem__(self, index: int) -> Item: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Item, ...]: ...

    def __getitem__(self, index: int | slice) -> Item | tuple[Item, ...]:
        return self._items[index]

    def one(self) -> Item:
        if len(self) != 1:
            raise ValueError(f"expected one match, found {len(self)}")
        return self._items[0]


@dataclass(frozen=True)
class MemberUse:
    """An exact membership consumed at a particular occurrence of a pinned output."""

    member: Member
    ref: SavedResultRef
    output_path: Path
    occurrence_path: Path
    role: str

    @property
    def origin(self) -> Origin:
        return self.member.origin

    def load(self, *, frame: str = "snapshot", location_id: int | None = None) -> Any:
        return self.member.load(frame=frame, location_id=location_id)

    @property
    def used_by(self) -> ResultHits:
        return self.member.used_by


class HitMembers(_Matches[MemberUse]):
    def __init__(self, items: Iterable[MemberUse], *, recorded: bool) -> None:
        super().__init__(items)
        self.recorded = recorded

    def where(self, *, role: str | None = None, filter: Any = None, occurrence_path: Path | None = None) -> HitMembers:
        rows = (
            item
            for item in self
            if (role is None or item.role == role)
            and (filter is None or matches_calculator(item.origin.filter_signature.payload, filter))
            and (occurrence_path is None or item.occurrence_path == occurrence_path)
        )
        return HitMembers(rows, recorded=self.recorded)

    def __repr__(self) -> str:
        return f"<{len(self)} member uses>" if self.recorded else "<membership unrecorded>"


@dataclass(frozen=True)
class ResultHit:
    """A metadata observation bound to one database, result version and named output."""

    _database: Database = field(repr=False, compare=False)
    object_locator: Path
    host_locator: Path | None
    input_key: str
    calculator_signature: CalculatorSignature = field(repr=False)
    output_path: Path
    ref: SavedResultRef
    _context_json: str | None = field(repr=False)
    occurrence_path: Path | None = None
    role: str | None = None
    operator: str | None = None
    origin: Origin | None = None

    @property
    def context_recorded(self) -> bool:
        return self._context_json is not None

    @property
    def context(self) -> Any:
        if self._context_json is None:
            return None
        from .member_storage import read_context

        self._database._check()
        with self._database._repo.session() as session:
            return read_context(session, self._context_json)

    @property
    def members(self) -> HitMembers:
        from .members import member_from_origins

        self._database._check()
        if not self.context_recorded:
            return HitMembers((), recorded=False)
        with self._database._repo.session() as session:
            rows = session.execute(
                select(SelectionUse, SelectionOrigin, DataObject.identity_key)
                .join(SelectionOrigin, SelectionOrigin.id == SelectionUse.origin_id)
                .join(DataObject, DataObject.id == SelectionOrigin.member_input_id)
                .where(
                    SelectionUse.version_id == self.ref.version_id,
                    SelectionUse.output_path_json == _json(self.output_path),
                )
                .order_by(SelectionUse.occurrence_path_json, SelectionUse.role, SelectionUse.origin_id)
            )
            return HitMembers(
                (
                    MemberUse(
                        member_from_origins(self._database, [origin], key=key),
                        self.ref,
                        self.output_path,
                        tuple(json.loads(use.occurrence_path_json)),
                        use.role,
                    )
                    for use, origin, key in rows
                ),
                recorded=True,
            )


class ResultHits(_Matches[ResultHit]):
    def __repr__(self) -> str:
        return f"<{len(self)} saved output matches>"

    def to_dataframe(self) -> Any:
        """Export one metadata row per match, including joinable scientific locators."""
        import pandas as pd

        columns = (
            "object_locator",
            "host_locator",
            "input_key",
            "calculator_signature",
            "output_path",
            "ref",
            "context_recorded",
            "occurrence_path",
            "role",
        )
        return pd.DataFrame.from_records(
            [tuple(getattr(hit, name) for name in columns) for hit in self], columns=columns
        )


def _objects(db: Database, objects: Any) -> dict[str, list[tuple[Path, Path | None]]] | None:
    from .catalogues import Halo, Halos

    if objects is None:
        return None
    namespace = db.database_id
    found: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    if isinstance(objects, Halos):
        if objects._catalogue._database is not db and objects._catalogue._database.database_id != namespace:
            raise ValueError("object collection belongs to another database")
        for entry in objects._inventory(resolve_inputs=True):
            if entry.input_key is not None:
                found[entry.input_key].append((entry.object_locator, entry.host_locator))
        return found
    owners = {db}
    seen_halos = set()
    seen_locators = set()
    for item in [objects] if isinstance(objects, Input) else objects:
        if not isinstance(item, Input):
            raise TypeError("objects must contain database input handles")
        if item._database not in owners:
            if item._database.database_id != namespace:
                raise ValueError("object belongs to another database")
            owners.add(item._database)
        if isinstance(item, Halo):
            handle = (item._database, item.object_id)
            if handle in seen_halos:
                continue
            seen_halos.add(handle)
            bound = item._bound()
            if bound is None:
                try:
                    key = sim_identity_key(item._exact_description().signature)
                except NotImplementedError:
                    continue
            else:
                key = bound.key
            address = (item.object_locator, item.host_locator)
        else:
            key = item.key
            # The standard locator can reuse the namespace already read for this collection.
            locator = (input_locator(namespace, key) if type(item).object_locator is Input.object_locator
                       else item.object_locator)
            address = (locator, None)
        if address[0] not in seen_locators:
            found[key].append(address)
            seen_locators.add(address[0])
    return found


def _indexed_declarations(session: Session, hashes: set[str]) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = defaultdict(list)
    if hashes and inspect(session.connection()).has_table(RecipeOutput.__tablename__):
        stmt = select(RecipeOutput).order_by(RecipeOutput.ordinal)
        for (row,) in matching_rows(session, stmt, RecipeOutput.recipe_hash, hashes):
            found[row.recipe_hash].append(
                (tuple(json.loads(row.path_json)), row.leaf, json.loads(row.declaration_json))
            )
    return found


@dataclass(frozen=True)
class _OutputSelection:
    """Recipe predicates are evaluated once per recipe, independently of the number of saved inputs."""

    path: Path | None = None
    filter: Any = None
    transform: Any = None
    scope: Any = None
    uses: Any = None

    def select(self, declarations: Iterable[Any], *, include_all_paths: bool) -> Iterator[tuple[Path, Any]]:
        for path, leaf, declaration in declarations:
            if not include_all_paths and (path != self.path if self.path is not None else not leaf):
                continue
            scopes = declaration["scopes"]
            direct = next((layer["filter"] for layer in reversed(scopes) if "filter" in layer), None)
            if self.filter is not None and not matches_calculator(direct, self.filter):
                continue
            if self.transform is not None:
                transforms = [step for layer in scopes for step in layer.get("transforms", ())]
                if not matches_transforms(transforms, self.transform):
                    continue
            if self.scope is not None and not matches_scope(scopes, self.scope):
                continue
            if self.uses is None:
                yield path, None
            else:
                for occurrence in declaration["occurrences"]:
                    if matches_calculator(occurrence["payload"], self.uses):
                        yield path, occurrence


def _selected_recipes(
    session: Session, hashes: set[str], selection: _OutputSelection, *, reverse: bool
) -> dict[str, Any]:
    index = _indexed_declarations(session, hashes)
    selected = {}
    for (row,) in matching_rows(session, select(Recipe), Recipe.hash, hashes):
        signature = CalculatorSignature.from_json(row.signature_text)
        if signature.hash != row.hash:
            raise ValueError("saved recipe metadata conflicts with its address")
        declarations = index[row.hash] if row.hash in index else output_declarations(signature)
        selected[row.hash] = (signature, list(selection.select(declarations, include_all_paths=reverse)))
    return selected


def _reverse_uses(session: Session, member: Member) -> dict[tuple[str, Path], list[Any]]:
    found: dict[tuple[str, Path], list[Any]] = defaultdict(list)
    if inspect(session.connection()).has_table(SelectionUse.__tablename__):
        origins = {origin.id: origin for origin in member.origins}
        for (use,) in matching_rows(session, select(SelectionUse), SelectionUse.origin_id, origins):
            found[use.version_id, tuple(json.loads(use.output_path_json))].append((use, origins[use.origin_id]))
        for uses in found.values():
            uses.sort(key=lambda pair: (pair[0].occurrence_path_json, pair[0].origin_id))
    return found


def _output_observations(session: Session, version_ids: list[str]) -> dict[tuple[str, Path], Any]:
    if not inspect(session.connection()).has_table(OutputObservation.__tablename__):
        return {}
    return {
        (row.version_id, tuple(json.loads(row.output_path_json))): row
        for (row,) in matching_rows(session, select(OutputObservation), OutputObservation.version_id, version_ids)
    }


def _matches_context(session: Session, observation: Any, requested: Any) -> bool:
    if requested is None:
        return True
    if observation is None or not requested.complete:
        return False
    from .member_storage import read_context

    actual = read_context(session, observation.context_json)
    return actual.complete and actual == requested


def _producer_locations(session: Session, namespace: str, observations: dict[Any, Any]) -> dict[str, tuple[Path, Path | None]]:
    from .catalogues import host_locators

    producers = {row.version_id: tuple(json.loads(row.source_locator_json))
                 for row in observations.values() if row.source_locator_json}
    hosts = host_locators(session, namespace, set(producers.values()))
    return {version: (locator, hosts.get(locator)) for version, locator in producers.items()}


def _saved_versions(
    session: Session,
    db: Database,
    *,
    ref: RecordRef | None,
    root_hash: str | None,
    input_keys: Collection[str] | None,
    member_versions: set[str] | None,
) -> Sequence[Any]:
    versions, current = db._store._versions, db._store._table
    stmt = select(
        versions.c.id,
        versions.c.version_id,
        versions.c.calculator_signature_hash,
        DataObject.identity_key,
        DataObject.identity_json,
    ).select_from(versions.join(DataObject, DataObject.identity_key == versions.c.sim_signature))
    if ref is not None:
        stmt = stmt.where(versions.c.version_id == ref.version_id, versions.c.record_key == ref.cache_key)
    elif member_versions is None:
        stmt = stmt.join(current, current.c.version_id == versions.c.version_id)
    if root_hash is not None:
        stmt = stmt.where(versions.c.calculator_signature_hash == root_hash)
    if member_versions is not None:
        rows = matching_rows(session, stmt, versions.c.version_id, member_versions)
    elif input_keys is not None:
        rows = matching_rows(session, stmt, DataObject.identity_key, input_keys)
    else:
        return session.execute(stmt.order_by(versions.c.id)).all()
    return sorted(rows, key=attrgetter("id"))


class ResultDiscovery:
    def __init__(self, database: Database) -> None:
        self.database = database

    def where(
        self,
        *,
        objects: Any = None,
        calculator: Any = None,
        ref: SavedResultRef | RecordRef | None = None,
        output: Path | None = None,
        transform: Any = None,
        filter: Any = None,
        scope: Any = None,
        uses: Any = None,
        context: Any = None,
    ) -> ResultHits:
        """Find current confirmed outputs, or select one exact historical receipt with ref."""
        db = self.database
        db._check()
        if output is not None and (not isinstance(output, tuple) or not all(isinstance(k, str) for k in output)):
            raise TypeError("output must be a tuple of named keys")
        if ref is not None:
            ref = db._owned_ref(ref)
            if ref.version_id is None:
                raise ValueError("result discovery requires a pinned version reference")
        return self._query(
            objects=_objects(db, objects),
            root_hash=signature_of(calculator).hash if calculator is not None else None,
            ref=ref,
            selection=_OutputSelection(output, filter, transform, scope, uses),
            context=context,
        )

    def _query(
        self,
        *,
        objects: Any = None,
        root_hash: str | None = None,
        ref: RecordRef | None = None,
        selection: _OutputSelection,
        context: Any = None,
        member: Member | None = None,
    ) -> ResultHits:
        db = self.database
        namespace = db.database_id
        with db._repo.session() as session:
            # SQLite's legacy transaction mode otherwise allows successive SELECTs to observe different refreshes.
            session.connection().exec_driver_sql("BEGIN")
            actual_uses = _reverse_uses(session, member) if member is not None else {}
            member_versions = {key[0] for key in actual_uses} if member is not None else None
            saved = _saved_versions(
                session, db, ref=ref, root_hash=root_hash, input_keys=objects, member_versions=member_versions
            )
            if not saved:
                return ResultHits(())
            hashes = {row.calculator_signature_hash for row in saved}
            recipes = _selected_recipes(session, hashes, selection, reverse=member is not None)
            observations = _output_observations(session, [s.version_id for s in saved])
            producers = _producer_locations(session, namespace, observations) if objects is None else {}
            pending = set(
                session.scalars(
                    select(SaveOperation.ref_json).where(
                        SaveOperation.completed.is_(False),
                        SaveOperation.reused.is_(False),
                        SaveOperation.ref_json.is_not(None),
                    )
                )
            )
            hits = []
            for row in saved:
                if row.calculator_signature_hash not in recipes:
                    continue
                recipe, paths = recipes[row.calculator_signature_hash]
                raw = RecordRef(tuple(json.loads(row.identity_json)), recipe.hash, version_id=row.version_id)
                if raw.to_json() in pending:
                    continue
                pinned = SavedResultRef(namespace, raw)
                for path, occurrence in paths:
                    if member is not None and (row.version_id, path) not in actual_uses:
                        continue
                    observed = observations.get((row.version_id, path))
                    if not _matches_context(session, observed, context):
                        continue
                    if objects is not None:
                        locations = objects[row.identity_key]
                    else:
                        locations = [producers.get(row.version_id, (input_locator(namespace, row.identity_key), None))]
                    for locator, host in locations:
                        hit = ResultHit(
                            db,
                            locator,
                            host,
                            row.identity_key,
                            recipe,
                            path,
                            pinned,
                            observed.context_json if observed is not None else None,
                        )
                        if member is not None:
                            for use, origin in actual_uses[row.version_id, path]:
                                hits.append(
                                    replace(
                                        hit,
                                        occurrence_path=tuple(json.loads(use.occurrence_path_json)),
                                        role=use.role,
                                        origin=origin,
                                    )
                                )
                        elif occurrence is not None:
                            hits.append(
                                replace(
                                    hit,
                                    occurrence_path=tuple(occurrence["path"]),
                                    role=occurrence["role"],
                                    operator=occurrence["operator"],
                                )
                            )
                        else:
                            hits.append(hit)
            return ResultHits(hits)


def used_by(member: Member) -> ResultHits:
    """Include each observed use of the selected origins, across confirmed versions."""
    return ResultDiscovery(member._database)._query(selection=_OutputSelection(), member=member)
