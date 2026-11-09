"""Metadata transactions around an existing calculator and independent result store.

A running row is committed before execution. Only the last metadata transaction
changes it to succeeded and inserts a pinned result link. Payload writes are not
part of that transaction; explicit recovery never guesses which version belongs
here. Query methods return the immediate run chain loaded; use session() for
collection navigation and arbitrary SQLAlchemy queries.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, create_engine, event, or_, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, joinedload, selectinload, sessionmaker

from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.core.calculate.runtime.logical_identity import LogicalSnapshotIdentity
from pynbodyext.core.calculate.store import RESULT_STORE_SCHEMA, RecordRef, ResultStore
from pynbodyext.core.calculate.store.identity import sim_identity_key

from .catalogues import CatalogueObjectKey, catalogue_identity_key, typed_object_key, validate_kind
from .environment import execution_context_json, validate_execution_metadata
from .loading import LoadedInput, SnapshotLoader, load_snapshot, validate_local_location
from .models import (
    Base,
    Catalogue,
    CatalogueObject,
    DataLocation,
    DataObject,
    Recipe,
    ResultLink,
    Run,
    Simulation,
    Timestep,
)
from .records import AssociationIssue, RecordedRun, RunExecutionError, RunPersistenceError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.result.result import Result
    from pynbodyext.core.calculate.runtime.options import RunOptions
    from pynbodyext.core.calculate.runtime.sim_identity import SimIdentityProvider


def _logical_input(
    value: LogicalSnapshotIdentity | tuple[Any, ...],
) -> tuple[tuple[Any, ...], LogicalSnapshotIdentity, str]:
    signature = value.signature() if isinstance(value, LogicalSnapshotIdentity) else value
    if type(signature) is not tuple or len(signature) < 7 or signature[:2] != ("sim", "pynbodyext.logical-snapshot/v1"):
        raise ValueError("ORM inputs require a configured logical snapshot identity")
    logical = LogicalSnapshotIdentity(*signature[2:7])
    suffix = signature[7:]
    kind = "snapshot"
    if suffix:
        if len(suffix) == 2 and suffix[0] == "subset":
            kind = "subset"
        elif len(suffix) == 3 and suffix[0] == "halo" and type(suffix[1]) in (str, int, float, bool):
            kind = "halo"
        else:
            raise ValueError("unsupported logical snapshot view suffix")
        if not isinstance(suffix[-1], str) or re.fullmatch(r"[0-9a-f]{16}", suffix[-1]) is None:
            raise ValueError("logical view requires the existing particle-index digest")
    sim_identity_key(signature)  # validate scalar types, including finite halo numbers
    return signature, logical, kind


def _nonempty(value: str, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _foreign_keys(connection: Any, record: Any, proxy: Any) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _run_query() -> Any:
    return select(Run).options(
        joinedload(Run.input).joinedload(DataObject.timestep).joinedload(Timestep.simulation),
        joinedload(Run.input).joinedload(DataObject.parent),
        joinedload(Run.input_location),
        joinedload(Run.recipe),
        joinedload(Run.result_link),
    )






def _catalogue_query() -> Any:
    return select(Catalogue).options(
        joinedload(Catalogue.root_input).joinedload(DataObject.timestep).joinedload(Timestep.simulation),
        joinedload(Catalogue.root_input).joinedload(DataObject.parent),
    )


def _catalogue_object_query() -> Any:
    return select(CatalogueObject).options(
        joinedload(CatalogueObject.catalogue)
        .joinedload(Catalogue.root_input)
        .joinedload(DataObject.timestep)
        .joinedload(Timestep.simulation),
        joinedload(CatalogueObject.catalogue).joinedload(Catalogue.root_input).joinedload(DataObject.parent),
        joinedload(CatalogueObject.host),
    )


class AnalysisRepository:
    """Relate scientific objects, recipes, runs and named-store result references.

    SQLite is the tested backend. Store and loader bindings are copied and kept
    in process; reopen with equivalent bindings. Loader callables accept one local
    absolute path/prefix and return a root SimSnap; pre-bind options in project
    code. No implicit loader, serialized callable or credentials stored in SQL.
    Explicitly importing this package requires the existing ``store`` extra.
    """

    def __init__(
        self,
        url_or_engine: str | Engine,
        *,
        stores: Mapping[str, ResultStore] | None = None,
        loaders: Mapping[str, SnapshotLoader] | None = None,
        create: bool = True,
    ) -> None:
        self.engine = url_or_engine if isinstance(url_or_engine, Engine) else create_engine(url_or_engine)
        if self.engine.dialect.name == "sqlite" and not event.contains(self.engine, "checkout", _foreign_keys):
            event.listen(self.engine, "checkout", _foreign_keys)
        self._stores = dict(stores or {})
        for name, store in self._stores.items():
            _nonempty(name, "store_name")
            if not isinstance(store, ResultStore):
                raise TypeError("store bindings must implement ResultStore")
        self._loaders = dict(loaders or {})
        for name, loader in self._loaders.items():
            _nonempty(name, "loader_id")
            if not callable(loader):
                raise TypeError("loader bindings must be callable")
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)
        if create:
            Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        """Open a caller-owned session; use a context manager and explicit commit."""
        return self._sessions()

    def _store(self, name: str) -> ResultStore:
        _nonempty(name, "store_name")
        if name not in self._stores:
            raise KeyError(f"result store {name!r} is not bound")
        return self._stores[name]

    def register_input(
        self, identity: LogicalSnapshotIdentity | tuple[Any, ...], *, location: str | None = None
    ) -> DataObject:
        """Idempotently register a logical root/view and optional root location."""
        signature, logical, kind = _logical_input(identity)
        if location is not None:
            _nonempty(location, "location")
        with self.session() as session, session.begin():
            simulation = session.scalar(
                select(Simulation).where(
                    Simulation.project_id == logical.project_id, Simulation.logical_id == logical.simulation_id
                )
            )
            if simulation is None:
                simulation = Simulation(project_id=logical.project_id, logical_id=logical.simulation_id)
                session.add(simulation)
                session.flush()
            timestep = session.scalar(
                select(Timestep).where(
                    Timestep.simulation_id == simulation.id, Timestep.logical_id == logical.timestep_id
                )
            )
            if timestep is None:
                timestep = Timestep(simulation=simulation, logical_id=logical.timestep_id)
                session.add(timestep)
                session.flush()
            root_sig = logical.signature()
            root = self._input_row(session, timestep, root_sig, "snapshot")
            data = root if kind == "snapshot" else self._input_row(session, timestep, signature, kind, root)
            if (
                location is not None
                and session.scalar(
                    select(DataLocation).where(DataLocation.data_object_id == root.id, DataLocation.uri == location)
                )
                is None
            ):
                session.add(DataLocation(data_object=root, uri=location))
            return data

    @staticmethod
    def _input_row(
        session: Session, timestep: Timestep, signature: tuple[Any, ...], kind: str, parent: DataObject | None = None
    ) -> DataObject:
        key = sim_identity_key(signature)
        data = session.scalar(select(DataObject).where(DataObject.identity_key == key))
        if data is None:
            data = DataObject(
                timestep=timestep, identity_key=key, identity_json=json.dumps(signature), kind=kind, parent=parent
            )
            session.add(data)
            session.flush()
        elif (
            sim_identity_key(data.signature) != key
            or data.timestep_id != timestep.id
            or data.kind != kind
            or data.parent_id != (None if parent is None else parent.id)
        ):
            raise ValueError("stored input metadata conflicts with its identity")
        return data

    def register_recipe(self, signature: CalculatorSignature) -> Recipe:
        """Register the existing stable DAG signature, not an alternative executor."""
        if not signature.stable:
            raise ValueError("ORM recipes must have a stable CalculatorSignature")
        with self.session() as session, session.begin():
            recipe = session.get(Recipe, signature.hash)
            if recipe is None:
                recipe = Recipe(hash=signature.hash, signature_text=signature.to_json())
                session.add(recipe)
            elif CalculatorSignature.from_json(recipe.signature_text).identity() != signature.identity():
                raise ValueError("stored recipe conflicts with its hash")
            return recipe

    def get_input(self, input_id: int) -> DataObject:
        """Return detached input metadata with hierarchy, parent and locations loaded."""
        with self.session() as session:
            data = session.scalar(
                select(DataObject)
                .where(DataObject.id == input_id)
                .options(
                    joinedload(DataObject.timestep).joinedload(Timestep.simulation),
                    joinedload(DataObject.parent),
                    selectinload(DataObject.locations),
                )
            )
            if data is None:
                raise KeyError(f"unknown input {input_id!r}")
            return data

    @staticmethod
    def _validate_catalogue(catalogue: Catalogue) -> None:
        root = catalogue.root_input
        signature = AnalysisRepository._validate_input_metadata(root)
        if root.kind != "snapshot":
            raise ValueError("catalogue root must be a registered root snapshot")
        key = catalogue_identity_key(
            signature, catalogue.catalogue_id, catalogue.finder, catalogue.revision, catalogue.loader_id
        )
        if catalogue.identity_key != key:
            raise ValueError("stored catalogue metadata conflicts with its identity")

    @staticmethod
    def _validate_catalogue_object(data: CatalogueObject) -> None:
        AnalysisRepository._validate_catalogue(data.catalogue)
        validate_kind(data.kind)
        if data.host_id is not None:
            host = data.host
            if data.kind != "subhalo" or host is None or host.kind != "halo" or host.catalogue_fk != data.catalogue_fk:
                raise ValueError("subhalo host must be a halo in the same catalogue")

    def register_catalogue(
        self,
        root_input_id: int,
        catalogue_id: str,
        *,
        finder: str,
        revision: str | None = None,
        loader_id: str = "default",
    ) -> Catalogue:
        """Idempotently register metadata for a validated root, without loading it.

        Scientific identity includes the root logical signature and all labels.
        An absent revision is part of this identity, not a SQL wildcard.
        """
        with self.session() as session, session.begin():
            root = session.get(DataObject, root_input_id)
            if root is None:
                raise KeyError(f"unknown input {root_input_id!r}")
            signature = self._validate_input_metadata(root)
            if root.kind != "snapshot":
                raise ValueError("catalogue root must be a registered root snapshot")
            key = catalogue_identity_key(signature, catalogue_id, finder, revision, loader_id)
            data = session.scalar(
                _catalogue_query().where(
                    or_(
                        Catalogue.identity_key == key,
                        and_(
                            Catalogue.root_input_id == root_input_id,
                            Catalogue.catalogue_id == catalogue_id,
                            Catalogue.finder == finder,
                            Catalogue.revision == revision,
                            Catalogue.loader_id == loader_id,
                        ),
                    )
                )
            )
            if data is None:
                data = Catalogue(
                    root_input=root,
                    identity_key=key,
                    catalogue_id=catalogue_id,
                    finder=finder,
                    revision=revision,
                    loader_id=loader_id,
                )
                session.add(data)
                session.flush()
            else:
                self._validate_catalogue(data)
                if data.root_input_id != root_input_id:
                    raise ValueError("stored catalogue root conflicts with its identity")
            catalogue_pk = data.id
        return self.get_catalogue(catalogue_pk)

    def get_catalogue(self, catalogue_pk: int) -> Catalogue:
        """Return detached metadata with root, parent and timestep/simulation loaded."""
        with self.session() as session:
            data = session.scalar(_catalogue_query().where(Catalogue.id == catalogue_pk))
            if data is None:
                raise KeyError(f"unknown catalogue {catalogue_pk!r}")
            self._validate_catalogue(data)
            return data

    def catalogues(self, root_input_id: int) -> list[Catalogue]:
        """List catalogues for a root by primary key, with get_catalogue relations."""
        with self.session() as session:
            rows = list(
                session.scalars(
                    _catalogue_query().where(Catalogue.root_input_id == root_input_id).order_by(Catalogue.id)
                )
            )
            for data in rows:
                self._validate_catalogue(data)
            return rows

    def register_catalogue_object(
        self, catalogue_pk: int, object_key: CatalogueObjectKey, *, kind: str = "halo", host_id: int | None = None
    ) -> CatalogueObject:
        """Register a typed object key; conflicting host metadata never reparents.

        Only subhalos can have hosts, and a supplied host must be a halo in the
        same catalogue. A subhalo without a known host is valid.
        """
        validate_kind(kind)
        key = typed_object_key(object_key)
        with self.session() as session, session.begin():
            catalogue = session.scalar(_catalogue_query().where(Catalogue.id == catalogue_pk))
            if catalogue is None:
                raise KeyError(f"unknown catalogue {catalogue_pk!r}")
            self._validate_catalogue(catalogue)
            if host_id is not None:
                host = session.get(CatalogueObject, host_id)
                if host is None:
                    raise KeyError(f"unknown catalogue host {host_id!r}")
                if kind != "subhalo" or host.kind != "halo" or host.catalogue_fk != catalogue_pk:
                    raise ValueError("subhalo host must be a halo in the same catalogue")
            data = session.scalar(
                _catalogue_object_query().where(
                    CatalogueObject.catalogue_fk == catalogue_pk,
                    CatalogueObject.kind == kind,
                    CatalogueObject.typed_object_key == key,
                )
            )
            if data is None:
                data = CatalogueObject(catalogue_fk=catalogue_pk, kind=kind, typed_object_key=key, host_id=host_id)
                session.add(data)
                session.flush()
            else:
                self._validate_catalogue_object(data)
                if data.host_id != host_id:
                    raise ValueError("stored catalogue object conflicts with requested host")
            object_pk = data.id
        return self.get_catalogue_object(object_pk)

    def get_catalogue_object(self, object_pk: int) -> CatalogueObject:
        """Return detached object with catalogue/root hierarchy and scalar host loaded."""
        with self.session() as session:
            data = session.scalar(_catalogue_object_query().where(CatalogueObject.id == object_pk))
            if data is None:
                raise KeyError(f"unknown catalogue object {object_pk!r}")
            self._validate_catalogue_object(data)
            return data

    def catalogue_objects(
        self, catalogue_pk: int, *, kind: str | None = None, host_id: int | None = None
    ) -> list[CatalogueObject]:
        """List objects by primary key; None omits a kind/host filter.

        Immediate relations match get_catalogue_object; collection navigation and
        unhosted-only queries use a caller-owned session.
        """
        stmt = (
            _catalogue_object_query().where(CatalogueObject.catalogue_fk == catalogue_pk).order_by(CatalogueObject.id)
        )
        if kind is not None:
            validate_kind(kind)
            stmt = stmt.where(CatalogueObject.kind == kind)
        if host_id is not None:
            stmt = stmt.where(CatalogueObject.host_id == host_id)
        with self.session() as session:
            rows = list(session.scalars(stmt))
            for data in rows:
                self._validate_catalogue_object(data)
            return rows








    def _prepare_load(
        self, input_id: int, location_id: int | None
    ) -> tuple[DataObject, DataLocation, LogicalSnapshotIdentity, SnapshotLoader]:
        data = self.get_input(input_id)
        signature = self._validate_input_metadata(data)
        _, logical, kind = _logical_input(signature)
        if kind != "snapshot":
            raise ValueError("only registered root snapshots can be loaded; view selection is not reconstructible")
        if logical.loader_id not in self._loaders:
            raise KeyError(f"snapshot loader {logical.loader_id!r} is not bound")
        location: DataLocation | None
        if location_id is None:
            if len(data.locations) != 1:
                raise ValueError("input needs exactly one location, or an explicit location_id")
            location = data.locations[0]
        else:
            location = next((item for item in data.locations if item.id == location_id), None)
            if location is None:
                raise ValueError("selected location does not belong to input")
        validate_local_location(location.uri)
        return data, location, logical, self._loaders[logical.loader_id]

    def load_input(self, input_id: int, *, location_id: int | None = None) -> LoadedInput:
        """Load a registered root using its loader_id binding, without creating a Run.

        One location is selected automatically; with multiple locations pass its
        database ``location_id`` explicitly. Local absolute paths/prefixes only.
        Errors propagate; no alternate location, retry or live-snapshot cache.
        """
        return load_snapshot(*self._prepare_load(input_id, location_id))

    def run_input(
        self,
        calculator: CalculatorBase,
        input_id: int,
        *,
        store_name: str,
        location_id: int | None = None,
        reuse: bool = True,
        options: RunOptions | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> RecordedRun:
        """Commit a Run, load a selected registered root, then execute the calculator.

        Invalid requests fail preflight. Loading/validation failures after Run
        creation are recorded like calculation errors. Even cache hits load the
        snapshot; result refresh and explicit recovery follow ``run`` semantics.
        ``execution_metadata`` records tool_versions/random_seed without seeding
        RNGs or changing keys; context describes this invocation, not its payload.
        """
        metadata = validate_execution_metadata(execution_metadata)
        backend = self._store(store_name)
        request = self._prepare_load(input_id, location_id)
        recipe = self.register_recipe(calculator.to_signature())
        run = self.begin_run(
            input_id, recipe.hash, store_name=store_name, input_location_id=request[1].id, execution_metadata=metadata
        )

        def execute() -> Result:
            loaded = load_snapshot(*request)
            return calculator.run(
                loaded.sim, store=backend, sim_identity=loaded.sim_identity, reuse=reuse, options=options
            )

        return self._execute_run(run, execute)

    def begin_run(
        self,
        input_id: int,
        recipe_hash: str,
        *,
        store_name: str,
        input_location_id: int | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> Run:
        """Commit running and its context before work starts.

        Unknown inputs/recipes/stores and invalid execution_metadata fail before
        creating a Run. Without an explicit location the context URI is null.
        """
        metadata = validate_execution_metadata(execution_metadata)
        self._store(store_name)
        run_id = str(uuid.uuid4())
        with self.session() as session, session.begin():
            data = session.get(DataObject, input_id)
            if data is None or session.get(Recipe, recipe_hash) is None:
                raise KeyError("input and recipe must be registered before beginning a run")
            location_owner = input_id
            location = None
            if input_location_id is not None:
                location = session.get(DataLocation, input_location_id)
                if location is None or location.data_object_id != location_owner:
                    raise ValueError("selected location does not belong to input")
            _, logical, _ = _logical_input(data.signature)
            context = execution_context_json(
                metadata,
                uri=None if location is None else location.uri,
                loader_id=logical.loader_id,
                revision=logical.revision,
            )
            session.add(
                Run(
                    id=run_id,
                    input_id=input_id,
                    input_location_id=input_location_id,
                    recipe_hash=recipe_hash,
                    store_name=store_name,
                    status="running",
                    started_at=time.time(),
                    _execution_context_json=context,
                )
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Run:
        with self.session() as session:
            run = session.scalar(_run_query().where(Run.id == run_id))
            if run is None:
                raise KeyError(f"unknown run {run_id!r}")
            return run

    def runs(
        self, *, input_id: int | None = None, recipe_hash: str | None = None, status: str | None = None
    ) -> list[Run]:
        stmt = _run_query().order_by(Run.started_at, Run.id)
        if input_id is not None:
            stmt = stmt.where(Run.input_id == input_id)
        if recipe_hash is not None:
            stmt = stmt.where(Run.recipe_hash == recipe_hash)
        if status is not None:
            if status not in ("running", "succeeded", "failed"):
                raise ValueError("unknown run status")
            stmt = stmt.where(Run.status == status)
        with self.session() as session:
            return list(session.scalars(stmt))

    @staticmethod
    def _validate_input_metadata(data: DataObject) -> tuple[Any, ...]:
        signature, logical, kind = _logical_input(data.signature)
        timestep = data.timestep
        simulation = timestep.simulation
        if (
            sim_identity_key(signature) != data.identity_key
            or data.kind != kind
            or timestep.logical_id != logical.timestep_id
            or simulation.logical_id != logical.simulation_id
            or simulation.project_id != logical.project_id
        ):
            raise ValueError("stored input hierarchy conflicts with its logical identity")
        if kind == "snapshot":
            if data.parent_id is not None:
                raise ValueError("root snapshot cannot have a parent")
        else:
            parent = data.parent
            if (
                parent is None
                or parent.parent_id is not None
                or parent.kind != "snapshot"
                or parent.timestep_id != data.timestep_id
                or sim_identity_key(parent.signature) != sim_identity_key(logical.signature())
                or parent.identity_key != sim_identity_key(logical.signature())
            ):
                raise ValueError("view parent conflicts with its logical root")
        return signature

    @staticmethod
    def _validate_run_metadata(run: Run) -> tuple[Any, ...]:
        signature = AnalysisRepository._validate_input_metadata(run.input)
        location_owner = run.input_id
        if run.input_location_id is not None:
            location = run.input_location
            if location is None or location.data_object_id != location_owner:
                raise ValueError("stored run location conflicts with its input")
        recipe = CalculatorSignature.from_json(run.recipe.signature_text)
        if not recipe.stable or recipe.hash != run.recipe_hash:
            raise ValueError("stored recipe metadata is inconsistent")
        return signature

    def _validate_ref(self, run: Run, ref: RecordRef) -> ResultStore:
        if ref.version_id is None:
            raise ValueError("ORM result associations require a pinned version")
        signature = self._validate_run_metadata(run)
        if ref.cache_key != RecordRef(signature, run.recipe_hash).cache_key:
            raise ValueError("result reference does not match run input/recipe")
        backend = self._store(run.store_name)
        if ref not in backend:  # metadata/identity validation, no value decode
            raise KeyError(f"missing saved version {ref.version_id!r}")
        return backend

    def finish_run(self, run_id: str, ref: RecordRef, *, reused: bool | None = None) -> Run:
        """Explicitly associate a matching saved version; never choose latest."""
        if reused is not None and type(reused) is not bool:
            raise ValueError("reused must be bool or None")
        run = self.get_run(run_id)
        self._validate_ref(run, ref)
        with self.session() as session, session.begin():
            changed = session.execute(
                update(Run)
                .where(Run.id == run_id, Run.status == "running")
                .values(status="succeeded", finished_at=time.time(), reused=reused)
                .returning(Run.id)
            ).scalar_one_or_none()
            if changed is not None:
                session.add(ResultLink(run_id=run_id, ref_json=ref.to_json(), store_schema=RESULT_STORE_SCHEMA))
            else:
                terminal = session.get(Run, run_id)
                link = session.get(ResultLink, run_id)
                if (
                    terminal is None
                    or terminal.status != "succeeded"
                    or link is None
                    or link.store_schema != RESULT_STORE_SCHEMA
                    or RecordRef.from_json(link.ref_json) != ref
                    or (reused is not None and terminal.reused != reused)
                ):
                    raise ValueError("run is terminal; conflicting completion is not allowed")
        return self.get_run(run_id)

    def fail_run(self, run_id: str, error: Exception) -> Run:
        with self.session() as session, session.begin():
            changed = session.execute(
                update(Run)
                .where(Run.id == run_id, Run.status == "running")
                .values(
                    status="failed", finished_at=time.time(), error_type=type(error).__name__, error_message=str(error)
                )
                .returning(Run.id)
            ).scalar_one_or_none()
            if changed is None:
                raise ValueError("only a running run can fail")
        return self.get_run(run_id)

    def run(
        self,
        calculator: CalculatorBase,
        sim: Any,
        *,
        sim_identity: SimIdentityProvider,
        store_name: str,
        reuse: bool = True,
        options: RunOptions | None = None,
        execution_metadata: Mapping[str, Any] | None = None,
    ) -> RecordedRun:
        """Track an invocation with copied tool_versions/random_seed metadata.

        Context records this invocation even on a cache hit. No managed location
        is selected for a caller-supplied snapshot, so context URI is null.
        """
        metadata = validate_execution_metadata(execution_metadata)
        backend = self._store(store_name)
        signature = tuple(sim_identity(sim))
        ancestor = getattr(sim, "ancestor", sim)
        filename = getattr(ancestor, "filename", None)
        location = None
        if filename:
            path = os.fspath(filename)
            if not (path.startswith("<") and path.endswith(">")):
                location = os.path.abspath(path)
        data = self.register_input(signature, location=location)
        recipe = self.register_recipe(calculator.to_signature())
        run = self.begin_run(data.id, recipe.hash, store_name=store_name, execution_metadata=metadata)
        return self._execute_run(
            run, lambda: calculator.run(sim, store=backend, sim_identity=sim_identity, reuse=reuse, options=options)
        )

    def _execute_run(self, run: Run, execute: Callable[[], Result]) -> RecordedRun:
        """Complete an already committed run, preserving one failure/recovery path."""
        try:
            result = execute()
        except Exception as exc:
            try:
                self.fail_run(run.id, exc)
            except Exception as metadata_error:
                raise RunPersistenceError(run.id) from metadata_error
            raise RunExecutionError(run.id) from exc
        if result.errors:
            try:
                failed = self.fail_run(run.id, RuntimeError(str(result.errors)))
            except Exception as exc:
                raise RunPersistenceError(run.id, result=result) from exc
            return RecordedRun(failed, result)
        try:
            if result.store_ref is None:
                raise ValueError("successful tracked computation did not return a saved version")
            completed = self.finish_run(run.id, result.store_ref, reused=result.from_store)
        except Exception as exc:
            raise RunPersistenceError(run.id, result_ref=result.store_ref, result=result) from exc
        return RecordedRun(completed, result)

    @staticmethod
    def _link_ref(run: Run) -> RecordRef:
        if run.status != "succeeded" or run.result_link is None:
            raise ValueError("run has no completed result association")
        if run.result_link.store_schema != RESULT_STORE_SCHEMA:
            raise ValueError("unsupported linked result-store schema")
        try:
            return RecordRef.from_json(run.result_link.ref_json)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid stored result reference") from exc

    def fetch_result(self, run_id: str) -> Result:
        run = self.get_run(run_id)
        ref = self._link_ref(run)
        return self._validate_ref(run, ref).fetch(ref)

    def audit(self) -> list[AssociationIssue]:
        """Inspect associations without decoding values or mutating either store."""
        issues: list[AssociationIssue] = []
        for run in self.runs():
            if run.status == "running":
                issues.append(AssociationIssue(run.id, "unfinished", "explicit inspection/recovery required"))
                continue
            if run.status == "failed":
                if run.result_link is not None:
                    issues.append(AssociationIssue(run.id, "invalid_link", "failed run has a result link"))
                continue
            if run.result_link is None:
                issues.append(AssociationIssue(run.id, "missing_link", "succeeded run lacks a result link"))
                continue
            if run.store_name not in self._stores:
                issues.append(AssociationIssue(run.id, "missing_store", run.store_name))
                continue
            try:
                self._validate_ref(run, self._link_ref(run))
            except KeyError as exc:
                issues.append(AssociationIssue(run.id, "missing_payload", str(exc)))
            except (ValueError, TypeError) as exc:
                issues.append(AssociationIssue(run.id, "invalid_link", str(exc)))
            except Exception as exc:
                issues.append(AssociationIssue(run.id, "unreadable_payload", str(exc)))
        return issues

    def unlinked_versions(self, run_id: str) -> list[RecordRef]:
        """Matching unreferenced versions are candidates, not proof of run ownership."""
        run = self.get_run(run_id)
        backend = self._store(run.store_name)
        address = RecordRef(self._validate_run_metadata(run), run.recipe_hash)
        with self.session() as session:
            linked = {
                RecordRef.from_json(text)
                for text in session.scalars(
                    select(ResultLink.ref_json).join(ResultLink.run).where(Run.store_name == run.store_name)
                )
            }
        return [ref for ref in backend.list_versions(address) if ref not in linked]
