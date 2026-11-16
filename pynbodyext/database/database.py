"""A single local database for calculation results and reloadable inputs."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pynbody
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import URL

from pynbodyext.core.calculate.interfaces import DATABASE_CONTEXT, SavedInput, is_stable_sim_identity
from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.core.calculate.runtime.logical_identity import ConfiguredSnapshotIdentity, LogicalSnapshotIdentity
from pynbodyext.core.calculate.runtime.persistence import capture_input, execute
from pynbodyext.core.calculate.runtime.sim_identity import particle_index_digest, snapshot_identity
from pynbodyext.core.calculate.store import RecordRef, SQLAlchemyResultStore
from pynbodyext.core.calculate.store.base import ResultRecord, SignatureMismatchError, _check_record
from pynbodyext.core.calculate.store.identity import sim_identity_key
from pynbodyext.core.calculate.study.table import SavedResultRef
from pynbodyext.orm import AnalysisRepository
from pynbodyext.orm.loading import parse_input_identity
from pynbodyext.orm.models import DataLocation, DataObject, Recipe, Selection
from pynbodyext.orm.selections import SELECTION_SCHEMA, SelectionSpec, descriptor_key

from .models import (
    Base,
    DatabaseNamespace,
    InputInfo,
    ObjectSelection,
    ResultAlias,
    ResultSummary,
    SaveOperation,
    StudyDefinitionRow,
    StudyName,
)


def encode_input(value: SavedInput) -> str:
    return json.dumps(asdict(value), sort_keys=True)


def decode_input(text: str) -> SavedInput:
    value = json.loads(text)
    for key in ("signature", "root_signature", "unit_basis"):
        value[key] = tuple(value[key])
    if value["indices"] is not None:
        value["indices"] = tuple(value["indices"])
    return SavedInput(**value)


class DatabasePersistenceError(RuntimeError):
    """An exact receipt survives metadata failure and can be recovered explicitly."""

    def __init__(self, ref: RecordRef | None, operation: str) -> None:
        self.result_ref = ref
        self.operation_id = operation
        super().__init__(f"could not finish save {operation}; use database.recover(result_ref) when a receipt exists")


@dataclass(frozen=True)
class _CurrentResult:
    ref: RecordRef
    pending_operation: str | None

    def confirmed_ref(self) -> RecordRef:
        if self.pending_operation is not None:
            raise DatabasePersistenceError(self.ref, self.pending_operation)
        return self.ref


class _TrackedStore(SQLAlchemyResultStore):
    """Record the already-encoded receipt before the owning store commits bytes."""

    def __init__(self, engine: Any, repo_session: Any, *, create: bool) -> None:
        super().__init__(engine, create=create)
        self._session = repo_session
        self.pending: ContextVar[str | None] = ContextVar(f"save_{id(self)}", default=None)

    def _put(self, record: Any) -> RecordRef:
        operation_id = self.pending.get()
        if operation_id is not None:
            with self._session() as session, session.begin():
                op = session.get(SaveOperation, operation_id)
                if op is None:
                    raise ValueError("missing save operation")
                op.ref_json = record.ref.to_json()
                summary = json.loads(op.summary_json)
                summary.update(saved_at=record.created_at, calculator=record.pretty_key)
                op.summary_json = json.dumps(summary)
        try:
            return super()._put(record)
        except Exception as exc:
            if operation_id is not None:
                # The receipt was recorded before the payload transaction. A
                # failed commit acknowledgment leaves its durability unknown.
                raise DatabasePersistenceError(record.ref, operation_id) from exc
            raise


@dataclass
class _Invocation:
    """Tracks actual progress for a Study report independently of cache state."""

    started: bool = False
    run_id: str | None = None
    failed: bool = False


class Database:
    """Save, browse and reuse results in one local SQLite file.

    Parameters
    ----------
    path : path-like
        SQLite file; initialized automatically. No particle files are copied.
    project : str
        Namespace for explicitly registered logical simulations.
    read_only : bool
        Open an existing SQLite file without creating tables or metadata.
    """

    def __init__(self, path: str | os.PathLike[str], *, project: str = "default", read_only: bool = False) -> None:
        if not isinstance(project, str) or not project.strip():
            raise ValueError("project must be a nonempty string")
        self.project = project
        self.path = Path(path).absolute()
        self._closed = False
        self._read_only = read_only
        if read_only and not self.path.is_file():
            raise FileNotFoundError(self.path)
        url = (
            URL.create("sqlite", database=self.path.as_uri(), query={"mode": "ro", "uri": "true"})
            if read_only
            else URL.create("sqlite", database=str(self.path))
        )
        self._engine = create_engine(url)
        try:
            self._repo = AnalysisRepository(
                self._engine, allow_path_inputs=True, loaders={"default": pynbody.load}, create=not read_only
            )
            if not read_only:
                Base.metadata.create_all(self._engine)
                with self._engine.begin() as connection:
                    connection.execute(
                        insert(DatabaseNamespace).values(id=1, uuid=str(uuid.uuid4())).on_conflict_do_nothing()
                    )
            self._store = _TrackedStore(self._engine, self._repo.session, create=not read_only)
        except BaseException:
            self._engine.dispose()
            raise
        self._repo._stores["default"] = self._store
        self._catalogue_live: dict[int, Any] = {}

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("database is closed")

    @property
    def read_only(self) -> bool:
        return self._read_only

    def _check_write(self) -> None:
        self._check()
        if self.read_only:
            raise PermissionError("database is read-only")

    @property
    def database_id(self) -> str:
        """Persistent namespace shared by byte-for-byte copies of this database."""
        self._check()
        if not inspect(self._engine).has_table(DatabaseNamespace.__tablename__):
            raise RuntimeError("database schema lacks Study metadata; initialize with a writable Database")
        with self._repo.session() as session:
            namespace = session.get(DatabaseNamespace, 1)
            if namespace is None:
                raise RuntimeError("database schema lacks a Study namespace")
            return namespace.uuid

    @property
    def studies(self) -> Any:
        """Browse saved Study definitions."""
        from .study import Studies

        return Studies(self, self._study_namespace())

    def _study_namespace(self) -> str:
        namespace = self.database_id
        schema = inspect(self._engine)
        if not all(schema.has_table(model.__tablename__) for model in (StudyDefinitionRow, StudyName)):
            raise RuntimeError("database schema lacks Study definitions; initialize with a writable Database")
        return namespace

    def study(self, *, objects: Any, calculations: Any) -> Any:
        """Freeze an ordered grid of exact registered inputs and calculators."""
        from .study import Study

        return Study.create(self, objects, calculations)

    def __enter__(self) -> Database:
        self._check()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release connections; already decoded Result values remain usable."""
        self._engine.dispose()
        self._catalogue_live.clear()
        self._closed = True

    @contextmanager
    def use(self, *, mode: str = "auto") -> Any:
        """Temporarily select this database and a manual/auto/refresh policy."""
        self._check()
        if mode not in ("manual", "auto", "refresh"):
            raise ValueError("mode must be manual, auto or refresh")
        token = DATABASE_CONTEXT.set((self, mode))
        try:
            yield self
        finally:
            DATABASE_CONTEXT.reset(token)

    @property
    def snapshots(self) -> Any:
        from .objects import InputCollection

        return InputCollection(self, roots=True)

    @property
    def simulations(self) -> Any:
        from .objects import SimulationCollection

        return SimulationCollection(self)

    @property
    def runs(self) -> list[Any]:
        self._check()
        return self._repo.runs()

    def bind_loader(self, name: str, loader: Any) -> None:
        """Bind trusted project code taking one local root path; rebind on reopen."""
        self._check()
        if not isinstance(name, str) or not name.strip() or not callable(loader):
            raise ValueError("loader requires a nonempty name and callable")
        self._repo._loaders[name] = loader

    def _provider(self, sim: Any) -> Any:
        path = os.path.abspath(os.fspath(sim.ancestor.filename))
        with self._repo.session() as session:
            rows = list(session.scalars(select(DataObject).join(DataLocation).where(DataLocation.uri == path)))
        logical = [parse_input_identity(row.signature).logical for row in rows if row.timestep_id is not None]
        if len(logical) > 1:
            raise ValueError("ambiguous logical revision for path; select a database Snapshot")
        if logical:
            identity = logical[0]
            assert identity is not None
            return ConfiguredSnapshotIdentity({path: identity})
        return snapshot_identity

    def _record_input(self, desc: SavedInput) -> Any:
        self._check_write()
        if not is_stable_sim_identity(desc.signature) or desc.location is None:
            raise ValueError("persistent results require a file-backed stable input identity")
        parse_input_identity(desc.signature)
        root = self._repo.register_input(desc.root_signature, location=desc.location)
        data = self._repo.register_input(desc.signature)
        root_desc = replace(desc, signature=desc.root_signature, indices=None)
        with self._repo.session() as session, session.begin():
            for row, info in ((root, root_desc), (data, desc)):
                existing = session.get(InputInfo, row.id)
                text = encode_input(info)
                if existing is None:
                    session.add(InputInfo(input_id=row.id, description=text))
                    session.flush()
                elif decode_input(existing.description) != info:
                    old = decode_input(existing.description)
                    # Logical locations are aliases; captured replay facts may not drift.
                    if replace(old, location=info.location) != info:
                        raise ValueError("input replay description changed; register a new revision")
            if desc.indices is not None:
                spec = SelectionSpec.indices(list(desc.indices))
                key = descriptor_key(
                    root.id,
                    desc.root_count,
                    spec,
                    view_signature=desc.signature if parse_input_identity(desc.signature).kind == "halo" else None,
                )
                selection = session.scalar(select(Selection).where(Selection.descriptor_key == key))
                if selection is None:
                    session.add(
                        Selection(
                            root_input_id=root.id,
                            view_input_id=data.id,
                            schema=SELECTION_SCHEMA,
                            definition_json=spec.to_json(),
                            expected_count=desc.root_count,
                            descriptor_key=key,
                        )
                    )
                elif selection.view_input_id != data.id:
                    raise ValueError("selection descriptor points at a different input")
        return self._repo.get_input(data.id)

    def _description(self, input_id: int) -> SavedInput:
        self._check()
        with self._repo.session() as session:
            row = session.get(InputInfo, input_id)
            if row is None:
                raise ValueError("input has no reload description")
            desc = decode_input(row.description)
        data = self._repo.get_input(input_id)
        self._repo._validate_input_metadata(data)
        parsed = parse_input_identity(desc.signature)
        if parsed.root_signature != desc.root_signature or type(desc.root_count) is not int or desc.root_count < 0:
            raise ValueError("invalid input replay description")
        if data.signature != desc.signature:
            raise ValueError("input description conflicts with registered identity")
        if parsed.kind == "snapshot":
            if desc.indices is not None:
                raise ValueError("root input cannot contain a particle selection")
        else:
            if desc.indices is None or any(type(i) is not int or i < 0 or i >= desc.root_count for i in desc.indices):
                raise ValueError("invalid particle selection indices")
            if particle_index_digest(desc.indices) != desc.signature[-1]:
                raise ValueError("particle selection digest conflicts with input identity")
            if data.parent_id is None:
                raise ValueError("selection has no root input")
            root_desc = self._description(data.parent_id)
            if root_desc.root_count != desc.root_count or root_desc.signature != desc.root_signature:
                raise ValueError("selection root description conflicts with view")
            spec = SelectionSpec.indices(list(desc.indices))
            key = descriptor_key(
                data.parent_id, desc.root_count, spec, view_signature=desc.signature if parsed.kind == "halo" else None
            )
            with self._repo.session() as session:
                selection = session.scalar(select(Selection).where(Selection.descriptor_key == key))
            if selection is None or selection.view_input_id != data.id:
                raise ValueError("input has no matching immutable Selection definition")
            self._repo.get_selection(selection.id)
        return desc

    def register(
        self, sim: Any, *, simulation: str, snapshot: str, revision: str | None = None, loader: str = "default"
    ) -> Any:
        """Explicitly declare logical identity; existing path results are not rekeyed."""
        self._check()
        if sim.ancestor is not sim:
            raise ValueError("register requires a root snapshot")
        logical = LogicalSnapshotIdentity(self.project, simulation, snapshot, revision, loader)
        provider = ConfiguredSnapshotIdentity({sim.filename: logical})
        data = self._record_input(capture_input(sim, provider))
        from .objects import Input

        return Input(self, data.id)

    def _attach(self, result: Any, desc: SavedInput, signature: CalculatorSignature) -> Any:
        result._save_input = desc
        result._save_signature = signature.to_json()
        result._database = self
        return result

    def _alias_check(self, input_id: int, name: str | None, signature: CalculatorSignature) -> None:
        if name is None:
            return
        if not isinstance(name, str) or not name.strip():
            raise ValueError("result alias must be a nonempty string")
        with self._repo.session() as session:
            alias = session.get(ResultAlias, (input_id, name))
            if alias is not None and alias.calculator_hash != signature.hash:
                raise ValueError(f"result alias {name!r} already refers to a different calculator")

    def save_result(self, result: Any, *, name: str | None = None, _run_id: str | None = None) -> RecordRef:
        """Append an immutable result version using facts captured at run time."""
        self._check_write()
        desc = result._save_input
        if desc is None or result._save_signature is None:
            raise ValueError("result has no captured input identity; run the calculator before saving")
        if result.errors:
            raise ValueError("cannot save a failed calculation")
        signature = CalculatorSignature.from_json(result._save_signature)
        if not signature.stable:
            raise ValueError("persistent results require a stable calculator signature")
        data = self._record_input(desc)
        self._repo.register_recipe(signature)
        self._alias_check(data.id, name, signature)
        self._associate_catalogue(result, data.id)
        op_id = str(uuid.uuid4())
        with self._repo.session() as session, session.begin():
            session.add(
                SaveOperation(
                    id=op_id,
                    input_id=data.id,
                    signature_text=signature.to_json(),
                    name=name,
                    run_id=_run_id,
                    reused=False,
                    completed=False,
                    summary_json=json.dumps(
                        {
                            "type": type(result.value).__name__,
                            "shape": list(getattr(result.value, "shape", ())),
                            "units": str(getattr(result.value, "units", "")),
                        }
                    ),
                )
            )
        token = self._store.pending.set(op_id)
        try:
            ref = self._store.store(result, sim_signature=desc.signature, calculator_signature=signature)
        finally:
            self._store.pending.reset(token)
        result.store_ref = ref
        self._attach(result, desc, signature)
        try:
            self._finish_save(op_id)
        except Exception as exc:
            raise DatabasePersistenceError(ref, op_id) from exc
        return ref

    def _associate_catalogue(self, result: Any, input_id: int) -> None:
        if parse_input_identity(result._save_input.signature).kind != "halo":
            return
        from .catalogues import associate_saved_catalogue

        link = result._save_catalogue
        if link is not None and link[0]() is not None:
            if associate_saved_catalogue(self, link[0](), link[1], input_id):
                return
        with self._repo.session() as session:
            existing = session.scalar(select(ObjectSelection).where(ObjectSelection.input_id == input_id))
        if existing is None:
            import warnings

            warnings.warn(
                "Halo catalogue is unavailable or unsupported; saved as a replayable selection. "
                "Register a supported catalogue to browse it through halos.",
                UserWarning,
                stacklevel=3,
            )

    def _finish_save(self, operation_id: str) -> None:
        with self._repo.session() as session:
            op = session.get(SaveOperation, operation_id)
            if op is None or op.ref_json is None:
                raise ValueError("save has no exact receipt")
        ref = RecordRef.from_json(op.ref_json)
        data = self._repo.get_input(op.input_id)
        signature = CalculatorSignature.from_json(op.signature_text)
        self._repo._validate_input_metadata(data)
        self._description(data.id)
        with self._repo.session() as session:
            recipe = session.get(Recipe, signature.hash)
            if (
                recipe is None
                or CalculatorSignature.from_json(recipe.signature_text).identity() != signature.identity()
            ):
                raise ValueError("saved recipe metadata conflicts with operation")
        if ref.version_id is None or ref.cache_key != RecordRef(data.signature, signature.hash).cache_key:
            raise ValueError("saved receipt conflicts with input/recipe")
        if ref not in self._store:
            raise KeyError("saved payload is absent")
        self._alias_check(data.id, op.name, signature)
        if op.run_id is not None:
            self._repo.finish_run(op.run_id, ref, reused=op.reused)
        with self._repo.session() as session, session.begin():
            summary = session.get(ResultSummary, ref.version_id)
            if summary is None:
                session.add(
                    ResultSummary(
                        version_id=ref.version_id,
                        input_id=data.id,
                        calculator_hash=signature.hash,
                        ref_json=ref.to_json(),
                        summary_json=op.summary_json,
                    )
                )
            if op.name is not None:
                alias = session.get(ResultAlias, (data.id, op.name))
                if alias is None:
                    session.add(ResultAlias(input_id=data.id, name=op.name, calculator_hash=signature.hash))
                elif alias.calculator_hash != signature.hash:
                    raise ValueError("result alias conflict")
            stored = session.get(SaveOperation, operation_id)
            if stored is None:
                raise ValueError("save operation disappeared")
            stored.completed = True

    def recover(self, ref: RecordRef | SavedResultRef) -> None:
        """Finish only operations already associated with this exact saved receipt."""
        self._check_write()
        ref = self._owned_ref(ref)
        if ref.version_id is None:
            raise ValueError("recovery requires a pinned receipt")
        with self._repo.session() as session:
            ops = list(session.scalars(select(SaveOperation).where(SaveOperation.ref_json == ref.to_json())))
        if not ops:
            raise KeyError("receipt has no registered save operation in this database")
        for op in ops:
            self._finish_save(op.id)

    def _owned_ref(self, ref: RecordRef | SavedResultRef) -> RecordRef:
        if isinstance(ref, SavedResultRef):
            if ref.database_id != self.database_id:
                raise ValueError("result reference belongs to another database")
            return ref.record_ref
        return ref

    def fetch(self, ref: RecordRef | SavedResultRef) -> Any:
        """Read a pinned historical version without loading particles."""
        self._check()
        scoped = isinstance(ref, SavedResultRef)
        ref = self._owned_ref(ref)
        if scoped:
            record = self._store._read_ref(ref)
            if record is None:
                raise KeyError(f"no stored result for {ref.key!r}")
            result = record.to_result(self._store.codec, rebuild_calculator=False)
        else:
            result = self._store.fetch(ref)
        with self._repo.session() as session:
            data = session.scalar(
                select(DataObject).where(DataObject.identity_key == sim_identity_key(ref.sim_signature))
            )
        if data is None:
            raise KeyError("receipt input is not registered in this database")
        signature_text = result.provenance.calculator_signature_text if result.provenance else None
        if signature_text is None:
            raise ValueError("stored result has no calculator description")
        return self._attach(result, self._description(data.id), CalculatorSignature.from_json(signature_text))

    def versions(self, ref: RecordRef | SavedResultRef) -> list[RecordRef]:
        self._check()
        return self._store.list_versions(self._owned_ref(ref))

    def _registered_input(self, signature: tuple[Any, ...]) -> Any:
        from .objects import Input

        with self._repo.session() as session:
            pk = session.scalar(select(DataObject.id).where(DataObject.identity_key == sim_identity_key(signature)))
        return None if pk is None else Input(self, pk)

    def _current_refs(self, addresses: Any) -> dict[str, _CurrentResult]:
        """Select current pinned addresses in one SQLite read snapshot, without decoding."""
        pending = {ref.cache_key: ref for ref in addresses}
        if not pending:
            return {}
        keys = tuple(pending)
        current, versions = self._store._table, self._store._versions
        found = {}
        with self._engine.connect() as connection:
            # sqlite3 legacy transaction mode does not begin for SELECT.
            connection.exec_driver_sql("BEGIN")
            for start in range(0, len(keys), 500):
                rows = connection.execute(
                    select(current.c.record_key, current.c.version_id, versions.c.record_key.label("version_key"))
                    .select_from(current.outerjoin(versions, current.c.version_id == versions.c.version_id))
                    .where(current.c.record_key.in_(keys[start : start + 500]))
                )
                refs = {}
                for row in rows:
                    if row.record_key != row.version_key:
                        raise SignatureMismatchError("current cache pointer conflicts with its saved version")
                    refs[row.record_key] = replace(pending[row.record_key], version_id=row.version_id)
                if refs:
                    receipts = {ref.to_json(): key for key, ref in refs.items()}
                    operations = connection.execute(
                        select(SaveOperation.ref_json, SaveOperation.id).where(
                            SaveOperation.completed.is_(False),
                            SaveOperation.reused.is_(False),
                            SaveOperation.ref_json.in_(receipts),
                        )
                    )
                    unfinished = {receipts[row.ref_json]: row.id for row in operations}
                    found.update({key: _CurrentResult(ref, unfinished.get(key)) for key, ref in refs.items()})
        return found

    def _read_version(self, ref: RecordRef, signature: CalculatorSignature) -> ResultRecord:
        record = _check_record(self._store._read_ref(ref), signature)
        if record is None:
            raise KeyError(f"no stored result for {ref.key!r}")
        if not record.calculator_signature_text:
            raise SignatureMismatchError("stored result has no calculator signature")
        return record

    def _read_current(self, desc: SavedInput, signature: CalculatorSignature) -> ResultRecord | None:
        address = RecordRef(desc.signature, signature.hash)
        current = self._current_refs([address]).get(address.cache_key)
        return None if current is None else self._read_version(current.confirmed_ref(), signature)

    def _load(self, input_id: int, *, location_id: int | None = None) -> Any:
        desc = self._description(input_id)
        return self._load_description(desc, location_id=location_id)

    def _load_description(self, desc: SavedInput, *, location_id: int | None = None) -> Any:
        root = self._registered_input(desc.root_signature)
        if root is None:
            raise ValueError("input root is not registered")
        loaded = self._repo.load_input(root.input_id, location_id=location_id)
        if len(loaded.sim) != desc.root_count:
            raise ValueError("loaded root particle count changed; use a new revision")
        if desc.unit_basis:
            loaded.sim.physical_units(distance=desc.unit_basis[0], velocity=desc.unit_basis[1], mass=desc.unit_basis[2])
        sim = loaded.sim
        if desc.indices is not None:
            sim = SelectionSpec.indices(list(desc.indices)).materialize(loaded)
            parsed = parse_input_identity(desc.signature)
            if parsed.kind == "halo":
                sim._halo_number = desc.signature[-2]
        if loaded.sim_identity(sim) != desc.signature:
            raise ValueError("reloaded particle selection does not match its saved identity")
        return sim, loaded.sim_identity

    def _begin_invocation(self, data: Any, desc: SavedInput, signature: CalculatorSignature) -> Any:
        """Record the selected root location and exact selection before execution."""
        root_id = data.id if data.parent_id is None else data.parent_id
        with self._repo.session() as session:
            location = session.scalar(
                select(DataLocation).where(DataLocation.data_object_id == root_id, DataLocation.uri == desc.location)
            )
            selection = (
                None
                if desc.indices is None
                else session.scalar(
                    select(Selection).where(
                        Selection.view_input_id == data.id,
                        Selection.definition_json == SelectionSpec.indices(list(desc.indices)).to_json(),
                    )
                )
            )
        return self._repo.begin_run(
            data.id,
            signature.hash,
            store_name="default",
            input_location_id=None if location is None else location.id,
            selection_id=None if selection is None else selection.id,
        )

    def run_calculator(self, calculator: Any, source: Any, *, options: Any, mode: str) -> Any:
        """Coordinate lookup before IO, then delegate calculation to EvalEngine."""
        self._check()
        if mode != "manual":
            self._check_write()
        from .catalogues import Halo
        from .objects import Input

        if isinstance(source, Input):
            if source._database is not self:
                raise ValueError("input belongs to another database")
            desc = source._exact_description() if isinstance(source, Halo) else self._description(source.input_id)

            def compute() -> Any:
                return self._load_description(desc), calculator, options
        else:
            provider = self._provider(source)
            desc = capture_input(source, provider)

            def compute() -> Any:
                return (source, provider), calculator, options

        signature = calculator.to_signature()
        return self._coordinate(desc, signature, mode, compute, source=source)

    def _coordinate(
        self,
        desc: SavedInput,
        signature: CalculatorSignature,
        mode: str,
        compute: Any,
        *,
        source: Any,
        invocation: _Invocation | None = None,
        prepare: Any = None,
    ) -> Any:
        """Share the exact same lookup, EvalEngine, ResultStore and Run lifecycle."""
        if mode != "manual" and not signature.stable:
            raise ValueError("persistent results require a stable calculator signature")
        record = self._read_current(desc, signature) if mode == "auto" else None
        if record is not None:
            hit = record.to_result(self._store.codec, rebuild_calculator=invocation is None)
        elif prepare is not None:
            # A cache entry may have disappeared after whole-grid preflight.
            prepare()

        if mode == "manual":
            if invocation is not None:
                invocation.started = True
            (sim, provider), calculator, options = compute()
            manual_result = execute(calculator, sim, options, provider=provider)
            manual_result._database = self
            return manual_result

        from .catalogues import Halo, _bind_object

        data = self._record_input(desc)
        if isinstance(source, Halo):
            _bind_object(self, source.object_id, data.id)
        self._repo.register_recipe(signature)
        run = self._begin_invocation(data, desc, signature)
        if invocation is not None:
            invocation.started, invocation.run_id = True, run.id
        if record is not None:
            self._attach(hit, desc, signature)
            self._record_reuse(data.id, run.id, record, hit, signature)
            return hit

        try:
            (sim, provider), calculator, options = compute()
            result = execute(calculator, sim, options, provider=provider)
        except Exception as exc:
            self._repo.fail_run(run.id, exc)
            if invocation is not None:
                invocation.failed = True
            raise
        self._attach(result, desc, signature)
        if result.errors:
            self._repo.fail_run(run.id, RuntimeError(str(result.errors)))
            if invocation is not None:
                invocation.failed = True
            return result
        try:
            self.save_result(result, _run_id=run.id)
        except DatabasePersistenceError:
            raise
        except Exception as exc:
            # Preserve the direct API's existing failure recording. Study
            # distinguishes a persistence failure from a scientific failure.
            if invocation is None:
                self._repo.fail_run(run.id, exc)
            raise
        return result

    def _record_reuse(
        self, input_id: int, run_id: str, record: ResultRecord, result: Any, signature: CalculatorSignature
    ) -> None:
        op_id = str(uuid.uuid4())
        with self._repo.session() as session, session.begin():
            session.add(
                SaveOperation(
                    id=op_id,
                    input_id=input_id,
                    signature_text=signature.to_json(),
                    ref_json=record.ref.to_json(),
                    run_id=run_id,
                    reused=True,
                    completed=False,
                    summary_json=json.dumps(
                        {
                            "type": type(result.value).__name__,
                            "shape": list(getattr(result.value, "shape", ())),
                            "units": str(getattr(result.value, "units", "")),
                            "calculator": signature.pretty(),
                            "saved_at": record.created_at,
                        }
                    ),
                )
            )
        try:
            self._finish_save(op_id)
        except Exception as exc:
            raise DatabasePersistenceError(record.ref, op_id) from exc

    def register_catalogue(self, snap: Any, catalogue: Any, *, name: str, revision: str | None = None) -> Any:
        from .catalogues import register_catalogue

        self._check_write()

        return register_catalogue(self, snap, catalogue, name=name, revision=revision)

    def set_default_catalogue(self, snap: Any, name: str) -> None:
        self._check_write()
        if snap._database is not self:
            raise ValueError("snapshot belongs to another database")
        catalogue = snap.catalogues[name]
        with self._repo.session() as session, session.begin():
            info = session.get(InputInfo, snap.input_id)
            if info is None:
                raise ValueError("snapshot has no reload description")
            info.default_catalogue_id = catalogue.catalogue_id
