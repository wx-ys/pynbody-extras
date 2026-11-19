"""Database-owned Study definitions and bindings."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from pynbodyext.core.calculate.interfaces import DATABASE_CONTEXT
from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.nodes.parameter_state import parameters_changed
from pynbodyext.core.calculate.result.result import ErrorInfo
from pynbodyext.core.calculate.result.signature import (
    CalculatorSignature,
    calculator_from_signature,
    calculator_to_signature,
)
from pynbodyext.core.calculate.store.base import RecordRef
from pynbodyext.core.calculate.store.identity import sim_identity_key
from pynbodyext.core.calculate.study.definition import (
    ObjectRef,
    StudyDefinition,
    canonical_json,
    check_name,
    unique_fields,
)
from pynbodyext.core.calculate.study.table import CellSelector, SavedResultRef, StudyCell, StudyTable
from pynbodyext.orm.jobs import _trusted_payload
from pynbodyext.orm.models import Catalogue as CatalogueRow, CatalogueObject, DataObject, Recipe

from .catalogues import Catalogue, Halo, _number
from .database import DatabasePersistenceError, _Invocation
from .models import StudyDefinitionRow, StudyName
from .objects import Input

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pynbodyext.core.calculate.interfaces import SavedInput

    from .database import Database, _CurrentResult


def _description(obj: Input) -> SavedInput:
    if isinstance(obj, Halo):
        return obj._exact_description()
    return obj._database._description(obj.input_id)


def _object_ref(db_id: str, obj: Input) -> ObjectRef:
    desc = _description(obj)
    locator = (
        ("catalogue", obj.catalogue.key, obj.kind, sim_identity_key((obj.number,)))
        if isinstance(obj, Halo)
        else ("input", obj.key)
    )
    return ObjectRef(db_id, desc.signature, locator, obj.name)


def _resolve(db: Database, db_id: str, ref: ObjectRef) -> Input:
    if ref.database_id != db_id:
        raise ValueError("study reference belongs to another database")
    with db._repo.session() as session:
        if ref.locator[0] == "input":
            pk = session.scalar(select(DataObject.id).where(DataObject.identity_key == ref.key))
            if pk is None:
                raise KeyError(f"study input is not registered: {ref.key}")
            obj: Input = Input(db, pk)
        else:
            _, catalogue_key, kind, typed_id = ref.locator
            row = session.execute(
                select(CatalogueObject.id, CatalogueRow.id)
                .join(CatalogueRow, CatalogueRow.id == CatalogueObject.catalogue_fk)
                .where(
                    CatalogueRow.identity_key == catalogue_key,
                    CatalogueObject.kind == kind,
                    CatalogueObject.typed_object_key == typed_id,
                )
            ).one_or_none()
            if row is None:
                raise KeyError(f"study catalogue object is not registered: {ref.locator}")
            obj = Halo(Catalogue(db, row[1]), row[0], _number(typed_id), kind=kind)
    if sim_identity_key(_description(obj).signature) != ref.key:
        raise ValueError("study locator no longer resolves to the exact input identity")
    return obj


class _BindingChanged(ValueError):
    """The calculator no longer matches the saved request."""


def _verify_signature(calculator: CalculatorBase, signature: CalculatorSignature, label: str) -> None:
    actual = calculator_to_signature(calculator)
    if parameters_changed(calculator) or not actual.stable or not actual.same_identity(signature):
        raise _BindingChanged(f"calculator parameters could not be verified against the frozen signature for {label!r}")


@dataclass(frozen=True, slots=True)
class StudyPlanCell:
    selector: CellSelector
    action: str
    reason: str


class StudyExecutionError(RuntimeError):
    """An ordinary cell failure with the partial invocation report."""

    def __init__(self, report: StudyTable, coordinate: CellSelector, cause: BaseException) -> None:
        self.report = report
        self.coordinate = coordinate
        super().__init__(f"Study cell {coordinate.label!r} failed: {cause}")


class StudyPersistenceError(RuntimeError):
    """An unconfirmed Study invocation with its separate recovery receipt."""

    def __init__(self, report: StudyTable, coordinate: CellSelector, run_id: str | None, cause: Exception) -> None:
        self.report = report
        self.coordinate = coordinate
        self.run_id = run_id
        self.receipt = cause if isinstance(cause, DatabasePersistenceError) else None
        self.result_ref = None if self.receipt is None else self.receipt.result_ref
        super().__init__(f"Study stopped at {coordinate.label!r}: {cause}")


class Studies:
    """Named immutable definitions in the owning database."""

    def __init__(self, database: Database, database_id: str) -> None:
        self._database = database
        self._database_id = database_id

    def __iter__(self) -> Any:
        self._database._check()
        with self._database._repo.session() as session:
            return iter(list(session.scalars(select(StudyName.name).order_by(StudyName.name))))

    def __getitem__(self, name: str) -> Study:
        self._database._check()
        with self._database._repo.session() as session:
            row = session.execute(
                select(StudyDefinitionRow.digest, StudyDefinitionRow.definition_json)
                .join(StudyName, StudyName.digest == StudyDefinitionRow.digest)
                .where(StudyName.name == name)
            ).one_or_none()
        if row is None:
            raise KeyError(name)
        definition = StudyDefinition.from_dict(json.loads(row.definition_json, object_pairs_hook=unique_fields))
        if definition.id != row.digest or definition.database_id != self._database_id:
            raise ValueError("saved Study definition integrity or ownership mismatch")
        return Study(self._database, definition)


class Study:
    """A frozen input × calculation request bound to its owning Database."""

    def __init__(
        self, database: Database, definition: StudyDefinition, bindings: Mapping[str, CalculatorBase] | None = None
    ) -> None:
        self._database = database
        self._definition = definition
        self._bindings = dict(bindings or {})
        self._check_bindings(self._bindings)

    @classmethod
    def create(
        cls,
        db: Database,
        objects: Iterable[Input | ObjectRef],
        calculations: Mapping[str, CalculatorBase | CalculatorSignature],
    ) -> Study:
        database_id = db._study_namespace()
        refs = []
        for source in objects:
            if isinstance(source, ObjectRef):
                _resolve(db, database_id, source)
                refs.append(source)
            elif isinstance(source, Input) and source._database is db:
                refs.append(_object_ref(database_id, source))
            else:
                raise ValueError("study objects must be registered handles or references from this database")
        columns = []
        bindings = {}
        for label, calculator in calculations.items():
            if isinstance(calculator, CalculatorSignature):
                signature = calculator
            elif isinstance(calculator, CalculatorBase):
                signature = calculator_to_signature(calculator)
                bindings[label] = calculator
            else:
                raise TypeError("study calculations must be calculators or CalculatorSignature values")
            columns.append((label, signature.to_json(full=True)))
        return cls(db, StudyDefinition(database_id, tuple(refs), tuple(columns)), bindings)

    @property
    def objects(self) -> tuple[ObjectRef, ...]:
        return self._definition.objects

    @property
    def calculations(self) -> Mapping[str, CalculatorSignature]:
        return self._definition.calculations

    @property
    def definition_id(self) -> str:
        return self._definition.id

    def _check_bindings(self, bindings: Mapping[str, CalculatorBase]) -> None:
        signatures = self.calculations
        for label, calculator in bindings.items():
            if label not in signatures:
                raise KeyError(f"unknown study column {label!r}")
            if not isinstance(calculator, CalculatorBase):
                raise TypeError("study bindings must be calculators")
            _verify_signature(calculator, signatures[label], label)

    def bind_calculators(self, calculations: Mapping[str, CalculatorBase]) -> Study:
        """Return a binding of the same definition to exact live calculators."""
        bindings = {**self._bindings, **calculations}
        self._check_bindings(bindings)
        return Study(self._database, self._definition, bindings)

    def save(self, *, name: str) -> str:
        """Atomically bind a name to this immutable definition and return its digest."""
        check_name(name)
        db = self._database
        db._check_write()
        digest, text = self.definition_id, canonical_json(self._definition.as_dict())
        with db._repo.session() as session, session.begin():
            session.execute(
                insert(StudyDefinitionRow).values(digest=digest, definition_json=text).on_conflict_do_nothing()
            )
            row = session.get(StudyDefinitionRow, digest)
            if row is None or row.definition_json != text:
                raise ValueError("saved definition digest conflicts with its content")
            session.execute(
                insert(StudyName).values(name=name, digest=digest, created_at=time.time()).on_conflict_do_nothing()
            )
            saved = session.get(StudyName, name)
            if saved is None or saved.digest != digest:
                raise ValueError(f"study name {name!r} already refers to a different definition")
        return digest

    def _mode(self, mode: str | None) -> str:
        self._database._check()
        context = DATABASE_CONTEXT.get()
        if mode is None:
            mode = context[1] if context is not None and context[0] is self._database else "auto"
        if mode not in ("manual", "auto", "refresh"):
            raise ValueError("mode must be manual, auto or refresh")
        return mode

    def _grid(self) -> dict[tuple[str, str], CellSelector]:
        digest = self.definition_id
        hashes = {label: sig.hash for label, sig in self.calculations.items()}
        return {
            (obj.key, label): CellSelector(obj.database_id, digest, obj.key, label, sig_hash)
            for obj in self.objects
            for label, sig_hash in hashes.items()
        }

    def _sources(self) -> dict[str, tuple[Input, SavedInput]]:
        sources = {}
        for ref in self.objects:
            obj = _resolve(self._database, self._definition.database_id, ref)
            sources[ref.key] = obj, _description(obj)
        return sources

    def _cache_refs(
        self, coordinates: Iterable[tuple[str, str]], signatures: Mapping[str, CalculatorSignature]
    ) -> dict[tuple[str, str], _CurrentResult | None]:
        by_key = {obj.key: obj for obj in self.objects}
        addresses = {
            (key, label): RecordRef(by_key[key].signature, signatures[label].hash) for key, label in coordinates
        }
        pinned = self._database._current_refs(addresses.values())
        return {coordinate: pinned.get(ref.cache_key) for coordinate, ref in addresses.items()}

    def _check_recipe(self, signature: CalculatorSignature) -> None:
        with self._database._repo.session() as session:
            row = session.get(Recipe, signature.hash)
            if row is not None and not CalculatorSignature.from_json(row.signature_text).same_identity(signature):
                raise ValueError("registered recipe conflicts with Study signature")

    def _computable(
        self,
        label: str,
        signature: CalculatorSignature,
        desc: SavedInput,
        *,
        reconstruct: bool,
        calculators: dict[str, CalculatorBase],
    ) -> None:
        db = self._database
        root = db._registered_input(desc.root_signature)
        if root is None:
            raise ValueError("input root is not registered")
        db._repo._prepare_load(root.input_id, None)
        if label in self._bindings:
            self._check_bindings({label: self._bindings[label]})
            calculators[label] = self._bindings[label]
        if label not in calculators:
            if not signature.constructible:
                raise ValueError("nonconstructible calculator needs an exact live binding")
            if reconstruct:
                _trusted_payload(signature.as_dict(full=True), {})
                calc = calculator_from_signature(signature)
                if not isinstance(calc, CalculatorBase) or not calculator_to_signature(calc).same_identity(signature):
                    raise ValueError("reconstructed calculator does not match its frozen signature")
                calculators[label] = calc
            else:
                _trusted_payload(signature.as_dict(full=True), {})
        calculator = calculators.get(label)
        if calculator is not None:
            _verify_signature(calculator, signature, label)
            if calculator._resolve_run_options().errors == "collect_partial":
                raise ValueError("Study cannot treat collect_partial as whole-cell success")

    def plan(self, *, mode: str | None = None) -> tuple[StudyPlanCell, ...]:
        """Inspect cache and binding metadata without constructing or loading anything."""
        mode = self._mode(mode)
        self._check_bindings(self._bindings)
        grid, signatures = self._grid(), self.calculations
        current = self._cache_refs(grid, signatures) if mode == "auto" else {}
        reasons = []
        refs = {obj.key: obj for obj in self.objects}
        descriptions: dict[str, SavedInput] = {}
        for coordinate, selector in grid.items():
            key, label = coordinate
            try:
                if mode != "manual":
                    self._database._check_write()
                    self._check_recipe(signatures[label])
                if key not in descriptions:
                    source = _resolve(self._database, self._definition.database_id, refs[key])
                    descriptions[key] = _description(source)
                cached = current.get(coordinate)
                if cached is not None:
                    cached.confirmed_ref()
                    action, reason = "reuse", "current saved version"
                else:
                    self._computable(label, signatures[label], descriptions[key], reconstruct=False, calculators={})
                    action, reason = "compute", "explicit recomputation" if mode != "auto" else "no current result"
            except (
                KeyError,
                ValueError,
                TypeError,
                PermissionError,
                NotImplementedError,
                DatabasePersistenceError,
            ) as exc:
                action, reason = "blocked", str(exc)
            reasons.append(StudyPlanCell(selector, action, reason))
        return tuple(reasons)

    def _report(
        self,
        origin: str,
        started: float,
        outcomes: Mapping[tuple[str, str], StudyCell],
        *,
        mode: str | None = None,
        errors: str | None = None,
        missing: str | None = None,
    ) -> StudyTable:
        return StudyTable(
            self._definition,
            origin,
            started,
            max(started, time.time()),
            mode,
            errors,
            missing,
            tuple(obj.key for obj in self.objects),
            outcomes,
        )

    def collect(self, *, missing: str = "raise") -> StudyTable:
        """Capture the current result versions, then read those exact values."""
        self._database._check()
        if missing not in ("raise", "keep"):
            raise ValueError("missing must be raise or keep")
        started = time.time()
        self._sources()
        grid, signatures = self._grid(), self.calculations
        current = self._cache_refs(grid, signatures)
        outcomes = {}
        db = self._database
        for coordinate, cached in current.items():
            if cached is None:
                outcomes[coordinate] = StudyCell("missing", reason="no current result")
            else:
                ref = cached.confirmed_ref()
                record = db._read_version(ref, signatures[coordinate[1]])
                # Collect only the public value, without constructing the recipe.
                outcomes[coordinate] = StudyCell(
                    "succeeded",
                    db._store.codec.decode(record.value),
                    SavedResultRef(self._definition.database_id, ref),
                    from_store=True,
                )
        absent = [coordinate for coordinate, cell in outcomes.items() if cell.status == "missing"]
        if absent and missing == "raise":
            raise KeyError(f"missing Study results for {absent!r}")
        return self._report("collect", started, outcomes, missing=missing)

    def _selection(
        self, grid: Mapping[tuple[str, str], CellSelector], cells: Iterable[CellSelector] | None
    ) -> list[tuple[str, str]]:
        if cells is None:
            return list(grid)
        selected: set[tuple[str, str]] = set()
        for cell in cells:
            if not isinstance(cell, CellSelector):
                raise TypeError("Study cell selectors must come from StudyTable.cells()")
            coordinate = cell.input_key, cell.label
            if grid.get(coordinate) != cell or coordinate in selected:
                raise ValueError("unknown, repeated or foreign Study cell selector")
            selected.add(coordinate)
        return [coordinate for coordinate in grid if coordinate in selected]

    def _preflight(
        self,
        ordered: list[tuple[str, str]],
        mode: str,
        sources: Mapping[str, tuple[Input, SavedInput]],
        signatures: Mapping[str, CalculatorSignature],
    ) -> dict[str, CalculatorBase]:
        db = self._database
        current = self._cache_refs(ordered, signatures) if mode == "auto" else {}
        calculators: dict[str, CalculatorBase] = {}
        for coordinate in ordered:
            key, label = coordinate
            signature = signatures[label]
            if mode != "manual":
                self._check_recipe(signature)
            cached = current.get(coordinate)
            if cached is not None:
                db._read_version(cached.confirmed_ref(), signature)
            else:
                try:
                    self._computable(label, signature, sources[key][1], reconstruct=True, calculators=calculators)
                except (ValueError, TypeError, KeyError, ImportError) as exc:
                    raise ValueError(f"Study cell {label!r} is blocked for {key}: {exc}") from exc
        return calculators

    def _execute_cell(
        self,
        source: Input,
        desc: SavedInput,
        label: str,
        signature: CalculatorSignature,
        mode: str,
        calculators: dict[str, CalculatorBase],
        invocation: _Invocation,
    ) -> StudyCell:
        db = self._database

        def prepare() -> None:
            self._computable(label, signature, desc, reconstruct=True, calculators=calculators)

        def compute() -> Any:
            loaded = db._load_description(desc)
            calculator = calculators[label]
            _verify_signature(calculator, signature, label)
            if calculator._resolve_run_options().errors == "collect_partial":
                raise _BindingChanged("Study cannot treat collect_partial as whole-cell success")
            return loaded, calculator, calculator._resolve_run_options(errors="raise")

        if label in self._bindings:
            self._check_bindings({label: self._bindings[label]})
        result = db._coordinate(desc, signature, mode, compute, source=source, invocation=invocation, prepare=prepare)
        return StudyCell(
            "succeeded", result.value, self._scoped_ref(result.store_ref), invocation.run_id, result.from_store
        )

    def run(
        self, *, mode: str | None = None, errors: str = "raise", cells: Iterable[CellSelector] | None = None
    ) -> StudyTable:
        """Run selected cells serially, sharing the single-object result cache."""
        started = time.time()
        db = self._database
        mode = self._mode(mode)
        if errors not in ("raise", "collect"):
            raise ValueError("errors must be raise or collect")
        if mode != "manual":
            db._check_write()
        self._check_bindings(self._bindings)
        grid, signatures = self._grid(), self.calculations
        ordered = self._selection(grid, cells)
        outcomes = dict.fromkeys(grid, StudyCell("not_run"))

        def report() -> StudyTable:
            return self._report("run", started, outcomes, mode=mode, errors=errors)

        if not ordered:
            return report()
        sources = self._sources()
        calculators = self._preflight(ordered, mode, sources, signatures)
        for coordinate in ordered:
            selector = grid[coordinate]
            key, label = coordinate
            source, desc = sources[key]
            signature = signatures[label]
            invocation = _Invocation()

            try:
                outcomes[coordinate] = self._execute_cell(source, desc, label, signature, mode, calculators, invocation)
            except Exception as exc:
                if invocation.failed or mode == "manual" and invocation.started:
                    outcomes[coordinate] = StudyCell(
                        "failed", run_id=invocation.run_id, _error=ErrorInfo(type(exc).__name__, str(exc))
                    )
                    if errors == "collect" and not isinstance(exc, _BindingChanged):
                        continue
                    raise StudyExecutionError(report(), selector, exc) from exc
                if invocation.started:
                    receipt = exc.result_ref if isinstance(exc, DatabasePersistenceError) else None
                    outcomes[coordinate] = StudyCell(
                        "incomplete",
                        run_id=invocation.run_id,
                        reason=f"{type(exc).__name__}: {exc}",
                        recovery_ref=self._scoped_ref(receipt),
                    )
                raise StudyPersistenceError(report(), selector, invocation.run_id, exc) from exc
            except (KeyboardInterrupt, SystemExit) as exc:
                if invocation.started:
                    outcomes[coordinate] = StudyCell("incomplete", run_id=invocation.run_id, reason=type(exc).__name__)
                exc.__dict__["report"] = report()
                raise
        return report()

    def _scoped_ref(self, ref: RecordRef | None) -> SavedResultRef | None:
        return None if ref is None else SavedResultRef(self._definition.database_id, ref)
