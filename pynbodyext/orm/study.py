"""Serial root studies through recorded runs, with read-only report checkpoints."""

from __future__ import annotations

import json
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.result.result import ErrorInfo
from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.core.calculate.store import RecordRef
from pynbodyext.core.calculate.store.codec import ValueCodec
from pynbodyext.core.calculate.store.identity import sim_identity_key

from .models import Recipe
from .records import RunExecutionError, RunPersistenceError

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from .repository import AnalysisRepository

STUDY_SCHEMA = "pynbodyext.orm.study/v1"
_REPORT_FIELDS = frozenset({"schema", "store_name", "input_ids", "identities", "columns", "cells"})
_CELL_FIELDS = frozenset({"input_id", "label", "run_id", "status", "value", "ref", "errors"})
_ERROR_FIELDS = frozenset({"error_type", "message", "phase", "traceback_text"})
_REF_FIELDS = frozenset({"schema", "sim_signature", "calculator_signature_hash", "version_id"})


def _positive_id(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer, excluding bool")


def _text(value: Any, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty built-in string")


def _fields(data: Any, expected: frozenset[str], name: str) -> None:
    if type(data) is not dict or set(data) != expected:
        raise ValueError(f"{name} requires exactly {', '.join(sorted(expected))}")


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate study JSON field {key!r}")
        data[key] = value
    return data


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not supported")


def _reference(text: str) -> RecordRef:
    # A ref is a JSON string nested inside report JSON; its own fields need the
    # same duplicate/schema boundary before the shared typed-ref parser runs.
    data = json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant)
    _fields(data, _REF_FIELDS, "study reference")
    return RecordRef.from_json(text)


def _error(error: ErrorInfo) -> dict[str, Any]:
    if not isinstance(error, ErrorInfo):
        raise TypeError("study errors must be ErrorInfo values")
    _text(error.error_type, "error_type")
    if type(error.message) is not str or any(
        value is not None and type(value) is not str for value in (error.phase, error.traceback_text)
    ):
        raise ValueError("study error message must be a string; phase and traceback_text must be strings or None")
    return {name: getattr(error, name) for name in _ERROR_FIELDS}


@dataclass(frozen=True, slots=True)
class StudyCell:
    """A completed attempt, distinguished by status even when its value is None.

    ``ref`` is the actual saved/reused pinned reference on success, owned by the
    report's named store. Failed cells retain Run IDs and structured diagnostics
    but never claim a saved result association.
    """

    input_id: int
    label: str
    run_id: str
    status: str
    value: Any = None
    ref: RecordRef | None = None
    errors: tuple[ErrorInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class StudyReport:
    """Requested axes and completed attempts, independent of live ORM objects.

    ``identities`` follows ``input_ids``; ``columns`` contains ordered
    (label, full calculator signature hash) pairs. A partial report has only
    completed cells. Loading restores values and validates references without
    reconstructing calculators, locating backends, loading inputs or changing
    Runs. Explicitly bind the owning store before fetching a historical ref.
    """

    store_name: str
    input_ids: tuple[int, ...]
    identities: tuple[tuple[Any, ...], ...]
    columns: tuple[tuple[str, str], ...]
    cells: tuple[StudyCell, ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _text(self.store_name, "store_name")
        if any(type(value) is not tuple for value in (self.input_ids, self.identities, self.columns, self.cells)):
            raise ValueError("study report axes and cells must be tuples")
        if len(self.input_ids) != len(self.identities):
            raise ValueError("study identities must follow the complete input_ids axis")
        identities: dict[int, str] = {}
        for input_id, identity in zip(self.input_ids, self.identities, strict=True):
            _positive_id(input_id, "input_id")
            if input_id in identities:
                raise ValueError("study input IDs must be unique")
            identities[input_id] = sim_identity_key(identity)
        columns: dict[str, str] = {}
        for column in self.columns:
            if type(column) is not tuple or len(column) != 2:
                raise ValueError("study columns must be (label, signature hash) pairs")
            label, key = column
            _text(label, "calculator label")
            if label in columns:
                raise ValueError("study calculator labels must be unique")
            if type(key) is not str or len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
                raise ValueError("study columns require full calculator signature hashes")
            columns[label] = key
        coordinates, run_ids = set(), set()
        for cell in self.cells:
            if not isinstance(cell, StudyCell):
                raise TypeError("study cells must be StudyCell values")
            _positive_id(cell.input_id, "cell input_id")
            _text(cell.label, "cell label")
            _text(cell.run_id, "cell run_id")
            coordinate = (cell.input_id, cell.label)
            if cell.input_id not in identities or cell.label not in columns:
                raise ValueError("study cell must belong to the requested axes")
            if coordinate in coordinates or cell.run_id in run_ids:
                raise ValueError("study cells must have unique coordinates and Run IDs")
            coordinates.add(coordinate)
            run_ids.add(cell.run_id)
            if type(cell.status) is not str or cell.status not in ("succeeded", "failed"):
                raise ValueError("study cell status must be succeeded or failed")
            if type(cell.errors) is not tuple:
                raise ValueError("study errors must be a tuple")
            for error in cell.errors:
                _error(error)
            if cell.status == "failed":
                if not cell.errors or cell.ref is not None:
                    raise ValueError("failed study cells require errors and no result reference")
            elif (
                cell.errors
                or not isinstance(cell.ref, RecordRef)
                or cell.ref.version_id is None
                or sim_identity_key(cell.ref.sim_signature) != identities[cell.input_id]
                or cell.ref.calculator_signature_hash != columns[cell.label]
            ):
                raise ValueError("successful study cells require an error-free pinned reference matching their axes")

    def cell(self, input_id: int, label: str) -> StudyCell:
        """Return a completed attempt; missing/unattempted coordinates raise KeyError."""
        _positive_id(input_id, "input_id")
        _text(label, "label")
        for cell in self.cells:
            if cell.input_id == input_id and cell.label == label:
                return cell
        raise KeyError((input_id, label))

    def to_dict(self, *, codec: ValueCodec | None = None) -> dict[str, Any]:
        """Encode only values, typed identities, pinned refs and report metadata."""
        self._validate()
        codec = codec or ValueCodec()
        return {
            "schema": STUDY_SCHEMA,
            "store_name": self.store_name,
            "input_ids": list(self.input_ids),
            "identities": [codec.encode(identity) for identity in self.identities],
            "columns": [{"label": label, "key": key} for label, key in self.columns],
            "cells": [
                {
                    "input_id": cell.input_id,
                    "label": cell.label,
                    "run_id": cell.run_id,
                    "status": cell.status,
                    "value": codec.encode(cell.value),
                    "ref": None if cell.ref is None else cell.ref.to_json(),
                    "errors": [_error(error) for error in cell.errors],
                }
                for cell in self.cells
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, codec: ValueCodec | None = None) -> StudyReport:
        """Restore a strict report without accessing a repository or backend."""
        _fields(data, _REPORT_FIELDS, "study report")
        if type(data["schema"]) is not str or data["schema"] != STUDY_SCHEMA:
            raise ValueError("unsupported study report schema")
        if any(type(data[key]) is not list for key in ("input_ids", "identities", "columns", "cells")):
            raise ValueError("study report axes and cells require JSON lists")
        codec = codec or ValueCodec()
        columns = []
        for column in data["columns"]:
            _fields(column, frozenset({"label", "key"}), "study column")
            columns.append((column["label"], column["key"]))
        cells = []
        for cell in data["cells"]:
            _fields(cell, _CELL_FIELDS, "study cell")
            if type(cell["errors"]) is not list:
                raise ValueError("study cell errors require a JSON list")
            errors = []
            for error in cell["errors"]:
                _fields(error, _ERROR_FIELDS, "study error")
                errors.append(ErrorInfo(**error))
            if cell["ref"] is not None and type(cell["ref"]) is not str:
                raise ValueError("study reference must be a RecordRef JSON string or None")
            cells.append(
                StudyCell(
                    cell["input_id"],
                    cell["label"],
                    cell["run_id"],
                    cell["status"],
                    codec.decode(cell["value"]),
                    None if cell["ref"] is None else _reference(cell["ref"]),
                    tuple(errors),
                )
            )
        return cls(
            data["store_name"],
            tuple(data["input_ids"]),
            tuple(codec.decode(identity) for identity in data["identities"]),
            tuple(columns),
            tuple(cells),
        )

    def to_json(self) -> str:
        """Serialize this checkpoint using the existing trusted local value codec."""
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> StudyReport:
        """Read versioned JSON, rejecting duplicate fields and non-finite constants."""
        if type(text) is not str:
            raise TypeError("study JSON must be a built-in string")
        return cls.from_dict(json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant))

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a JSON checkpoint; no Run or result-store state is changed."""
        text = self.to_json()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> StudyReport:
        """Read saved values and refs only; no execution or recovery is performed."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_json(handle.read())


class StudyPersistenceError(RuntimeError):
    """A halted study plus the exact original receipt for explicit recovery.

    ``report`` includes prior completed attempts, not the interrupted cell.
    ``receipt`` retains its metadata cause; result/ref/run_id aliases are copied
    unchanged. The caller must inspect and explicitly recover the pending Run.
    """

    def __init__(self, report: StudyReport, input_id: int, label: str, receipt: RunPersistenceError) -> None:
        self.report = report
        self.input_id = input_id
        self.label = label
        self.receipt = receipt
        self.run_id = receipt.run_id
        self.result = receipt.result
        self.result_ref = receipt.result_ref
        super().__init__(f"study stopped at input {input_id}, calculator {label!r}; inspect run {receipt.run_id}")


def _preflight(
    repo: AnalysisRepository,
    input_ids: Iterable[int],
    calculators: Mapping[str, CalculatorBase],
    store_name: str,
    location_ids: Mapping[int, int] | None,
    reuse: bool,
    errors: str,
) -> tuple[tuple[int, ...], tuple[tuple[Any, ...], ...], tuple[tuple[str, CalculatorBase, str], ...], dict[int, int]]:
    """Read every requested binding/identity/recipe before any Run or science."""
    _text(store_name, "store_name")
    if type(reuse) is not bool:
        raise ValueError("study reuse must be a built-in bool")
    if type(errors) is not str or errors not in ("raise", "record"):
        raise ValueError("study errors must be raise or record")
    repo._store(store_name)
    ids = tuple(input_ids)
    for input_id in ids:
        _positive_id(input_id, "input_id")
    if len(set(ids)) != len(ids):
        raise ValueError("study input IDs must be unique")
    if not isinstance(calculators, Mapping):
        raise TypeError("study calculators must be an ordered label-to-Calculator mapping")
    columns = []
    labels: set[str] = set()
    with repo.session() as session:
        for label, calculator in tuple(calculators.items()):
            _text(label, "calculator label")
            if label in labels:
                raise ValueError("study calculator labels must be unique")
            labels.add(label)
            if not isinstance(calculator, CalculatorBase):
                raise TypeError("study calculators must be CalculatorBase values")
            signature = calculator.to_signature()
            if not signature.stable:
                raise ValueError("study recipes must have stable CalculatorSignatures")
            recipe = session.get(Recipe, signature.hash)
            if (
                recipe is not None
                and CalculatorSignature.from_json(recipe.signature_text).identity() != signature.identity()
            ):
                raise ValueError("stored recipe conflicts with its hash")
            columns.append((label, calculator, signature.hash))
    selected: dict[int, int] = {}
    if location_ids is not None:
        if not isinstance(location_ids, Mapping):
            raise TypeError("study location_ids must map input IDs to location IDs")
        for input_id, location_id in tuple(location_ids.items()):
            _positive_id(input_id, "location input_id")
            _positive_id(location_id, "location_id")
            if input_id not in ids or input_id in selected:
                raise ValueError("study location mapping keys must be unique requested input IDs")
            selected[input_id] = location_id
    identities, locations = [], {}
    for input_id in ids:
        data, location, logical, _ = repo._prepare_load(input_id, selected.get(input_id))
        if data.kind != "snapshot" or data.parent_id is not None:
            raise ValueError("studies require registered root snapshots, not views or selections")
        identities.append(logical.signature())
        locations[input_id] = location.id
    return ids, tuple(identities), tuple(columns), locations


def run_study(
    repo: AnalysisRepository,
    input_ids: Iterable[int],
    calculators: Mapping[str, CalculatorBase],
    store_name: str,
    location_ids: Mapping[int, int] | None = None,
    reuse: bool = True,
    errors: str = "raise",
) -> StudyReport:
    """Run a fully preflighted root-input × calculator request serially.

    Inputs are unique positive built-in metadata IDs. Calculators are an ordered
    mapping with unique nonempty labels and stable signatures; configured live
    options and nonconstructible stable recipes remain supported. Locations map
    requested roots to owned location IDs; missing entries use the single-location
    rule. All bindings/recipes/inputs are checked before any Run or loader work.

    Execution is input-major, then mapping order, through ``repo.run_input``.
    ``errors="record"`` continues only a recorded execution failure or a returned
    failed RecordedRun. Metadata finalization failure always raises
    StudyPersistenceError carrying completed progress and the exact recovery
    receipt. Other exceptions propagate. No recovery or deletion is automatic.

    Empty axes create no work after validating the supplied request. Resume is a
    new explicit invocation: every requested cell creates a new Run, with existing
    cache reuse controlled by ``reuse``. Report loading does not resume execution.
    """
    ids, identities, columns, locations = _preflight(
        repo, input_ids, calculators, store_name, location_ids, reuse, errors
    )
    axes = tuple((label, key) for label, _, key in columns)
    cells: list[StudyCell] = []
    for input_id in ids:
        for label, calculator, _ in columns:
            try:
                recorded = repo.run_input(
                    calculator, input_id, store_name=store_name, location_id=locations[input_id], reuse=reuse
                )
            except RunPersistenceError as receipt:
                report = StudyReport(store_name, ids, identities, axes, tuple(cells))
                raise StudyPersistenceError(report, input_id, label, receipt) from receipt
            except RunExecutionError as exc:
                if errors == "raise":
                    raise
                failed = repo.get_run(exc.run_id)
                if failed.status != "failed":
                    raise ValueError("only a recorded failed Run may continue a study") from exc
                diagnostic = ErrorInfo(
                    failed.error_type or type(exc.__cause__).__name__,
                    failed.error_message or str(exc.__cause__),
                    traceback_text="".join(traceback.format_exception(exc.__cause__ or exc)),
                )
                cells.append(StudyCell(input_id, label, exc.run_id, "failed", errors=(diagnostic,)))
                continue
            if recorded.run.status == "failed":
                if errors == "raise":
                    raise RunExecutionError(recorded.run.id) from RuntimeError(str(recorded.result.errors))
                cells.append(
                    StudyCell(
                        input_id,
                        label,
                        recorded.run.id,
                        "failed",
                        recorded.result.value,
                        errors=tuple(recorded.result.errors),
                    )
                )
            elif recorded.run.status == "succeeded":
                cells.append(
                    StudyCell(
                        input_id, label, recorded.run.id, "succeeded", recorded.result.value, recorded.result.store_ref
                    )
                )
            else:
                raise ValueError("run_input returned an unfinished study attempt")
    return StudyReport(store_name, ids, identities, axes, tuple(cells))
