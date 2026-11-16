"""Read-only, detached observations of a fixed Study definition."""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody import units as pynbody_units

from pynbodyext.core.calculate.result.result import ErrorInfo
from pynbodyext.core.calculate.store.codec import ValueCodec
from pynbodyext.core.calculate.store.identity import sim_identity_key

from .definition import (
    ObjectRef,
    StudyDefinition,
    canonical_json,
    check_hash,
    check_name,
    check_uuid,
    json_object,
    read_record_ref,
    unique_fields,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from pynbodyext.core.calculate.result.signature import CalculatorSignature
    from pynbodyext.core.calculate.store.base import RecordRef

_SCHEMA = "pynbodyext.study.table/v1"
_STATUSES = frozenset({"succeeded", "missing", "failed", "not_run", "incomplete"})
_CODEC = ValueCodec(portable_only=True)


@dataclass(frozen=True, slots=True)
class SavedResultRef:
    """A pinned ResultStore receipt in its owning Database namespace."""

    database_id: str
    record_ref: RecordRef

    def __post_init__(self) -> None:
        check_uuid(self.database_id)
        check_hash(self.record_ref.calculator_signature_hash)
        if self.record_ref.version_id is None:
            raise ValueError("a Study result reference requires a pinned version")

    @property
    def cache_key(self) -> str:
        return self.record_ref.cache_key

    @property
    def version_id(self) -> str:
        version = self.record_ref.version_id
        assert version is not None
        return version

    def as_dict(self) -> dict[str, Any]:
        return {"database_id": self.database_id, "record_ref": json.loads(self.record_ref.to_json())}

    @classmethod
    def from_dict(cls, data: Any) -> SavedResultRef:
        data = json_object(data, "database_id record_ref")
        return cls(data["database_id"], read_record_ref(data["record_ref"]))


@dataclass(frozen=True, slots=True)
class CellSelector:
    database_id: str
    definition_id: str
    input_key: str
    label: str
    calculator_hash: str


def _portable(encoded: Any) -> None:
    if isinstance(encoded, dict):
        if encoded.get("__kind") in ("opaque", "unencoded") or "pickle_b64" in encoded:
            raise ValueError("StudyTable requires portable values without opaque or pickle payloads")
        for item in encoded.values():
            _portable(item)
    elif isinstance(encoded, (list, tuple)):
        for item in encoded:
            _portable(item)


@dataclass(frozen=True, slots=True, eq=False)
class StudyCell:
    """One coordinate; successful None is distinct from an absent value."""

    status: str
    _value: Any = None
    ref: SavedResultRef | None = None
    run_id: str | None = None
    from_store: bool | None = None
    _error: ErrorInfo | None = None
    reason: str | None = None
    recovery_ref: SavedResultRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_value", deepcopy(self._value))
        object.__setattr__(self, "_error", deepcopy(self._error))

    @property
    def has_value(self) -> bool:
        return self.status == "succeeded"

    @property
    def value(self) -> Any:
        if not self.has_value:
            raise ValueError(f"Study cell has no value (status={self.status})")
        return deepcopy(self._value)

    @property
    def error(self) -> ErrorInfo | None:
        return deepcopy(self._error)

    def _validate(self, origin: str, mode: str | None, obj: ObjectRef, signature: CalculatorSignature) -> None:
        if type(self.status) is not str or self.status not in _STATUSES:
            raise ValueError(f"unknown Study cell status {self.status!r}")
        if origin == "collect" and (self.status not in ("succeeded", "missing") or self.run_id is not None):
            raise ValueError("invalid collect cell")
        if origin == "run" and self.status == "missing":
            raise ValueError("missing is a collect-only cell status")
        if self.run_id is not None:
            check_name(self.run_id)
        if mode == "manual" and (self.run_id is not None or self.recovery_ref is not None):
            raise ValueError("manual cell cannot carry persistence metadata")
        if self.reason is not None:
            check_name(self.reason)
        if self.has_value:
            if (
                type(self.from_store) is not bool
                or self._error is not None
                or self.reason is not None
                or self.recovery_ref is not None
                or (self.ref is None) != (mode == "manual")
                or (origin == "collect" and not self.from_store)
                or (mode in ("refresh", "manual") and self.from_store)
                or (origin == "run" and mode != "manual" and self.run_id is None)
            ):
                raise ValueError("invalid successful Study cell metadata")
        elif self._value is not None or self.ref is not None or self.from_store is not None:
            raise ValueError("only successful Study cells carry a value, reference or reuse flag")
        if (self.status == "failed") != (self._error is not None):
            raise ValueError("failed cells require structured error details")
        if self.status == "failed" and mode != "manual" and self.run_id is None:
            raise ValueError("recorded failure requires a Run ID")
        if self.status in ("missing", "incomplete") and self.reason is None:
            raise ValueError("missing and incomplete cells need a reason")
        if self.status == "not_run" and self.run_id is not None:
            raise ValueError("unattempted cells cannot have a Run ID")
        if self.recovery_ref is not None and self.status != "incomplete":
            raise ValueError("only incomplete cells can carry a recovery receipt")
        for ref in (self.ref, self.recovery_ref):
            if ref is not None and (
                ref.database_id != obj.database_id
                or sim_identity_key(ref.record_ref.sim_signature) != obj.key
                or ref.record_ref.calculator_signature_hash != signature.hash
            ):
                raise ValueError("result reference conflicts with its Study coordinate")

    def _encode(self) -> dict[str, Any]:
        encoded = _CODEC.encode(self._value) if self.has_value else None
        _portable(encoded)
        return {
            "status": self.status,
            "has_value": self.has_value,
            "value": encoded,
            "ref": self.ref.as_dict() if self.ref else None,
            "run_id": self.run_id,
            "from_store": self.from_store,
            "error": asdict(self._error) if self._error else None,
            "reason": self.reason,
            "recovery_ref": self.recovery_ref.as_dict() if self.recovery_ref else None,
        }

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> StudyCell:
        present = data["status"] == "succeeded"
        if data["has_value"] is not present or (not present and data["value"] is not None):
            raise ValueError("invalid value presence in Study cell")
        error = data["error"]
        if error is not None:
            json_object(error, "error_type message phase traceback_text")
            if any(type(error[key]) is not str for key in ("error_type", "message")) or any(
                error[key] is not None and type(error[key]) is not str for key in ("phase", "traceback_text")
            ):
                raise ValueError("invalid Study cell diagnostic")
        return cls(
            data["status"],
            _CODEC.decode(data["value"]) if present else None,
            SavedResultRef.from_dict(data["ref"]) if data["ref"] is not None else None,
            data["run_id"],
            data["from_store"],
            ErrorInfo(**error) if error is not None else None,
            data["reason"],
            SavedResultRef.from_dict(data["recovery_ref"]) if data["recovery_ref"] is not None else None,
        )


def _object_address(obj: Any) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(obj, ObjectRef):
        reference = getattr(obj, "_study_address", None)
        if reference is None:
            raise TypeError("expected a registered Database handle or ObjectRef")
        return reference()
    return obj.database_id, obj.key, obj.locator


@dataclass(frozen=True, slots=True)
class StudyRow:
    _table: StudyTable
    object: ObjectRef

    def cell(self, label: str) -> StudyCell:
        return self._table._cells[self.object.key, label]

    def __getitem__(self, label: str) -> Any:
        return self.cell(label).value


@dataclass(frozen=True, slots=True)
class StudyTable:
    """An invocation or cache observation, with detached values and exact refs."""

    _definition: StudyDefinition
    origin: str
    started_at: float
    finished_at: float
    mode: str | None
    errors: str | None
    missing: str | None
    _rows: tuple[str, ...]
    _cells: Mapping[tuple[str, str], StudyCell]

    def __post_init__(self) -> None:
        if self.origin == "run":
            if (
                self.mode not in ("manual", "auto", "refresh")
                or self.errors not in ("raise", "collect")
                or self.missing is not None
            ):
                raise ValueError("invalid Study run policies")
        elif (
            self.origin != "collect"
            or self.mode is not None
            or self.errors is not None
            or self.missing not in ("raise", "keep")
        ):
            raise ValueError("invalid Study collect policies")
        if (
            any(type(t) not in (int, float) or not math.isfinite(t) for t in (self.started_at, self.finished_at))
            or self.finished_at < self.started_at
        ):
            raise ValueError("invalid Study capture times")
        signatures = self.calculations
        by_key = {obj.key: obj for obj in self._definition.objects}
        if tuple(key for key in by_key if key in set(self._rows)) != self._rows:
            raise ValueError("invalid displayed Study rows")
        grid = {(key, label) for key in self._rows for label in signatures}
        if set(self._cells) != grid:
            raise ValueError("Study table must have exactly one cell per displayed coordinate")
        for (key, label), cell in self._cells.items():
            cell._validate(self.origin, self.mode, by_key[key], signatures[label])
        object.__setattr__(self, "_cells", MappingProxyType(dict(self._cells)))

    @property
    def definition_id(self) -> str:
        return self._definition.id

    @property
    def objects(self) -> tuple[ObjectRef, ...]:
        shown = set(self._rows)
        return tuple(obj for obj in self._definition.objects if obj.key in shown)

    @property
    def calculations(self) -> Mapping[str, CalculatorSignature]:
        return self._definition.calculations

    def __iter__(self) -> Iterator[StudyRow]:
        return (StudyRow(self, obj) for obj in self.objects)

    def __getitem__(self, obj_or_ref: Any) -> StudyRow:
        database_id, key, locator = _object_address(obj_or_ref)
        if database_id != self._definition.database_id:
            raise ValueError("object belongs to another database")
        for obj in self.objects:
            if obj.key == key and obj.locator == locator:
                return StudyRow(self, obj)
        raise KeyError(locator)

    def __repr__(self) -> str:
        counts: dict[str, int] = {}
        for cell in self._cells.values():
            counts[cell.status] = counts.get(cell.status, 0) + 1
        states = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        return f"StudyTable({self.origin}, {len(self._rows)} rows × {len(self._definition.columns)} columns; {states})"

    def cell(self, obj_or_ref: Any, label: str) -> StudyCell:
        """Return the outcome at an exact displayed coordinate."""
        return self._cells[self[obj_or_ref].object.key, label]

    def cells(self, *, status: str | None = None) -> tuple[CellSelector, ...]:
        """Select exact displayed coordinates for a later explicit invocation."""
        if status is not None and status not in _STATUSES:
            raise ValueError(f"unknown Study cell status {status!r}")
        digest = self.definition_id
        signatures = self.calculations
        hashes = {label: signature.hash for label, signature in signatures.items()}
        return tuple(
            CellSelector(self._definition.database_id, digest, key, label, hashes[label])
            for key in self._rows
            for label in signatures
            if status is None or self._cells[key, label].status == status
        )

    def where(self, predicate: Callable[[StudyRow], bool]) -> StudyTable:
        """Keep rows satisfying a predicate, preserving their source definition and refs."""
        keys = tuple(row.object.key for row in self if predicate(row))
        selected = set(keys)
        return StudyTable(
            self._definition,
            self.origin,
            self.started_at,
            self.finished_at,
            self.mode,
            self.errors,
            self.missing,
            keys,
            {key: cell for key, cell in self._cells.items() if key[0] in selected},
        )

    def column(self, label: str, *, units: str | None = None) -> np.ma.MaskedArray:
        """Extract numeric scalars, masking cells whose status is not succeeded."""
        if label not in self.calculations:
            raise KeyError(label)
        cells = [self._cells[key, label] for key in self._rows]
        values = [cell.value for cell in cells if cell.has_value]
        if any(np.ndim(value) != 0 or np.asarray(value).dtype.kind not in "biufc" for value in values):
            raise ValueError("Study column requires numeric scalar values")
        unit_labels = {str(unit) for value in values if pynbody_units.has_units(value) for unit in (value.units,)}
        if unit_labels and any(not pynbody_units.has_units(value) for value in values):
            raise ValueError("cannot mix unitless and unit-bearing values")
        if units is None and len(unit_labels) > 1:
            raise ValueError("mixed units require an explicit target unit")
        if units is not None:
            if values and not unit_labels:
                raise ValueError("cannot convert unitless Study values")
            try:
                values = [value.in_units(units) for value in values]
            except pynbody_units.UnitsException as exc:
                raise ValueError(f"incompatible Study column units: {units}") from exc
        mask = np.array([not cell.has_value for cell in cells], dtype=bool)
        data = np.zeros(len(cells), dtype=np.asarray(values).dtype)
        data[~mask] = values
        projected = np.ma.array(data, mask=mask)
        projected.units = pynbody_units.Unit(units or next(iter(unit_labels))) if units or unit_labels else None
        return projected

    def to_dataframe(
        self, *, units: Mapping[str, str] | None = None, include_status: bool = True, include_refs: bool = True
    ) -> Any:
        """Project values and optional status/ref columns into a pandas DataFrame."""
        import pandas as pd

        conversions = {} if units is None else dict(units)
        data: dict[str, list[Any]] = {}
        column_units = dict(conversions)
        labels = tuple(self.calculations)
        if set(conversions) - set(labels):
            raise KeyError("unknown column in unit conversions")
        for label in labels:
            cells = [self._cells[key, label] for key in self._rows]
            if label in conversions:
                projected = self.column(label, units=conversions[label])
                data[label] = [
                    None if missing else value
                    for value, missing in zip(projected.data, np.ma.getmaskarray(projected), strict=True)
                ]
            else:
                data[label] = [cell.value if cell.has_value else None for cell in cells]
                unit_labels = {
                    str(value.units) for value in data[label] if value is not None and pynbody_units.has_units(value)
                }
                successful_values = [value for value, cell in zip(data[label], cells, strict=True) if cell.has_value]
                if unit_labels and any(not pynbody_units.has_units(value) for value in successful_values):
                    raise ValueError(f"column {label!r} mixes unitless and unit-bearing values")
                if len(unit_labels) > 1:
                    raise ValueError(f"mixed units in {label!r} require an explicit target unit")
                if unit_labels:
                    column_units[label] = next(iter(unit_labels))
            if include_status:
                if f"{label}.status" in labels:
                    raise ValueError("status column name collides with a Study column")
                data[f"{label}.status"] = [cell.status for cell in cells]
            if include_refs:
                if f"{label}.ref" in labels:
                    raise ValueError("reference column name collides with a Study column")
                data[f"{label}.ref"] = [cell.ref for cell in cells]
        frame = pd.DataFrame(data, index=pd.Index(self._rows, name="input_key"))
        frame.attrs["pynbodyext"] = {"definition_id": self.definition_id, "units": column_units}
        return frame

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> None:
        """Encode completely and atomically publish a portable JSON artifact."""
        target = Path(path)
        text = canonical_json(
            {
                "schema": _SCHEMA,
                "definition": self._definition.as_dict(),
                "definition_id": self.definition_id,
                "origin": self.origin,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "mode": self.mode,
                "errors": self.errors,
                "missing": self.missing,
                "rows": self._rows,
                "columns": tuple(self.calculations),
                "cells": [
                    {"input_key": key, "label": label, **self._cells[key, label]._encode()}
                    for key in self._rows
                    for label in self.calculations
                ],
            }
        )
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as temp:
            staged = Path(temp.name)
            try:
                temp.write(text)
                temp.close()
                if overwrite:
                    staged.replace(target)
                else:
                    os.link(staged, target)
            finally:
                staged.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> StudyTable:
        """Read a portable table without locating a database or rebuilding recipes."""
        data = json_object(
            json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_fields),
            "schema definition definition_id origin started_at finished_at mode errors missing rows columns cells",
        )
        if data["schema"] != _SCHEMA:
            raise ValueError("unsupported StudyTable schema")
        definition = StudyDefinition.from_dict(data["definition"])
        if data["definition_id"] != definition.id:
            raise ValueError("StudyTable definition digest mismatch")
        if data["columns"] != list(definition.calculations):
            raise ValueError("invalid displayed Study columns")
        if any(type(data[axis]) is not list for axis in ("rows", "cells")):
            raise ValueError("invalid StudyTable rows or cells")
        if any(type(key) is not str for key in data["rows"]):
            raise ValueError("invalid StudyTable row key")
        # Screen the entire artifact before ValueCodec can decode any cell.
        for entry in data["cells"]:
            json_object(entry, "input_key label status has_value value ref run_id from_store error reason recovery_ref")
            _portable(entry["value"])
        cells = {}
        for entry in data["cells"]:
            key, label = entry["input_key"], entry["label"]
            if type(key) is not str or type(label) is not str or (key, label) in cells:
                raise ValueError("invalid or duplicate Study cell coordinate")
            cells[key, label] = StudyCell._decode(entry)
        return cls(
            definition,
            data["origin"],
            data["started_at"],
            data["finished_at"],
            data["mode"],
            data["errors"],
            data["missing"],
            tuple(data["rows"]),
            cells,
        )
