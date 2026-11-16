"""The frozen definition shared by Database studies and detached tables."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pynbodyext.core.calculate.interfaces import is_stable_sim_identity
from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.core.calculate.store.base import RecordRef
from pynbodyext.core.calculate.store.identity import sim_identity_key

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFINITION_SCHEMA = "pynbodyext.database.study/v1"


def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"duplicate JSON field {key!r}")
        fields[key] = value
    return fields


def json_object(data: Any, fields: str) -> dict[str, Any]:
    if type(data) is not dict or set(data) != set(fields.split()):
        raise ValueError(f"expected fields: {fields}")
    return data


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def check_uuid(value: str) -> None:
    if type(value) is not str or str(UUID(value)) != value:
        raise ValueError("expected a canonical database UUID")


def check_name(value: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("expected a nonempty name")


def check_hash(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected a full SHA-256 hash")


def read_record_ref(data: Any) -> RecordRef:
    json_object(data, "schema sim_signature calculator_signature_hash version_id")
    ref = RecordRef.from_json(canonical_json(data))
    if data != json.loads(ref.to_json()):
        raise ValueError("noncanonical result reference")
    return ref


@dataclass(frozen=True, slots=True, eq=False)
class ObjectRef:
    """A typed exact input and its registered locator in one database namespace."""

    database_id: str
    signature: tuple[Any, ...]
    locator: tuple[str, ...]
    label: str

    def __post_init__(self) -> None:
        check_uuid(self.database_id)
        key = self.key
        check_name(self.label)
        if not is_stable_sim_identity(self.signature):
            raise ValueError("study inputs require stable identities")
        if type(self.locator) is not tuple or not all(type(x) is str and x for x in self.locator):
            raise ValueError("invalid object locator")
        if self.locator == ("input", key):
            return
        if len(self.locator) != 4 or self.locator[0] != "catalogue" or self.locator[2] not in ("halo", "subhalo"):
            raise ValueError("invalid object locator")

    @property
    def key(self) -> str:
        return sim_identity_key(self.signature)

    def as_dict(self) -> dict[str, Any]:
        # Use the existing typed tuple encoding used by RecordRef.
        address = json.loads(RecordRef(self.signature, "input").to_json())
        return {
            "database_id": self.database_id,
            "identity": address["sim_signature"],
            "locator": list(self.locator),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ObjectRef:
        data = json_object(data, "database_id identity locator label")
        address = json.loads(RecordRef((), "input").to_json())
        address["sim_signature"] = data["identity"]
        if type(data["locator"]) is not list:
            raise ValueError("invalid object locator")
        return cls(data["database_id"], read_record_ref(address).sim_signature, tuple(data["locator"]), data["label"])

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ObjectRef) and (self.database_id, self.key, self.locator, self.label) == (
            other.database_id,
            other.key,
            other.locator,
            other.label,
        )

    def __hash__(self) -> int:
        return hash((self.database_id, self.key, self.locator, self.label))


@dataclass(frozen=True, slots=True)
class StudyDefinition:
    database_id: str
    objects: tuple[ObjectRef, ...]
    columns: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        check_uuid(self.database_id)
        if any(obj.database_id != self.database_id for obj in self.objects):
            raise ValueError("study object belongs to another database")
        if len({obj.key for obj in self.objects}) != len(self.objects):
            raise ValueError("duplicate exact study input")
        labels: set[str] = set()
        for label, text in self.columns:
            check_name(label)
            if label in labels:
                raise ValueError("duplicate study column")
            labels.add(label)
            signature = CalculatorSignature.from_json(text)
            if not signature.stable or text != signature.to_json(full=True):
                raise ValueError("study requires full, stable calculator signatures")

    @property
    def calculations(self) -> Mapping[str, CalculatorSignature]:
        return MappingProxyType({label: CalculatorSignature.from_json(text) for label, text in self.columns})

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": DEFINITION_SCHEMA,
            "database_id": self.database_id,
            "objects": [obj.as_dict() for obj in self.objects],
            "columns": [
                {"label": label, "hash": sig.hash, "signature": sig.as_dict(full=True)}
                for label, sig in self.calculations.items()
            ],
        }

    @property
    def id(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Any) -> StudyDefinition:
        data = json_object(data, "schema database_id objects columns")
        if data["schema"] != DEFINITION_SCHEMA or any(type(data[axis]) is not list for axis in ("objects", "columns")):
            raise ValueError("unsupported or invalid Study definition")
        columns = []
        for item in data["columns"]:
            entry = json_object(item, "label hash signature")
            signature = CalculatorSignature.from_dict(
                json_object(
                    entry["signature"], "schema payload stable constructible unstable_paths non_constructible_paths"
                )
            )
            if entry["hash"] != signature.hash or canonical_json(entry["signature"]) != canonical_json(
                signature.as_dict(full=True)
            ):
                raise ValueError("invalid Study calculator signature or hash")
            columns.append((entry["label"], signature.to_json(full=True)))
        return cls(data["database_id"], tuple(ObjectRef.from_dict(obj) for obj in data["objects"]), tuple(columns))
