"""Versioned root-input recipes; bindings and execution stay in project code."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate import interfaces
from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.result.signature import CalculatorSignature, calculator_from_signature
from pynbodyext.core.calculate.runtime.options import RunOptions

if TYPE_CHECKING:
    from .records import RecordedRun
    from .repository import AnalysisRepository

JOB_SCHEMA = "pynbodyext.orm.job/v1"
_JOB_FIELDS = frozenset({"schema", "calculator_signature_text", "input_id", "location_id", "store_name", "reuse"})
_CLASS_NODES = frozenset({"dataclass", "generic", "pipeline", "transform_chain"})


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate job JSON field {key!r}")
        data[key] = value
    return data


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not supported")


def _json_object(text: str) -> dict[str, Any]:
    if type(text) is not str:
        raise TypeError("job JSON and calculator_signature_text must be built-in strings")
    data = json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_invalid_constant)
    if type(data) is not dict:
        raise ValueError("job JSON and calculator signatures require objects")
    return data


def _trusted_payload(value: Any, trusted_types: dict[str, type], *, initial: bool = True) -> None:
    """Check the entire graph before any decoder can invoke a constructor.

    The general signature decoder has import fallbacks for calculator and enum
    classes. Jobs accept only explicit registered calculator names/aliases and
    already loaded Enum classes, so those fallbacks cannot load manifest types.
    This private registry read deliberately narrows the ORM entry point without
    changing general signature reconstruction or importing calculator plugins.
    Incoming names must be explicitly registered. Rebuilt canonical names may
    refer only to the types already vetted in that complete incoming graph.
    """
    if isinstance(value, list):
        for item in value:
            _trusted_payload(item, trusted_types, initial=initial)
    elif isinstance(value, dict):
        if "payload" in value:
            for flag in ("stable", "constructible"):
                if flag in value and (type(value[flag]) is not bool or not value[flag]):
                    raise ValueError("job signatures must be stable and constructible with boolean flags")
        if value.get("node") in _CLASS_NODES:
            name = value.get("class")
            if type(name) is not str:
                raise ValueError("job calculator type must name an explicitly registered class")
            available = interfaces._SIGNATURE_TYPES if initial else trusted_types
            candidate = available.get(name)
            if candidate is None:
                raise ValueError(
                    f"job calculator type {name!r} is not explicitly registered or already vetted; import project code first"
                )
            if initial:
                for trusted_name in (name, interfaces.signature_type_name(candidate)):
                    previous = trusted_types.get(trusted_name)
                    if previous is not None and previous is not candidate:
                        raise ValueError(f"ambiguous trusted job calculator name {trusted_name!r}")
                    trusted_types[trusted_name] = candidate
            version = value.get("version", 0)
            current = interfaces.signature_type_version(candidate)
            if type(version) is not int or version != current:
                raise ValueError(
                    f"job calculator {name!r} signature version {version!r} differs from current {current}"
                )
        if value.get("type") == "enum":
            name = value.get("class")
            module_name, _, attribute = name.rpartition(".") if type(name) is str else ("", "", "")
            module = sys.modules.get(module_name)
            candidate = vars(module).get(attribute) if module is not None else None
            if not isinstance(candidate, type) or not issubclass(candidate, Enum):
                raise ValueError(f"job enum {name!r} is not an already loaded enum class")
        for item in value.values():
            _trusted_payload(item, trusted_types, initial=initial)


def _recipe_payload(value: Any, trusted_types: dict[str, type]) -> Any:
    """Normalize registered aliases, retaining every scientific parameter.

    Version zero is normally omitted by the encoder; an explicit zero is the
    same version. CalculatorSignature's canonical hashing also makes integral
    floats equivalent to integers while keeping booleans distinct.
    """
    if isinstance(value, list):
        return [_recipe_payload(item, trusted_types) for item in value]
    if isinstance(value, dict):
        result = {key: _recipe_payload(item, trusted_types) for key, item in value.items()}
        if value.get("node") in _CLASS_NODES:
            result["class"] = interfaces.signature_type_name(trusted_types[value["class"]])
            if result.get("version", 0) == 0:
                result.pop("version", None)
        return result
    return value


def _default_options(calculator: CalculatorBase) -> None:
    """Reject graph options that signatures would otherwise silently discard."""
    defaults = RunOptions()
    seen: set[int] = set()
    pending = [calculator]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        options = node.default_options
        if type(options) is not RunOptions or any(
            type(getattr(options, field.name)) is not type(getattr(defaults, field.name))
            or getattr(options, field.name) != getattr(defaults, field.name)
            for field in fields(defaults)
        ):
            raise ValueError("jobs support only default RunOptions on every calculator")
        pending.extend(node.dependencies())


def _calculator(text: str) -> CalculatorBase:
    data = _json_object(text)
    trusted_types: dict[str, type] = {}
    _trusted_payload(data, trusted_types)
    signature = CalculatorSignature.from_dict(data)
    if not signature.stable or not signature.constructible:
        raise ValueError("job calculator signature must be stable and constructible")
    calculator = calculator_from_signature(signature)
    if not isinstance(calculator, CalculatorBase):
        raise TypeError("job recipe must reconstruct a CalculatorBase")
    _default_options(calculator)
    rebuilt = calculator.to_signature()
    if not rebuilt.stable or not rebuilt.constructible:
        raise ValueError("rebuilt job calculator must be stable and constructible")
    _trusted_payload(rebuilt.as_dict(full=True), trusted_types, initial=False)
    expected_key = CalculatorSignature(payload=_recipe_payload(signature.payload, trusted_types)).hash
    rebuilt_key = CalculatorSignature(payload=_recipe_payload(rebuilt.payload, trusted_types)).hash
    if expected_key != rebuilt_key:
        raise ValueError("reconstructed job recipe identity differs from the manifest")
    return calculator


@dataclass(frozen=True, slots=True)
class JobSpec:
    """An immutable root job in one explicitly supplied metadata repository.

    IDs are local to that repository. Project code must import/register required
    calculator names or aliases before reading or executing a job; class-path-only
    custom calculators are excluded. Reconstruction invokes those trusted types,
    never automatically imports application modules. Only default RunOptions are
    supported. Loader/store bindings, credentials and live snapshots stay outside
    the manifest; scientific parameter strings are preserved without redaction.

    Parameters
    ----------
    calculator_signature_text : str
        A stable, constructible calculator signature JSON string.
    input_id : int
        Positive built-in metadata ID of a registered root snapshot.
    store_name : str
        Nonempty name of an explicitly bound ResultStore.
    location_id : int or None, optional
        Positive built-in ID of the chosen root location. None uses the runner's
        single-location rule.
    reuse : bool, default True
        Reuse the current payload or save a new immutable version when False.
    schema : str, default "pynbodyext.orm.job/v1"
        Exact manifest version. Unknown versions are rejected.
    """

    calculator_signature_text: str
    input_id: int
    store_name: str
    location_id: int | None = None
    reuse: bool = True
    schema: str = JOB_SCHEMA

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != JOB_SCHEMA:
            raise ValueError("unsupported job schema")
        for name, value in (("input_id", self.input_id), ("location_id", self.location_id)):
            if name == "location_id" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"job {name} must be a positive built-in integer, excluding bool")
        if type(self.store_name) is not str or not self.store_name.strip():
            raise ValueError("job store_name must be a nonempty built-in string")
        if type(self.reuse) is not bool:
            raise ValueError("job reuse must be a built-in bool")
        _calculator(self.calculator_signature_text)

    @classmethod
    def from_calculator(
        cls,
        calculator: CalculatorBase,
        input_id: int,
        *,
        store_name: str,
        location_id: int | None = None,
        reuse: bool = True,
    ) -> JobSpec:
        """Copy a stable recipe and root address without loading or calculating.

        Non-default graph RunOptions fail before signature conversion can discard
        them. The copied signature is checked against explicitly registered types.
        """
        if not isinstance(calculator, CalculatorBase):
            raise TypeError("job recipe must be a CalculatorBase")
        _default_options(calculator)
        return cls(calculator.to_signature().to_json(full=True), input_id, store_name, location_id, reuse)

    def to_json(self) -> str:
        """Serialize only the six recipe/address fields; no runtime bindings."""
        return json.dumps(
            {name: getattr(self, name) for name in _JOB_FIELDS}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_json(cls, text: str) -> JobSpec:
        """Validate strict JSON and trusted recipe types before accepting a job.

        Unknown/missing/duplicate fields, schemas and non-finite constants fail
        here, without repository access or scientific evaluation.
        """
        data = _json_object(text)
        if set(data) != _JOB_FIELDS:
            raise ValueError(
                "job JSON requires exactly schema, calculator_signature_text, input_id, location_id, store_name, reuse"
            )
        return cls(**data)

    def execute(self, repo: AnalysisRepository) -> RecordedRun:
        """Rebuild the trusted root recipe and delegate to ``repo.run_input``.

        Type availability is rechecked, and views/selections are rejected before
        loading or Run creation. The runner owns binding/location preflight and
        all later loading, cache reuse, refresh and tracked failure semantics.
        """
        calculator = _calculator(self.calculator_signature_text)
        data = repo.get_input(self.input_id)
        if data.kind != "snapshot" or data.parent_id is not None:
            raise ValueError("job v1 requires a registered root snapshot, not a view or selection")
        return repo.run_input(
            calculator, self.input_id, store_name=self.store_name, location_id=self.location_id, reuse=self.reuse
        )
