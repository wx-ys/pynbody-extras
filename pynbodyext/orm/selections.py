"""Versioned root-relative selection definitions; no SQL or alternate executor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.family import get_family

from pynbodyext.core.calculate.nodes.filters import FilterBase
from pynbodyext.core.calculate.result.signature import SIGNATURE_SCHEMA, CalculatorSignature, calculator_from_signature

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap

    from .loading import LoadedInput

SELECTION_SCHEMA = "pynbodyext.orm.selection/v1"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate selection JSON field {key!r}")
        result[key] = value
    return result


def _has_scoped_transform(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("node") == "scoped":
            scope = value.get("scope")
            if not isinstance(scope, dict):
                raise ValueError("selection filter has an invalid scope definition")
            if scope.get("transforms"):
                return True
        return any(_has_scoped_transform(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_scoped_transform(item) for item in value)
    return False


def _filter(signature_data: Any) -> FilterBase:
    fields = {"schema", "payload", "constructible", "stable", "non_constructible_paths", "unstable_paths"}
    if (
        not isinstance(signature_data, dict)
        or set(signature_data) != fields
        or signature_data["schema"] != SIGNATURE_SCHEMA
        or any(type(signature_data[key]) is not bool for key in ("stable", "constructible"))
        or any(
            type(signature_data[key]) is not list or any(type(path) is not str for path in signature_data[key])
            for key in ("non_constructible_paths", "unstable_paths")
        )
    ):
        raise ValueError("selection filter requires a complete versioned calculator signature")
    signature = CalculatorSignature.from_dict(signature_data)
    if not signature.stable or not signature.constructible:
        raise ValueError("selection filter must be stable and constructible")
    if _has_scoped_transform(signature.payload):
        raise ValueError("selection filters must evaluate in original coordinates without scoped transforms")
    calculator = calculator_from_signature(signature)
    if not isinstance(calculator, FilterBase):
        raise TypeError("selection filter requires a FilterBase calculator")
    rebuilt = calculator.to_signature()
    if not rebuilt.stable or not rebuilt.constructible or rebuilt.identity() != signature.identity():
        raise ValueError("selection filter signature cannot be stably reconstructed")
    return calculator


def _definition(text: str) -> dict[str, Any]:
    data = json.loads(text, object_pairs_hook=_unique_fields)
    if not isinstance(data, dict) or data.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unsupported selection schema")
    kind = data.get("kind")
    if type(kind) is not str:
        raise ValueError("selection kind must be a string")
    field = {"indices": "indices", "family": "family", "filter": "calculator_signature"}.get(kind)
    if field is None or set(data) != {"schema", "kind", field}:
        raise ValueError("invalid selection kind or definition fields")
    value = data[field]
    if kind == "indices":
        if type(value) is not list or any(type(index) is not int or index < 0 for index in value):
            raise ValueError("selection indices require a list of nonnegative built-in integers")
    elif kind == "family":
        if type(value) is not str or not value.strip():
            raise ValueError("selection family requires a nonempty string")
    else:
        calculator = _filter(value)
        data[field] = calculator.to_signature().as_dict(full=True)
    return data


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    """A copied, immutable replay definition for an untransformed root snapshot.

    Indices preserve order and duplicates within root family-block ordering;
    cross-block reordering is rejected, never sorted. A family must exist in
    the loaded root. Filters use the existing stable, constructible CalculatorSignature;
    scoped transforms are excluded so selection uses original coordinates.
    """

    _text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "_text", _json(_definition(self._text)))

    @classmethod
    def indices(cls, indices: list[int]) -> SelectionSpec:
        """Copy a strict built-in integer list; bounds are checked against a root."""
        if type(indices) is not list or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("selection indices require a list of nonnegative built-in integers")
        return cls(_json({"schema": SELECTION_SCHEMA, "kind": "indices", "indices": indices}))

    @classmethod
    def family(cls, family: str) -> SelectionSpec:
        """Select a named family, failing if it is absent from the loaded root."""
        return cls(_json({"schema": SELECTION_SCHEMA, "kind": "family", "family": family}))

    @classmethod
    def filter(cls, calculator: FilterBase) -> SelectionSpec:
        """Copy a stable, reconstructible filter without evaluating it."""
        if not isinstance(calculator, FilterBase):
            raise TypeError("selection filter requires a FilterBase calculator")
        return cls(
            _json(
                {
                    "schema": SELECTION_SCHEMA,
                    "kind": "filter",
                    "calculator_signature": calculator.to_signature().as_dict(full=True),
                }
            )
        )

    def to_json(self) -> str:
        """Return the complete canonical versioned definition."""
        return self._text

    @classmethod
    def from_json(cls, text: str) -> SelectionSpec:
        """Validate schema, fields and reconstructibility without evaluating."""
        return cls(text)

    def materialize(self, loaded: LoadedInput) -> SimSnap:
        """Evaluate once against a previously validated root; never cache a mask."""
        data = json.loads(self._text)
        root = loaded.sim
        if data["kind"] == "indices":
            indices = data["indices"]
            if any(index >= len(root) for index in indices):
                raise ValueError("selection index is out of range for root particle count")
            # Pynbody represents family membership as contiguous view slices.
            # Check public family lengths in root appearance order without
            # sorting or changing the caller's index sequence (ADR-016).
            boundaries = np.cumsum([len(root[family]) for family in root.families()])
            family_indices = np.searchsorted(boundaries, indices, side="right")
            if np.any(np.diff(family_indices) < 0):
                raise ValueError("selection indices must follow root family-block order; cross-block reordering is unsupported")
            return root[np.asarray(indices, dtype=np.int64)]
        if data["kind"] == "family":
            family = get_family(data["family"], create=False)
            if family not in root.families():
                raise ValueError(f"selection family {data['family']!r} is absent from root")
            return root[family]
        result = _filter(data["calculator_signature"]).run(root, sim_identity=loaded.sim_identity)
        if result.errors:
            raise ValueError(f"selection filter failed: {result.errors}")
        mask = np.asarray(result.value)
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(root):
            raise ValueError("selection filter mask must be one-dimensional boolean with root particle count length")
        return root[mask]


def descriptor_key(
    root_input_id: int, expected_count: int, spec: SelectionSpec, *, view_signature: tuple[Any, ...] | None = None
) -> str:
    """Hash the replay definition, retaining existing subset descriptor addresses.

    A halo view adds its typed full signature to distinguish halo numbers from
    each other and from an ordinary subset selecting the same particles.
    """
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("selection expected_count must be a nonnegative root particle count")
    descriptor = {
        "root_input_id": root_input_id,
        "expected_count": expected_count,
        "schema": SELECTION_SCHEMA,
        "definition": json.loads(spec.to_json()),
    }
    if view_signature is not None:
        from pynbodyext.core.calculate.store.identity import sim_identity_key

        from .loading import parse_input_identity

        if parse_input_identity(view_signature).kind != "halo":
            raise ValueError("only halo replay descriptors accept a view signature discriminator")
        descriptor["halo_identity"] = sim_identity_key(view_signature)
    return hashlib.sha256(_json(descriptor).encode("utf-8")).hexdigest()
