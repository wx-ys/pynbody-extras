"""Portable calculator signatures for reconstruction and indexing.

This module is intentionally separate from ``CalculatorBase.signature()``.
The existing runtime signature remains optimized for in-run cache keys, while
``CalculatorSignature`` stores enough structured data to reconstruct supported
calculator objects in the current Python environment.

Storage pattern
---------------
For persisting calculators alongside results, use the two-field pattern::

    sig = calculator_to_signature(calc)
    key = sig.pretty()  # human-readable, searchable
    data = sig.to_json()  # full signature for reconstruction

    # ... later ...
    calc = calculator_from_signature(data)

The convenience functions :func:`calculator_pack` and :func:`calculator_unpack`
streamline this workflow.

Classes
-------
_Encoder
    Encodes Python calculator graphs into JSON-compatible payloads.
_Decoder
    Decodes JSON-compatible payloads back into Python objects.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields as dataclass_fields, is_dataclass
from enum import Enum
from threading import RLock
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4
from weakref import ReferenceType, ref

import numpy as np
from pynbody import units
from pynbody.array import SimArray
from pynbody.family import Family, get_family

from pynbodyext.core.calculate.interfaces import (
    decode_calculator_payload,
    encode_calculator_payload,
    is_calculator_node,
    signature_type_name,
)
from pynbodyext.core.calculate.params.fields import captured_init_value, collect_param_specs
from pynbodyext.util._type import class_path, import_object
from pynbodyext.util.arrays import dtype_has_metadata

if TYPE_CHECKING:
    from collections.abc import Callable

SIGNATURE_SCHEMA = "pynbodyext.calculator.signature/v2"
#: Schemas whose payloads this module can still read.  v2 changed what the hash
#: covers (schema + payload, not the JSON envelope) and added the ``pipeline``
#: node kind; v1 payloads decode unchanged.
_READABLE_SCHEMAS = frozenset({"pynbodyext.calculator.signature/v1", SIGNATURE_SCHEMA})
DEFAULT_INLINE_ARRAY_BYTES = 128
__all__ = [
    "CalculatorSignature",
    "calculator_from_signature",
    "calculator_to_signature",
    "calculator_pack",
    "calculator_unpack",
    "calculator_pretty_init_args",
    "calculator_parameter_rows",
]


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _make_unit(value: Any) -> Any:
    return units.Unit(value)


def _pipeline_output_values(payload: dict[str, Any], decode: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Resolve ordered output bindings, optionally decoding each distinct recipe once."""
    outputs: dict[str, Any] = {}
    for item in payload["outputs"]:
        if "same_as" in item:
            value = outputs[item["same_as"]]
        else:
            value = item["value"] if decode is None else decode(item["value"])
        outputs[item["key"]] = value
    return outputs


_session_tokens: dict[int, tuple[ReferenceType[Any], str]] = {}
_session_token_lock = RLock()


def _session_token(value: Any) -> str:
    """Identify a live node or array without retaining it or recycling its token."""
    address = id(value)

    def forget(reference: ReferenceType[Any]) -> None:
        with _session_token_lock:
            entry = _session_tokens.get(address)
            if entry is not None and entry[0] is reference:
                del _session_tokens[address]

    with _session_token_lock:
        entry = _session_tokens.get(address)
        if entry is None or entry[0]() is not value:
            entry = ref(value, forget), uuid4().hex
            _session_tokens[address] = entry
        return entry[1]


def _contains_session_identity(value: Any) -> bool:
    """Recognise session-only leaves in compact v1 payloads, which omit the schema and stability."""
    if isinstance(value, list):
        return any(_contains_session_identity(item) for item in value)
    if not isinstance(value, dict):
        return False
    kind = value.get("node")
    if (
        kind == "lambda_property"
        or value.get("type") == "unsupported"
        or (value.get("type") == "array" and ("sha256" not in value or value.get("storage") == "object_hash"))
    ):
        return True
    for key, item in value.items():
        if (kind, key) in (("dataclass", "init"), ("generic", "identity")) and isinstance(item, dict):
            # These mappings hold scientific parameter names. Only their values
            # are encoded nodes, even if a parameter happens to be named 'type'.
            if (kind == "generic" and "opaque_id" in item) or any(_contains_session_identity(v) for v in item.values()):
                return True
        elif _contains_session_identity(item):
            return True
    return False


def _freeze_signature_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return tuple(_freeze_signature_value(item) for item in value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_signature_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (str(key), _freeze_signature_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        )
    raise TypeError(f"unsupported signature payload value: {type(value)!r}")


# ---------------------------------------------------------------------------
# CalculatorSignature — public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalculatorSignature:
    """Structured signature for one calculator graph.

    ``payload`` is JSON-compatible and can be hashed or stored.  Two independent
    properties describe what the signature is good for:

    ``constructible``
        The payload can rebuild the calculator (:func:`calculator_from_signature`).
        False when a part can only be identified, not rebuilt — a large array
        kept by its content hash, a callable, a custom node without a decoder.
        ``non_constructible_paths`` names those parts.
    ``stable``
        The payload — hence :attr:`hash` — is the same in every Python session
        for an equal calculator, so it can key data that outlives the process
        (a persistent result store, a table of galaxies).  False when a part is
        identified by something session-specific, such as an opaque node's token or
        the ``repr`` of a function; such a signature still separates calculators
        within one run.  ``unstable_paths`` names those parts.
    """

    payload: dict[str, Any]
    constructible: bool = True
    non_constructible_paths: tuple[str, ...] = ()
    stable: bool = True
    unstable_paths: tuple[str, ...] = ()
    #: Memoised immutable payload.  ``None`` means "not yet computed"; the frozen
    #: payload is never ``None``.  Excluded from equality/hash so the memo does
    #: not affect signature identity.
    _frozen_cache: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def hash(self) -> str:
        """Return the content hash of this signature: sha256 hex of schema + identity.

        Only what identifies the computation is hashed — the schema version and
        :meth:`identity`, the canonical form of the payload.  The descriptive flags
        (``constructible``, ``stable`` and their paths) are derived from the
        payload and are left out, so they can evolve without moving any key.
        Equal for equal calculators in every session when :attr:`stable` is true.
        """
        identity = {"schema": SIGNATURE_SCHEMA, "payload": self.identity()}
        return hashlib.sha256(_canonical_payload(identity).encode("utf-8")).hexdigest()

    def identity(self) -> Any:
        """The payload in canonical form: what two signatures must share to be equal.

        Numbers compare by value — an integral float reads as the integer it
        equals, so ``Sphere(30)`` and ``Sphere(30.0)`` share a key — while the
        payload itself keeps the values as given, for rebuilding and display.
        """
        return _canonical_identity(self.payload)

    def same_identity(self, other: CalculatorSignature) -> bool:
        """Compare the full canonical payloads, including corresponding NaN values."""
        return _canonical_payload(self.identity()) == _canonical_payload(other.identity())

    def short_hash(self, *, length: int = 12) -> str:
        return self.hash[:length]

    def frozen_payload(self) -> Any:
        """Return the frozen (canonicalised) payload, computed at most once.

        The payload is immutable, so the deep freeze that walks nested values
        only needs to happen on the first request; subsequent ``cache_key()``
        calls reuse the memoised result instead of re-walking the whole payload.
        """
        cached = self._frozen_cache
        if cached is None:
            cached = _freeze_signature_value(self.payload)
            object.__setattr__(self, "_frozen_cache", cached)
        return cached

    def cache_key(self) -> tuple[Any, ...]:
        return ("calculator_signature", self.frozen_payload())

    def as_dict(self, *, full: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        data: dict[str, Any] = {"payload": self.payload}
        if full:
            data["schema"] = SIGNATURE_SCHEMA
        if full or not self.constructible:
            data["constructible"] = self.constructible
        if full or self.non_constructible_paths:
            data["non_constructible_paths"] = list(self.non_constructible_paths)
        if full or not self.stable:
            data["stable"] = self.stable
        if full or self.unstable_paths:
            data["unstable_paths"] = list(self.unstable_paths)
        return data

    def to_json(self, *, indent: int | None = None, full: bool = False) -> str:
        return json.dumps(
            self.as_dict(full=full), sort_keys=True, separators=(",", ":") if indent is None else None, indent=indent
        )

    def pretty(self) -> str:
        """Return a compact canonical expression for this signature."""
        from .render import SignaturePrinter

        return SignaturePrinter.calculator(self.payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculatorSignature:
        schema = data.get("schema")
        if schema is not None and schema not in _READABLE_SCHEMAS:
            raise ValueError(f"unsupported calculator signature schema: {data.get('schema')!r}")
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("calculator signature payload must be a dictionary")
        if "stable" in data:
            stable = bool(data["stable"])
        else:
            stable = schema != "pynbodyext.calculator.signature/v1" and not _contains_session_identity(payload)
        return cls(
            payload=payload,
            constructible=bool(data.get("constructible", True)),
            non_constructible_paths=tuple(str(p) for p in data.get("non_constructible_paths", ())),
            stable=stable,
            unstable_paths=tuple(str(p) for p in data.get("unstable_paths", ())),
        )

    @classmethod
    def from_json(cls, text: str) -> CalculatorSignature:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("calculator signature JSON must decode to a dictionary")
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Internal container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Encoded:
    value: Any
    constructible: bool = True
    #: parts that cannot be rebuilt
    paths: tuple[str, ...] = ()
    #: parts identified by something session-specific (``id()``, a function repr)
    unstable: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Module-level utilities (pure helpers, no encode/decode)
# ---------------------------------------------------------------------------


def _merge_encoded(value: Any, children: list[_Encoded]) -> _Encoded:
    paths: list[str] = []
    unstable: list[str] = []
    constructible = True
    for child in children:
        if not child.constructible:
            constructible = False
        paths.extend(child.paths)
        unstable.extend(child.unstable)
    return _Encoded(value=value, constructible=constructible, paths=tuple(paths), unstable=tuple(unstable))


def _non_constructible(value: Any, path: str, reason: str, **metadata: Any) -> _Encoded:
    """Encode a value the signature can neither rebuild nor identify across sessions.

    The value is identified by its ``repr``, which separates two values within
    one run but is not a durable identity: a function's repr carries its memory
    address, and an arbitrary object's repr need not reflect its state.
    """
    payload = {"type": "unsupported", "class": class_path(value), "repr": repr(value), "reason": reason}
    payload.update(metadata)
    return _Encoded(payload, constructible=False, paths=(path,), unstable=(path,))


def _canonical_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _canonical_identity(value: Any) -> Any:
    """Return *value* with every integral float replaced by the int it equals.

    ``30`` and ``30.0`` (and ``-0.0`` and ``0``) are the same number to every
    calculator parameter that accepts both, so they must not split a key.  Booleans
    stay booleans, and non-integral or non-finite floats are kept as they are.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {key: _canonical_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_identity(item) for item in value]
    return value


_COMMUTATIVE_PROPERTY_OPS = {"add", "mul", "eq", "ne"}


def _short_class_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _payload_sort_label(value: Any) -> str:  # noqa: PLR0911
    if not isinstance(value, dict):
        return type(value).__name__
    node_type = value.get("node")
    if node_type == "dataclass":
        return _short_class_name(str(value.get("class", "")))
    if node_type == "calculator_value_property":
        return _payload_sort_label(value.get("calculator"))
    if node_type == "op_property":
        return f"OpProperty.{value.get('op_name', '')}"
    if node_type == "filter_op":
        return f"FilterOp.{value.get('op', '')}"
    if "class" in value:
        return _short_class_name(str(value["class"]))
    return str(node_type or value.get("type") or type(value).__name__)


def _payload_sort_category(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    if value.get("node") == "constant_property":
        return 0
    if "node" not in value:
        return 0
    return 1


def _payload_sort_key(value: Any) -> tuple[Any, ...]:
    return (_payload_sort_category(value), _payload_sort_label(value).casefold(), _canonical_payload(value))


def _sort_encoded_commutative(items: list[_Encoded]) -> list[_Encoded]:
    return sorted(items, key=lambda item: _payload_sort_key(item.value))


def _object_array_element(value: Any, path: str, ancestors: frozenset[int]) -> Any:  # noqa: PLR0911
    """Encode one Python element for hashing, preserving its type and container boundaries."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if type(value) is float:
        return ["float", struct.pack("!d", value).hex()]
    if type(value) is complex:
        return ["complex", struct.pack("!dd", value.real, value.imag).hex()]
    if id(value) in ancestors:
        raise ValueError("recursive object array element")
    if type(value) in (list, tuple):
        ancestors = ancestors | {id(value)}
        return [class_path(value), [_object_array_element(item, path, ancestors) for item in value]]
    if type(value) is dict:
        ancestors = ancestors | {id(value)}
        return [
            "dict",
            [
                [_object_array_element(k, path, ancestors), _object_array_element(v, path, ancestors)]
                for k, v in value.items()
            ],
        ]
    if isinstance(value, (np.ndarray, np.generic)) and dtype_has_metadata(value.dtype):
        raise ValueError("object array element has dtype metadata")
    if isinstance(value, np.void) and value.dtype.hasobject:
        return [
            "record",
            str(value.dtype),
            [_object_array_element(value[k], path, ancestors) for k in value.dtype.names or ()],
        ]
    if isinstance(value, np.generic):
        return [class_path(value), str(value.dtype), value.tobytes().hex()]
    if type(value) in (np.ndarray, SimArray):
        return _Encoder.array_payload(value, path, 0, ancestors=ancestors).value
    raise ValueError(f"cannot identify object array element of type {class_path(value)} at {path}")


# ---------------------------------------------------------------------------
# _Encoder — encodes Python objects into JSON-compatible payloads
# ---------------------------------------------------------------------------


class _Encoder:
    """Encodes Python *values* of a calculator graph into JSON-compatible payloads.

    All methods are static.  Calculator payloads themselves — which node kinds exist
    and how each is written — live in the layer that owns the classes,
    :mod:`pynbodyext.core.calculate.nodes.serde`, and reach this module through
    :func:`pynbodyext.core.calculate.interfaces.encode_calculator_payload`.
    """

    # ------------------------------------------------------------------
    # Value encoding
    # ------------------------------------------------------------------

    @staticmethod
    def array_payload(
        value: np.ndarray, path: str, inline_bytes: int, *, ancestors: frozenset[int] = frozenset()
    ) -> _Encoded:
        array = np.asarray(value)
        base: dict[str, Any] = {
            "type": "array",
            "class": class_path(value),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "simarray": isinstance(value, SimArray),
        }
        if isinstance(value, SimArray):
            base["units"] = str(value.units)
        buf = io.BytesIO()
        try:
            if array.dtype.hasobject:
                if type(value) not in (np.ndarray, SimArray) or dtype_has_metadata(array.dtype):
                    raise ValueError("object array subclass or dtype metadata")
                digest = hashlib.sha256()
                ancestors = ancestors | {id(value)}
                for item in array.flat:
                    digest.update(_canonical_payload(_object_array_element(item, path, ancestors)).encode("utf-8"))
                    digest.update(b"\n")
                base.update(storage="object_hash", sha256=digest.hexdigest())
                return _Encoded(base, constructible=False, paths=(path,), unstable=(path,))
            np.save(buf, array, allow_pickle=False)
        except (TypeError, ValueError, NotImplementedError, RecursionError):
            # A manual calculation can still use the array. Distinct arrays with
            # opaque elements or layouts must never share a transient store key.
            base.update(storage="opaque", session_token=_session_token(value))
            return _Encoded(base, constructible=False, paths=(path,), unstable=(path,))
        data = buf.getvalue()
        base["sha256"] = hashlib.sha256(data).hexdigest()
        if array.nbytes <= inline_bytes:
            base["storage"] = "inline"
            base["data"] = base64.b64encode(data).decode("ascii")
            return _Encoded(base)
        base["storage"] = "hash"
        return _Encoded(base, constructible=False, paths=(path,))

    @staticmethod
    def encode_calculator_value(value: Any, path: str, inline_bytes: int) -> _Encoded:
        sig = calculator_to_signature(value, inline_array_bytes=inline_bytes, _path=path)
        return _Encoded(
            sig.payload, constructible=sig.constructible, paths=sig.non_constructible_paths, unstable=sig.unstable_paths
        )

    @staticmethod
    def encode_value(value: Any, path: str, inline_bytes: int, *, output_names: dict[int, str] | None = None) -> _Encoded:  # noqa: PLR0911
        """Encode an arbitrary Python value into a JSON-compatible payload."""
        if is_calculator_node(value):
            if output_names is not None and id(value) in output_names:
                return _Encoded({"type": "pipeline_output", "key": output_names[id(value)]})
            return _Encoder.encode_calculator_value(value, path, inline_bytes)
        if value is None or isinstance(value, (bool, int, float, str)):
            return _Encoded(value)
        if isinstance(value, np.generic):
            return _Encoded({"type": "numpy_scalar", "dtype": str(value.dtype), "value": value.item()})
        if isinstance(value, Enum):
            return _Encoded({"type": "enum", "class": class_path(value), "value": value.value})
        if isinstance(value, Family):
            return _Encoded({"type": "family", "name": value.name})
        if isinstance(value, units.UnitBase):
            return _Encoded({"type": "unit", "value": str(value)})
        if isinstance(value, np.ndarray):
            return _Encoder.array_payload(value, path, inline_bytes)

        def encode_child(item: Any, child_path: str) -> _Encoded:
            return _Encoder.encode_value(item, child_path, inline_bytes, output_names=output_names)

        if isinstance(value, (tuple, list)):
            items = [encode_child(item, f"{path}[{i}]") for i, item in enumerate(value)]
            kind = "tuple" if isinstance(value, tuple) else "list"
            return _merge_encoded({"type": kind, "items": [e.value for e in items]}, items)
        if isinstance(value, dict) or (output_names is not None and isinstance(value, Mapping)):
            pairs: list[dict[str, Any]] = []
            children: list[_Encoded] = []
            entries = value.items() if output_names is not None else sorted(value.items(), key=lambda p: repr(p[0]))
            for i, (k, v) in enumerate(entries):
                ke = encode_child(k, f"{path}.key[{i}]")
                ve = encode_child(v, f"{path}[{repr(k)}]")
                pairs.append({"key": ke.value, "value": ve.value})
                children.extend((ke, ve))
            return _merge_encoded({"type": "dict", "items": pairs}, children)
        if callable(value):
            return _non_constructible(value, path, "callables cannot be reconstructed from signature")
        return _non_constructible(value, path, "unsupported value type")


# ---------------------------------------------------------------------------
# Module-level helper that depends on _Encoder
# ---------------------------------------------------------------------------


def _encoded_values_equal(left: Any, right: Any, inline_array_bytes: int) -> bool:
    le = _Encoder.encode_value(left, "default", inline_array_bytes)
    re = _Encoder.encode_value(right, "default", inline_array_bytes)
    return _canonical_payload(_canonical_identity(le.value)) == _canonical_payload(_canonical_identity(re.value))


# ---------------------------------------------------------------------------
# _Decoder — decodes JSON-compatible payloads back into Python objects
# ---------------------------------------------------------------------------


class _Decoder:
    """Decodes JSON-compatible signature payloads back into Python objects.

    All methods are static.  Entry points used by ``calculator_from_signature``::

        _Decoder.decode_dataclass(payload)
        _Decoder.decode_special(payload)
    """

    #: Dispatch table for value-type payloads (keyed by ``payload["type"]``).
    _VALUE_DECODERS: ClassVar[dict[str, Callable[[dict[str, Any]], Any]]] = {}
    #: Dispatch table for special calculator nodes (keyed by ``payload["node"]``).
    _SPECIAL_DECODERS: ClassVar[dict[str, Callable[[dict[str, Any]], Any]]] = {}

    # ------------------------------------------------------------------
    # Atomic value decoders
    # ------------------------------------------------------------------

    @staticmethod
    def array(payload: dict[str, Any]) -> Any:
        if payload.get("storage") != "inline":
            raise ValueError("cannot reconstruct hash-only array signature")
        raw = base64.b64decode(payload["data"].encode("ascii"))
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        value: Any = arr
        if payload.get("simarray"):
            sa = SimArray(arr)
            if payload.get("units") is not None:
                sa.units = _make_unit(payload["units"])
            value = sa
        return value

    @staticmethod
    def numpy_scalar(payload: dict[str, Any]) -> Any:
        return np.dtype(payload["dtype"]).type(payload["value"])

    @staticmethod
    def enum_(payload: dict[str, Any]) -> Any:
        return import_object(payload["class"])(payload["value"])

    @staticmethod
    def family(payload: dict[str, Any]) -> Any:
        return get_family(payload["name"], False)

    @staticmethod
    def unit(payload: dict[str, Any]) -> Any:
        return _make_unit(payload["value"])

    @staticmethod
    def tuple_(payload: dict[str, Any]) -> Any:
        return tuple(_Decoder.decode_value(item) for item in payload["items"])

    @staticmethod
    def list_(payload: dict[str, Any]) -> Any:
        return [_Decoder.decode_value(item) for item in payload["items"]]

    @staticmethod
    def dict_(payload: dict[str, Any]) -> Any:
        return {_Decoder.decode_value(i["key"]): _Decoder.decode_value(i["value"]) for i in payload["items"]}

    @staticmethod
    def unsupported(payload: dict[str, Any]) -> Any:
        raise ValueError(f"cannot reconstruct unsupported value at {payload.get('class')}: {payload.get('reason')}")

    # ------------------------------------------------------------------
    # Value dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def decode_value(payload: Any) -> Any:
        """Decode any value payload back into a Python object."""
        if not isinstance(payload, dict):
            return payload
        if "node" in payload:
            return calculator_from_signature(CalculatorSignature(payload=payload))
        vtype = payload.get("type")
        handler = _Decoder._VALUE_DECODERS.get(vtype) if isinstance(vtype, str) else None
        if handler is None:
            raise ValueError(f"unsupported encoded value type: {vtype!r}")
        return handler(payload)

    @staticmethod
    def decode_calculator_value(payload: dict[str, Any]) -> Any:
        if "payload" in payload:
            return calculator_from_signature(CalculatorSignature.from_dict(payload))
        return calculator_from_signature(CalculatorSignature(payload=payload))

# Populate handler dicts after class definition.
_Decoder._VALUE_DECODERS = {
    "scalar": lambda p: p.get("value"),
    "numpy_scalar": _Decoder.numpy_scalar,
    "enum": _Decoder.enum_,
    "family": _Decoder.family,
    "unit": _Decoder.unit,
    "array": _Decoder.array,
    "tuple": _Decoder.tuple_,
    "list": _Decoder.list_,
    "dict": _Decoder.dict_,
    "calculator": lambda p: _Decoder.decode_calculator_value(p["value"]),
    "unsupported": _Decoder.unsupported,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculator_to_signature(
    calculator: Any, *, inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES, _path: str = "calculator"
) -> CalculatorSignature:
    """Return a structured signature for a calculator graph."""
    payload, constructible, paths, unstable = encode_calculator_payload(
        calculator, path=_path, inline_array_bytes=inline_array_bytes
    )
    return CalculatorSignature(
        payload=payload,
        constructible=constructible,
        non_constructible_paths=paths,
        stable=not unstable,
        unstable_paths=unstable,
    )


def calculator_from_signature(signature: CalculatorSignature | dict[str, Any] | str) -> Any:
    """Reconstruct a calculator from a trusted, constructible signature.

    Reconstruction can import the classes named in the signature and run their
    constructors or custom reconstruction hooks. Supply only recipes from a
    source whose Python code you trust.
    """
    if isinstance(signature, str):
        signature = CalculatorSignature.from_json(signature)
    elif isinstance(signature, dict):
        signature = CalculatorSignature.from_dict(signature)

    if not isinstance(signature, CalculatorSignature):
        raise TypeError(f"expected CalculatorSignature, dict, or JSON string; got {type(signature)!r}")
    if not signature.constructible:
        paths = ", ".join(signature.non_constructible_paths) or "unknown"
        raise ValueError(f"calculator signature is not constructible; non-constructible paths: {paths}")

    return decode_calculator_payload(signature.payload)


def _argument_value(calculator: Any, name: str) -> Any:
    """Return the value the constructor was given for *name*.

    ``params.fields`` records it before ``__post_init__`` runs, because that hook may
    replace a value with an equivalent object (``"ssc"`` -> ``CenPos("ssc")``); the
    recorded argument is what a label should print and what a signature should
    carry, so either way the same text describes the node.  Without a record (a
    slots-based subclass) the live value is used.
    """
    captured, value = captured_init_value(calculator, name)
    return value if captured else getattr(calculator, name, None)


def _encode_init_entries(calculator: Any, path: str, inline_array_bytes: int) -> list[tuple[str, _Encoded]]:
    """Encoded init entries: the arguments that are not the declared default.

    The one implementation of the rule.  It feeds the signature payload (identity)
    and the display label, so the two cannot drift; parameters left at their declared
    default are omitted, because the renderer spells them out from the class.
    A factory has no fixed default value. Record its initialized value explicitly
    so reconstruction and display do not need to call the factory again.
    """
    field_map = {f.name: f for f in dataclass_fields(type(calculator))}
    entries: list[tuple[str, _Encoded]] = []
    for spec in collect_param_specs(type(calculator)):
        item = field_map.get(spec.name)
        if item is None or not item.init or not spec.signature:
            continue
        value = _argument_value(calculator, spec.name)
        if item.default is not MISSING and _encoded_values_equal(value, item.default, inline_array_bytes):
            continue
        encoded = _Encoder.encode_value(value, f"{path}.init.{spec.name}", inline_array_bytes)
        entries.append((spec.name, encoded))
    return entries


def calculator_pretty_init_args(calculator: Any, *, inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES) -> str:
    """Return a pretty-printed string of the calculator's init arguments.

    Every positional parameter is rendered, including when it has its declared
    default. Keyword-only parameters at a fixed declared default are omitted.
    Values produced by a default factory are rendered explicitly. Both halves
    come from :func:`_encode_init_entries` and :func:`display_init_payload`,
    the same pair the signature payload and a stored signature use.
    """
    if not is_dataclass(calculator):
        return ""

    from .render import TreePrinter, display_init_payload

    path = signature_type_name(calculator)
    payload = {
        "class": path,
        "init": {name: enc.value for name, enc in _encode_init_entries(calculator, "init", inline_array_bytes)},
    }
    init_payload = display_init_payload(payload, inline_array_bytes=inline_array_bytes)
    if not init_payload:
        return ""
    return TreePrinter.dataclass_args({"class": path, "init": init_payload})


def calculator_parameter_rows(
    calculator: Any, *, inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES
) -> list[tuple[str, str, bool]]:
    """Return ``(name, value, is_default)`` for every signature parameter of *calculator*.

    The table counterpart of :func:`calculator_pretty_init_args`: every parameter
    that takes part in the signature, in declaration order — keyword-only ones
    too, since a table has room for them — with the value spelled the way the
    signature expression spells it and a flag for "left at its declared default".
    Hand-written nodes can also declare constructor values through their
    ``signature_payload`` mapping.
    """
    from .render import SignaturePrinter, default_init_payload

    if not is_dataclass(calculator):
        identity = calculator.signature_payload() if is_calculator_node(calculator) else None
        return [
            (
                name,
                SignaturePrinter.value(_Encoder.encode_value(value, f"init.{name}", inline_array_bytes).value),
                False,
            )
            for name, value in (identity or {}).items()
        ]

    explicit = {name: enc.value for name, enc in _encode_init_entries(calculator, "init", inline_array_bytes)}
    defaults = default_init_payload(type(calculator), inline_array_bytes=inline_array_bytes)
    field_map = {item.name: item for item in dataclass_fields(calculator)}
    rows: list[tuple[str, str, bool]] = []
    for spec in collect_param_specs(type(calculator)):
        item = field_map.get(spec.name)
        if item is None or not item.init or not spec.signature:
            continue
        if spec.name in explicit:
            rows.append((spec.name, SignaturePrinter.value(explicit[spec.name]), False))
        elif spec.name in defaults:
            rows.append((spec.name, SignaturePrinter.value(defaults[spec.name]), True))
    return rows


# ---------------------------------------------------------------------------
# Storage helpers (two-field pattern)
# ---------------------------------------------------------------------------


def calculator_pack(calculator: Any, *, inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES) -> tuple[str, str]:
    """Pack a calculator into a ``(pretty_key, signature_json)`` pair for storage.

    This is the recommended persistence pattern:

    - *pretty_key* — a compact, human-readable string suitable for display,
      search, and manual inspection (see :meth:`CalculatorSignature.pretty`).
    - *signature_json* — the full, lossless JSON representation that can be
      passed to :func:`calculator_unpack` or :func:`calculator_from_signature`
      to reconstruct the original calculator.

    Parameters
    ----------
    calculator : CalculatorBase
        The calculator to pack for storage.
    inline_array_bytes : int, optional
        Maximum array size in bytes to inline in the signature payload.

    Returns
    -------
    tuple[str, str]
        ``(pretty_key, signature_json)``.
    """
    sig = calculator_to_signature(calculator, inline_array_bytes=inline_array_bytes)
    return sig.pretty(), sig.to_json()


def calculator_unpack(signature_json: str, *, pretty_key: str | None = None) -> Any:
    """Reconstruct a calculator from its stored signature JSON.

    This is a thin wrapper around :func:`calculator_from_signature` that
    accepts the *signature_json* field produced by :func:`calculator_pack`.

    Parameters
    ----------
    signature_json : str
        Trusted JSON previously returned by :func:`calculator_pack`. Reconstruction
        may import classes and run their Python constructors or hooks.
    pretty_key : str, optional
        The corresponding human-readable key.  Not used for reconstruction;
        accepted only for API symmetry / self-documenting call sites.

    Returns
    -------
    Any
        The reconstructed calculator object.

    Raises
    ------
    ValueError
        If the signature is not constructible.
    TypeError
        If *signature_json* is not a valid signature JSON string.
    """
    return calculator_from_signature(signature_json)
