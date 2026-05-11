"""Portable calculator signatures for reconstruction and indexing.

This module is intentionally separate from ``CalculatorBase.signature()``.
The existing runtime signature remains optimized for in-run cache keys, while
``CalculatorSignature`` stores enough structured data to reconstruct supported
calculator objects in the current Python environment.

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
import importlib
import io
import json
from dataclasses import MISSING, Field, dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pynbody import units
from pynbody.array import SimArray
from pynbody.family import Family, get_family

from pynbodyext.core.calculate.params.fields import collect_param_specs

if TYPE_CHECKING:
    from collections.abc import Callable

SIGNATURE_SCHEMA = "pynbodyext.calculator.signature/v1"
DEFAULT_INLINE_ARRAY_BYTES = 128
__all__ = [
    "CalculatorSignature",
    "calculator_from_signature",
    "calculator_to_signature",
    "calculator_pretty_init_args",
]


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _make_unit(value: Any) -> Any:
    return units.Unit(value)


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

    ``payload`` is JSON-compatible and can be hashed or stored. If
    ``constructible`` is false, the payload can still identify the calculator,
    but cannot be used to reconstruct it without external data.
    """

    payload: dict[str, Any]
    constructible: bool = True
    non_constructible_paths: tuple[str, ...] = ()

    @property
    def hash(self) -> str:
        """Return a stable hash for this signature payload."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def short_hash(self, *, length: int = 12) -> str:
        return self.hash[:length]

    def frozen_payload(self) -> Any:
        return _freeze_signature_value(self.payload)

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
        return data

    def to_json(self, *, indent: int | None = None, full: bool = False) -> str:
        return json.dumps(
            self.as_dict(full=full),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    def pretty(self) -> str:
        """Return a compact canonical expression for this signature."""
        from .render import _pretty_calculator
        return _pretty_calculator(self.payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculatorSignature:
        schema = data.get("schema")
        if schema is not None and schema != SIGNATURE_SCHEMA:
            raise ValueError(f"unsupported calculator signature schema: {data.get('schema')!r}")
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("calculator signature payload must be a dictionary")
        return cls(
            payload=payload,
            constructible=bool(data.get("constructible", True)),
            non_constructible_paths=tuple(str(p) for p in data.get("non_constructible_paths", ())),
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
    paths: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Module-level utilities (pure helpers, no encode/decode)
# ---------------------------------------------------------------------------


def _class_path(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_object(path: str) -> Any:
    module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"invalid object path: {path!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _merge_encoded(value: Any, children: list[_Encoded]) -> _Encoded:
    paths: list[str] = []
    constructible = True
    for child in children:
        if not child.constructible:
            constructible = False
        paths.extend(child.paths)
    return _Encoded(value=value, constructible=constructible, paths=tuple(paths))


def _non_constructible(value: Any, path: str, reason: str, **metadata: Any) -> _Encoded:
    payload = {
        "type": "unsupported",
        "class": _class_path(value),
        "repr": repr(value),
        "reason": reason,
    }
    payload.update(metadata)
    return _Encoded(payload, constructible=False, paths=(path,))


def _canonical_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_COMMUTATIVE_PROPERTY_OPS = {"add", "mul", "eq", "ne"}


def _short_class_name(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def _payload_sort_label(value: Any) -> str: # noqa: PLR0911
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


def _field_default(item: Field[Any]) -> tuple[bool, Any]:
    if item.default is not MISSING:
        return True, item.default
    if item.default_factory is not MISSING:
        return True, item.default_factory()
    return False, None


# ---------------------------------------------------------------------------
# _Encoder — encodes Python objects into JSON-compatible payloads
# ---------------------------------------------------------------------------


class _Encoder:
    """Encodes Python calculator graphs into JSON-compatible signature payloads.

    All methods are static.  Entry points used by ``calculator_to_signature``::

        _Encoder.encode_special(calculator, path, inline_bytes)
        _Encoder.encode_dataclass(calculator, path, inline_bytes)
        _Encoder.encode_generic(calculator, path, inline_bytes)
    """

    # ------------------------------------------------------------------
    # Transform / scope helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_transform_base(value: Any) -> bool:
        try:
            from pynbodyext.core.calculate.nodes.transforms import TransformBase
        except Exception:
            return False
        return isinstance(value, TransformBase)

    @staticmethod
    def transform_state(
        calculator: Any, inline_bytes: int, path: str
    ) -> tuple[dict[str, Any] | None, list[_Encoded]]:
        if not _Encoder.is_transform_base(calculator):
            return None, []
        from .enums import RevertPolicy
        state: dict[str, Any] = {}
        children: list[_Encoded] = []
        if calculator.revert_policy != RevertPolicy.ALWAYS:
            rp = _Encoder.encode_value(calculator.revert_policy, f"{path}.revert_policy", inline_bytes)
            state["revert_policy"] = rp.value
            children.append(rp)
        if calculator.measure_filter is not None:
            mf = _Encoder.encode_value(calculator.measure_filter, f"{path}.measure_filter", inline_bytes)
            state["measure_filter"] = mf.value
            children.append(mf)
        if not state:
            return None, []
        return state, children

    @staticmethod
    def encode_scope(scope: Any, path: str, inline_bytes: int) -> _Encoded:
        from .enums import RevertPolicy
        transforms = [
            _Encoder.encode_calculator_value(t, f"{path}.transforms[{idx}]", inline_bytes)
            for idx, t in enumerate(scope.transforms)
        ]
        children: list[_Encoded] = [*transforms]
        payload: dict[str, Any] = {}
        if transforms:
            payload["transforms"] = [item.value for item in transforms]
        if scope.filter is not None:
            fe = _Encoder.encode_calculator_value(scope.filter, f"{path}.filter", inline_bytes)
            payload["filter"] = fe.value
            children.append(fe)
        if scope.revert_policy != RevertPolicy.ALWAYS:
            payload["revert_policy"] = scope.revert_policy.value
        return _merge_encoded(payload, children)

    # ------------------------------------------------------------------
    # Value encoding
    # ------------------------------------------------------------------

    @staticmethod
    def array_payload(value: np.ndarray, path: str, inline_bytes: int) -> _Encoded:
        array = np.asarray(value)
        base: dict[str, Any] = {
            "type": "array",
            "class": _class_path(value),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "simarray": isinstance(value, SimArray),
        }
        if isinstance(value, SimArray):
            base["units"] = str(value.units)
        buf = io.BytesIO()
        try:
            np.save(buf, array, allow_pickle=False)
        except Exception as exc:
            base["reason"] = f"array cannot be serialized without pickle: {exc}"
            return _Encoded(base, constructible=False, paths=(path,))
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
        return _Encoded(sig.payload, constructible=sig.constructible, paths=sig.non_constructible_paths)

    @staticmethod
    def encode_value(value: Any, path: str, inline_bytes: int) -> _Encoded: # noqa: PLR0911
        """Encode an arbitrary Python value into a JSON-compatible payload."""
        from pynbodyext.core.calculate.nodes.base import CalculatorBase
        if isinstance(value, CalculatorBase):
            return _Encoder.encode_calculator_value(value, path, inline_bytes)
        if value is None or isinstance(value, (bool, int, float, str)):
            return _Encoded(value)
        if isinstance(value, np.generic):
            return _Encoded({"type": "numpy_scalar", "dtype": str(value.dtype), "value": value.item()})
        if isinstance(value, Enum):
            return _Encoded({"type": "enum", "class": _class_path(value), "value": value.value})
        if isinstance(value, Family):
            return _Encoded({"type": "family", "name": value.name})
        if isinstance(value, units.UnitBase):
            return _Encoded({"type": "unit", "value": str(value)})
        if isinstance(value, np.ndarray):
            return _Encoder.array_payload(value, path, inline_bytes)
        if isinstance(value, tuple):
            items = [_Encoder.encode_value(item, f"{path}[{i}]", inline_bytes) for i, item in enumerate(value)]
            return _merge_encoded({"type": "tuple", "items": [e.value for e in items]}, items)
        if isinstance(value, list):
            items = [_Encoder.encode_value(item, f"{path}[{i}]", inline_bytes) for i, item in enumerate(value)]
            return _merge_encoded({"type": "list", "items": [e.value for e in items]}, items)
        if isinstance(value, dict):
            pairs: list[dict[str, Any]] = []
            children: list[_Encoded] = []
            for i, (k, v) in enumerate(sorted(value.items(), key=lambda p: repr(p[0]))):
                ke = _Encoder.encode_value(k, f"{path}.key[{i}]", inline_bytes)
                ve = _Encoder.encode_value(v, f"{path}[{repr(k)}]", inline_bytes)
                pairs.append({"key": ke.value, "value": ve.value})
                children.extend((ke, ve))
            return _merge_encoded({"type": "dict", "items": pairs}, children)
        if callable(value):
            return _non_constructible(value, path, "callables cannot be reconstructed from signature")
        return _non_constructible(value, path, "unsupported value type")

    # ------------------------------------------------------------------
    # Calculator encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode_dataclass(calculator: Any, path: str, inline_bytes: int) -> _Encoded:
        """Encode a dataclass-based calculator."""
        ts, children = _Encoder.transform_state(calculator, inline_bytes, path)
        init_payload: dict[str, Any] = {}
        field_map = {f.name: f for f in dataclass_fields(type(calculator))}
        for spec in collect_param_specs(type(calculator)):
            item = field_map[spec.name]
            if not item.init or not spec.signature:
                continue
            value = getattr(calculator, spec.name)
            has_default, default = _field_default(item)
            if has_default and _encoded_values_equal(value, default, inline_bytes):
                continue
            enc = _Encoder.encode_value(value, f"{path}.init.{spec.name}", inline_bytes)
            init_payload[spec.name] = enc.value
            children.append(enc)
        payload: dict[str, Any] = {"node": "dataclass", "class": _class_path(calculator)}
        if init_payload:
            payload["init"] = init_payload
        if ts is not None:
            payload["transform"] = ts
        return _merge_encoded(payload, children)

    @staticmethod
    def encode_special(calculator: Any, path: str, inline_bytes: int) -> _Encoded | None:
        """Encode known special calculator types; returns None for unknown types."""
        from pynbodyext.core.calculate.nodes.base import BoundCalculator, CombinedCalculator
        from pynbodyext.core.calculate.nodes.expr import (
            CalculatorValueProperty,
            ConstantProperty,
            LambdaProperty,
            OpProperty,
        )
        from pynbodyext.core.calculate.nodes.filters import AndFilter, NotFilter, OrFilter
        from pynbodyext.core.calculate.nodes.transforms import TransformChain

        ts, ts_children = _Encoder.transform_state(calculator, inline_bytes, path)
        encoded: _Encoded | None = None

        if isinstance(calculator, BoundCalculator):
            base = _Encoder.encode_calculator_value(calculator.base, f"{path}.base", inline_bytes)
            scope = _Encoder.encode_scope(calculator.scope, f"{path}.scope", inline_bytes)
            encoded = _merge_encoded(
                {"node": "bound", "base": base.value, "scope": scope.value}, [base, scope]
            )

        elif isinstance(calculator, CombinedCalculator):
            items = [
                _Encoder.encode_calculator_value(item, f"{path}.items[{i}]", inline_bytes)
                for i, item in enumerate(calculator.items)
            ]
            encoded = _merge_encoded({"node": "combined", "items": [e.value for e in items]}, items)

        elif isinstance(calculator, TransformChain):
            transforms = [
                _Encoder.encode_calculator_value(t, f"{path}.transforms[{i}]", inline_bytes)
                for i, t in enumerate(calculator.transforms)
            ]
            payload: dict[str, Any] = {
                "node": "transform_chain",
                "class": _class_path(calculator),
                "transforms": [e.value for e in transforms],
            }
            if ts is not None:
                payload["transform"] = ts
            encoded = _merge_encoded(payload, [*ts_children, *transforms])

        elif isinstance(calculator, (AndFilter, OrFilter)):
            op = "and" if isinstance(calculator, AndFilter) else "or"
            left = _Encoder.encode_calculator_value(calculator.left, f"{path}.left", inline_bytes)
            right = _Encoder.encode_calculator_value(calculator.right, f"{path}.right", inline_bytes)
            left, right = _sort_encoded_commutative([left, right])
            encoded = _merge_encoded(
                {"node": "filter_op", "op": op, "left": left.value, "right": right.value}, [left, right]
            )

        elif isinstance(calculator, NotFilter):
            child = _Encoder.encode_calculator_value(calculator.child, f"{path}.child", inline_bytes)
            encoded = _merge_encoded({"node": "filter_op", "op": "not", "child": child.value}, [child])

        elif isinstance(calculator, ConstantProperty):
            v = _Encoder.encode_value(calculator._value, f"{path}.value", inline_bytes)
            encoded = _merge_encoded({"node": "constant_property", "value": v.value}, [v])

        elif isinstance(calculator, CalculatorValueProperty):
            nested = _Encoder.encode_calculator_value(calculator.calculator, f"{path}.calculator", inline_bytes)
            encoded = _merge_encoded({"node": "calculator_value_property", "calculator": nested.value}, [nested])

        elif isinstance(calculator, OpProperty):
            operands = [
                _Encoder.encode_calculator_value(op, f"{path}.operands[{i}]", inline_bytes)
                for i, op in enumerate(calculator.operands)
            ]
            if calculator.op_name in _COMMUTATIVE_PROPERTY_OPS:
                operands = _sort_encoded_commutative(operands)
            encoded = _merge_encoded(
                {"node": "op_property", "op_name": calculator.op_name, "operands": [e.value for e in operands]},
                operands,
            )

        elif isinstance(calculator, LambdaProperty):
            payload = {
                "node": "lambda_property",
                "class": _class_path(calculator),
                "func_repr": repr(calculator._func),
            }
            encoded = _Encoded(payload, constructible=False, paths=(f"{path}.func",))

        return encoded

    @staticmethod
    def encode_generic(calculator: Any, path: str, inline_bytes: int) -> _Encoded:
        """Encode a generic (non-dataclass, non-special) calculator."""
        payload_items = calculator.signature_payload()
        children: list[_Encoded] = []
        identity: dict[str, Any] = {}
        if payload_items is None:
            identity["opaque_id"] = id(calculator)
        else:
            for key, value in payload_items.items():
                enc = _Encoder.encode_value(value, f"{path}.identity.{key}", inline_bytes)
                identity[key] = enc.value
                children.append(enc)
        deps = [
            _Encoder.encode_calculator_value(dep, f"{path}.deps[{i}]", inline_bytes)
            for i, dep in enumerate(calculator.dependencies())
        ]
        children.extend(deps)
        ts, ts_children = _Encoder.transform_state(calculator, inline_bytes, path)
        children.extend(ts_children)
        payload: dict[str, Any] = {"node": "generic", "class": _class_path(calculator), "identity": identity}
        if deps:
            payload["deps"] = [e.value for e in deps]
        if ts is not None:
            payload["transform"] = ts
        merged = _merge_encoded(payload, children)
        return _Encoded(merged.value, constructible=False, paths=(*merged.paths, path))


# ---------------------------------------------------------------------------
# Module-level helper that depends on _Encoder
# ---------------------------------------------------------------------------


def _encoded_values_equal(left: Any, right: Any, inline_array_bytes: int) -> bool:
    le = _Encoder.encode_value(left, "default", inline_array_bytes)
    re = _Encoder.encode_value(right, "default", inline_array_bytes)
    return _canonical_payload(le.value) == _canonical_payload(re.value)


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
        return _import_object(payload["class"])(payload["value"])

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
        raise ValueError(
            f"cannot reconstruct unsupported value at {payload.get('class')}: {payload.get('reason')}"
        )

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

    # ------------------------------------------------------------------
    # Calculator node decoders
    # ------------------------------------------------------------------

    @staticmethod
    def decode_bound(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.nodes.base import BoundCalculator
        return BoundCalculator(
            base=_Decoder.decode_value(payload["base"]),
            scope=_Decoder.decode_scope(payload["scope"]),
        )

    @staticmethod
    def decode_combined(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.nodes.base import CombinedCalculator
        return CombinedCalculator(*(_Decoder.decode_value(item) for item in payload["items"]))

    @staticmethod
    def decode_transform_chain(payload: dict[str, Any]) -> Any:
        cls = _import_object(payload["class"])
        return _Decoder.apply_transform_state(
            cls(*(_Decoder.decode_value(item) for item in payload["transforms"])),
            payload.get("transform"),
        )

    @staticmethod
    def decode_filter_op(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.nodes.filters import AndFilter, NotFilter, OrFilter
        op = payload["op"]
        if op == "and":
            return AndFilter(_Decoder.decode_value(payload["left"]), _Decoder.decode_value(payload["right"]))
        if op == "or":
            return OrFilter(_Decoder.decode_value(payload["left"]), _Decoder.decode_value(payload["right"]))
        if op == "not":
            return NotFilter(_Decoder.decode_value(payload["child"]))
        raise ValueError(f"unsupported filter op: {op!r}")

    @staticmethod
    def decode_constant_property(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.nodes.expr import ConstantProperty
        return ConstantProperty(_Decoder.decode_value(payload["value"]))

    @staticmethod
    def decode_calculator_value_property(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.nodes.expr import CalculatorValueProperty
        return CalculatorValueProperty(_Decoder.decode_value(payload["calculator"]))

    @staticmethod
    def decode_op_property(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.nodes.expr import OpProperty
        return OpProperty(payload["op_name"], [_Decoder.decode_value(item) for item in payload["operands"]])

    # ------------------------------------------------------------------
    # Calculator dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def decode_special(payload: dict[str, Any]) -> Any:
        """Decode a special (non-dataclass) calculator payload."""
        node_type = payload["node"]
        if node_type == "lambda_property":
            raise ValueError("cannot reconstruct LambdaProperty from signature")
        handler = _Decoder._SPECIAL_DECODERS.get(node_type)
        if handler is None:
            raise ValueError(f"unsupported special calculator node: {node_type!r}")
        return handler(payload)

    @staticmethod
    def decode_dataclass(payload: dict[str, Any]) -> Any:
        """Decode a dataclass calculator payload."""
        cls = _import_object(payload["class"])
        values = {name: _Decoder.decode_value(enc) for name, enc in payload.get("init", {}).items()}
        return _Decoder.apply_transform_state(cls(**values), payload.get("transform"))

    # ------------------------------------------------------------------
    # Scope / transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode_scope(payload: dict[str, Any]) -> Any:
        from pynbodyext.core.calculate.runtime.scopes import ScopeSpec

        from .enums import RevertPolicy
        transforms = tuple(_Decoder.decode_value(item) for item in payload.get("transforms", ()))
        filter_node = _Decoder.decode_value(payload["filter"]) if "filter" in payload else None
        revert_policy = RevertPolicy(payload.get("revert_policy", RevertPolicy.ALWAYS.value))
        return ScopeSpec(transforms=transforms, filter=filter_node, revert_policy=revert_policy)

    @staticmethod
    def apply_transform_state(calculator: Any, payload: dict[str, Any] | None) -> Any:
        if payload is not None:
            if "revert_policy" in payload:
                calculator.revert_policy = _Decoder.decode_value(payload["revert_policy"])
            if "measure_filter" in payload:
                calculator.measure_filter = _Decoder.decode_value(payload["measure_filter"])
        return calculator


# Populate handler dicts after class definition.
_Decoder._VALUE_DECODERS = {
    "scalar":      lambda p: p.get("value"),
    "numpy_scalar": _Decoder.numpy_scalar,
    "enum":        _Decoder.enum_,
    "family":      _Decoder.family,
    "unit":        _Decoder.unit,
    "array":       _Decoder.array,
    "tuple":       _Decoder.tuple_,
    "list":        _Decoder.list_,
    "dict":        _Decoder.dict_,
    "calculator":  lambda p: _Decoder.decode_calculator_value(p["value"]),
    "unsupported": _Decoder.unsupported,
}
_Decoder._SPECIAL_DECODERS = {
    "bound":                     _Decoder.decode_bound,
    "combined":                  _Decoder.decode_combined,
    "transform_chain":           _Decoder.decode_transform_chain,
    "filter_op":                 _Decoder.decode_filter_op,
    "constant_property":         _Decoder.decode_constant_property,
    "calculator_value_property": _Decoder.decode_calculator_value_property,
    "op_property":               _Decoder.decode_op_property,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculator_to_signature(
    calculator: Any,
    *,
    inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES,
    _path: str = "calculator",
) -> CalculatorSignature:
    """Return a structured signature for a calculator graph."""
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

    if not isinstance(calculator, CalculatorBase):
        raise TypeError(f"expected CalculatorBase, got {type(calculator)!r}")

    encoded = _Encoder.encode_special(calculator, _path, inline_array_bytes)
    if encoded is None:
        if is_dataclass(calculator):
            encoded = _Encoder.encode_dataclass(calculator, _path, inline_array_bytes) # type: ignore[unreachable]
        else:
            encoded = _Encoder.encode_generic(calculator, _path, inline_array_bytes)

    return CalculatorSignature(
        payload=encoded.value,
        constructible=encoded.constructible,
        non_constructible_paths=encoded.paths,
    )


def calculator_from_signature(signature: CalculatorSignature | dict[str, Any] | str) -> Any:
    """Reconstruct a calculator from a constructible signature."""
    if isinstance(signature, str):
        signature = CalculatorSignature.from_json(signature)
    elif isinstance(signature, dict):
        signature = CalculatorSignature.from_dict(signature)

    if not isinstance(signature, CalculatorSignature):
        raise TypeError(f"expected CalculatorSignature, dict, or JSON string; got {type(signature)!r}")
    if not signature.constructible:
        paths = ", ".join(signature.non_constructible_paths) or "unknown"
        raise ValueError(f"calculator signature is not constructible; non-constructible paths: {paths}")

    payload = signature.payload
    if payload.get("node") == "dataclass":
        return _Decoder.decode_dataclass(payload)
    return _Decoder.decode_special(payload)


def calculator_pretty_init_args(
    calculator: Any,
    *,
    inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES,
) -> str:
    """Return a pretty-printed string of the calculator's init arguments."""
    if not is_dataclass(calculator):
        return ""

    field_map = {f.name: f for f in dataclass_fields(type(calculator))}  # type: ignore[arg-type]
    init_payload: dict[str, Any] = {}

    for spec in collect_param_specs(type(calculator)):
        item = field_map.get(spec.name)
        if item is None or not item.init or not spec.signature:
            continue
        value = getattr(calculator, spec.name)
        has_default, default = _field_default(item)
        if has_default and _encoded_values_equal(value, default, inline_array_bytes):
            continue
        enc = _Encoder.encode_value(value, f"init.{spec.name}", inline_array_bytes)
        init_payload[spec.name] = enc.value

    if not init_payload:
        return ""

    from .render import _tree_dataclass_args
    return _tree_dataclass_args({"class": _class_path(calculator), "init": init_payload})
