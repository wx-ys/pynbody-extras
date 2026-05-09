"""Portable calculator signatures for reconstruction and indexing.

This module is intentionally separate from ``CalculatorBase.signature()``.
The existing runtime signature remains optimized for in-run cache keys, while
``CalculatorSignature`` stores enough structured data to reconstruct supported
calculator objects in the current Python environment.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
from dataclasses import MISSING, Field, dataclass, fields as dataclass_fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

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
        """Return a short hash prefix."""
        return self.hash[:length]

    def frozen_payload(self) -> Any:
        """Return a hashable representation of the payload."""
        return _freeze_signature_value(self.payload)

    def cache_key(self) -> tuple[Any, ...]:
        return ("calculator_signature", self.frozen_payload())

    def as_dict(self, *, full: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible dictionary.

        The default representation is compact and intended for keys/storage.
        Use ``full=True`` to include schema and reconstruction diagnostics.
        """
        data: dict[str, Any] = {"payload": self.payload}
        if full:
            data["schema"] = SIGNATURE_SCHEMA
        if full or not self.constructible:
            data["constructible"] = self.constructible
        if full or self.non_constructible_paths:
            data["non_constructible_paths"] = list(self.non_constructible_paths)
        return data

    def to_json(self, *, indent: int | None = None, full: bool = False) -> str:
        """Return a canonical JSON representation."""
        return json.dumps(
            self.as_dict(full=full),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
        )

    def pretty(self) -> str:
        """Return a compact canonical expression for this signature."""
        return _pretty_calculator(self.payload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculatorSignature:
        """Create a signature object from :meth:`as_dict` output."""
        schema = data.get("schema")
        if schema is not None and schema != SIGNATURE_SCHEMA:
            raise ValueError(f"unsupported calculator signature schema: {data.get('schema')!r}")
        payload = data.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("calculator signature payload must be a dictionary")
        return cls(
            payload=payload,
            constructible=bool(data.get("constructible", True)),
            non_constructible_paths=tuple(str(item) for item in data.get("non_constructible_paths", ())),
        )

    @classmethod
    def from_json(cls, text: str) -> CalculatorSignature:
        """Create a signature object from JSON."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("calculator signature JSON must decode to a dictionary")
        return cls.from_dict(data)


@dataclass(slots=True)
class _Encoded:
    value: Any
    constructible: bool = True
    paths: tuple[str, ...] = ()


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


def _payload_sort_label(value: Any) -> str:
    if not isinstance(value, dict):
        label = type(value).__name__
    else:
        node_type = value.get("node")
        if node_type == "dataclass":
            label = _short_class_name(str(value.get("class", "")))
        elif node_type == "calculator_value_property":
            label = _payload_sort_label(value.get("calculator"))
        elif node_type == "op_property":
            label = f"OpProperty.{value.get('op_name', '')}"
        elif node_type == "filter_op":
            label = f"FilterOp.{value.get('op', '')}"
        elif "class" in value:
            label = _short_class_name(str(value["class"]))
        else:
            label = str(node_type or value.get("type") or type(value).__name__)
    return label


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


def _encoded_values_equal(left: Any, right: Any, inline_array_bytes: int) -> bool:
    left_encoded = _encode_value(left, "default", inline_array_bytes)
    right_encoded = _encode_value(right, "default", inline_array_bytes)
    return _canonical_payload(left_encoded.value) == _canonical_payload(right_encoded.value)


def _field_default(item: Field[Any]) -> tuple[bool, Any]:
    if item.default is not MISSING:
        return True, item.default
    if item.default_factory is not MISSING:
        return True, item.default_factory()
    return False, None


def _array_payload(value: np.ndarray, path: str, inline_array_bytes: int) -> _Encoded:
    array = np.asarray(value)
    base_payload: dict[str, Any] = {
        "type": "array",
        "class": _class_path(value),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "simarray": isinstance(value, SimArray),
    }
    if isinstance(value, SimArray):
        base_payload["units"] = str(value.units)

    buffer = io.BytesIO()
    try:
        np.save(buffer, array, allow_pickle=False)
    except Exception as exc:
        base_payload["reason"] = f"array cannot be serialized without pickle: {exc}"
        return _Encoded(base_payload, constructible=False, paths=(path,))

    data = buffer.getvalue()
    base_payload["sha256"] = hashlib.sha256(data).hexdigest()
    if array.nbytes <= inline_array_bytes:
        base_payload["storage"] = "inline"
        base_payload["data"] = base64.b64encode(data).decode("ascii")
        return _Encoded(base_payload)

    base_payload["storage"] = "hash"
    return _Encoded(base_payload, constructible=False, paths=(path,))


def _encode_calculator_value(value: Any, path: str, inline_array_bytes: int) -> _Encoded:
    signature = calculator_to_signature(value, inline_array_bytes=inline_array_bytes, _path=path)
    return _Encoded(
        signature.payload,
        constructible=signature.constructible,
        paths=signature.non_constructible_paths,
    )


def _encode_value(value: Any, path: str, inline_array_bytes: int) -> _Encoded:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

    if isinstance(value, CalculatorBase):
        encoded = _encode_calculator_value(value, path, inline_array_bytes)
    elif value is None or isinstance(value, (bool, int, float, str)):
        encoded = _Encoded(value)
    elif isinstance(value, np.generic):
        encoded = _Encoded({"type": "numpy_scalar", "dtype": str(value.dtype), "value": value.item()})
    elif isinstance(value, Enum):
        encoded = _Encoded({"type": "enum", "class": _class_path(value), "value": value.value})
    elif isinstance(value, Family):
        encoded = _Encoded({"type": "family", "name": value.name})
    elif isinstance(value, units.UnitBase):
        encoded = _Encoded({"type": "unit", "value": str(value)})
    elif isinstance(value, np.ndarray):
        encoded = _array_payload(value, path, inline_array_bytes)
    elif isinstance(value, tuple):
        items = [_encode_value(item, f"{path}[{idx}]", inline_array_bytes) for idx, item in enumerate(value)]
        encoded = _merge_encoded({"type": "tuple", "items": [item.value for item in items]}, items)
    elif isinstance(value, list):
        items = [_encode_value(item, f"{path}[{idx}]", inline_array_bytes) for idx, item in enumerate(value)]
        encoded = _merge_encoded({"type": "list", "items": [item.value for item in items]}, items)
    elif isinstance(value, dict):
        pairs: list[dict[str, Any]] = []
        children: list[_Encoded] = []
        for idx, (key, item_value) in enumerate(sorted(value.items(), key=lambda pair: repr(pair[0]))):
            key_encoded = _encode_value(key, f"{path}.key[{idx}]", inline_array_bytes)
            item_encoded = _encode_value(item_value, f"{path}[{repr(key)}]", inline_array_bytes)
            pairs.append({"key": key_encoded.value, "value": item_encoded.value})
            children.extend((key_encoded, item_encoded))
        encoded = _merge_encoded({"type": "dict", "items": pairs}, children)
    elif callable(value):
        encoded = _non_constructible(value, path, "callables cannot be reconstructed from signature")
    else:
        encoded = _non_constructible(value, path, "unsupported value type")
    return encoded


def _decode_array_value(payload: dict[str, Any]) -> Any:
    if payload.get("storage") != "inline":
        raise ValueError("cannot reconstruct hash-only array signature")
    raw = base64.b64decode(payload["data"].encode("ascii"))
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    value: Any = array
    if payload.get("simarray"):
        sim_array = SimArray(array)
        if payload.get("units") is not None:
            sim_array.units = _make_unit(payload["units"])
        value = sim_array
    return value
def _decode_numpy_scalar(payload: dict[str, Any]) -> Any:
    return np.dtype(payload["dtype"]).type(payload["value"])


def _decode_enum_value(payload: dict[str, Any]) -> Any:
    return _import_object(payload["class"])(payload["value"])


def _decode_family_value(payload: dict[str, Any]) -> Any:
    return get_family(payload["name"], False)


def _decode_unit_value(payload: dict[str, Any]) -> Any:
    return _make_unit(payload["value"])


def _decode_tuple_value(payload: dict[str, Any]) -> Any:
    return tuple(_decode_value(item) for item in payload["items"])


def _decode_list_value(payload: dict[str, Any]) -> Any:
    return [_decode_value(item) for item in payload["items"]]


def _decode_dict_value(payload: dict[str, Any]) -> Any:
    return {
        _decode_value(item["key"]): _decode_value(item["value"])
        for item in payload["items"]
    }


def _raise_unsupported_value(payload: dict[str, Any]) -> Any:
    raise ValueError(
        f"cannot reconstruct unsupported value at {payload.get('class')}: "
        f"{payload.get('reason')}"
    )


_VALUE_DECODERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "scalar": lambda payload: payload.get("value"),
    "numpy_scalar": _decode_numpy_scalar,
    "enum": _decode_enum_value,
    "family": _decode_family_value,
    "unit": _decode_unit_value,
    "array": _decode_array_value,
    "tuple": _decode_tuple_value,
    "list": _decode_list_value,
    "dict": _decode_dict_value,
    "calculator": lambda payload: _decode_calculator_value(payload["value"]),
    "unsupported": _raise_unsupported_value,
}


def _decode_value(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "node" in payload:
        return calculator_from_signature(CalculatorSignature(payload=payload))

    value_type = payload.get("type")
    handler = _VALUE_DECODERS.get(value_type) if isinstance(value_type, str) else None
    if handler is None:
        raise ValueError(f"unsupported encoded value type: {value_type!r}")
    return handler(payload)

def _decode_calculator_value(payload: dict[str, Any]) -> Any:
    if "payload" in payload:
        value = calculator_from_signature(CalculatorSignature.from_dict(payload))
    else:
        value = calculator_from_signature(CalculatorSignature(payload=payload))
    return value


def _quote_string(value: str) -> str:
    return json.dumps(value)


def _pretty_numpy_scalar(payload: dict[str, Any]) -> str:
    return repr(payload.get("value"))


def _pretty_enum(payload: dict[str, Any]) -> str:
    class_name = _short_class_name(str(payload.get("class", "")))
    return f"{class_name}.{payload.get('value')}"


def _pretty_family(payload: dict[str, Any]) -> str:
    return _quote_string(str(payload.get("name", "")))


def _pretty_unit(payload: dict[str, Any]) -> str:
    return f"Unit({_quote_string(str(payload.get('value', '')))})"


def _pretty_array(payload: dict[str, Any]) -> str:
    class_name = "SimArray" if payload.get("simarray") else "array"
    shape = tuple(payload.get("shape", ()))
    dtype = payload.get("dtype")
    suffix = f", units={_quote_string(str(payload['units']))}" if payload.get("units") is not None else ""
    return f"{class_name}(shape={shape!r}, dtype={dtype!r}{suffix})"


def _pretty_tuple(payload: dict[str, Any]) -> str:
    items = [_pretty_value(item) for item in payload.get("items", ())]
    text = f"({', '.join(items)})"
    if len(items) == 1:
        text = f"({items[0]},)"
    return text


def _pretty_list(payload: dict[str, Any]) -> str:
    return f"[{', '.join(_pretty_value(item) for item in payload.get('items', ()))}]"


def _pretty_dict(payload: dict[str, Any]) -> str:
    items = [
        f"{_pretty_value(item['key'])}: {_pretty_value(item['value'])}"
        for item in payload.get("items", ())
    ]
    return f"{{{', '.join(items)}}}"


def _pretty_unsupported(payload: dict[str, Any]) -> str:
    class_name = _short_class_name(str(payload.get("class", "unsupported")))
    return f"<unsupported {class_name}>"


_VALUE_PRETTY_HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "numpy_scalar": _pretty_numpy_scalar,
    "enum": _pretty_enum,
    "family": _pretty_family,
    "unit": _pretty_unit,
    "array": _pretty_array,
    "tuple": _pretty_tuple,
    "list": _pretty_list,
    "dict": _pretty_dict,
    "unsupported": _pretty_unsupported,
}


def _pretty_value(payload: Any) -> str:
    if isinstance(payload, str):
        text = _quote_string(payload)
    elif not isinstance(payload, dict):
        text = repr(payload)
    elif "node" in payload:
        text = _pretty_calculator(payload)
    else:
        value_type = payload.get("type")
        handler: Callable[[dict[str, Any]], str] | None = None
        if isinstance(value_type, str) and value_type in _VALUE_PRETTY_HANDLERS:
            handler = _VALUE_PRETTY_HANDLERS[value_type]
        text = handler(payload) if handler is not None else repr(payload)
    return text


def _pretty_dataclass_args(payload: dict[str, Any]) -> str:
    init_payload = payload.get("init", {})
    if not init_payload:
        return ""

    field_order: list[str] = []
    try:
        cls = _import_object(payload["class"])
        field_order = [
            spec.name
            for spec in collect_param_specs(cls)
            if spec.signature
        ]
    except Exception:
        field_order = list(init_payload)

    init_names = list(init_payload)
    positional_prefix = field_order[: len(init_names)] == init_names
    parts: list[str] = []
    if positional_prefix:
        parts.extend(_pretty_value(init_payload[name]) for name in init_names)
    else:
        ordered_names = [name for name in field_order if name in init_payload]
        ordered_names.extend(sorted(name for name in init_payload if name not in ordered_names))
        parts.extend(f"{name}={_pretty_value(init_payload[name])}" for name in ordered_names)
    return ", ".join(parts)


def _pretty_scope_suffix(scope: dict[str, Any]) -> str:
    suffix = ""
    revert_policy = scope.get("revert_policy")
    revert_arg = ", revert=False" if revert_policy == "never" else ""
    for transform in scope.get("transforms", ()):
        suffix += f".transform({_pretty_calculator(transform)}{revert_arg})"
        revert_arg = ""
    if "filter" in scope:
        suffix += f".filter({_pretty_filter(scope['filter'])})"
    return suffix


def _pretty_filter(payload: dict[str, Any], *, parent_op: str | None = None) -> str:
    if payload.get("node") != "filter_op":
        return _pretty_calculator(payload)
    op_name = payload.get("op")
    if op_name == "not":
        child = _pretty_filter(payload["child"], parent_op="not")
        text = f"~{child}"
    elif op_name in {"and", "or"}:
        separator = " & " if op_name == "and" else " | "
        left = _pretty_filter(payload["left"], parent_op=op_name)
        right = _pretty_filter(payload["right"], parent_op=op_name)
        text = separator.join((left, right))
    else:
        text = _pretty_calculator(payload)

    if parent_op == "not" and op_name in {"and", "or"}:
        return f"({text})"
    if parent_op == "and" and op_name == "or":
        return f"({text})"
    return text


def _pretty_op(payload: dict[str, Any], *, parent_op: str | None = None) -> str:
    op_name = payload.get("op_name")
    operands = payload.get("operands", ())
    symbol_map = {
        "add": " + ",
        "mul": " * ",
        "sub": " - ",
        "truediv": " / ",
        "pow": " ** ",
        "lt": " < ",
        "le": " <= ",
        "gt": " > ",
        "ge": " >= ",
        "eq": " == ",
        "ne": " != ",
    }
    precedence = {"add": 10, "sub": 10, "mul": 20, "truediv": 20, "pow": 30}
    unary_symbol = {"neg": "-", "pos": "+", "abs": "abs"}

    if op_name in {"add", "mul"}:
        text = symbol_map[op_name].join(_pretty_property_operand(item, parent_op=op_name) for item in operands)
    elif op_name in symbol_map and len(operands) == 2:
        left = _pretty_property_operand(operands[0], parent_op=op_name)
        right = _pretty_property_operand(operands[1], parent_op=op_name)
        text = f"{left}{symbol_map[op_name]}{right}"
    elif op_name in unary_symbol and len(operands) == 1:
        child = _pretty_property_operand(operands[0], parent_op=op_name)
        text = f"{unary_symbol[op_name]}({child})" if op_name == "abs" else f"{unary_symbol[op_name]}{child}"
    elif op_name == "clip" and len(operands) == 3:
        text = f"{_pretty_property_operand(operands[0])}.clip({_pretty_value(operands[1])}, {_pretty_value(operands[2])})"
    else:
        text = f"OpProperty({_quote_string(str(op_name))}, [{', '.join(_pretty_value(item) for item in operands)}])"

    if parent_op in precedence and op_name in precedence and precedence[op_name] < precedence[parent_op]:
        return f"({text})"
    return text


def _pretty_property_operand(payload: Any, *, parent_op: str | None = None) -> str:
    if isinstance(payload, dict) and payload.get("node") == "op_property":
        return _pretty_op(payload, parent_op=parent_op)
    return _pretty_value(payload)


def _pretty_dataclass_calculator(payload: dict[str, Any]) -> str:
    class_name = _short_class_name(str(payload.get("class", "Calculator")))
    text = f"{class_name}({_pretty_dataclass_args(payload)})"
    if "transform" in payload:
        text += _pretty_scope_suffix(payload["transform"])
    return text


def _pretty_bound_calculator(payload: dict[str, Any]) -> str:
    return f"{_pretty_calculator(payload['base'])}{_pretty_scope_suffix(payload.get('scope', {}))}"


def _pretty_combined_calculator(payload: dict[str, Any]) -> str:
    items = ", ".join(_pretty_calculator(item) for item in payload.get("items", ()))
    return f"CombinedCalculator({items})"


def _pretty_transform_chain(payload: dict[str, Any]) -> str:
    class_name = _short_class_name(str(payload.get("class", "TransformChain")))
    transforms = ", ".join(_pretty_calculator(item) for item in payload.get("transforms", ()))
    text = f"{class_name}({transforms})"
    if "transform" in payload:
        text += _pretty_scope_suffix(payload["transform"])
    return text


def _pretty_constant_property(payload: dict[str, Any]) -> str:
    return _pretty_value(payload["value"])


def _pretty_calculator_value_property(payload: dict[str, Any]) -> str:
    return _pretty_calculator(payload["calculator"])


def _pretty_lambda_property(payload: dict[str, Any]) -> str:
    return "LambdaProperty(<function>)"

def _pretty_generic_calculator(payload: dict[str, Any]) -> str:
    return _short_class_name(str(payload.get("class", "Calculator")))

_CALCULATOR_PRETTY_HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "dataclass": _pretty_dataclass_calculator,
    "bound": _pretty_bound_calculator,
    "combined": _pretty_combined_calculator,
    "transform_chain": _pretty_transform_chain,
    "filter_op": _pretty_filter,
    "constant_property": _pretty_constant_property,
    "calculator_value_property": _pretty_calculator_value_property,
    "op_property": _pretty_op,
    "lambda_property": _pretty_lambda_property,
    "generic": _pretty_generic_calculator,
}


def _pretty_calculator(payload: dict[str, Any]) -> str:
    node_type = payload.get("node")
    handler: Callable[[dict[str, Any]], str] | None = None
    if isinstance(node_type, str) and node_type in _CALCULATOR_PRETTY_HANDLERS:
        handler = _CALCULATOR_PRETTY_HANDLERS[node_type]
    text = handler(payload) if handler is not None else repr(payload)
    return text


def _transform_state(calculator: Any, inline_array_bytes: int, path: str) -> tuple[dict[str, Any] | None, list[_Encoded]]:
    if not _is_transform_base_instance(calculator):
        return None, []

    from .enums import RevertPolicy

    state: dict[str, Any] = {}
    children: list[_Encoded] = []
    if calculator.revert_policy != RevertPolicy.ALWAYS:
        revert_policy = _encode_value(calculator.revert_policy, f"{path}.revert_policy", inline_array_bytes)
        state["revert_policy"] = revert_policy.value
        children.append(revert_policy)
    if calculator.measure_filter is not None:
        measure_filter = _encode_value(calculator.measure_filter, f"{path}.measure_filter", inline_array_bytes)
        state["measure_filter"] = measure_filter.value
        children.append(measure_filter)
    if not state:
        return None, []
    return state, children


def _is_transform_base_instance(value: Any) -> bool:
    try:
        from pynbodyext.core.calculate.nodes.transforms import TransformBase
    except Exception:
        return False
    return isinstance(value, TransformBase)


def _apply_transform_state(calculator: Any, payload: dict[str, Any] | None) -> Any:
    if payload is not None:
        if "revert_policy" in payload:
            calculator.revert_policy = _decode_value(payload["revert_policy"])
        if "measure_filter" in payload:
            calculator.measure_filter = _decode_value(payload["measure_filter"])
    return calculator


def _encode_scope(scope: Any, path: str, inline_array_bytes: int) -> _Encoded:
    from .enums import RevertPolicy

    transforms = [
        _encode_calculator_value(transform, f"{path}.transforms[{idx}]", inline_array_bytes)
        for idx, transform in enumerate(scope.transforms)
    ]
    children: list[_Encoded] = [*transforms]
    payload: dict[str, Any] = {}
    if transforms:
        payload["transforms"] = [item.value for item in transforms]
    if scope.filter is not None:
        filter_encoded = _encode_calculator_value(scope.filter, f"{path}.filter", inline_array_bytes)
        payload["filter"] = filter_encoded.value
        children.append(filter_encoded)
    if scope.revert_policy != RevertPolicy.ALWAYS:
        payload["revert_policy"] = scope.revert_policy.value
    return _merge_encoded(payload, children)


def _decode_scope(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.runtime.scopes import ScopeSpec

    from .enums import RevertPolicy

    transforms = tuple(_decode_value(item) for item in payload.get("transforms", ()))
    filter_node = _decode_value(payload["filter"]) if "filter" in payload else None
    revert_policy = RevertPolicy(payload.get("revert_policy", RevertPolicy.ALWAYS.value))
    return ScopeSpec(transforms=transforms, filter=filter_node, revert_policy=revert_policy)


def _encode_dataclass_calculator(calculator: Any, path: str, inline_array_bytes: int) -> _Encoded:
    transform_state, children = _transform_state(calculator, inline_array_bytes, path)
    init_payload: dict[str, Any] = {}
    dataclass_field_map = {item.name: item for item in dataclass_fields(type(calculator))}
    for spec in collect_param_specs(type(calculator)):
        item = dataclass_field_map[spec.name]
        if not item.init or not spec.signature:
            continue
        value = getattr(calculator, spec.name)
        has_default, default = _field_default(item)
        if has_default and _encoded_values_equal(value, default, inline_array_bytes):
            continue
        encoded = _encode_value(getattr(calculator, spec.name), f"{path}.init.{spec.name}", inline_array_bytes)
        init_payload[spec.name] = encoded.value
        children.append(encoded)
    payload: dict[str, Any] = {}
    payload["node"] = "dataclass"
    payload["class"] = _class_path(calculator)
    if init_payload:
        payload["init"] = init_payload
    if transform_state is not None:
        payload["transform"] = transform_state
    return _merge_encoded(payload, children)


def _decode_dataclass_calculator(payload: dict[str, Any]) -> Any:
    cls = _import_object(payload["class"])
    values = {name: _decode_value(encoded) for name, encoded in payload.get("init", {}).items()}
    calculator = cls(**values)
    return _apply_transform_state(calculator, payload.get("transform"))


def _encode_special_calculator(calculator: Any, path: str, inline_array_bytes: int) -> _Encoded | None:
    from pynbodyext.core.calculate.nodes.base import BoundCalculator, CombinedCalculator
    from pynbodyext.core.calculate.nodes.expr import (
        CalculatorValueProperty,
        ConstantProperty,
        LambdaProperty,
        OpProperty,
    )
    from pynbodyext.core.calculate.nodes.filters import AndFilter, NotFilter, OrFilter
    from pynbodyext.core.calculate.nodes.transforms import TransformChain

    transform_state, transform_children = _transform_state(calculator, inline_array_bytes, path)
    encoded: _Encoded | None = None

    if isinstance(calculator, BoundCalculator):
        base = _encode_calculator_value(calculator.base, f"{path}.base", inline_array_bytes)
        scope = _encode_scope(calculator.scope, f"{path}.scope", inline_array_bytes)
        payload = {"node": "bound", "base": base.value, "scope": scope.value}
        encoded = _merge_encoded(payload, [base, scope])

    elif isinstance(calculator, CombinedCalculator):
        items = [_encode_calculator_value(item, f"{path}.items[{idx}]", inline_array_bytes) for idx, item in enumerate(calculator.items)]
        payload = {"node": "combined", "items": [item.value for item in items]}
        encoded = _merge_encoded(payload, items)

    elif isinstance(calculator, TransformChain):
        transforms = [
            _encode_calculator_value(transform, f"{path}.transforms[{idx}]", inline_array_bytes)
            for idx, transform in enumerate(calculator.transforms)
        ]
        payload = {"node": "transform_chain", "class": _class_path(calculator), "transforms": [item.value for item in transforms]}
        if transform_state is not None:
            payload["transform"] = transform_state
        encoded = _merge_encoded(payload, [*transform_children, *transforms])

    elif isinstance(calculator, (AndFilter, OrFilter)):
        op_name = "and" if isinstance(calculator, AndFilter) else "or"
        left = _encode_calculator_value(calculator.left, f"{path}.left", inline_array_bytes)
        right = _encode_calculator_value(calculator.right, f"{path}.right", inline_array_bytes)
        left, right = _sort_encoded_commutative([left, right])
        payload = {"node": "filter_op", "op": op_name, "left": left.value, "right": right.value}
        encoded = _merge_encoded(payload, [left, right])

    elif isinstance(calculator, NotFilter):
        child = _encode_calculator_value(calculator.child, f"{path}.child", inline_array_bytes)
        payload = {"node": "filter_op", "op": "not", "child": child.value}
        encoded = _merge_encoded(payload, [child])

    elif isinstance(calculator, ConstantProperty):
        value = _encode_value(calculator._value, f"{path}.value", inline_array_bytes)
        payload = {"node": "constant_property", "value": value.value}
        encoded = _merge_encoded(payload, [value])

    elif isinstance(calculator, CalculatorValueProperty):
        nested = _encode_calculator_value(calculator.calculator, f"{path}.calculator", inline_array_bytes)
        payload = {"node": "calculator_value_property", "calculator": nested.value}
        encoded = _merge_encoded(payload, [nested])

    elif isinstance(calculator, OpProperty):
        operands = [
            _encode_calculator_value(operand, f"{path}.operands[{idx}]", inline_array_bytes)
            for idx, operand in enumerate(calculator.operands)
        ]
        if calculator.op_name in _COMMUTATIVE_PROPERTY_OPS:
            operands = _sort_encoded_commutative(operands)
        payload = {"node": "op_property", "op_name": calculator.op_name, "operands": [item.value for item in operands]}
        encoded = _merge_encoded(payload, operands)

    elif isinstance(calculator, LambdaProperty):
        payload = {"node": "lambda_property", "class": _class_path(calculator), "func_repr": repr(calculator._func)}
        encoded = _Encoded(payload, constructible=False, paths=(f"{path}.func",))

    return encoded

def _encode_generic_calculator(calculator: Any, path: str, inline_array_bytes: int) -> _Encoded:
    identity = _encode_value(calculator.instance_signature(), f"{path}.identity", inline_array_bytes)
    deps = [
        _encode_calculator_value(dep, f"{path}.deps[{idx}]", inline_array_bytes)
        for idx, dep in enumerate(calculator.dependencies())
    ]
    transform_state, transform_children = _transform_state(calculator, inline_array_bytes, path)

    payload: dict[str, Any] = {
        "node": "generic",
        "class": _class_path(calculator),
        "identity": identity.value,
    }
    if deps:
        payload["deps"] = [item.value for item in deps]
    if transform_state is not None:
        payload["transform"] = transform_state

    merged = _merge_encoded(payload, [identity, *deps, *transform_children])
    paths = merged.paths if path in merged.paths else (*merged.paths, path)
    return _Encoded(merged.value, constructible=False, paths=paths)


def _decode_bound_node(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.nodes.base import BoundCalculator
    return BoundCalculator(
        base=_decode_value(payload["base"]),
        scope=_decode_scope(payload["scope"]),
    )


def _decode_combined_node(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.nodes.base import CombinedCalculator
    return CombinedCalculator(*(_decode_value(item) for item in payload["items"]))


def _decode_transform_chain_node(payload: dict[str, Any]) -> Any:
    cls = _import_object(payload["class"])
    return _apply_transform_state(
        cls(*(_decode_value(item) for item in payload["transforms"])),
        payload.get("transform"),
    )


def _decode_filter_op_node(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.nodes.filters import AndFilter, NotFilter, OrFilter

    op_name = payload["op"]
    if op_name == "and":
        return AndFilter(_decode_value(payload["left"]), _decode_value(payload["right"]))
    if op_name == "or":
        return OrFilter(_decode_value(payload["left"]), _decode_value(payload["right"]))
    if op_name == "not":
        return NotFilter(_decode_value(payload["child"]))
    raise ValueError(f"unsupported filter op: {op_name!r}")


def _decode_constant_property_node(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.nodes.expr import ConstantProperty
    return ConstantProperty(_decode_value(payload["value"]))


def _decode_calculator_value_property_node(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.nodes.expr import CalculatorValueProperty
    return CalculatorValueProperty(_decode_value(payload["calculator"]))


def _decode_op_property_node(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.nodes.expr import OpProperty
    return OpProperty(payload["op_name"], [_decode_value(item) for item in payload["operands"]])


_SPECIAL_CALCULATOR_DECODERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "bound": _decode_bound_node,
    "combined": _decode_combined_node,
    "transform_chain": _decode_transform_chain_node,
    "filter_op": _decode_filter_op_node,
    "constant_property": _decode_constant_property_node,
    "calculator_value_property": _decode_calculator_value_property_node,
    "op_property": _decode_op_property_node,
}


def _decode_special_calculator(payload: dict[str, Any]) -> Any:
    node_type = payload["node"]
    if node_type == "lambda_property":
        raise ValueError("cannot reconstruct LambdaProperty from signature")

    handler = _SPECIAL_CALCULATOR_DECODERS.get(node_type)
    if handler is None:
        raise ValueError(f"unsupported special calculator node: {node_type!r}")
    return handler(payload)

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

    encoded = _encode_special_calculator(calculator, _path, inline_array_bytes)
    if encoded is None:
        if is_dataclass(calculator):
            encoded = _encode_dataclass_calculator(calculator, _path, inline_array_bytes)   # type: ignore
        else:
            encoded = _encode_generic_calculator(calculator, _path, inline_array_bytes)

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
        return _decode_dataclass_calculator(payload)
    return _decode_special_calculator(payload)



def _tree_tuple(payload: dict[str, Any]) -> str:
    items = [_tree_value(item) for item in payload.get("items", ())]
    text = f"({', '.join(items)})"
    if len(items) == 1:
        text = f"({items[0]},)"
    return text


def _tree_list(payload: dict[str, Any]) -> str:
    return f"[{', '.join(_tree_value(item) for item in payload.get('items', ()))}]"


def _tree_dict(payload: dict[str, Any]) -> str:
    items = [
        f"{_tree_value(item['key'])}: {_tree_value(item['value'])}"
        for item in payload.get("items", ())
    ]
    return f"{{{', '.join(items)}}}"


def _tree_array(payload: dict[str, Any]) -> str:
    class_name = "SimArray" if payload.get("simarray") else "array"
    shape = tuple(payload.get("shape", ()))

    if not shape:
        return class_name
    if len(shape) == 1:
        return f"{class_name}[{shape[0]}]"
    return f"{class_name}[{shape!r}]"


def _tree_unsupported(payload: dict[str, Any]) -> str:
    class_name = _short_class_name(str(payload.get("class", "?")))
    if class_name == "function":
        class_name = "fn"
    return f"<unsupported {class_name}>"

_TREE_VALUE_HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "tuple": _tree_tuple,
    "list": _tree_list,
    "dict": _tree_dict,
    "array": _tree_array,
    "unsupported": _tree_unsupported,
}


def _tree_value(payload: Any) -> str:
    if isinstance(payload, str):
        return _quote_string(payload)
    if not isinstance(payload, dict):
        return repr(payload)
    if "node" in payload:
        return _tree_calculator_head(payload)

    value_type = payload.get("type")
    handler = None
    if isinstance(value_type, str):
        handler = _TREE_VALUE_HANDLERS.get(value_type) or _VALUE_PRETTY_HANDLERS.get(value_type)
    return handler(payload) if handler is not None else repr(payload)


def _tree_property_operand(payload: Any, *, parent_op: str | None = None) -> str:
    if isinstance(payload, dict) and payload.get("node") == "op_property":
        return _tree_op(payload, parent_op=parent_op)
    return _tree_value(payload)


def _tree_op(payload: dict[str, Any], *, parent_op: str | None = None) -> str:
    op_name = payload.get("op_name")
    operands = payload.get("operands", ())
    symbol_map = {
        "add": " + ",
        "mul": " * ",
        "sub": " - ",
        "truediv": " / ",
        "pow": " ** ",
        "lt": " < ",
        "le": " <= ",
        "gt": " > ",
        "ge": " >= ",
        "eq": " == ",
        "ne": " != ",
    }
    precedence = {"add": 10, "sub": 10, "mul": 20, "truediv": 20, "pow": 30}
    unary_symbol = {"neg": "-", "pos": "+", "abs": "abs"}

    if op_name in {"add", "mul"}:
        text = symbol_map[op_name].join(
            _tree_property_operand(item, parent_op=op_name)
            for item in operands
        )
    elif op_name in symbol_map and len(operands) == 2:
        left = _tree_property_operand(operands[0], parent_op=op_name)
        right = _tree_property_operand(operands[1], parent_op=op_name)
        text = f"{left}{symbol_map[op_name]}{right}"
    elif op_name in unary_symbol and len(operands) == 1:
        child = _tree_property_operand(operands[0], parent_op=op_name)
        text = f"{unary_symbol[op_name]}({child})" if op_name == "abs" else f"{unary_symbol[op_name]}{child}"
    elif op_name == "clip" and len(operands) == 3:
        text = (
            f"{_tree_property_operand(operands[0])}.clip("
            f"{_tree_value(operands[1])}, {_tree_value(operands[2])})"
        )
    else:
        text = f"OpProperty({_quote_string(str(op_name))})"

    if parent_op in precedence and op_name in precedence and precedence[op_name] < precedence[parent_op]:
        return f"({text})"
    return text

def _tree_dataclass_head(payload: dict[str, Any]) -> str:
    return _short_class_name(str(payload.get("class", "Calculator")))


def _tree_bound_head(payload: dict[str, Any]) -> str:
    return _tree_calculator_head(payload["base"])


def _tree_transform_chain_head(payload: dict[str, Any]) -> str:
    return _short_class_name(str(payload.get("class", "TransformChain")))


def _tree_filter_op_head(payload: dict[str, Any]) -> str:
    return {
        "and": "AndFilter",
        "or": "OrFilter",
        "not": "NotFilter",
    }.get(payload.get("op"), "FilterOp")    # type: ignore[arg-type]


def _tree_constant_property_head(payload: dict[str, Any]) -> str:
    return _tree_value(payload["value"])


def _tree_calculator_value_property_head(payload: dict[str, Any]) -> str:
    return _tree_calculator_head(payload["calculator"])

def _tree_generic_head(payload: dict[str, Any]) -> str:
    return _short_class_name(str(payload.get("class", "Calculator")))


_TREE_CALCULATOR_HEADERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "dataclass": _tree_dataclass_head,
    "bound": _tree_bound_head,
    "transform_chain": _tree_transform_chain_head,
    "combined": lambda payload: "CombinedCalculator",
    "filter_op": _tree_filter_op_head,
    "constant_property": _tree_constant_property_head,
    "calculator_value_property": _tree_calculator_value_property_head,
    "op_property": _tree_op,
    "lambda_property": lambda payload: "LambdaProperty",
    "generic": _tree_generic_head,
}


def _tree_calculator_head(payload: dict[str, Any]) -> str:
    node_type = payload.get("node")
    handler = _TREE_CALCULATOR_HEADERS.get(node_type) if isinstance(node_type, str) else None
    return handler(payload) if handler is not None else repr(payload)


def _tree_dataclass_args(payload: dict[str, Any]) -> str:
    init_payload = payload.get("init", {})
    if not init_payload:
        return ""

    field_order: list[str] = []
    try:
        cls = _import_object(payload["class"])
        field_order = [
            spec.name
            for spec in collect_param_specs(cls)
            if spec.signature
        ]
    except Exception:
        field_order = list(init_payload)

    init_names = list(init_payload)
    positional_prefix = field_order[: len(init_names)] == init_names

    parts: list[str] = []
    if positional_prefix:
        parts.extend(_tree_value(init_payload[name]) for name in init_names)
    else:
        ordered_names = [name for name in field_order if name in init_payload]
        ordered_names.extend(sorted(name for name in init_payload if name not in ordered_names))
        parts.extend(f"{name}={_tree_value(init_payload[name])}" for name in ordered_names)

    return ", ".join(parts)

def calculator_pretty_init_args(
    calculator: Any,
    *,
    inline_array_bytes: int = DEFAULT_INLINE_ARRAY_BYTES,
) -> str:
    if not is_dataclass(calculator):
        return ""

    dataclass_field_map = {item.name: item for item in dataclass_fields(type(calculator))} # type: ignore[arg-type]
    init_payload: dict[str, Any] = {}

    for spec in collect_param_specs(type(calculator)):
        item = dataclass_field_map.get(spec.name)
        if item is None or not item.init or not spec.signature:
            continue

        value = getattr(calculator, spec.name)
        has_default, default = _field_default(item)
        if has_default and _encoded_values_equal(value, default, inline_array_bytes):
            continue

        encoded = _encode_value(value, f"init.{spec.name}", inline_array_bytes)
        init_payload[spec.name] = encoded.value

    if not init_payload:
        return ""

    payload = {
        "class": _class_path(calculator),
        "init": init_payload,
    }
    return _tree_dataclass_args(payload)

