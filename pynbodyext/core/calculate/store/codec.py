"""Portable value codec for persisting calculator results.

A calculator result value can be a scalar, a numpy array, a pynbody
:class:`SimArray` carrying units, or an arbitrary nested structure built from
those.  :class:`ValueCodec` turns such values into a self-describing,
JSON-compatible ``dict`` so a store (in-memory, or a relational database in the
future) can persist them without knowing about numpy or pynbody.

Every encoded value is tagged with a ``"__kind"`` discriminator. Numeric arrays
use raw C-contiguous bytes, dtype and shape; object arrays use shaped, flat
elements, and structured arrays use NPY with pickle disabled. Sim arrays also
carry their units string. Structured object fields, dtype metadata and layouts
that NPY cannot represent losslessly are rejected during encoding. Other values
that are not natively representable fall back to a pickled opaque blob, keeping
common cases portable and diff-friendly. ExternalFileRef has an explicit
versioned descriptor tag; encoding, decoding and hashing it never inspect files.

The codec depends on ``numpy``, the stdlib-only external-file value type (and a
lazy import of ``pynbody.array.SimArray``), never on calculators or store adapters.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import math
import pickle
from typing import Any

import numpy as np

from .external import ExternalFileRef

__all__ = ["ValueCodec"]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _sim_array_units(value: Any) -> Any:
    """Return a units label for a value, or ``None`` if it has none.

    Dimensionless pynbody arrays carry a ``NoUnit`` that has no re-parseable
    text form (``str()`` yields ``"NoUnit()"``), so those are encoded as plain
    arrays.  Only arrays with a genuine, named unit retain the unit label.
    """
    units = getattr(value, "units", None)
    if units is None:
        return None
    try:
        from pynbody.units import NoUnit

        if isinstance(units, NoUnit):
            return None
    except Exception:
        pass
    return str(units)


def _canonical_payload(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reject_dtype_metadata(dtype: np.dtype[Any]) -> None:
    if dtype.metadata is not None:
        raise TypeError("structured dtype metadata is unsupported")
    if dtype.subdtype is not None:
        _reject_dtype_metadata(dtype.subdtype[0])
    if dtype.fields is not None:
        for name in dtype.names or ():
            _reject_dtype_metadata(dtype.fields[name][0])


def _validate_structured_dtype(dtype: np.dtype[Any]) -> None:
    """Check header readability without allocating another full value array."""
    _reject_dtype_metadata(dtype)
    buffer = io.BytesIO()
    try:
        np.save(buffer, np.empty(0, dtype=dtype), allow_pickle=False)
        size = buffer.tell()
        buffer.seek(0)
        restored = np.load(buffer, allow_pickle=False, max_header_size=size)
        if restored.dtype != dtype:
            raise ValueError("NPY changes the structured dtype")
    except (TypeError, ValueError) as exc:
        raise TypeError("structured dtype cannot be represented losslessly") from exc


class ValueCodec:
    """Encode arbitrary result values to portable dicts and back.

    Parameters
    ----------
    raise_on_opaque : bool, default: True
        When True, values that are not natively representable and are not
        pickle-able raise :class:`TypeError`.  When False, such values are
        encoded as a non-portable marker instead of failing.

    Notes
    -----
    Round-tripping through the codec is stable: ``decode(encode(x))`` returns a
    value equal to ``x`` for supported types (numpy arrays compare element-wise
    via ``==``; units are preserved for :class:`SimArray`).

    NumPy scalars retain their dtype. Dictionary keys support Python strings,
    integers, floats, booleans, None and tuples of Python literals. The existing
    tuple-key format stores their literal text; nonliteral representations
    (such as ``(nan,)`` or a custom object's repr) raise :class:`ValueError`
    during decoding rather than silently changing the key's identity.
    """

    def __init__(self, *, raise_on_opaque: bool = True) -> None:
        self.raise_on_opaque = raise_on_opaque

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, value: Any) -> dict[str, Any]:
        """Return a JSON-compatible dict describing ``value``."""
        return self._encode(value)

    def decode(self, data: dict[str, Any]) -> Any:
        """Reconstruct a value from a dict produced by :meth:`encode`."""
        if not isinstance(data, dict):
            raise TypeError(f"encoded value must be a dict, got {type(data)!r}")
        return self._decode(data)

    def content_hash(self, value: Any) -> str:
        """Hash the encoded value; external-file hashes describe descriptors only."""
        return hashlib.sha256(_canonical_payload(self.encode(value)).encode("utf-8")).hexdigest()

    def size_bytes(self, value: Any) -> int:
        """Return the encoded payload size in bytes (for store classification)."""
        return len(_canonical_payload(self.encode(value)).encode("utf-8"))

    def kind(self, value: Any) -> str:
        """Return the ``__kind`` tag for ``value`` without fully encoding cheaply."""
        return str(self._encode(value).get("__kind", "unknown"))

    # ------------------------------------------------------------------
    # Encoding internals
    # ------------------------------------------------------------------

    def _encode(self, value: Any) -> dict[str, Any]:  # noqa: PLR0911
        if value is None:
            return {"__kind": "none"}

        if isinstance(value, bool):
            return {"__kind": "bool", "value": value}

        if isinstance(value, np.ndarray):
            return self._encode_array(value)

        if isinstance(value, np.generic):
            # numpy scalar (e.g. np.float64, np.int64) -> native scalar + dtype
            return {"__kind": "np_scalar", "dtype": str(value.dtype), "value": self._encode(value.item())}

        if isinstance(value, str):
            return {"__kind": "str", "value": value}

        if isinstance(value, bytes):
            return {"__kind": "bytes", "data_b64": _b64(value)}

        if isinstance(value, int):
            return {"__kind": "int", "value": int(value)}

        if isinstance(value, float):
            return self._encode_float(value)

        if isinstance(value, (list, tuple)):
            return {"__kind": "seq", "seq_kind": type(value).__name__, "items": [self._encode(v) for v in value]}

        if isinstance(value, dict):
            return {"__kind": "dict", "items": self._encode_dict(value)}

        if isinstance(value, ExternalFileRef):
            return {"__kind": "external_file_ref_v1", "descriptor": value._to_payload()}

        return self._encode_opaque(value)

    def _encode_array(self, value: np.ndarray) -> dict[str, Any]:
        units = _sim_array_units(value)
        kind = "sim_array" if units is not None else "array"

        if value.dtype.fields is not None:
            if value.dtype.hasobject:
                raise TypeError("structured arrays containing object fields are unsupported")
            _validate_structured_dtype(value.dtype)
            buffer = io.BytesIO()
            try:
                np.save(buffer, np.asarray(value), allow_pickle=False)
            except (TypeError, ValueError) as exc:
                raise TypeError("structured dtype cannot be represented losslessly") from exc
            payload: dict[str, Any] = {"__kind": "structured_array_v1", "data_b64": _b64(buffer.getvalue())}
            if units is not None:
                payload["units"] = units
            return payload

        if value.dtype.kind == "O":
            payload = {
                "__kind": "array_objects_v2",
                "shape": list(value.shape),
                "items": [self._encode(item) for item in value.flat],
            }
            if units is not None:
                payload["units"] = units
            return payload

        contiguous = np.ascontiguousarray(value)
        payload = {
            "__kind": kind,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data_b64": _b64(contiguous.tobytes()),
        }
        if units is not None:
            payload["units"] = units
        return payload

    def _encode_float(self, value: float) -> dict[str, Any]:
        if math.isnan(value):
            return {"__kind": "float", "special": "nan"}
        if math.isinf(value):
            return {"__kind": "float", "special": "inf" if value > 0 else "-inf"}
        return {"__kind": "float", "value": value}

    def _encode_opaque(self, value: Any) -> dict[str, Any]:
        try:
            payload = pickle.dumps(value)
        except Exception as exc:  # noqa: BLE001
            if not self.raise_on_opaque:
                return {"__kind": "unencoded", "type": f"{type(value).__module__}.{type(value).__qualname__}"}
            raise TypeError(f"cannot encode value of type {type(value)!r} for result store") from exc
        cls = type(value)
        return {"__kind": "opaque", "type": f"{cls.__module__}.{cls.__qualname__}", "pickle_b64": _b64(payload)}

    def _encode_dict(self, value: dict[Any, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Encode a dict with typed tags for scalar and literal tuple keys.

        JSON object keys are always strings, so to round-trip an ``int``,
        ``float``, ``bool``, ``tuple``, or ``None`` key we tag each key with its
        type.  The sort uses the type name then the string form so the order is
        stable regardless of key type.
        """
        items: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for key, item in value.items():
            items.append((self._encode_dict_key(key), self._encode(item)))
        items.sort(key=lambda pair: (pair[0]["kind"], pair[0]["text"]))
        return items

    def _encode_dict_key(self, key: Any) -> dict[str, Any]:
        return {"kind": type(key).__name__, "text": str(key)}

    def _decode_dict_key(self, data: dict[str, Any]) -> Any:  # noqa: PLR0911
        kind = data.get("kind", "str")
        text = data.get("text", "")
        if kind == "str":
            return text
        if kind == "int":
            return int(text)
        if kind == "float":
            return float(text)
        if kind == "bool":
            return text.lower() == "true"
        if kind in ("none", "NoneType"):
            return None
        if kind == "tuple":
            try:
                key = ast.literal_eval(text)
                if not isinstance(key, tuple):
                    raise ValueError("expected a tuple")
                hash(key)
            except (SyntaxError, ValueError, TypeError) as exc:
                raise ValueError("cannot decode tuple key: expected a hashable Python literal tuple") from exc
            return key
        # unknown key type: fall back to the string form (keeps data lossless-ish)
        return text

    # ------------------------------------------------------------------
    # Decoding internals
    # ------------------------------------------------------------------

    def _decode(self, data: dict[str, Any]) -> Any:  # noqa: PLR0911
        kind = data.get("__kind")
        if kind == "none":
            return None
        if kind == "bool":
            return bool(data["value"])
        if kind == "str":
            return str(data["value"])
        if kind == "bytes":
            return _unb64(data["data_b64"])
        if kind == "int":
            return int(data["value"])
        if kind == "float":
            special = data.get("special")
            if special == "nan":
                return float("nan")
            if special == "inf":
                return float("inf")
            if special == "-inf":
                return float("-inf")
            return float(data["value"])
        if kind == "np_scalar":
            value = self._decode(data["value"])
            dtype = np.dtype(data["dtype"])
            # Array scalar extraction strips trailing NULs and narrows string
            # dtypes. The stored width restores padding lost by encode's item().
            if dtype.kind == "U":
                return np.str_(value.ljust(dtype.itemsize // np.dtype("U1").itemsize, "\0"))
            if dtype.kind == "S":
                return np.bytes_(value.ljust(dtype.itemsize, b"\0"))
            return np.asarray(value, dtype=dtype)[()]
        if kind in ("seq",):
            items = [self._decode(v) for v in data["items"]]
            return tuple(items) if data.get("seq_kind") == "tuple" else items
        if kind == "dict":
            return {self._decode_dict_key(k): self._decode(v) for k, v in data["items"]}
        if kind in ("array", "sim_array"):
            return self._decode_array(data, sim=kind == "sim_array")
        if kind == "array_objects_v2":
            return self._decode_object_array(data)
        if kind == "structured_array_v1":
            return self._decode_structured_array(data)
        if kind == "array_objects":
            return np.array([self._decode(v) for v in data["items"]], dtype=object)
        if kind == "external_file_ref_v1":
            if set(data) != {"__kind", "descriptor"}:
                raise ValueError("invalid external-file reference envelope fields")
            return ExternalFileRef._from_payload(data["descriptor"])
        if kind == "opaque":
            return pickle.loads(_unb64(data["pickle_b64"]))
        if kind == "unencoded":
            raise ValueError(f"value was stored unencoded ({data.get('type')}); cannot decode")
        raise ValueError(f"unknown encoded value kind: {kind!r}")

    def _decode_structured_array(self, data: dict[str, Any]) -> Any:
        raw = _unb64(data["data_b64"])
        with io.BytesIO(raw) as buffer:
            # Wide scientific records may legitimately exceed NumPy's default
            # header limit. Bound it by this payload, while keeping pickle off.
            array = np.load(buffer, allow_pickle=False, max_header_size=len(raw))
        if not isinstance(array, np.ndarray) or array.dtype.fields is None or array.dtype.hasobject:
            raise ValueError("invalid structured-array payload")
        if "units" in data:
            from pynbody.array import SimArray

            return SimArray(array, units=data["units"])
        return array

    def _decode_object_array(self, data: dict[str, Any]) -> Any:
        shape = data.get("shape")
        items = data.get("items")
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or not isinstance(items, list)
            or math.prod(shape) != len(items)
        ):
            raise ValueError("invalid object-array shape or element count")
        array = np.empty(tuple(shape), dtype=object)
        for index, item in enumerate(items):
            array.flat[index] = self._decode(item)
        if "units" in data:
            from pynbody.array import SimArray

            return SimArray(array, units=data["units"])
        return array

    def _decode_array(self, data: dict[str, Any], *, sim: bool) -> Any:
        dtype = np.dtype(data["dtype"])
        raw = np.frombuffer(_unb64(data["data_b64"]), dtype=dtype)
        shape = tuple(int(s) for s in data["shape"])
        arr = raw.reshape(shape).copy()  # detach from the immutable bytes buffer
        if sim:
            from pynbody.array import SimArray

            return SimArray(arr, units=data.get("units"))
        return arr
