"""Portable value codec for persisting calculator results.

A calculator result value can be a scalar, a numpy array, a pynbody
:class:`SimArray` carrying units, or an arbitrary nested structure built from
those.  :class:`ValueCodec` turns such values into a self-describing,
JSON-compatible ``dict`` so a store (in-memory, or a relational database in the
future) can persist them without knowing about numpy or pynbody.

Every encoded value is tagged with a ``"__kind"`` discriminator.  Arrays are
stored as base64 of the raw C-contiguous bytes plus dtype and shape; sim arrays
additionally carry their units string.  Anything that is not
natively representable falls back to a pickled opaque blob, so the codec is
lossless for any pickle-able object while keeping the common cases portable and
diff-friendly.

The codec is intentionally a leaf module: it depends only on ``numpy`` (and a
lazy import of ``pynbody.array.SimArray``) and exposes no calculator or store
concepts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import pickle
from typing import Any

import numpy as np

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
        """Return a stable content fingerprint for ``value``."""
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

        return self._encode_opaque(value)

    def _encode_array(self, value: np.ndarray) -> dict[str, Any]:
        units = _sim_array_units(value)
        kind = "sim_array" if units is not None else "array"

        if value.dtype.kind == "O":
            # object arrays are not byte-representable; encode element-by-element
            return {"__kind": "array_objects", "items": [self._encode(v) for v in value.tolist()]}

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
        """Encode a dict preserving arbitrary (hashable) keys via typed tags.

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
        if kind == "none":
            return None
        if kind == "tuple":
            inner = text.strip("()").strip()
            return tuple(part.strip() for part in inner.split(",")) if inner else ()
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
            return np.asarray(data["value"], dtype=data["dtype"]).item()
        if kind in ("seq",):
            items = [self._decode(v) for v in data["items"]]
            return tuple(items) if data.get("seq_kind") == "tuple" else items
        if kind == "dict":
            return {self._decode_dict_key(k): self._decode(v) for k, v in data["items"]}
        if kind in ("array", "sim_array"):
            return self._decode_array(data, sim=kind == "sim_array")
        if kind == "array_objects":
            return np.array([self._decode(v) for v in data["items"]], dtype=object)
        if kind == "opaque":
            return pickle.loads(_unb64(data["pickle_b64"]))
        if kind == "unencoded":
            raise ValueError(f"value was stored unencoded ({data.get('type')}); cannot decode")
        raise ValueError(f"unknown encoded value kind: {kind!r}")

    def _decode_array(self, data: dict[str, Any], *, sim: bool) -> Any:
        dtype = np.dtype(data["dtype"])
        raw = np.frombuffer(_unb64(data["data_b64"]), dtype=dtype)
        shape = tuple(int(s) for s in data["shape"])
        arr = raw.reshape(shape).copy()  # detach from the immutable bytes buffer
        if sim:
            from pynbody.array import SimArray

            return SimArray(arr, units=data.get("units"))
        return arr
