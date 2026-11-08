"""Canonical, typed simulation addresses shared by stores and sample tables."""

from __future__ import annotations

import json
import math
from typing import Any


def _encode_part(value: Any) -> Any:
    value_type = type(value)
    if value is None:
        return ["none"]
    if value_type is tuple:
        return ["tuple", [_encode_part(part) for part in value]]
    if value_type in (str, bool, int):
        return [value_type.__name__, value]
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("simulation identity floats must be finite")
        # A stable exact representation, with equal signed zeros sharing a key.
        return ["float", (0.0 if value == 0 else value).hex()]
    raise TypeError(
        f"unsupported simulation identity component {value_type.__name__}; "
        "use built-in str, bool, int, finite float, None, or tuples of these"
    )


def sim_identity_key(identity: tuple[Any, ...]) -> str:
    """Return versioned JSON identifying a simulation tuple without type loss.

    Strings, booleans, integers, finite floats, None and recursive tuples are
    supported. Types are distinct (1, True and 1.0 are different identities),
    while positive/negative floating zero are equivalent. Providers must convert
    NumPy IDs to built-ins explicitly. No repr, pickle or mutable values are used.
    """
    if type(identity) is not tuple:
        raise TypeError("simulation identity must be a tuple")
    return json.dumps(
        ["pynbodyext.sim-identity/v1", _encode_part(identity)],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )
