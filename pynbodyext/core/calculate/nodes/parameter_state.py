"""Detect in-place changes to declared scientific calculator parameters."""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any

from pynbodyext.core.calculate.params.fields import collect_param_specs
from pynbodyext.core.calculate.result.signature import _Encoder


def _state(value: Any) -> str | None:
    try:
        encoded = _Encoder.encode_value(value, "parameter", 0)
        if not encoded.unstable:
            return json.dumps(encoded.value, sort_keys=True, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        # Legacy calculations can use arbitrary normalized objects. Study
        # preflight must enforce the stricter identity contract at its boundary.
        return None
    return None


def remember_parameters(node: Any) -> None:
    """Capture the normalized fields, after any constructor conversions."""
    constructor_params = {field.name for field in fields(node) if field.init}
    node._normalized_parameter_state = {
        spec.name: _state(getattr(node, spec.name))
        for spec in collect_param_specs(type(node))
        if spec.signature and spec.name in constructor_params and hasattr(node, spec.name)
    }


def update_parameter(node: Any, name: str, value: Any) -> None:
    states = getattr(node, "_normalized_parameter_state", {})
    if name in states:
        node._normalized_parameter_state = {**states, name: _state(value)}


def parameters_changed(calculator: Any) -> bool:
    """Read the live graph without relying on its cached constructor signature."""
    seen: set[int] = set()
    pending = [calculator]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for name, state in getattr(node, "_normalized_parameter_state", {}).items():
            if state is None or not hasattr(node, name) or _state(getattr(node, name)) != state:
                return True
        pending.extend(node.dependencies())
    return False
