"""Known invocation facts, without loading native code or configuring execution."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping
from typing import Any

EXECUTION_CONTEXT_SCHEMA = "pynbodyext.orm.execution-context/v1"


def validate_execution_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the explicit fields and detach them from caller-owned mappings."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("execution_metadata must be a mapping or None")
    if set(value) - {"tool_versions", "random_seed"}:
        raise ValueError("execution_metadata accepts only tool_versions and random_seed")
    tools = value.get("tool_versions", {})
    if not isinstance(tools, Mapping) or any(type(k) is not str or type(v) is not str for k, v in tools.items()):
        raise ValueError("execution_metadata.tool_versions must map strings to strings")
    seed = value.get("random_seed")
    if seed is not None and type(seed) is not int:
        raise ValueError("execution_metadata.random_seed must be an integer or None")
    return dict(json.loads(json.dumps({"tool_versions": dict(tools), "random_seed": seed}, allow_nan=False)))


def execution_context_json(
    metadata: Mapping[str, Any], *, uri: str | None, loader_id: str, revision: str | None
) -> str:
    """Snapshot this invocation, never an inferred cached-payload producer.

    An absent native module means loaded-module information is unknown, not
    that the extension is unavailable. Collection itself never imports it.
    """
    packages: dict[str, str | None] = {}
    for name in ("pynbodyext-core", "pynbodyext", "numpy", "pynbody"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    native = None
    module = sys.modules.get("pynbodyext._native")
    if module is not None:
        version = vars(module).get("__version__")
        native = {"module": "pynbodyext._native", "version": version if type(version) is str else None}
    context = {
        "schema": EXECUTION_CONTEXT_SCHEMA,
        "scope": "invocation",
        "python": platform.python_version(),
        "packages": packages,
        "native": native,
        "input": {"uri": uri, "loader_id": loader_id, "revision": revision},
        **metadata,
    }
    return json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False)
