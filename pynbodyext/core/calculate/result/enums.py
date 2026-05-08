"""Enums and normalized string kinds used by calculator nodes.

This module centralizes the small set of enums and string-normalization helpers
used throughout the calculator framework.  These values make execution and
result behavior more explicit and easier to inspect.

What This Module Defines
------------------------
The main exported concepts are:

- :class:`NodeStatus` for result-node state such as pending, ok, and error
- :class:`RecordPolicy` for controlling stored raw/public values
- :class:`EffectPolicy` for side-effect classification
- :class:`CachePolicy` for runtime cache behavior
- :class:`ErrorPolicy` for run-level error handling
- :class:`RevertPolicy` for transform cleanup behavior
- :class:`BuiltinKinds` and :func:`normalize_kind` for node kind labels

Why These Enums Matter
----------------------
These enums encode framework decisions that would otherwise appear as ad hoc
strings or booleans scattered across the codebase.  Using explicit enum values
makes APIs clearer and debugging easier.

For example:

- ``RecordPolicy.FULL`` means a node keeps raw and public values
- ``ErrorPolicy.COLLECT_PARTIAL`` allows a pipeline to retain successful outputs
  even if one child fails
- ``RevertPolicy.NEVER`` keeps a transform applied after a scoped run

Common Usage Examples
---------------------
A calculator can override how much result data is retained::

    calc = calc.record(RecordPolicy.FULL)

Run-level error handling can also be controlled explicitly::

    result = calc.run(sim, errors=ErrorPolicy.COLLECT)

Transforms can opt out of automatic cleanup::

    shifted = shift.revert(RevertPolicy.NEVER)

Node Kinds
----------
Node kinds are short normalized strings used for display, signatures, and
result inspection.  Most users rely on the built-in kinds supplied by
:class:`BuiltinKinds`, while custom framework extensions may define additional
kinds that still satisfy the normalization rules.

Normalization Helpers
---------------------
The helper functions such as :func:`normalize_kind`,
:func:`normalize_error_policy`, and :func:`normalize_revert_policy` convert
friendly user input into validated enum values.

This is why public APIs can often accept either an enum value or a string.

When This Module Matters
------------------------
This module is most relevant when you are:

- writing new calculator subclasses
- customizing result retention behavior
- tuning error-handling behavior for larger pipelines
- reasoning about transform cleanup and mutability

Notes
-----
Most end users do not need to import this module often, but it is useful to
understand because many public APIs accept or expose these enum values directly.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final, TypeAlias

NodeKind: TypeAlias = str

_NODE_KIND_RE = re.compile(r"^[a-z][a-z0-9_.:-]*$")


class BuiltinKinds:
    """Built-in node kind strings."""

    CALCULATOR: Final[NodeKind] = "calculator"
    FILTER: Final[NodeKind] = "filter"
    TRANSFORM: Final[NodeKind] = "transform"
    PROPERTY: Final[NodeKind] = "property"
    PROFILE: Final[NodeKind] = "profile"
    COMBINED: Final[NodeKind] = "combined"
    OP: Final[NodeKind] = "op"


def normalize_kind(kind: str | None, default: NodeKind = BuiltinKinds.CALCULATOR) -> NodeKind:
    """Normalize and validate a node kind string."""
    if kind is None:
        return default
    value = kind.strip().lower()
    if not value:
        return default
    if not _NODE_KIND_RE.fullmatch(value):
        raise ValueError(
            f"Invalid node kind {kind!r}. "
            "Expected pattern: ^[a-z][a-z0-9_.:-]*$"
        )
    return value


class NodeStatus(str, Enum):
    """Execution status for a result node."""

    PENDING = "pending"
    OK = "ok"
    ERROR = "error"


class RecordPolicy(str, Enum):
    """Policy controlling how much node value data is retained."""

    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"
    ERROR_ONLY = "error_only"


class EffectPolicy(str, Enum):
    """Side-effect classification for cache and execution decisions."""

    PURE = "pure"
    CONTEXTUAL = "contextual"
    MUTATING = "mutating"


class CachePolicy(str, Enum):
    """Runtime cache storage policy for a calculator node."""

    AUTO = "auto"
    FULL = "full"
    NONE = "none"
    SMALL_ONLY = "small_only"


class ErrorPolicy(str, Enum):
    """Run-level error handling policy."""

    RAISE = "raise"
    COLLECT = "collect"
    COLLECT_PARTIAL = "collect_partial"


def normalize_error_policy(value: ErrorPolicy | str) -> ErrorPolicy:
    """Return an :class:`ErrorPolicy` from an enum or string value."""
    if isinstance(value, ErrorPolicy):
        return value
    return ErrorPolicy(value)


class RevertPolicy(str, Enum):
    """Policy controlling transform cleanup."""

    ALWAYS = "always"
    NEVER = "never"


def normalize_revert_policy(value: RevertPolicy | str | bool) -> RevertPolicy:
    """Return a :class:`RevertPolicy` from an enum, string, or bool."""
    if isinstance(value, RevertPolicy):
        return value
    if isinstance(value, bool):
        return RevertPolicy.ALWAYS if value else RevertPolicy.NEVER
    return RevertPolicy(value)
