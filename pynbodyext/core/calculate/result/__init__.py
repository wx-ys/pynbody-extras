"""Structured result, signature, enum, and exception types."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BuiltinKinds",
    "CachePolicy",
    "EffectPolicy",
    "ErrorPolicy",
    "NodeKind",
    "NodeStatus",
    "RecordPolicy",
    "RevertPolicy",
    "normalize_error_policy",
    "normalize_kind",
    "normalize_revert_policy",
    "CalculatorError",
    "CycleError",
    "ErrorInfo",
    "PerfSummary",
    "PhaseRecord",
    "ProvenanceInfo",
    "Result",
    "ResultNode",
    "ValueSummary",
    "CalculatorSignature",
    "calculator_from_signature",
    "calculator_to_signature",
    "calculator_pack",
    "calculator_unpack",
    "ResultQuery",
    "ResultRepr",
]


def __getattr__(name: str) -> Any:
    if name in {
        "BuiltinKinds",
        "CachePolicy",
        "EffectPolicy",
        "ErrorPolicy",
        "NodeKind",
        "NodeStatus",
        "RecordPolicy",
        "RevertPolicy",
        "normalize_error_policy",
        "normalize_kind",
        "normalize_revert_policy",
    }:
        from . import enums

        return getattr(enums, name)
    if name in {"CalculatorError", "CycleError"}:
        from . import exceptions

        return getattr(exceptions, name)
    if name in {
        "ErrorInfo",
        "PerfSummary",
        "PhaseRecord",
        "ProvenanceInfo",
        "Result",
        "ResultNode",
        "ValueSummary",
    }:
        from . import result

        return getattr(result, name)
    if name in {
        "CalculatorSignature",
        "calculator_from_signature",
        "calculator_to_signature",
        "calculator_pack",
        "calculator_unpack",
    }:
        from . import signature

        return getattr(signature, name)
    if name == "ResultQuery":
        from .query import ResultQuery

        return ResultQuery
    if name == "ResultRepr":
        from .repr import ResultRepr

        return ResultRepr
    raise AttributeError(name)
