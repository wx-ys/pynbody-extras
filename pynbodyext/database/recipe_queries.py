"""Index declared output scopes directly from canonical CalculatorSignature payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.interfaces import signature_type_name
from pynbodyext.core.calculate.result.signature import CalculatorSignature

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

Payload = dict[str, Any]
Path = tuple[str, ...]


def signature_of(value: Any) -> CalculatorSignature:
    return value if isinstance(value, CalculatorSignature) else value.to_signature()


def _node_class(node: Payload) -> str | None:
    if node.get("node") == "scoped":
        return _node_class(node["base"])
    if node.get("node") == "filter_op":
        from pynbodyext.core.calculate.nodes.filters import AndFilter, NotFilter, OrFilter

        return signature_type_name({"and": AndFilter, "or": OrFilter, "not": NotFilter}[node["op"]])
    return node.get("class")


def matches_calculator(payload: Payload | None, selector: Any) -> bool:
    """Match the root type of an expression or its complete canonical signature."""
    if payload is None:
        return False
    if isinstance(selector, type):
        return _node_class(payload) == signature_type_name(selector)
    return CalculatorSignature(payload).hash == signature_of(selector).hash


def _steps(payloads: Sequence[Payload]) -> Iterator[Payload]:
    for payload in payloads:
        if payload.get("node") == "transform_chain":
            yield from _steps(payload["transforms"])
        else:
            yield payload


def matches_transforms(payloads: Sequence[Payload], selector: Any) -> bool:
    """Match the complete ordered transform method, allowing a chain or a sequence."""
    requested = selector if isinstance(selector, (tuple, list)) else (selector,)
    expected = [
        step
        for item in requested
        for step in (_steps([signature_of(item).payload]) if not isinstance(item, type) else (item,))
    ]
    actual = tuple(_steps(payloads))
    if len(actual) != len(expected):
        return False
    return all(
        matches_calculator(left, right if isinstance(right, type) else CalculatorSignature(right))
        for left, right in zip(actual, expected, strict=True)
    )


def _scope_payload(value: Any) -> Payload:
    if isinstance(value, dict):
        return value
    spec = getattr(value, "spec", value)
    scope: Payload = {}
    if spec.transforms:
        scope["transforms"] = [item.to_signature().payload for item in spec.transforms]
    if spec.filter is not None:
        scope["filter"] = spec.filter.to_signature().payload
    if spec.revert_policy.value != "always":
        scope["revert_policy"] = spec.revert_policy.value
    return scope


def matches_scope(scopes: Sequence[Payload], selector: Any) -> bool:
    requested = selector if isinstance(selector, (tuple, list)) else (selector,)
    expected = [payload for item in requested if (payload := _scope_payload(item))]
    return CalculatorSignature({"scopes": list(scopes)}).hash == CalculatorSignature({"scopes": expected}).hash


def _occurrences(value: Any, path: Path, role: str, operator: str | None = None) -> Iterator[Payload]:
    """Walk only encoded calculator nodes. Each edge keeps its absolute DAG path."""
    if isinstance(value, list):
        for i, item in enumerate(value):
            yield from _occurrences(item, (*path, str(i)), role, operator)
        return
    if not isinstance(value, dict):
        return
    kind = value.get("node")
    if kind:
        yield {"path": path, "role": role, "operator": operator, "payload": value}
    if kind == "scoped":
        yield from _scope_occurrences(value["scope"], path)
        # The wrapped expression is this occurrence's base, not a second use.
        value = value["base"]
        kind = value.get("node")
    for key, child in value.items():
        if key == "outputs" and kind == "pipeline":
            for item in child:
                yield from _occurrences(item["value"], (*path, "outputs", item["key"]), "dependency")
        elif key == "transform" and isinstance(child, dict):
            if "measure_filter" in child:
                yield from _occurrences(child["measure_filter"], (*path, "measure_filter"), "transform_measure")
        elif kind == "filter_op" and key in ("left", "right", "child"):
            yield from _occurrences(child, (*path, key), "filter_operand", value["op"])
        elif isinstance(child, (dict, list)):
            yield from _occurrences(child, (*path, key), "parameter" if key == "init" else role, operator)


def _scope_occurrences(scope: Payload, path: Path) -> Iterator[Payload]:
    for index, item in enumerate(scope.get("transforms", ())):
        yield from _occurrences(item, (*path, "scope", "transforms", str(index)), "output_transform")
    if "filter" in scope:
        yield from _occurrences(scope["filter"], (*path, "scope", "filter"), "output_filter")


def output_declarations(signature: CalculatorSignature) -> Iterator[tuple[Path, bool, Payload]]:
    """Yield root and named outputs in recipe order, inheriting enclosing scope layers."""

    def visit(
        node: Payload, output: Path, path: Path, scopes: list[Payload], uses: list[Payload]
    ) -> Iterator[tuple[Path, bool, Payload]]:
        while node.get("node") == "scoped":
            scope = node["scope"]
            scopes = [*scopes, scope]
            uses = [*uses, *_scope_occurrences(scope, path)]
            node = node["base"]
        pipeline = node.get("node") == "pipeline"
        occurrences = uses if pipeline else [*uses, *_occurrences(node, path, "output")]
        yield output, not pipeline, {"scopes": scopes, "occurrences": occurrences}
        if pipeline:
            for item in node["outputs"]:
                key = item["key"]
                yield from visit(item["value"], (*output, key), (*path, "outputs", key), scopes, uses)

    yield from visit(signature.payload, (), (), [], [])
