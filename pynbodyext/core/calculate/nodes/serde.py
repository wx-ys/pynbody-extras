"""Which node kinds a signature can carry, and how each one rebuilds.

``result/signature.py`` owns the generic machinery: the payload container
(``CalculatorSignature``), hashing, JSON, and the value codecs.  *This* module owns
the half that knows the node classes — the kind table, how an object is recognised
as a kind, and how a payload becomes a calculator again — and installs it through
:mod:`pynbodyext.core.calculate.interfaces`.

The split is what keeps ``result`` (a layer *below* ``nodes``) from importing the
node layer: the lower module asks the registry instead of reaching up.  See
``tests/test_layering_contract.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import is_dataclass
from typing import Any

from pynbodyext.core.calculate.interfaces import (
    register_calculator_layer,
    resolve_signature_type,
    signature_type_name,
    signature_type_version,
)
from pynbodyext.core.calculate.result.enums import RevertPolicy
from pynbodyext.core.calculate.result.signature import (
    _COMMUTATIVE_PROPERTY_OPS,
    _Decoder,
    _encode_init_entries,
    _Encoded,
    _Encoder,
    _merge_encoded,
    _sort_encoded_commutative,
)


def _is_calculator(value: Any) -> bool:
    """Whether *value* is a calculator node."""
    from .base import CalculatorBase

    return isinstance(value, CalculatorBase)


def _is_transform_base(value: Any) -> bool:
    from .transforms import TransformBase

    return isinstance(value, TransformBase)


def _is_plain_pipeline(value: Any) -> bool:
    """Whether *value* is a :class:`Pipeline` built by ``Pipeline.__init__``.

    Only then is ``cls(outputs)`` the way to rebuild it; a subclass with its own
    constructor is a generic node and describes itself through
    ``signature_payload``.
    """
    from .pipeline import Pipeline

    return isinstance(value, Pipeline) and type(value).__init__ is Pipeline.__init__


def _class_identity(calculator: Any) -> dict[str, Any]:
    """The ``class`` (and ``version``, when declared) entries of a payload.

    ``class`` is the class's signature name (:func:`signature_type_name`), so a
    class that declares ``signature_name`` keeps its key when its module moves;
    ``version`` is present only when the class declares a non-zero
    ``signature_version``, so declaring none leaves existing keys alone.
    """
    identity: dict[str, Any] = {"class": signature_type_name(calculator)}
    version = signature_type_version(calculator)
    if version:
        identity["version"] = version
    return identity


def _resolve_class(payload: dict[str, Any]) -> Any:
    """The class a payload names, warning when it was recorded at another version.

    The class is rebuilt either way — the parameters still apply — but it is the
    current algorithm that will run, and its signature (hence key) differs from
    the recorded one.
    """
    cls = resolve_signature_type(str(payload["class"]))
    recorded = int(payload.get("version", 0))
    current = signature_type_version(cls)
    if recorded != current:
        warnings.warn(
            f"{payload['class']} was recorded at signature_version {recorded}; rebuilding it at version "
            f"{current}, which computes with the current code and has a different signature",
            UserWarning,
            stacklevel=4,
        )
    return cls


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _transform_state(calculator: Any, inline_bytes: int, path: str) -> tuple[dict[str, Any] | None, list[_Encoded]]:
    if not _is_transform_base(calculator):
        return None, []

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


def _encode_scope(scope: Any, path: str, inline_bytes: int) -> _Encoded:
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


def _encode_dataclass(calculator: Any, path: str, inline_bytes: int) -> _Encoded:
    """Encode a dataclass-based calculator."""
    ts, children = _transform_state(calculator, inline_bytes, path)
    init_entries = _encode_init_entries(calculator, path, inline_bytes)
    init_payload: dict[str, Any] = {name: enc.value for name, enc in init_entries}
    children.extend(enc for _, enc in init_entries)
    payload: dict[str, Any] = {"node": "dataclass", **_class_identity(calculator)}
    if init_payload:
        payload["init"] = init_payload
    if ts is not None:
        payload["transform"] = ts
    return _merge_encoded(payload, children)


def _encode_special(calculator: Any, path: str, inline_bytes: int) -> _Encoded | None:
    """Encode known special calculator types; returns None for unknown types."""
    from .base import CombinedCalculator
    from .expr import CalculatorValueProperty, ConstantProperty, LambdaProperty, OpProperty
    from .filters import AndFilter, NotFilter, OrFilter
    from .transforms import TransformChain

    ts, ts_children = _transform_state(calculator, inline_bytes, path)
    encoded: _Encoded | None = None

    if _is_plain_pipeline(calculator):
        # Output names are part of what a pipeline computes (they key its value
        # dict), so they are identity; the pipeline's own ``name`` is a label and
        # is not, like every other node's.  Order is kept: it is the order of
        # the returned dict.
        outputs = [
            (key, _Encoder.encode_calculator_value(child, f"{path}.outputs[{key!r}]", inline_bytes))
            for key, child in calculator.outputs.items()
        ]
        pipeline_payload = {
            "node": "pipeline",
            **_class_identity(calculator),
            "outputs": [{"key": key, "value": enc.value} for key, enc in outputs],
        }
        encoded = _merge_encoded(pipeline_payload, [enc for _, enc in outputs])

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
            **_class_identity(calculator),
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
        # The function is identified by its repr, which carries its address: it
        # separates two lambdas within a run, but not across sessions.
        payload = {"node": "lambda_property", **_class_identity(calculator), "func_repr": repr(calculator._func)}
        encoded = _Encoded(payload, constructible=False, paths=(f"{path}.func",), unstable=(f"{path}.func",))

    return encoded


def _encode_generic(calculator: Any, path: str, inline_bytes: int) -> _Encoded:
    """Encode a generic (non-dataclass, non-special) calculator."""
    payload_items = calculator.signature_payload()
    children: list[_Encoded] = []
    identity: dict[str, Any] = {}
    opaque = payload_items is None
    if opaque:
        # No declared identity: fall back to the object itself, which keeps two
        # such nodes apart within a run but means nothing in another session.
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
    ts, ts_children = _transform_state(calculator, inline_bytes, path)
    children.extend(ts_children)
    payload: dict[str, Any] = {"node": "generic", **_class_identity(calculator), "identity": identity}
    if deps:
        payload["deps"] = [e.value for e in deps]
    if ts is not None:
        payload["transform"] = ts
    merged = _merge_encoded(payload, children)
    unstable = (*merged.unstable, path) if opaque else merged.unstable
    # Rebuildable when the class says how: ``from_signature_payload`` is the
    # inverse of ``signature_payload``.  The deps are encoded for identity only;
    # a node derives them from its own state again when rebuilt.
    rebuildable = not opaque and callable(getattr(type(calculator), "from_signature_payload", None))
    paths = merged.paths if rebuildable else (*merged.paths, path)
    return _Encoded(merged.value, constructible=rebuildable and merged.constructible, paths=paths, unstable=unstable)


def encode_calculator(
    calculator: Any, *, path: str = "calculator", inline_array_bytes: int = 128
) -> tuple[Any, bool, tuple[str, ...], tuple[str, ...]]:
    """Encode *calculator* as ``(payload, constructible, non_constructible_paths, unstable_paths)``."""
    if not _is_calculator(calculator):
        raise TypeError(f"expected CalculatorBase, got {type(calculator)!r}")

    encoded = _encode_special(calculator, path, inline_array_bytes)
    if encoded is None:
        if is_dataclass(calculator):
            encoded = _encode_dataclass(calculator, path, inline_array_bytes)
        else:
            encoded = _encode_generic(calculator, path, inline_array_bytes)

    # The output-scope lifetime changes what descendants measure. Version it
    # independently of the recipe class's own signature_version, including
    # Pipeline subclasses encoded through custom constructors or payload hooks.
    if calculator._manages_output_scopes:
        encoded.value["branch_scope_version"] = 1

    # A non-empty scope on a calculator must be part of its identity so a scoped
    # clone gets a different cache key than the unscoped one.
    scope_spec = getattr(calculator, "scope", None)
    if scope_spec is not None and not scope_spec.is_empty:
        scope_e = _encode_scope(scope_spec, f"{path}.scope", inline_array_bytes)
        encoded = _merge_encoded({"node": "scoped", "base": encoded.value, "scope": scope_e.value}, [encoded, scope_e])

    return encoded.value, encoded.constructible, encoded.paths, encoded.unstable


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_scoped(payload: dict[str, Any]) -> Any:
    """Decode a plain calculator wrapped with a scope (see ``encode_calculator``)."""
    base = _Decoder.decode_value(payload["base"])
    base.scope = _decode_scope(payload["scope"])
    return base


def _decode_generic(payload: dict[str, Any]) -> Any:
    """Rebuild a hand-written node through its ``from_signature_payload`` classmethod."""
    cls = _resolve_class(payload)
    rebuild = getattr(cls, "from_signature_payload", None)
    if not callable(rebuild) or "opaque_id" in payload.get("identity", {}):
        raise ValueError(f"cannot reconstruct {payload['class']}: it defines no from_signature_payload()")
    identity = {key: _Decoder.decode_value(value) for key, value in payload.get("identity", {}).items()}
    return _apply_transform_state(rebuild(identity), payload.get("transform"))


def _decode_pipeline(payload: dict[str, Any]) -> Any:
    cls = _resolve_class(payload)
    return cls({item["key"]: _Decoder.decode_value(item["value"]) for item in payload["outputs"]})


def _decode_combined(payload: dict[str, Any]) -> Any:
    from .base import CombinedCalculator

    return CombinedCalculator(*(_Decoder.decode_value(item) for item in payload["items"]))


def _decode_transform_chain(payload: dict[str, Any]) -> Any:
    cls = _resolve_class(payload)
    return _apply_transform_state(
        cls(*(_Decoder.decode_value(item) for item in payload["transforms"])), payload.get("transform")
    )


def _decode_filter_op(payload: dict[str, Any]) -> Any:
    from .filters import AndFilter, NotFilter, OrFilter

    op = payload["op"]
    if op == "and":
        return AndFilter(_Decoder.decode_value(payload["left"]), _Decoder.decode_value(payload["right"]))
    if op == "or":
        return OrFilter(_Decoder.decode_value(payload["left"]), _Decoder.decode_value(payload["right"]))
    if op == "not":
        return NotFilter(_Decoder.decode_value(payload["child"]))
    raise ValueError(f"unsupported filter op: {op!r}")


def _decode_constant_property(payload: dict[str, Any]) -> Any:
    from .expr import ConstantProperty

    return ConstantProperty(_Decoder.decode_value(payload["value"]))


def _decode_calculator_value_property(payload: dict[str, Any]) -> Any:
    from .expr import CalculatorValueProperty

    return CalculatorValueProperty(_Decoder.decode_value(payload["calculator"]))


def _decode_op_property(payload: dict[str, Any]) -> Any:
    from .expr import OpProperty

    return OpProperty(payload["op_name"], [_Decoder.decode_value(item) for item in payload["operands"]])


def _decode_scope(payload: dict[str, Any]) -> Any:
    from pynbodyext.core.calculate.result.enums import RevertPolicy
    from pynbodyext.core.calculate.runtime.scopes import ScopeSpec

    transforms = tuple(_Decoder.decode_value(item) for item in payload.get("transforms", ()))
    filter_node = _Decoder.decode_value(payload["filter"]) if "filter" in payload else None
    revert_policy = RevertPolicy(payload.get("revert_policy", RevertPolicy.ALWAYS.value))
    return ScopeSpec(transforms=transforms, filter=filter_node, revert_policy=revert_policy)


def _apply_transform_state(calculator: Any, payload: dict[str, Any] | None) -> Any:
    if payload is not None:
        if "revert_policy" in payload:
            calculator.revert_policy = _Decoder.decode_value(payload["revert_policy"])
        if "measure_filter" in payload:
            calculator.measure_filter = _Decoder.decode_value(payload["measure_filter"])
    return calculator


def _decode_special(payload: dict[str, Any]) -> Any:
    """Decode a special (non-dataclass) calculator payload."""
    node_type = payload["node"]
    if node_type == "lambda_property":
        raise ValueError("cannot reconstruct LambdaProperty from signature")
    handler = _SPECIAL_DECODERS.get(node_type)
    if handler is None:
        raise ValueError(f"unsupported special calculator node: {node_type!r}")
    return handler(payload)


def _decode_dataclass(payload: dict[str, Any]) -> Any:
    """Decode a dataclass calculator payload."""
    cls = _resolve_class(payload)
    values = {name: _Decoder.decode_value(enc) for name, enc in payload.get("init", {}).items()}
    return _apply_transform_state(cls(**values), payload.get("transform"))


_SPECIAL_DECODERS: dict[str, Any] = {
    "generic": _decode_generic,
    "pipeline": _decode_pipeline,
    "scoped": _decode_scoped,
    "combined": _decode_combined,
    "transform_chain": _decode_transform_chain,
    "filter_op": _decode_filter_op,
    "constant_property": _decode_constant_property,
    "calculator_value_property": _decode_calculator_value_property,
    "op_property": _decode_op_property,
}


def decode_calculator(payload: dict[str, Any]) -> Any:
    """Rebuild a calculator from its signature payload."""
    if payload.get("node") == "dataclass":
        return _decode_dataclass(payload)
    return _decode_special(payload)


register_calculator_layer(is_node=_is_calculator, encode=encode_calculator, decode=decode_calculator)
