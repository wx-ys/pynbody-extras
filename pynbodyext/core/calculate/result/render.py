"""Pretty and tree rendering helpers for calculator signatures.

Extracted from ``signature.py`` to keep encoding/decoding concerns separate
from display/rendering concerns.

All functions here are pure string formatters operating on JSON-compatible
signature payloads.

Classes
-------
SignaturePrinter
    Renders payloads as compact Python-like expressions,
    e.g. ``MyCalc(42, mass=True)``.
TreePrinter
    Renders payloads as short node labels for multi-line tree displays,
    e.g. ``MyCalc``.

Both classes are stateless — every method is a ``@staticmethod``.
Handler dispatch dicts are populated after each class definition.

Backward-compatibility aliases ``_pretty_calculator`` and
``_tree_dataclass_args`` are provided for the lazy imports used by
``signature.py``.
"""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING, Any, ClassVar

from pynbodyext.core.calculate.params.fields import collect_param_specs

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Shared module-level utilities
# ---------------------------------------------------------------------------


def _short_class_name(path: str) -> str:
    """Return the unqualified class name from a dotted path."""
    return path.rsplit(".", 1)[-1]


def _quote_string(value: str) -> str:
    """Return a JSON-quoted string literal."""
    return json.dumps(value)


def _import_object(path: str) -> Any:
    """Import and return an object by its fully-qualified dotted path."""
    module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"invalid object path: {path!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# SignaturePrinter
# ---------------------------------------------------------------------------


class SignaturePrinter:
    """Renders calculator signature payloads as compact Python-like expressions.

    All methods are static; use them as::

        SignaturePrinter.calculator(payload)  # -> "MyCalc(42, mass=True)"
        SignaturePrinter.value(payload)  # -> "42" / "Unit('kpc')"

    Handler dispatch dicts are populated after the class definition.
    """

    #: Dispatch table for value-type payloads (keyed by ``payload["type"]``).
    _VALUE_HANDLERS: ClassVar[dict[str, Callable[[dict[str, Any]], str]]] = {}
    #: Dispatch table for calculator payloads (keyed by ``payload["node"]``).
    _CALCULATOR_HANDLERS: ClassVar[dict[str, Callable[[dict[str, Any]], str]]] = {}

    # ------------------------------------------------------------------
    # Atomic value renderers
    # ------------------------------------------------------------------

    @staticmethod
    def numpy_scalar(payload: dict[str, Any]) -> str:
        return repr(payload.get("value"))

    @staticmethod
    def enum_(payload: dict[str, Any]) -> str:
        name = _short_class_name(str(payload.get("class", "")))
        return f"{name}.{payload.get('value')}"

    @staticmethod
    def family(payload: dict[str, Any]) -> str:
        return _quote_string(str(payload.get("name", "")))

    @staticmethod
    def unit(payload: dict[str, Any]) -> str:
        return f"Unit({_quote_string(str(payload.get('value', '')))})"

    @staticmethod
    def array(payload: dict[str, Any]) -> str:
        cls_name = "SimArray" if payload.get("simarray") else "array"
        shape = tuple(payload.get("shape", ()))
        dtype = payload.get("dtype")
        suffix = f", units={_quote_string(str(payload['units']))}" if payload.get("units") is not None else ""
        return f"{cls_name}(shape={shape!r}, dtype={dtype!r}{suffix})"

    @staticmethod
    def tuple_(payload: dict[str, Any]) -> str:
        items = [SignaturePrinter.value(i) for i in payload.get("items", ())]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"

    @staticmethod
    def list_(payload: dict[str, Any]) -> str:
        return "[" + ", ".join(SignaturePrinter.value(i) for i in payload.get("items", ())) + "]"

    @staticmethod
    def dict_(payload: dict[str, Any]) -> str:
        pairs = [
            f"{SignaturePrinter.value(i['key'])}: {SignaturePrinter.value(i['value'])}"
            for i in payload.get("items", ())
        ]
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def unsupported(payload: dict[str, Any]) -> str:
        return f"<unsupported {_short_class_name(str(payload.get('class', 'unsupported')))}>"

    # ------------------------------------------------------------------
    # Value dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def value(payload: Any) -> str:
        """Render any value payload as a compact expression string."""
        if isinstance(payload, str):
            return _quote_string(payload)
        if not isinstance(payload, dict):
            return repr(payload)
        if "node" in payload:
            return SignaturePrinter.calculator(payload)
        vtype = payload.get("type")
        handler = SignaturePrinter._VALUE_HANDLERS.get(vtype) if isinstance(vtype, str) else None
        return handler(payload) if handler is not None else repr(payload)

    # ------------------------------------------------------------------
    # Compound calculator renderers
    # ------------------------------------------------------------------

    @staticmethod
    def dataclass_args(payload: dict[str, Any]) -> str:
        """Render a dataclass calculator's init arguments."""
        init = payload.get("init", {})
        if not init:
            return ""
        field_order: list[str] = []
        try:
            cls = _import_object(payload["class"])
            field_order = [s.name for s in collect_param_specs(cls) if s.signature]
        except Exception:
            field_order = list(init)
        names = list(init)
        positional = field_order[: len(names)] == names
        if positional:
            parts = [SignaturePrinter.value(init[n]) for n in names]
        else:
            ordered = [n for n in field_order if n in init]
            ordered += sorted(n for n in init if n not in ordered)
            parts = [f"{n}={SignaturePrinter.value(init[n])}" for n in ordered]
        return ", ".join(parts)

    @staticmethod
    def scope_suffix(scope: dict[str, Any]) -> str:
        suffix = ""
        revert_arg = ", revert=False" if scope.get("revert_policy") == "never" else ""
        for t in scope.get("transforms", ()):
            suffix += f".transform({SignaturePrinter.calculator(t)}{revert_arg})"
            revert_arg = ""
        if "filter" in scope:
            suffix += f".filter({SignaturePrinter.filter_(scope['filter'])})"
        return suffix

    @staticmethod
    def filter_(payload: dict[str, Any], *, parent_op: str | None = None) -> str:
        if payload.get("node") != "filter_op":
            return SignaturePrinter.calculator(payload)
        op = payload.get("op")
        if op == "not":
            text = f"~{SignaturePrinter.filter_(payload['child'], parent_op='not')}"
        elif op in {"and", "or"}:
            sep = " & " if op == "and" else " | "
            left = SignaturePrinter.filter_(payload["left"], parent_op=op)
            right = SignaturePrinter.filter_(payload["right"], parent_op=op)
            text = sep.join((left, right))
        else:
            text = SignaturePrinter.calculator(payload)
        if parent_op == "not" and op in {"and", "or"}:
            return f"({text})"
        if parent_op == "and" and op == "or":
            return f"({text})"
        return text

    @staticmethod
    def op(payload: dict[str, Any], *, parent_op: str | None = None) -> str:
        op_name = payload.get("op_name")
        operands = payload.get("operands", ())
        _SYM = {
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
        _PREC = {"add": 10, "sub": 10, "mul": 20, "truediv": 20, "pow": 30}
        _UNARY = {"neg": "-", "pos": "+", "abs": "abs"}
        if op_name in {"add", "mul"}:
            text = _SYM[op_name].join(SignaturePrinter.property_operand(i, parent_op=op_name) for i in operands)
        elif op_name in _SYM and len(operands) == 2:
            l = SignaturePrinter.property_operand(operands[0], parent_op=op_name)
            r = SignaturePrinter.property_operand(operands[1], parent_op=op_name)
            text = f"{l}{_SYM[op_name]}{r}"
        elif op_name in _UNARY and len(operands) == 1:
            child = SignaturePrinter.property_operand(operands[0], parent_op=op_name)
            text = f"{_UNARY[op_name]}({child})" if op_name == "abs" else f"{_UNARY[op_name]}{child}"
        elif op_name == "clip" and len(operands) == 3:
            text = (
                f"{SignaturePrinter.property_operand(operands[0])}.clip("
                f"{SignaturePrinter.value(operands[1])}, {SignaturePrinter.value(operands[2])})"
            )
        else:
            ops_str = ", ".join(SignaturePrinter.value(i) for i in operands)
            text = f"OpProperty({_quote_string(str(op_name))}, [{ops_str}])"
        if parent_op in _PREC and op_name in _PREC and _PREC[op_name] < _PREC[parent_op]:
            return f"({text})"
        return text

    @staticmethod
    def property_operand(payload: Any, *, parent_op: str | None = None) -> str:
        if isinstance(payload, dict) and payload.get("node") == "op_property":
            return SignaturePrinter.op(payload, parent_op=parent_op)
        return SignaturePrinter.value(payload)

    @staticmethod
    def dataclass_calculator(payload: dict[str, Any]) -> str:
        name = _short_class_name(str(payload.get("class", "Calculator")))
        text = f"{name}({SignaturePrinter.dataclass_args(payload)})"
        if "transform" in payload:
            text += SignaturePrinter.scope_suffix(payload["transform"])
        return text

    @staticmethod
    def bound_calculator(payload: dict[str, Any]) -> str:
        return SignaturePrinter.calculator(payload["base"]) + SignaturePrinter.scope_suffix(payload.get("scope", {}))

    @staticmethod
    def combined_calculator(payload: dict[str, Any]) -> str:
        items = ", ".join(SignaturePrinter.calculator(i) for i in payload.get("items", ()))
        return f"CombinedCalculator({items})"

    @staticmethod
    def transform_chain(payload: dict[str, Any]) -> str:
        name = _short_class_name(str(payload.get("class", "TransformChain")))
        ts = ", ".join(SignaturePrinter.calculator(i) for i in payload.get("transforms", ()))
        text = f"{name}({ts})"
        if "transform" in payload:
            text += SignaturePrinter.scope_suffix(payload["transform"])
        return text

    @staticmethod
    def constant_property(payload: dict[str, Any]) -> str:
        return SignaturePrinter.value(payload["value"])

    @staticmethod
    def calculator_value_property(payload: dict[str, Any]) -> str:
        return SignaturePrinter.calculator(payload["calculator"])

    @staticmethod
    def lambda_property(payload: dict[str, Any]) -> str:
        return "LambdaProperty(<function>)"

    @staticmethod
    def generic_calculator(payload: dict[str, Any]) -> str:
        name = _short_class_name(str(payload.get("class", "Calculator")))
        identity = payload.get("identity", {})
        if not identity or identity == {"opaque_id": identity.get("opaque_id")}:
            return name
        parts = [f"{k}={SignaturePrinter.value(v)}" for k, v in identity.items() if k != "opaque_id"]
        return f"{name}({', '.join(parts)})" if parts else name

    # ------------------------------------------------------------------
    # Calculator dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def calculator(payload: dict[str, Any]) -> str:
        """Render a calculator payload as a compact Python-like expression."""
        ntype = payload.get("node")
        handler = SignaturePrinter._CALCULATOR_HANDLERS.get(ntype) if isinstance(ntype, str) else None
        return handler(payload) if handler is not None else repr(payload)


# Populate handler dicts after class definition so we can reference bound staticmethods.
SignaturePrinter._VALUE_HANDLERS = {
    "numpy_scalar": SignaturePrinter.numpy_scalar,
    "enum": SignaturePrinter.enum_,
    "family": SignaturePrinter.family,
    "unit": SignaturePrinter.unit,
    "array": SignaturePrinter.array,
    "tuple": SignaturePrinter.tuple_,
    "list": SignaturePrinter.list_,
    "dict": SignaturePrinter.dict_,
    "unsupported": SignaturePrinter.unsupported,
}
SignaturePrinter._CALCULATOR_HANDLERS = {
    "dataclass": SignaturePrinter.dataclass_calculator,
    "bound": SignaturePrinter.bound_calculator,
    "combined": SignaturePrinter.combined_calculator,
    "transform_chain": SignaturePrinter.transform_chain,
    "filter_op": SignaturePrinter.filter_,
    "constant_property": SignaturePrinter.constant_property,
    "calculator_value_property": SignaturePrinter.calculator_value_property,
    "op_property": SignaturePrinter.op,
    "lambda_property": SignaturePrinter.lambda_property,
    "generic": SignaturePrinter.generic_calculator,
}


# ---------------------------------------------------------------------------
# TreePrinter
# ---------------------------------------------------------------------------


class TreePrinter:
    """Renders calculator signature payloads as short tree node labels.

    Labels are intentionally compact (often just the class name) for use
    in multi-line tree displays.  Use as::

        TreePrinter.calculator_head(payload)  # -> "MyCalc"
        TreePrinter.dataclass_args(payload)  # -> "42, mass=True"

    Handler dispatch dicts are populated after the class definition.
    ``value()`` falls back to :attr:`SignaturePrinter._VALUE_HANDLERS` for
    types that ``TreePrinter`` does not override (e.g. scalars, enums).
    """

    #: Dispatch table for value-type payloads (overrides only).
    _VALUE_HANDLERS: ClassVar[dict[str, Callable[[dict[str, Any]], str]]] = {}
    #: Dispatch table for calculator-head payloads.
    _CALCULATOR_HEADERS: ClassVar[dict[str, Callable[[dict[str, Any]], str]]] = {}

    # ------------------------------------------------------------------
    # Atomic value renderers (tree-style overrides)
    # ------------------------------------------------------------------

    @staticmethod
    def tuple_(payload: dict[str, Any]) -> str:
        items = [TreePrinter.value(i) for i in payload.get("items", ())]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"

    @staticmethod
    def list_(payload: dict[str, Any]) -> str:
        return "[" + ", ".join(TreePrinter.value(i) for i in payload.get("items", ())) + "]"

    @staticmethod
    def dict_(payload: dict[str, Any]) -> str:
        pairs = [f"{TreePrinter.value(i['key'])}: {TreePrinter.value(i['value'])}" for i in payload.get("items", ())]
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def array(payload: dict[str, Any]) -> str:
        cls_name = "SimArray" if payload.get("simarray") else "array"
        shape = tuple(payload.get("shape", ()))
        if not shape:
            return cls_name
        return f"{cls_name}[{shape[0]}]" if len(shape) == 1 else f"{cls_name}[{shape!r}]"

    @staticmethod
    def unsupported(payload: dict[str, Any]) -> str:
        name = _short_class_name(str(payload.get("class", "?")))
        return f"<unsupported {'fn' if name == 'function' else name}>"

    # ------------------------------------------------------------------
    # Value dispatch (with SignaturePrinter fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def value(payload: Any) -> str:
        """Render a value payload as a short tree-display string."""
        if isinstance(payload, str):
            return _quote_string(payload)
        if not isinstance(payload, dict):
            return repr(payload)
        if "node" in payload:
            return TreePrinter.calculator_head(payload)
        vtype = payload.get("type")
        handler: Callable[[dict[str, Any]], str] | None = None
        if isinstance(vtype, str):
            handler = TreePrinter._VALUE_HANDLERS.get(vtype) or SignaturePrinter._VALUE_HANDLERS.get(vtype)
        return handler(payload) if handler is not None else repr(payload)

    # ------------------------------------------------------------------
    # Op renderer (tree-style, shorter unknown-op format)
    # ------------------------------------------------------------------

    @staticmethod
    def property_operand(payload: Any, *, parent_op: str | None = None) -> str:
        if isinstance(payload, dict) and payload.get("node") == "op_property":
            return TreePrinter.op(payload, parent_op=parent_op)
        return TreePrinter.value(payload)

    @staticmethod
    def op(payload: dict[str, Any], *, parent_op: str | None = None) -> str:
        op_name = payload.get("op_name")
        operands = payload.get("operands", ())
        _SYM = {
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
        _PREC = {"add": 10, "sub": 10, "mul": 20, "truediv": 20, "pow": 30}
        _UNARY = {"neg": "-", "pos": "+", "abs": "abs"}
        if op_name in {"add", "mul"}:
            text = _SYM[op_name].join(TreePrinter.property_operand(i, parent_op=op_name) for i in operands)
        elif op_name in _SYM and len(operands) == 2:
            l = TreePrinter.property_operand(operands[0], parent_op=op_name)
            r = TreePrinter.property_operand(operands[1], parent_op=op_name)
            text = f"{l}{_SYM[op_name]}{r}"
        elif op_name in _UNARY and len(operands) == 1:
            child = TreePrinter.property_operand(operands[0], parent_op=op_name)
            text = f"{_UNARY[op_name]}({child})" if op_name == "abs" else f"{_UNARY[op_name]}{child}"
        elif op_name == "clip" and len(operands) == 3:
            text = (
                f"{TreePrinter.property_operand(operands[0])}.clip("
                f"{TreePrinter.value(operands[1])}, {TreePrinter.value(operands[2])})"
            )
        else:
            text = f"OpProperty({_quote_string(str(op_name))})"
        if parent_op in _PREC and op_name in _PREC and _PREC[op_name] < _PREC[parent_op]:
            return f"({text})"
        return text

    # ------------------------------------------------------------------
    # Calculator head renderers
    # ------------------------------------------------------------------

    @staticmethod
    def dataclass_head(payload: dict[str, Any]) -> str:
        return _short_class_name(str(payload.get("class", "Calculator")))

    @staticmethod
    def bound_head(payload: dict[str, Any]) -> str:
        return TreePrinter.calculator_head(payload["base"])

    @staticmethod
    def transform_chain_head(payload: dict[str, Any]) -> str:
        return _short_class_name(str(payload.get("class", "TransformChain")))

    @staticmethod
    def filter_op_head(payload: dict[str, Any]) -> str:
        op = payload.get("op")
        if not isinstance(op, str):
            return "FilterOp"
        return {"and": "AndFilter", "or": "OrFilter", "not": "NotFilter"}.get(op, "FilterOp")

    @staticmethod
    def constant_property_head(payload: dict[str, Any]) -> str:
        return TreePrinter.value(payload["value"])

    @staticmethod
    def calculator_value_property_head(payload: dict[str, Any]) -> str:
        return TreePrinter.calculator_head(payload["calculator"])

    @staticmethod
    def generic_head(payload: dict[str, Any]) -> str:
        return _short_class_name(str(payload.get("class", "Calculator")))

    # ------------------------------------------------------------------
    # Calculator head dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def calculator_head(payload: dict[str, Any]) -> str:
        """Render a calculator payload as a short tree node label."""
        ntype = payload.get("node")
        handler = TreePrinter._CALCULATOR_HEADERS.get(ntype) if isinstance(ntype, str) else None
        return handler(payload) if handler is not None else repr(payload)

    # ------------------------------------------------------------------
    # Dataclass init args (for tree display, calls TreePrinter.value)
    # ------------------------------------------------------------------

    @staticmethod
    def dataclass_args(payload: dict[str, Any]) -> str:
        """Render a dataclass calculator's init arguments for tree display."""
        init = payload.get("init", {})
        if not init:
            return ""
        field_order: list[str] = []
        try:
            cls = _import_object(payload["class"])
            field_order = [s.name for s in collect_param_specs(cls) if s.signature]
        except Exception:
            field_order = list(init)
        names = list(init)
        positional = field_order[: len(names)] == names
        if positional:
            parts = [TreePrinter.value(init[n]) for n in names]
        else:
            ordered = [n for n in field_order if n in init]
            ordered += sorted(n for n in init if n not in ordered)
            parts = [f"{n}={TreePrinter.value(init[n])}" for n in ordered]
        return ", ".join(parts)


# Populate handler dicts.
TreePrinter._VALUE_HANDLERS = {
    "tuple": TreePrinter.tuple_,
    "list": TreePrinter.list_,
    "dict": TreePrinter.dict_,
    "array": TreePrinter.array,
    "unsupported": TreePrinter.unsupported,
}
TreePrinter._CALCULATOR_HEADERS = {
    "dataclass": TreePrinter.dataclass_head,
    "bound": TreePrinter.bound_head,
    "transform_chain": TreePrinter.transform_chain_head,
    "combined": lambda p: "CombinedCalculator",
    "filter_op": TreePrinter.filter_op_head,
    "constant_property": TreePrinter.constant_property_head,
    "calculator_value_property": TreePrinter.calculator_value_property_head,
    "op_property": TreePrinter.op,
    "lambda_property": lambda p: "LambdaProperty",
    "generic": TreePrinter.generic_head,
}


# ---------------------------------------------------------------------------
# Backward-compatibility aliases used by signature.py lazy imports
# ---------------------------------------------------------------------------

_pretty_calculator = SignaturePrinter.calculator
_tree_dataclass_args = TreePrinter.dataclass_args
