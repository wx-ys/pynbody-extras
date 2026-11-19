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
"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields as dataclass_fields
from typing import TYPE_CHECKING, Any, ClassVar

from pynbodyext.core.calculate.interfaces import resolve_signature_type
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


def default_init_payload(
    calculator_or_path: Any, *, inline_array_bytes: int | None = None
) -> dict[str, Any]:
    """Encode the declared defaults of a calculator's signature parameters.

    Signature payloads omit parameters at their declared default, so this is the
    other half of :func:`display_init_payload`: a node that was given no argument
    still renders as a full call — ``ParamContain`` reads as ``ParamContain(0.5,
    "r", "mass")`` rather than as a bare, uninformative class name.
    """
    try:
        from .signature import DEFAULT_INLINE_ARRAY_BYTES, _Encoder

        inline_bytes = DEFAULT_INLINE_ARRAY_BYTES if inline_array_bytes is None else inline_array_bytes
        cls = (
            calculator_or_path
            if isinstance(calculator_or_path, type)
            else resolve_signature_type(str(calculator_or_path))
        )
        field_map = {field.name: field for field in dataclass_fields(cls)}
        defaults: dict[str, Any] = {}
        for spec in collect_param_specs(cls):
            item = field_map.get(spec.name)
            if item is None or not item.init or not spec.signature:
                continue
            if item.default is MISSING:
                continue
            encoded = _Encoder.encode_value(item.default, f"init.{spec.name}", inline_bytes)
            defaults[spec.name] = encoded.value
        return defaults
    except Exception:
        return {}


def display_init_payload(payload: dict[str, Any], *, inline_array_bytes: int | None = None) -> dict[str, Any]:
    """The init payload to display: positional defaults filled in, arguments on top.

    A signature payload carries only the arguments that differ from the declared
    default, so the label is completed from the class declaration:

    - a **positional** parameter is always listed, as the argument it was given or
      its declared default, so ``ShiftPosTo("com")`` and ``ShiftPosTo("ssc")`` differ
      only in that value and a label always reads as a full constructor call;
    - a **keyword-only** parameter is listed only when it differs from its declared
      default, because it is an optional flag rather than part of what the node
      computes: ``WrapBox(None, "minirange")`` instead of repeating ``move_all=True``
      on every line of a tree.

    The single implementation of the rule; ``signature.calculator_pretty_init_args``
    routes a live calculator through it as well, so the label of a live object and of
    a stored payload cannot drift apart.
    """
    explicit = payload.get("init") or {}
    defaults = default_init_payload(payload.get("class"), inline_array_bytes=inline_array_bytes)
    if not defaults:
        return explicit
    kw_only = _kw_only_names(payload)
    merged = {name: value for name, value in defaults.items() if name not in kw_only}
    merged.update(explicit)
    order = _field_order(payload, merged)
    ordered = {name: merged[name] for name in order if name in merged}
    ordered.update({name: value for name, value in merged.items() if name not in ordered})
    return ordered


def _field_order(payload: dict[str, Any], init: dict[str, Any]) -> list[str]:
    """Return *payload*'s signature field names in declaration order."""
    try:
        cls = resolve_signature_type(str(payload["class"]))
        return [spec.name for spec in collect_param_specs(cls) if spec.signature]
    except Exception:
        return list(init)


def _kw_only_names(payload: dict[str, Any]) -> set[str]:
    """Return the calculator's keyword-only init parameter names.

    A keyword-only parameter (``move_all`` on transforms) is accepted only as
    ``name=value``, so rendering it positionally would print a call that no longer
    runs.
    """
    try:
        cls = resolve_signature_type(str(payload["class"]))
        return {item.name for item in dataclass_fields(cls) if item.kw_only}
    except Exception:
        return set()


def _dataclass_arg_parts(
    init: dict[str, Any],
    field_order: list[str],
    render_value: Callable[[Any], str],
    *,
    kw_only: set[str] | frozenset[str] = frozenset(),
) -> list[str]:
    """Render init entries, positional while the call form allows it.

    Positional rendering only applies to the non-keyword-only fields, and only
    while they form a leading run of the signature; everything else is rendered as
    ``name=value``.
    """
    names = list(init)
    positional_names = [name for name in names if name not in kw_only]
    positional_order = [name for name in field_order if name not in kw_only]
    if positional_order[: len(positional_names)] == positional_names:
        positional = set(positional_names)
        return [
            render_value(init[name]) if name in positional else f"{name}={render_value(init[name])}" for name in names
        ]
    ordered = [name for name in field_order if name in init]
    ordered += sorted(name for name in init if name not in ordered)
    return [f"{name}={render_value(init[name])}" for name in ordered]


# ---------------------------------------------------------------------------
# OpProperty expressions (shared by both printers)
# ---------------------------------------------------------------------------

_OP_SYMBOLS: dict[str, str] = {
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
_OP_PRECEDENCE: dict[str, int] = {"add": 10, "sub": 10, "mul": 20, "truediv": 20, "pow": 30}
_UNARY_SYMBOLS: dict[str, str] = {"neg": "-", "pos": "+", "abs": "abs"}


def _op_expression(
    op_name: Any,
    operands: Any,
    *,
    operand: Callable[..., str],
    value: Callable[[Any], str],
    parent_op: str | None = None,
) -> str:
    """Render an ``OpProperty`` payload as an infix expression.

    The operator table, the precedence rule that decides when parentheses are
    needed, and the ``clip``/unary spellings live here once: ``operand`` renders
    one nested operand and ``value`` a leaf, which are the only two things the
    signature and the tree renderings disagree on.
    """
    if op_name in {"add", "mul"}:
        text = _OP_SYMBOLS[op_name].join(operand(i, parent_op=op_name) for i in operands)
    elif op_name in _OP_SYMBOLS and len(operands) == 2:
        left = operand(operands[0], parent_op=op_name)
        right = operand(operands[1], parent_op=op_name)
        text = f"{left}{_OP_SYMBOLS[op_name]}{right}"
    elif op_name in _UNARY_SYMBOLS and len(operands) == 1:
        child = operand(operands[0], parent_op=op_name)
        text = f"{_UNARY_SYMBOLS[op_name]}({child})" if op_name == "abs" else f"{_UNARY_SYMBOLS[op_name]}{child}"
    elif op_name == "clip" and len(operands) == 3:
        text = f"{operand(operands[0])}.clip({value(operands[1])}, {value(operands[2])})"
    else:
        args = ", ".join(value(i) for i in operands)
        text = f"OpProperty({_quote_string(str(op_name))}, [{args}])"
    if parent_op in _OP_PRECEDENCE and op_name in _OP_PRECEDENCE and _OP_PRECEDENCE[op_name] < _OP_PRECEDENCE[parent_op]:
        return f"({text})"
    return text


# ---------------------------------------------------------------------------
# _ValuePrinter — the structural half both printers share
# ---------------------------------------------------------------------------


class _ValuePrinter:
    """How a *structural* value payload reads — shared by both printers.

    A one-item tuple keeps its trailing comma, a dict keeps its key order, a
    dataclass's arguments are ordered by its own fields: both printers answer that
    the same way, and disagree in exactly two places — how a bare value reads
    (``Unit('kpc')`` for the signature, ``kpc`` for the tree) and how a calculator
    reads (its expression, or its head).  A subclass supplies those two and
    inherits the structure, so a new value kind cannot be rendered two ways by
    accident.

    ``_VALUE_HANDLERS`` holds **method names**, looked up on ``cls``: storing the
    bound callables would pin the base's methods, and a nested tuple in the tree
    would then render its children the signature's way.  A kind this class does not
    name is looked up in ``_FALLBACK_VALUE_HANDLERS``, which is how the tree shows
    a unit or an enum exactly as the signature spells it.
    """

    #: value kind -> the name of the method on this class that renders it
    _VALUE_HANDLERS: ClassVar[dict[str, str]] = {}
    #: consulted for a kind this class does not name
    _FALLBACK_VALUE_HANDLERS: ClassVar[dict[str, str]] = {}

    @classmethod
    def value(cls, payload: Any) -> str:
        """Render any value payload, at whatever depth it sits."""
        if isinstance(payload, str):
            return _quote_string(payload)
        if not isinstance(payload, dict):
            return repr(payload)
        if "node" in payload:
            return cls._render_calculator(payload)
        vtype = payload.get("type")
        name = cls._VALUE_HANDLERS.get(vtype) if isinstance(vtype, str) else None
        if name is None and isinstance(vtype, str):
            name = cls._FALLBACK_VALUE_HANDLERS.get(vtype)
        handler = getattr(cls, name) if name is not None else None
        return handler(payload) if handler is not None else repr(payload)

    @classmethod
    def _render_calculator(cls, payload: dict[str, Any]) -> str:
        """Render one calculator payload; each printer answers this its own way."""
        raise NotImplementedError

    @staticmethod
    def pipeline_output(payload: dict[str, Any]) -> str:
        return f"outputs[{_quote_string(payload['key'])}]"

    @classmethod
    def tuple_(cls, payload: dict[str, Any]) -> str:
        items = [cls.value(i) for i in payload.get("items", ())]
        return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"

    @classmethod
    def list_(cls, payload: dict[str, Any]) -> str:
        return "[" + ", ".join(cls.value(i) for i in payload.get("items", ())) + "]"

    @classmethod
    def dict_(cls, payload: dict[str, Any]) -> str:
        pairs = [f"{cls.value(i['key'])}: {cls.value(i['value'])}" for i in payload.get("items", ())]
        return "{" + ", ".join(pairs) + "}"

    @classmethod
    def dataclass_args(cls, payload: dict[str, Any]) -> str:
        """Render a dataclass calculator's init arguments."""
        init = display_init_payload(payload)
        if not init:
            return ""
        return ", ".join(
            _dataclass_arg_parts(init, _field_order(payload, init), cls.value, kw_only=_kw_only_names(payload))
        )

    @classmethod
    def property_operand(cls, payload: Any, *, parent_op: str | None = None) -> str:
        if isinstance(payload, dict) and payload.get("node") == "op_property":
            return cls.op(payload, parent_op=parent_op)
        return cls.value(payload)

    @classmethod
    def op(cls, payload: dict[str, Any], *, parent_op: str | None = None) -> str:
        return _op_expression(
            payload.get("op_name"),
            payload.get("operands", ()),
            operand=cls.property_operand,
            value=cls.value,
            parent_op=parent_op,
        )


# ---------------------------------------------------------------------------
# SignaturePrinter
# ---------------------------------------------------------------------------


class SignaturePrinter(_ValuePrinter):
    """Renders calculator signature payloads as compact Python-like expressions.

    Use it as::

        SignaturePrinter.calculator(payload)  # -> "MyCalc(42, mass=True)"
        SignaturePrinter.value(payload)  # -> "42" / "Unit('kpc')"

    The structural value rendering is inherited from :class:`_ValuePrinter`; what
    this class decides is how a leaf reads (a value as an expression) and how a
    calculator reads (its expression).  ``_CALCULATOR_HANDLERS`` is populated after
    the class definition, because it names methods defined below it.
    """

    #: value kind -> method name, keyed by ``payload["type"]``
    _VALUE_HANDLERS: ClassVar[dict[str, str]] = {
        "numpy_scalar": "numpy_scalar",
        "enum": "enum_",
        "family": "family",
        "unit": "unit",
        "array": "array",
        "tuple": "tuple_",
        "list": "list_",
        "dict": "dict_",
        "unsupported": "unsupported",
        "pipeline_output": "pipeline_output",
    }
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

    # ------------------------------------------------------------------
    # Compound calculator renderers
    # ------------------------------------------------------------------

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
    def dataclass_calculator(payload: dict[str, Any]) -> str:
        name = _short_class_name(str(payload.get("class", "Calculator")))
        text = f"{name}({SignaturePrinter.dataclass_args(payload)})"
        if "transform" in payload:
            text += SignaturePrinter.scope_suffix(payload["transform"])
        return text

    @staticmethod
    def scoped_calculator(payload: dict[str, Any]) -> str:
        """Render a scope-wrapped calculator as ``<base>.filter(...)``/``.transform(...)``.

        ``ScopeSpec`` (from ``.filter(...)``/``.then(...)`` on a non-transform
        calculator) is encoded as a ``scoped`` wrapper rather than as fields on
        the base payload, so the base and the scope are rendered separately and
        then joined by the scope's own suffix.
        """
        return SignaturePrinter.calculator(payload["base"]) + SignaturePrinter.scope_suffix(payload.get("scope", {}))

    @staticmethod
    def combined_calculator(payload: dict[str, Any]) -> str:
        items = ", ".join(SignaturePrinter.calculator(i) for i in payload.get("items", ()))
        return f"CombinedCalculator({items})"

    @staticmethod
    def pipeline(payload: dict[str, Any]) -> str:
        """Render a pipeline as ``Pipeline({"m": MassSum(), ...})``."""
        name = _short_class_name(str(payload.get("class", "Pipeline")))
        outputs = ", ".join(
            f"{_quote_string(str(item['key']))}: {SignaturePrinter.calculator(item['value'])}"
            for item in payload.get("outputs", ())
        )
        return f"{name}({{{outputs}}})"

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

    @classmethod
    def _render_calculator(cls, payload: dict[str, Any]) -> str:
        """A nested calculator reads as its expression."""
        return cls.calculator(payload)

    @staticmethod
    def calculator(payload: dict[str, Any]) -> str:
        """Render a calculator payload as a compact Python-like expression."""
        ntype = payload.get("node")
        handler = SignaturePrinter._CALCULATOR_HANDLERS.get(ntype) if isinstance(ntype, str) else None
        return handler(payload) if handler is not None else repr(payload)


# Populate the calculator table after the class definition: it names methods
# defined below it, and the value table is declared inline (see the class body).
SignaturePrinter._CALCULATOR_HANDLERS = {
    "pipeline": SignaturePrinter.pipeline,
    "dataclass": SignaturePrinter.dataclass_calculator,
    "scoped": SignaturePrinter.scoped_calculator,
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


class TreePrinter(_ValuePrinter):
    """Renders calculator signature payloads as short tree node labels.

    A node label is its full constructor call: every signature parameter, as the
    argument it was given or its declared default (see
    :func:`display_init_payload`).  Listing the same parameters for every node —
    rather than hiding the ones that happen to match a default — is what makes a
    label predictable to read and to diff.  Use as::

        TreePrinter.calculator_head(payload)  # -> "MyCalc"
        TreePrinter.dataclass_args(payload)  # -> '42, "ssc", mass=True'

    The structural value rendering is inherited from :class:`_ValuePrinter`; what
    this class decides is how a calculator reads (its head) and two value kinds —
    an array by its shape, an unsupported value shortly.  Every other kind falls
    back to the signature's spelling through ``_FALLBACK_VALUE_HANDLERS``, so a
    unit reads ``Unit('kpc')`` here exactly as it does there.
    """

    #: value kind -> method name; only the kinds this printer overrides
    _VALUE_HANDLERS: ClassVar[dict[str, str]] = {"array": "array", "unsupported": "unsupported"}
    #: the signature's table, consulted for every kind not named above
    _FALLBACK_VALUE_HANDLERS: ClassVar[dict[str, str]] = SignaturePrinter._VALUE_HANDLERS
    #: Dispatch table for calculator-head payloads.
    _CALCULATOR_HEADERS: ClassVar[dict[str, Callable[[dict[str, Any]], str]]] = {}

    # ------------------------------------------------------------------
    # Atomic value renderers (the two tree-style overrides)
    # ------------------------------------------------------------------

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
    # Calculator head renderers
    # ------------------------------------------------------------------

    @staticmethod
    def dataclass_head(payload: dict[str, Any]) -> str:
        """Label a nested dataclass node as ``Name(args)``.

        Arguments are included here (unlike a bare :meth:`dataclass_args` call)
        so that a calculator nested inside another node's arguments reads like
        the expression that built it: ``Sphere(0.5 * ParamContain(0.5, "r", "mass"))``.
        """
        name = _short_class_name(str(payload.get("class", "Calculator")))
        args = TreePrinter.dataclass_args(payload)
        return f"{name}({args})" if args else name

    @staticmethod
    def scoped_head(payload: dict[str, Any]) -> str:
        """Label a scope-wrapped node by its base; the scope renders as a child."""
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

    @staticmethod
    def pipeline_head(payload: dict[str, Any]) -> str:
        """Label a nested pipeline by its output names: ``Pipeline(m, t)``."""
        name = _short_class_name(str(payload.get("class", "Pipeline")))
        return f"{name}({', '.join(str(item['key']) for item in payload.get('outputs', ()))})"

    # ------------------------------------------------------------------
    # Calculator head dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def calculator_head(payload: dict[str, Any]) -> str:
        """Render a calculator payload as a short tree node label."""
        ntype = payload.get("node")
        handler = TreePrinter._CALCULATOR_HEADERS.get(ntype) if isinstance(ntype, str) else None
        return handler(payload) if handler is not None else repr(payload)

    @classmethod
    def _render_calculator(cls, payload: dict[str, Any]) -> str:
        """A nested calculator reads as its head, not as its full expression."""
        return cls.calculator_head(payload)

# Populate the calculator table after the class definition: it names methods
# defined below it, and the value table is declared inline.
TreePrinter._CALCULATOR_HEADERS = {
    "pipeline": TreePrinter.pipeline_head,
    "dataclass": TreePrinter.dataclass_head,
    "scoped": TreePrinter.scoped_head,
    "transform_chain": TreePrinter.transform_chain_head,
    "combined": lambda p: "CombinedCalculator",
    "filter_op": TreePrinter.filter_op_head,
    "constant_property": TreePrinter.constant_property_head,
    "calculator_value_property": TreePrinter.calculator_value_property_head,
    "op_property": TreePrinter.op,
    "lambda_property": lambda p: "LambdaProperty",
    "generic": TreePrinter.generic_head,
}
