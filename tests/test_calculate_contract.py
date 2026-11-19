"""Regression tests for the ``CalculatorBase`` mixin contract.

``CalculatorBase`` is composed from focused mixins.  Earlier these mixins
declared type-only method stubs that raised ``NotImplementedError`` at runtime;
the real implementations come from other mixins via the MRO.  The contract is
now declared once (``_CalculatorContract``) and defines nothing at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pynbody
import pytest

from pynbodyext.core.calculate import PropertyBase


@PropertyBase.dataclass
class _Sum(PropertyBase[float]):
    def calculate(self, sim, params=None) -> float:
        return float(np.asarray(sim["x"]).sum())


def _sim():
    sim = pynbody.new(4)
    sim["x"] = np.arange(4.0)
    return sim


def test_contract_members_resolve_via_mro() -> None:
    """The members that used to be NotImplementedError stubs must resolve."""
    calc = _Sum()
    assert calc.kind is not None
    assert isinstance(calc.dependencies(), list)
    assert isinstance(calc.children(), list)
    assert isinstance(calc.signature(), tuple)
    assert calc.to_signature() is not None
    assert isinstance(calc.signature_hash(), str)
    assert calc.signature_payload() is None
    assert calc.run(_sim()).value == 6.0


def test_calculatorbase_remains_instantiable() -> None:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase

    assert CalculatorBase.__abstractmethods__ == frozenset()


def test_mixin_contract_defines_nothing_at_runtime() -> None:
    from pynbodyext.core.calculate.nodes.mixins import _CalculatorContract

    for name in ("kind", "dependencies", "children", "signature", "to_signature", "signature_payload"):
        assert name not in vars(_CalculatorContract), f"{name} should be TYPE_CHECKING-only"


# ---------------------------------------------------------------------------
# Drift guards
#
# Both of these pin a list that is written out in several places. They exist so
# that adding a run option or a node kind cannot silently miss one of the copies.
# ---------------------------------------------------------------------------


def test_run_option_kwargs_match_run_options() -> None:
    """The three signatures accept exactly the options ``RunOptions`` declares.

    ``run``, ``__call__`` and ``_resolve_run_options`` take ``**overrides`` typed by
    ``RunOptionKwargs`` rather than listing every option each time. This pins that
    shorthand against the dataclass, so an option added to one and not the other
    fails here instead of being silently ignored (or silently untyped) at run time.
    """
    import inspect

    from pynbodyext.core.calculate import RunOptions
    from pynbodyext.core.calculate.nodes import mixins
    from pynbodyext.core.calculate.runtime.options import RunOptionKwargs

    fields = {field.name for field in dataclasses.fields(RunOptions)}
    assert set(RunOptionKwargs.__annotations__) == fields

    def _passes_options_through(fn) -> bool:
        parameters = inspect.signature(fn).parameters
        return (
            any(param.kind is param.VAR_KEYWORD for param in parameters.values())
            and "options" in parameters
        )

    for fn in (
        mixins._CalculatorRunMixin.__call__,
        mixins._CalculatorRunMixin.run,
        mixins._CalculatorRunMixin._resolve_run_options,
    ):
        assert _passes_options_through(fn), fn.__qualname__


def test_unknown_run_option_is_refused_by_name() -> None:
    """A typo in an override is an error, not a silently ignored keyword."""
    from calculate_helpers import MassSum, make_sim

    with pytest.raises(TypeError, match="unknown run option"):
        MassSum().run(make_sim(), perf_memroy=True)


def _payload_kinds(payload, out: set[str] | None = None) -> set[str]:
    """Collect every ``node`` kind in a signature payload."""
    out = set() if out is None else out
    if isinstance(payload, dict):
        node = payload.get("node")
        if isinstance(node, str):
            out.add(node)
        for value in payload.values():
            _payload_kinds(value, out)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _payload_kinds(value, out)
    return out


def test_every_node_kind_has_both_renderers() -> None:
    """An encoded node kind must be renderable by both printers, and vice versa.

    ``result/render.py`` keeps a signature handler and a tree handler per kind,
    and ``result/signature.py`` decides which kinds exist. Rendering a kind with
    no handler degrades to a raw ``repr`` of the payload, which no test would
    otherwise catch.
    """
    from pynbodyext.core.calculate.result.render import SignaturePrinter, TreePrinter
    from pynbodyext.core.calculate.result.signature import calculator_to_signature

    seen: set[str] = set()
    from calculate_helpers import kind_samples

    for calculator in kind_samples():
        seen |= _payload_kinds(calculator_to_signature(calculator).payload)

    assert set(SignaturePrinter._CALCULATOR_HANDLERS) == seen
    assert set(TreePrinter._CALCULATOR_HEADERS) == seen


def _value_kind_samples() -> list[Any]:
    """Calculators carrying every value kind the encoder can emit.

    The kinds are read off the payload, so this stays honest if the encoder grows
    a kind: a value type with no renderer would print as a raw ``repr`` in the
    signature and in every tree, which no other test would notice.
    """
    from enum import Enum

    from pynbody import units
    from pynbody.array import SimArray
    from pynbody.family import get_family

    from pynbodyext.core.calculate import Param, Pipeline, PropertyBase

    class _Choice(Enum):
        FIRST = "first"

    class _OutputRecipe(Pipeline):
        def signature_payload(self):
            return {"selected": self.outputs["sum"]}

    @PropertyBase.dataclass
    class _Carrier(PropertyBase[str]):
        # Every field is required (and passed in) because the encoder leaves a
        # value out of the payload when it equals its declared default — the
        # sample would then carry no value of that kind at all.
        scalar: Param[Any]
        numpy_scalar: Param[Any]
        enum: Param[Any]
        family: Param[Any]
        unit: Param[Any]
        array: Param[Any]
        items: Param[Any]  # a tuple
        listing: Param[Any]
        mapping: Param[Any]
        callable_: Param[Any]

        def calculate(self, sim, params=None) -> str:
            return ""

    return [
        _Carrier(
            scalar=0.5,  # a bare scalar carries no "type" key
            # float32, not float64: numpy's float64 *is* a Python float, so it
            # encodes as a bare scalar and never reaches the numpy_scalar kind.
            numpy_scalar=np.float32(1.5),
            enum=_Choice.FIRST,
            family=get_family("star"),
            unit=units.Unit("kpc"),
            array=SimArray([1.0, 2.0], "Msol"),
            items=(1, 2),
            listing=[1, 2],
            mapping={"a": 1},
            callable_=_Choice,
        ),
        _OutputRecipe({"sum": _Sum()}),
    ]


def test_every_value_kind_has_a_renderer_in_both_printers() -> None:
    """A value kind the encoder emits must not fall through to ``repr``."""
    from pynbodyext.core.calculate.result.render import SignaturePrinter, TreePrinter
    from pynbodyext.core.calculate.result.signature import calculator_to_signature

    seen: set[str] = set()
    for calculator in _value_kind_samples():
        payload = calculator_to_signature(calculator).payload
        seen |= {kind for kind in _payload_value_kinds(payload) if kind != "scalar"}

    assert seen, "no value kinds were collected — the sample builder went stale"
    assert set(SignaturePrinter._VALUE_HANDLERS) == seen
    # The tree overrides two kinds and falls back for the rest, so what matters is
    # that every kind is reachable through one table or the other.
    assert set(TreePrinter._VALUE_HANDLERS) | set(TreePrinter._FALLBACK_VALUE_HANDLERS) == seen


def _payload_value_kinds(payload: Any, out: set[str] | None = None) -> set[str]:
    """Collect every ``type`` key in a signature payload."""
    out = set() if out is None else out
    if isinstance(payload, dict):
        vtype = payload.get("type")
        if isinstance(vtype, str):
            out.add(vtype)
        for value in payload.values():
            _payload_value_kinds(value, out)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _payload_value_kinds(value, out)
    return out
