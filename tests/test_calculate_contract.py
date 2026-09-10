"""Regression tests for the ``CalculatorBase`` mixin contract.

``CalculatorBase`` is composed from focused mixins.  Earlier these mixins
declared type-only method stubs that raised ``NotImplementedError`` at runtime;
the real implementations come from other mixins via the MRO.  The contract is
now declared once (``_CalculatorContract``) and defines nothing at runtime.
"""

from __future__ import annotations

import numpy as np
import pynbody

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
