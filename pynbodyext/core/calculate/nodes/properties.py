"""Role base class for read-only derived values.

:class:`PropertyBase` is the standard base class for calculators that read
from the active snapshot view and return a derived value.

In this project, most concrete properties are written in dataclass style with
:class:`Param` fields and a :meth:`calculate` method. See the user-facing
property modules for examples such as :class:`ParamSum`,
:class:`ParamContain`, :class:`CenPos`, :class:`KappaRot`, and
:class:`VirialRadius`.

Use :class:`PropertyBase` when the node:

- reads one or more fields from the active simulation view
- returns a scalar, array, or small structured value
- does not define a boolean mask
- does not mutate snapshot state

Recommended Authoring Style
---------------------------
For most new properties in this codebase, prefer:

- :meth:`PropertyBase.dataclass`
- class fields for constructor arguments
- :class:`Param` for runtime-resolved values
- :meth:`calculate` as the primary user hook

Simple Example
--------------
A small field-summing property in the same style as
:class:`pynbodyext.properties.ParamSum`::

    @PropertyBase.dataclass
    class ParamSum(PropertyBase[Any]):
        parameter: str

        def calculate(self, sim, params=None):
            return sim[params.parameter].sum()


    result = ParamSum("mass").run(sim)
    print(result.value)

Dynamic Parameter Example
-------------------------
A property can accept runtime-resolved parameters through :class:`Param` fields, in the same style as :class:`pynbodyext.properties.ParamContain`::

    @PropertyBase.dataclass
    class ParamContain(PropertyBase[Any]):
        frac: Param[float] = 0.5
        cal_key: str = "r"
        parameter: str = "mass"

        def calculate(self, sim, params=None):
            key = sim[params.cal_key]
            weight = sim[params.parameter]
            order = np.argsort(np.asarray(key))
            key_sorted = np.asarray(key)[order]
            weight_sorted = np.asarray(weight)[order]
            cumulative = np.cumsum(weight_sorted)
            cumulative = (cumulative - cumulative[0]) / float(cumulative[-1] - cumulative[0])
            return np.interp(params.frac, cumulative, key_sorted)

This style matches the concrete property modules more closely than older
examples based on hand-written :meth:`instance_signature`.

Scope Composition
-----------------
Properties compose naturally with filters and transforms::

    hot_mass = ParamSum("mass").filter(TemperatureAbove(1.0e5))
    centred_kappa = KappaRot().transform(ShiftPosTo("ssc"))

    print(hot_mass.run(sim).value)
    print(centred_kappa.run(sim).value)

Which Hook To Implement
-----------------------
Most :class:`PropertyBase` subclasses should implement :meth:`calculate`.

Use :meth:`calculate_with_params` when you want the explicit prepared
parameter shape. Simple ``calculate`` hooks can call ``self.log(...)`` or the
``self.debug/info/warning/error(...)`` helpers during a run without accepting
``ctx`` directly.

Only drop to the runtime-level hooks when the property truly needs direct
access to the execution context or node input.

Notes
-----
If a constructor argument is not resolving as expected, inspect the
:class:`Param` declarations and the dynamic parameter helpers in
:mod:`.params`.

If the node stops looking like a read-only derived value, reconsider whether
it should instead be a :class:`FilterBase`, :class:`TransformBase`,
:class:`RuntimeCalculatorBase`, or :class:`CalculatorBase` subclass.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from pynbodyext.core.calculate.result.enums import BuiltinKinds

from .runtime_base import RuntimeCalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbody.array import SimArray
    from pynbody.snapshot import SimSnap

    from pynbodyext.core.calculate.runtime import CalcRuntime, ExecutionContext, NodeInput


TProp = TypeVar("TProp")


class PropertyBase(RuntimeCalculatorBase[TProp, TProp], Generic[TProp], ABC):
    """Base class for calculators that read a property from a snapshot.

    Subclasses usually implement :meth:`calculate`; the framework wraps it in
    the standard execution context and records a ``calculate`` phase.
    """

    node_kind = BuiltinKinds.PROPERTY

    def calculate(self, sim: SimSnap, params: Any = None) -> TProp:
        """Calculate a value from the active simulation view.

        This is the preferred subclass interface.  ``params`` is the prepared
        dynamic parameter object, or ``None`` when no dynamic parameters were
        declared.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement calculate(), "
            "calculate_with_params(), _calculate_runtime(), or compute()."
        )

    def calculate_with_params(self, sim: SimSnap, params: Mapping[str, Any] | None = None) -> TProp:
        """Calculate a value from the active simulation view and prepared params.

        This is the medium-level subclass interface for properties with dynamic
        parameters.  New subclasses should implement ``calculate(sim, params=None)``.
        """
        return self.calculate(sim, params)

    def _calculate_runtime(
        self, sim: SimSnap, params: Mapping[str, Any] | None, ctx: ExecutionContext, input: NodeInput
    ) -> TProp:
        """Runtime calculation hook with access to context and node input."""
        return self.calculate_with_params(sim, params)

    def compute(self, runtime: CalcRuntime, params: Any) -> TProp:
        """Compute the property value through the transition runtime hook."""
        return self._calculate_runtime(runtime.sim, params, runtime.ctx, runtime.input)

    def _resolve_params_runtime(self, ctx: ExecutionContext, input: NodeInput) -> Mapping[str, Any] | None:
        sim = input.active_sim
        values = self.resolve_dynamic_params(ctx, input)
        return self.prepare_params(sim, values)

    @classmethod
    def as_property(cls, other: object | None) -> PropertyBase[Any]:
        """Coerce constants and calculators into property expression nodes."""
        from .expr import as_property

        return as_property(other)

    def _make_op(self, op_name: str, other: object, *, reverse: bool = False) -> PropertyBase[Any]:
        from .expr import make_associative_op, make_binary_op

        left: PropertyBase[Any] = self
        right = self.as_property(other)

        if reverse:
            left, right = right, left

        if op_name in {"add", "mul"}:
            return make_associative_op(op_name, left, right)

        return make_binary_op(op_name, left, right)

    def clip(self, vmin: object | None = None, vmax: object | None = None) -> PropertyBase[TProp]:
        """Return a symbolic clipped property expression."""
        from .expr import make_clip_op

        return make_clip_op(self, vmin=vmin, vmax=vmax)

    def __add__(self, other: object) -> PropertyBase[TProp]:
        return self._make_op("add", other)

    def __radd__(self, other: object) -> PropertyBase[TProp]:
        return self._make_op("add", other, reverse=True)

    def __sub__(self, other: object) -> PropertyBase[TProp]:
        from .expr import make_binary_op

        return make_binary_op("sub", self, self.as_property(other))

    def __rsub__(self, other: object) -> PropertyBase[TProp]:
        from .expr import make_binary_op

        return make_binary_op("sub", self.as_property(other), self)

    def __mul__(self, other: object) -> PropertyBase[TProp]:
        return self._make_op("mul", other)

    def __rmul__(self, other: object) -> PropertyBase[TProp]:
        return self._make_op("mul", other, reverse=True)

    @overload
    def __truediv__(self: PropertyBase[SimArray], other: object) -> PropertyBase[SimArray]: ...

    @overload
    def __truediv__(self: PropertyBase[int], other: object) -> PropertyBase[float]: ...

    @overload
    def __truediv__(self: PropertyBase[float], other: object) -> PropertyBase[float]: ...

    def __truediv__(self, other: object) -> PropertyBase[Any]:
        from .expr import make_binary_op

        return make_binary_op("truediv", self, self.as_property(other))

    def __pow__(self, other: object) -> PropertyBase[Any]:
        from .expr import make_binary_op

        return make_binary_op("pow", self, self.as_property(other))

    def __rpow__(self, other: object) -> PropertyBase[Any]:
        from .expr import make_binary_op

        return make_binary_op("pow", self.as_property(other), self)

    def __neg__(self) -> PropertyBase[TProp]:
        from .expr import make_unary_op

        return make_unary_op("neg", self)

    def __pos__(self) -> PropertyBase[TProp]:
        from .expr import make_unary_op

        return make_unary_op("pos", self)

    def __abs__(self) -> PropertyBase[TProp]:
        from .expr import make_unary_op

        return make_unary_op("abs", self)

    def ne(self, other: object) -> PropertyBase[Any]:
        """Return a symbolic ``self != other`` comparison."""
        from .expr import make_binary_op

        return make_binary_op("ne", self, self.as_property(other))

    def eq_(self, other: object) -> PropertyBase[Any]:
        """Return a symbolic ``self == other`` comparison.

        The method is named ``eq_`` because Python's ``__eq__`` is intentionally
        not overloaded for symbolic truth testing.
        """
        from .expr import make_binary_op

        return make_binary_op("eq", self, self.as_property(other))

    def lt(self, other: object) -> PropertyBase[Any]:
        """Return a symbolic ``self < other`` comparison."""
        from .expr import make_binary_op

        return make_binary_op("lt", self, self.as_property(other))

    def le(self, other: object) -> PropertyBase[Any]:
        """Return a symbolic ``self <= other`` comparison."""
        from .expr import make_binary_op

        return make_binary_op("le", self, self.as_property(other))

    def gt(self, other: object) -> PropertyBase[Any]:
        """Return a symbolic ``self > other`` comparison."""
        from .expr import make_binary_op

        return make_binary_op("gt", self, self.as_property(other))

    def ge(self, other: object) -> PropertyBase[Any]:
        """Return a symbolic ``self >= other`` comparison."""
        from .expr import make_binary_op

        return make_binary_op("ge", self, self.as_property(other))

    __hash__ = object.__hash__

    def __bool__(self) -> bool:
        raise TypeError("PropertyBase is symbolic; evaluate with run() or value().")
