"""Base class for calculators that read a derived property from a snapshot.

Subclasses typically implement :meth:`calculate` for simple read-only
calculations.  The framework wraps that logic in the standard execution model,
records a ``calculate`` phase, and makes the resulting value composable with
filters, transforms, symbolic expressions, and pipelines.

When To Use PropertyBase
------------------------
Subclass :class:`PropertyBase` when your node:

- reads fields from the active simulation view
- returns a scalar, array, or small structured value
- does not represent a boolean mask
- does not mutate the simulation state

Simple Subclass
---------------
The simplest custom property implements :meth:`instance_signature` and
:meth:`calculate`::

    class StellarMass(PropertyBase[float]):
        def instance_signature(self):
            return ("stellar_mass",)

        def calculate(self, sim):
            return float(sim["mass"].sum())

This is the preferred style when the property only depends on the active
snapshot view and constructor arguments.

Property With Dynamic Parameters
--------------------------------
Use ``dynamic_param_specs`` together with :meth:`prepare_params` and
:meth:`calculate_with_params` when constructor arguments may be calculator
outputs, callables, constants, or unit-like values resolved at run time::

    class MassInsideRadius(PropertyBase[float]):
        dynamic_param_specs = {"radius": "r"}

        def __init__(self, radius):
            super().__init__()
            self.radius = radius

        def instance_signature(self):
            return ("mass_inside_radius", self.radius)

        def prepare_params(self, sim, values):
            return {"radius": values["radius"]}

        def calculate_with_params(self, sim, params=None):
            radius = params["radius"]
            mask = sim["r"] < radius
            return float(sim["mass"][mask].sum())

This allows expressions such as ``MassInsideRadius(2 * re_calc)``.

Full Runtime Hook
-----------------
Override :meth:`_calculate_runtime` only when the property needs direct access
to :class:`ExecutionContext` or :class:`NodeInput`::

    class MeanRelativeToCentre(PropertyBase[float]):
        def __init__(self, field, centre_calc):
            super().__init__()
            self.field = field
            self.centre_calc = centre_calc

        def instance_signature(self):
            return ("mean_relative_to_centre", self.field, self.centre_calc.signature())

        def declared_dependencies(self):
            return [self.centre_calc]

        def _calculate_runtime(self, sim, params, ctx, input):
            centre = ctx.public_value(self.centre_calc, input)
            values = sim[self.field] - centre
            return float(values.mean())

Composition Example
-------------------
Property nodes compose naturally with filters, transforms, and symbolic
operators::

    mass = StellarMass()
    hot_mass = mass.filter(TemperatureAbove(1.0e5))
    shifted_mass = mass.transform(XShift(1.0))

    result = hot_mass.run(sim)
    print(result.value)

They also participate in symbolic expressions::

    ratio = StellarMass() / MassInsideRadius(10.0)
    print(ratio.run(sim).value)

Notes
-----
Most custom properties should override :meth:`calculate` or
:meth:`calculate_with_params`, not :meth:`execute`.  If the node is primarily a
mask or a mutation, use :class:`FilterBase` or :class:`TransformBase` instead.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from .enums import BuiltinKinds
from .template import RuntimeCalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbody.array import SimArray
    from pynbody.snapshot import SimSnap

    from .context import ExecutionContext, NodeInput
    from .runtime import CalcRuntime


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
        self,
        sim: SimSnap,
        params: Mapping[str, Any] | None,
        ctx: ExecutionContext,
        input: NodeInput,
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

    def _make_op(
        self,
        op_name: str,
        other: object,
        *,
        reverse: bool = False,
    ) -> PropertyBase[Any]:
        from .expr import make_associative_op, make_binary_op

        left: PropertyBase[Any] = self
        right = self.as_property(other)

        if reverse:
            left, right = right, left

        if op_name in ("add", "mul"):
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
