"""Base class for calculators that produce boolean selection masks.

A filter returns a :class:`FilterResult` as its raw runtime value.  That object
stores both the boolean mask and the filtered snapshot view used by scoped
calculators.  The public value of a filter is the mask itself.

When To Use FilterBase
----------------------
Subclass :class:`FilterBase` when your node:

- computes a boolean selection mask
- narrows the active simulation view for another calculator
- may depend on runtime-resolved parameters
- should compose with ``&``, ``|``, and ``~``

Simple Subclass
---------------
The simplest filter implements :meth:`instance_signature` and
:meth:`build_mask`::

    class TemperatureAbove(FilterBase):
        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold

        def instance_signature(self):
            return ("temperature_above", self.threshold)

        def build_mask(self, sim, params):
            return sim["temp"] > self.threshold

This is the preferred interface for filters that only depend on constructor
arguments and the active snapshot view.

Filter With Dynamic Parameters
------------------------------
Use ``dynamic_param_specs`` when threshold values, apertures, or other limits
can be calculators or callables resolved at run time::

    class RadiusBetween(FilterBase):
        dynamic_param_specs = {"rmin": "r", "rmax": "r"}

        def __init__(self, rmin, rmax):
            super().__init__()
            self.rmin = rmin
            self.rmax = rmax

        def instance_signature(self):
            return ("radius_between", self.rmin, self.rmax)

        def build_mask(self, sim, params):
            return (sim["r"] >= params["rmin"]) & (sim["r"] < params["rmax"])

Full Runtime Hook
-----------------
Override :meth:`_build_mask_runtime` only when the filter needs direct access to
:class:`ExecutionContext`, :class:`NodeInput`, or child calculator evaluation::

    class SphereAroundCentre(FilterBase):
        dynamic_param_specs = {"radius": "r"}

        def __init__(self, radius, centre_calc):
            super().__init__()
            self.radius = radius
            self.centre_calc = centre_calc

        def instance_signature(self):
            return ("sphere_around_centre", self.radius, self.centre_calc.signature())

        def declared_dependencies(self):
            return [self.centre_calc]

        def _build_mask_runtime(self, sim, params, ctx, input):
            centre = ctx.public_value(self.centre_calc, input)
            dx = sim["x"] - centre[0]
            dy = sim["y"] - centre[1]
            dz = sim["z"] - centre[2]
            return dx * dx + dy * dy + dz * dz < params["radius"] ** 2

Composition Example
-------------------
Filters are usually scoped onto another calculator::

    hot = TemperatureAbove(1.0e5)
    calc = GasMass().filter(hot)
    result = calc.run(sim)
    print(result.value)

Filters can also be combined into reusable boolean expressions::

    inner_hot = RadiusBetween(0.0, 10.0) & TemperatureAbove(1.0e5)
    result = MeanTemperature().filter(inner_hot).run(sim)

Notes
-----
Most filters should not override :meth:`execute`; let :class:`FilterBase`
manage mask normalization, filtered-view construction, and public-value
handling.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, NoReturn, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pynbody.filt import Filter as PynbodyFilter

from .context import ExecutionContext, FilterResult, NodeInput, resolve_value
from .declarative import dataclass_calc
from .enums import BuiltinKinds
from .template import RuntimeCalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbody.snapshot import SimSnap

    from .base import CalculatorBase
    from .runtime import CalcRuntime

MaskArray: TypeAlias = npt.NDArray[np.bool_]

class FilterBase(RuntimeCalculatorBase[FilterResult, MaskArray],PynbodyFilter, ABC):
    """Base class for calculators that produce a boolean selection mask.

    Subclasses usually override :meth:`build_mask`.
    The framework materializes the mask, creates a
    filtered simulation view, and returns the mask as the public value.
    """

    node_kind = BuiltinKinds.FILTER
    dataclass = staticmethod(dataclass_calc)

    __eq__ = object.__eq__
    __hash__ = object.__hash__
    def public_value(self, value: FilterResult) -> MaskArray:
        return value.mask

    def where(self, sim: SimSnap) -> tuple[Any, ...]:
        return np.where(self(sim))

    def materialize(self, ctx: ExecutionContext, value: FilterResult) -> FilterResult:
        value.mask = self.normalize_mask(value.source_sim, value.mask)
        value.mask_summary = ctx.engine.summarize_value(value.mask)
        return value

    def normalize_mask(self, sim: SimSnap, mask: Any) -> Any:
        """Normalize full-length 0/1 filter outputs into boolean masks."""
        shape = getattr(mask, "shape", None)
        dtype = getattr(mask, "dtype", None)

        if shape is None or dtype is None:
            return mask

        if len(shape) != 1 or shape[0] != len(sim):
            return mask

        if np.dtype(dtype) == np.dtype(np.bool_):
            return mask

        astype = getattr(mask, "astype", None)
        if astype is None:
            return mask

        try:
            return astype(np.bool_, copy=False)
        except TypeError:
            return astype(np.bool_)

    def materialize_public(self, ctx: ExecutionContext, value: MaskArray) -> MaskArray:
        return value


    def apply_mask(self, sim: SimSnap, mask: Any) -> SimSnap:
        """Apply a boolean mask to a simulation object."""
        return cast("SimSnap", sim[mask])

    def prepare_params(self, sim: SimSnap, values: Mapping[str, Any]) -> Any:
        """Convert resolved dynamic values into calculate params."""
        return super().prepare_params(sim, dict(values))

    def calculate(self, sim: SimSnap, params: Any = None) -> Any:
        """Calculate and return a boolean mask for ``sim``."""
        return self.build_mask(sim, params)

    def build_mask(self, sim: SimSnap, params: Any) -> Any:
        """Build a boolean mask for sim.

        Parameters
        ----------
        sim : SimSnap
            Active snapshot view after any bound transform or filter has been applied.
        params : object
            Prepared dynamic parameters resolved at runtime.  This is the output of ``prepare_params``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement calculate(), build_mask(), or compute().")

    def _build_mask_runtime(
        self,
        sim: SimSnap,
        params: Any,
        ctx: ExecutionContext,
        input: NodeInput,
    ) -> Any:
        return self.calculate(sim, params)

    def compute(self, runtime: CalcRuntime, params: Any) -> Any:
        """Compute the mask through the transition runtime hook."""
        return self._build_mask_runtime(runtime.sim, params, runtime.ctx, runtime.input)

    def wrap_raw(self, runtime: CalcRuntime, computed: Any) -> FilterResult:
        mask = self.normalize_mask(runtime.sim, computed)
        return FilterResult(mask=mask, source_sim=runtime.sim)

    def _resolve_params_runtime(self, ctx: ExecutionContext, input: NodeInput) -> Any:
        sim = input.active_sim
        values = self.resolve_dynamic_params(ctx, input)
        return self.prepare_params(sim, values)

    @staticmethod
    def resolve_value(sim: SimSnap, value: Any) -> Any:
        """Resolve a legacy constant-or-callable filter value."""
        return value(sim) if callable(value) else value

    def resolve_runtime_value(
        self,
        ctx: ExecutionContext,
        input: NodeInput,
        value: Any,
        *,
        field_name: str | None = None,
        target_units: Any | None = None,
    ) -> Any:
        """Resolve constants, callables, or calculator-valued parameters."""
        return resolve_value(ctx, input, value, field_name=field_name, target_units=target_units)

    def resolve_value_in_units(self, sim: SimSnap, value: Any, field_name: str) -> Any:
        """Resolve a value and convert it into the units of ``field_name``."""
        raw = self.resolve_value(sim, value)
        return self._in_sim_units(raw, field_name, sim)

    def with_filter(self, filt: FilterBase) -> NoReturn:
        """Reject pre-filtering filters; compose filters with boolean ops."""
        raise TypeError("FilterBase do not support pre-filter; please use &, |, ~ to compose filters.")

    def __and__(self, other: object) -> AndFilter:
        if not isinstance(other, FilterBase):
            raise TypeError(f"unsupported operand for &: {type(other)!r}")
        return AndFilter(self, other)

    def __or__(self, other: object) -> OrFilter:
        if not isinstance(other, FilterBase):
            raise TypeError(f"unsupported operand for |: {type(other)!r}")
        return OrFilter(self, other)

    def __xor__(self, other: object) -> NoReturn:
        raise TypeError("FilterBase do not support ^ operator; use & and | instead.")

    def __invert__(self) -> NotFilter:
        return NotFilter(self)


class AndFilter(FilterBase):
    """Logical intersection of two filters."""

    def __init__(self, left: FilterBase, right: FilterBase) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return [self.left, self.right]

    def instance_signature(self) -> tuple[Any, ...]:
        return ("and", self.left.signature(), self.right.signature())

    def _build_mask_runtime(self, sim: SimSnap, params: Any, ctx: ExecutionContext, input: NodeInput) -> Any:
        return ctx.public_value(self.left, input) & ctx.public_value(self.right, input)


class OrFilter(FilterBase):
    """Logical union of two filters."""

    def __init__(self, left: FilterBase, right: FilterBase) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return [self.left, self.right]

    def instance_signature(self) -> tuple[Any, ...]:
        return ("or", self.left.signature(), self.right.signature())

    def _build_mask_runtime(self, sim: SimSnap, params: Any, ctx: ExecutionContext, input: NodeInput) -> Any:
        return ctx.public_value(self.left, input) | ctx.public_value(self.right, input)


class NotFilter(FilterBase):
    """Logical negation of a filter."""

    def __init__(self, child: FilterBase) -> None:
        super().__init__()
        self.child = child

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return [self.child]

    def instance_signature(self) -> tuple[Any, ...]:
        return ("not", self.child.signature())

    def _build_mask_runtime(self, sim: SimSnap, params: Any, ctx: ExecutionContext, input: NodeInput) -> Any:
        return ~ctx.public_value(self.child, input)
