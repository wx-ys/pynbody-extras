"""Role base class for boolean selection masks.

:class:`FilterBase` is the standard base class for calculators that compute
boolean selection masks and narrow the active simulation view for downstream
work.

In this project, most concrete filters under :mod:`pynbodyext.filters` are
written in dataclass style with :class:`Param` fields and a
:meth:`calculate` method. Examples include :class:`Sphere`,
:class:`FamilyFilter`, :class:`BandPass`, :class:`Annulus`, and
:class:`SolarNeighborhood`.

A filter keeps a richer raw runtime value internally, but its public value is
the boolean mask itself.

Use :class:`FilterBase` when the node:

- computes a boolean mask
- scopes another calculator onto a subset of particles
- accepts runtime-resolved bounds, radii, or centers
- composes with boolean filter operators

Recommended Authoring Style
---------------------------
For most new filters in this codebase, prefer:

- :meth:`FilterBase.dataclass`
- :class:`Param` for resolved bounds and coordinates
- :meth:`calculate` as the main user hook
- letting :class:`FilterBase` normalize the mask and build the filtered view

Simple Example
--------------
A band-pass style filter in the same shape as the concrete filter module::

    @FilterBase.dataclass
    class BandPass(FilterBase):
        prop: str
        min: Param[float]
        max: Param[float]

        def calculate(self, sim, params=None):
            values = sim[params.prop]
            return (values >= params.min) & (values < params.max)

    mask = BandPass("temp", 1.0e5, 1.0e6).run(sim).value
    print(mask.sum())

Scoping Another Calculator
--------------------------
The most common use of a filter is to scope another calculator::

    hot_stellar_mass = ParamSum("mass").filter(
        FamilyFilter("star") & BandPass("temp", 1.0e5, 1.0e6)
    )
    print(hot_stellar_mass.run(sim).value)

Sphere-Style Example
--------------------
Many real filters in this project accept unit-aware positions or radii
through :class:`Param` field metadata. A simplified sphere-style example
looks like this::

    @FilterBase.dataclass
    class Sphere(FilterBase):
        radius: Param[float] = Param(field_name="pos")
        cen: Param[tuple[float, float, float]] = Param(
            default=(0, 0, 0),
            field_name="pos",
        )

        def calculate(self, sim, params=None):
            dx = sim["x"] - params.cen[0]
            dy = sim["y"] - params.cen[1]
            dz = sim["z"] - params.cen[2]
            return dx * dx + dy * dy + dz * dz < params.radius ** 2

Boolean Composition
-------------------
Filters compose naturally::

    aperture = Sphere("30 kpc") & FamilyFilter("star")
    result = KappaRot().filter(aperture).run(sim)
    print(result.value)

Which Hook To Implement
-----------------------
In this repository, :meth:`calculate` is the common authoring hook for
concrete filters.

:meth:`build_mask` is still available as the semantic low-level hook, but
most project filters are simpler and more consistent when they implement
:meth:`calculate` directly.

Use the runtime-level hooks only when the filter needs direct access to the
execution context or dependency evaluation.

Notes
-----
If a downstream calculator appears not to respect a filter, inspect the
scope composition and the trace tree.

If a mask shape is wrong, verify that the filter returns a full-length
one-dimensional mask for the active view.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, NoReturn, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pynbody.filt import Filter as PynbodyFilter

from pynbodyext.core.calculate.result.enums import BuiltinKinds
from pynbodyext.core.calculate.runtime.context import ExecutionContext, resolve_value
from pynbodyext.core.calculate.runtime.input import FilterResult, NodeInput

from .runtime_base import RuntimeCalculatorBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbody.snapshot import SimSnap

    from pynbodyext.core.calculate.runtime import CalcRuntime

    from .base import CalculatorBase

MaskArray: TypeAlias = npt.NDArray[np.bool_]

class FilterBase(RuntimeCalculatorBase[FilterResult, MaskArray],PynbodyFilter, ABC):
    """Base class for calculators that produce a boolean selection mask.

    Subclasses usually override :meth:`build_mask`.
    The framework materializes the mask, creates a
    filtered simulation view, and returns the mask as the public value.
    """

    node_kind = BuiltinKinds.FILTER

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
        return ("and",)

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
        return ("or",)

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
        return ("not",)

    def _build_mask_runtime(self, sim: SimSnap, params: Any, ctx: ExecutionContext, input: NodeInput) -> Any:
        return ~ctx.public_value(self.child, input)
