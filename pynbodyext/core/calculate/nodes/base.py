"""Base abstractions for executable calculator nodes.

This module defines :class:`CalculatorBase`, the lowest-level public base class
in the calculator framework, together with helper types used for scoped
calculators and grouped execution.

Most user-defined calculators should not inherit from :class:`CalculatorBase`
directly.

Choosing A Base Class
---------------------
Choose the narrowest base class that matches the node.

1. Use :class:`PropertyBase` for a read-only derived value.
2. Use :class:`FilterBase` for a boolean mask.
3. Use :class:`TransformBase` for a temporary mutation.
4. If none of those fit, but the node still follows the standard runtime
   lifecycle, subclass :class:`RuntimeCalculatorBase`.
5. Only subclass :class:`CalculatorBase` directly when you need to implement
   :meth:`execute` yourself.

Why CalculatorBase Is The Last Resort
-------------------------------------
Direct subclasses of :class:`CalculatorBase` own their execution model. That
usually means the node needs custom orchestration rather than the standard
property, filter, transform, or runtime-template workflow.

Typical reasons to inherit from :class:`CalculatorBase` directly are:

- evaluating child calculators in a custom order
- exposing a richer raw payload and a different public value
- handling failures or branching in a custom way
- acting like a small workflow rather than a single calculation

Direct Subclass Example
-----------------------
A direct subclass usually implements :meth:`signature_payload`,
:meth:`declared_dependencies`, and :meth:`execute`::

    class HotMassFraction(CalculatorBase[dict[str, float], float]):
        def __init__(self, hot_mass, total_mass):
            super().__init__()
            self.hot_mass = hot_mass
            self.total_mass = total_mass

        def signature_payload(self):
            return {"hot_mass": self.hot_mass, "total_mass": self.total_mass}

        def declared_dependencies(self):
            return [self.hot_mass, self.total_mass]

        def execute(self, ctx, input):
            hot = ctx.public_value(self.hot_mass, input)
            total = ctx.public_value(self.total_mass, input)
            if total == 0:
                raise ValueError("total mass is zero")
            return {"hot_mass": float(hot), "total_mass": float(total), "fraction": float(hot / total)}

        def public_value(self, value):
            return value["fraction"]


    result = HotMassFraction(ParamSum("mass").filter(TemperatureAbove(1.0e5)), ParamSum("mass")).run(sim)

    print(result.value)

Raw Value Versus Public Value
-----------------------------
A calculator node may keep a raw runtime payload that is richer than the public
value exposed at the root of the result. Override :meth:`public_value` when the
root output should be a projection of that raw payload.

This pattern is useful when the node needs to preserve extra diagnostics or
intermediate statistics while still behaving like a scalar-valued calculator.

Composition
-----------
A direct :class:`CalculatorBase` subclass still composes with filters,
transforms, naming, and pipelines like any other calculator.

Notes
-----
Keep :meth:`signature_payload` stable across structurally equivalent
instances. If the node depends on child calculators, always expose them through
:meth:`declared_dependencies` so traversal, caching, and result graphs stay
correct.

Most subclasses should not override :meth:`run` or :meth:`__call__`.
"""

from __future__ import annotations

from abc import ABC
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    TypeVar,
    TypeVarTuple,
    Unpack,
    cast,
    dataclass_transform,
    overload,
)

from pynbodyext.core.calculate.nodes.mixins import (
    _CalculatorComposeMixin,
    _CalculatorDisplayMixin,
    _CalculatorGraphMixin,
    _CalculatorLoggingMixin,
    _CalculatorRunMixin,
    _CalculatorSignatureMixin,
)
from pynbodyext.core.calculate.params.fields import Param
from pynbodyext.core.calculate.result.enums import (
    BuiltinKinds,
    CachePolicy,
    EffectPolicy,
    NodeKind,
    RecordPolicy,
    normalize_kind,
)
from pynbodyext.core.calculate.runtime.options import RunOptions
from pynbodyext.core.calculate.runtime.scopes import ScopeSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pynbodyext.core.calculate.params import DynamicParamSpec
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.runtime.input import NodeInput

T = TypeVar("T")
U = TypeVar("U")
Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
TCalc = TypeVar("TCalc", bound="CalculatorBase[Any, Any]")

TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")

@dataclass_transform(field_specifiers=(Param, Param.static))
class CalculatorBase(
    _CalculatorSignatureMixin,
    _CalculatorGraphMixin,
    _CalculatorLoggingMixin,
    _CalculatorDisplayMixin,
    _CalculatorRunMixin[TRaw, TPublic],
    _CalculatorComposeMixin[TRaw, TPublic],
    Generic[TRaw, TPublic],
    ABC,
):
    """Abstract base class for executable calculator nodes.

    Parameters
    ----------
    name : str, optional
        Human-readable node name.  Named nodes are registered in
        :attr:`Result.named` and can be retrieved with :meth:`Result.get` when
        their value is available.
    record_policy : RecordPolicy, optional
        Controls whether raw and public values are retained in the returned
        :class:`Result`.
    default_options : RunOptions, optional
        Default execution options used by :meth:`run`, :meth:`__call__`, and
        :meth:`value`.

    Notes
    -----
    Subclasses should make :meth:`signature_payload` stable for calculators
    that should share cache entries across equivalent instances.
    """

    node_kind: NodeKind = BuiltinKinds.CALCULATOR
    effect: EffectPolicy = EffectPolicy.CONTEXTUAL
    cacheable: bool = True
    parallel_safe: bool = True
    cache_policy: CachePolicy = CachePolicy.AUTO

    # Mapping of dynamic parameter names to unit metadata used for runtime resolution.
    # See :meth:`resolve_dynamic_param` and :mod:`.params` for details.
    dynamic_param_specs: ClassVar[Mapping[str, DynamicParamSpec | str | None]] = {}

    @overload
    @classmethod
    def dataclass(cls, target: type[TCalc], **dataclass_kwargs: Any) -> type[TCalc]: ...

    @overload
    @classmethod
    def dataclass(cls, target: None = None, **dataclass_kwargs: Any) -> Callable[[type[TCalc]], type[TCalc]]: ...

    @classmethod
    @dataclass_transform(field_specifiers=(Param, Param.static))
    def dataclass(
        cls, target: type[TCalc] | None = None, **dataclass_kwargs: Any
    ) -> type[TCalc] | Callable[[type[TCalc]], type[TCalc]]:
        """Decorate an explicit subclass with dataclass-style calculator fields."""
        from pynbodyext.core.calculate.params.declarative import dataclass_calc

        def wrap(raw_cls: type[TCalc]) -> type[TCalc]:
            if not issubclass(raw_cls, cls):
                raise TypeError(f"{cls.__name__}.dataclass can only decorate {cls.__name__} subclasses.")
            decorator: Any = dataclass_calc
            return decorator(raw_cls, **dataclass_kwargs)

        if target is None:
            return wrap
        return wrap(target)

    def _init_dataclass_base(self) -> None:
        CalculatorBase.__init__(
            self,
            name=getattr(self, "name", None),
            record_policy=getattr(self, "record_policy", None),
            default_options=getattr(self, "default_options", None),
        )

    def __init__(
        self,
        *,
        name: str | None = None,
        record_policy: RecordPolicy | None = None,
        default_options: RunOptions | None = None,
    ) -> None:
        self.name = name
        self.record_policy = record_policy
        self.default_options = default_options or RunOptions()
        self.scope = ScopeSpec()

    @property
    def kind(self) -> NodeKind:
        """Normalized node kind used for display and signatures."""
        return normalize_kind(getattr(self, "node_kind", BuiltinKinds.CALCULATOR))

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> TRaw:
        """Execute the calculator against an active :class:`NodeInput`.

        Parameters
        ----------
        ctx : ExecutionContext
            Runtime context that owns cache, trace, perf, and dependency
            evaluation state.
        input : NodeInput
            Snapshot view and active scope for this evaluation.

        Returns
        -------
        TRaw
            Raw calculator value before public-value conversion.
        """
        raise NotImplementedError

    def materialize(self, ctx: ExecutionContext, value: TRaw) -> TRaw:
        return value

    def public_value(self, value: TRaw) -> TPublic:
        return cast("TPublic", value)

    def materialize_public(self, ctx: ExecutionContext, value: TPublic) -> TPublic:
        return value


class _BatchCaller(Generic[TPublic]):
    """Lightweight callable returned by :meth:`CalculatorBase.batch`.

    Holds a pre-computed node signature and a single
    :class:`~pynbodyext.core.calculate.runtime.engine.EvalEngine` so that
    repeated calls to :meth:`__call__` pay only the unavoidable per-sim
    cost (``ExecutionContext`` creation + actual ``execute()`` work).
    """

    __slots__ = ("_node", "_engine", "_options", "_node_sig", "_structured_sig")

    def __init__(
        self,
        node: CalculatorBase[Any, TPublic],
        engine: EvalEngine,
        options: RunOptions,
        node_sig: tuple[Any, ...],
        structured_sig: Any,
    ) -> None:
        self._node = node
        self._engine = engine
        self._options = options
        self._node_sig = node_sig
        self._structured_sig = structured_sig

    def __call__(self, sim: Any) -> TPublic:
        """Evaluate the node on *sim* using the lightweight engine path."""
        return self._engine.run_light(
            self._node,
            sim,
            self._options,
            _precomputed_node_sig=self._node_sig,
            _precomputed_structured_sig=self._structured_sig,
        )


class CombinedCalculator(CalculatorBase[tuple[Unpack[Ts]], tuple[Unpack[Ts]]], Generic[Unpack[Ts]]):
    """Calculator that evaluates several calculators and returns a tuple.

    Examples
    --------
    Combine two outputs with the ``&`` operator::

        calc = ParamSum("mass") & KappaRot()
        mass, krot = calc.value(sim)
    """

    node_kind = BuiltinKinds.COMBINED
    items: tuple[CalculatorBase[Any, Any], ...]

    def __init__(self, *items: CalculatorBase[Any, Any], name: str | None = None) -> None:
        super().__init__(name=name)
        flat: list[CalculatorBase[Any, Any]] = []
        for item in items:
            if isinstance(item, CombinedCalculator):
                flat.extend(item.items)
            else:
                flat.append(item)
        self.items = tuple(flat)

    @overload  # type: ignore[override]
    def __and__(self, other: CalculatorBase[Any, U]) -> CombinedCalculator[Unpack[Ts], U]: ...

    @overload
    def __and__(self, other: CombinedCalculator[Any]) -> CombinedCalculator[Any]: ...

    def __and__(self, other: object) -> CombinedCalculator[Any]:
        if not isinstance(other, CalculatorBase):
            raise TypeError(f"unsupported operand for &: {type(other)!r}")
        return CombinedCalculator(self, other)

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return list(self.items)

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("items", len(self.items))]
        if self.name is not None:
            fields.append(("name", self.name))
        return fields

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> tuple[Unpack[Ts]]:
        with ctx.phase(self, "calculate"):
            return tuple(ctx.public_value(item, input) for item in self.items)
