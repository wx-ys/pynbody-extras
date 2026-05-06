"""Abstract base class for executable calculator nodes.

Parameters
----------
name : str, optional
    Human-readable node name.  Named nodes are easier to identify in progress
    output, trace trees, and result graphs.
record_policy : RecordPolicy, optional
    Controls whether raw and public values are retained in the returned
    :class:`Result`.
default_options : RunOptions, optional
    Default execution options used by :meth:`run`, :meth:`__call__`, and
    :meth:`value`.

When To Subclass This Directly
------------------------------
Use :class:`CalculatorBase` directly only when your node needs custom
orchestration that does not fit the simpler property/filter/transform patterns.

Typical reasons to subclass it directly are:

- the node evaluates several child calculators in a custom order
- the node exposes a raw value and a different public value
- the node needs explicit control over runtime phases
- the node acts like a small workflow rather than a single calculation

For simple read-only values, prefer :class:`PropertyBase`.
For boolean masks, prefer :class:`FilterBase`.
For temporary mutations, prefer :class:`TransformBase`.

Minimal Direct Subclass
-----------------------
A minimal direct subclass normally implements :meth:`instance_signature`,
:meth:`declared_dependencies`, and :meth:`execute`::

    class Ratio(CalculatorBase[float, float]):
        def __init__(self, numerator, denominator):
            super().__init__()
            self.numerator = numerator
            self.denominator = denominator

        def instance_signature(self):
            return (
                "ratio",
                self.numerator.signature(),
                self.denominator.signature(),
            )

        def declared_dependencies(self):
            return [self.numerator, self.denominator]

        def execute(self, ctx, input):
            num = ctx.public_value(self.numerator, input)
            den = ctx.public_value(self.denominator, input)
            if den == 0:
                raise ValueError("denominator is zero")
            return float(num / den)

Custom Raw/Public Value Pair
----------------------------
Override :meth:`public_value` when the raw runtime value should preserve more
information than the public root value::

    class MeanWithStats(CalculatorBase[dict[str, float], float]):
        def __init__(self, field):
            super().__init__()
            self.field = field

        def instance_signature(self):
            return ("mean_with_stats", self.field)

        def execute(self, ctx, input):
            values = input.active_sim[self.field]
            return {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "count": float(len(values)),
            }

        def public_value(self, value):
            return value["mean"]

The returned :class:`Result` will expose ``result.value`` as the mean, while
the raw node payload can still be retained according to the selected
``record_policy``.

Composition Example
-------------------
Direct subclasses still compose with filters, transforms, pipelines, and named
outputs like any other calculator::

    combo = (
        Ratio(StellarMass(), MeanTemperature())
        .filter(TemperatureAbove(1.0e5))
        .named("mass_over_temp")
    )

    result = combo.run(sim, progress="node")
    print(result.value)

Extension Notes
---------------
Keep :meth:`instance_signature` stable across structurally equivalent
instances.  If your node depends on other calculators, use
:meth:`declared_dependencies` so cache keys, graph traversal, and result trees
stay correct.

Most subclasses should not override :meth:`run` or :meth:`__call__`.  Put
calculator-specific logic in :meth:`execute`, :meth:`materialize`,
:meth:`public_value`, or :meth:`materialize_public`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from enum import Enum
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

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from .context import (
    ExecutionContext,
    FilterResult,
    NodeInput,
    ProgressSink,
    ProgressVerbosity,
    RunOptions,
    TransformResult,
)
from .display import (
    compact_repr,
    display_value,
    html_card,
    html_pre,
    mimebundle,
)
from .engine import EvalEngine
from .enums import (
    BuiltinKinds,
    CachePolicy,
    EffectPolicy,
    ErrorPolicy,
    NodeKind,
    RecordPolicy,
    normalize_kind,
)
from .fields import dynamic, static
from .params import (
    DynamicParamSpec,
    RuntimeValueResolver,
    StandaloneValueResolver,
    ValueResolver,
    dynamic_value_dependencies,
    dynamic_value_signature,
    resolve_value_for,
)
from .scopes import ScopeSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pynbodyext.util._type import SingleElementArray, UnitLike

    from .filters import FilterBase
    from .properties import PropertyBase
    from .result import Result
    from .transforms import TransformBase
T = TypeVar("T")
U = TypeVar("U")
Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
TBase = TypeVar("TBase", bound="CalculatorBase[Any, Any]")
TCalc = TypeVar("TCalc", bound="CalculatorBase[Any, Any]")

TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")

def _coerce_unit(value: UnitLike) -> units.UnitBase:
    return units.Unit(value)

def _merge_dependencies(*groups: list[CalculatorBase[Any, Any]]) -> list[CalculatorBase[Any, Any]]:
    merged: list[CalculatorBase[Any, Any]] = []
    seen: set[int] = set()
    for group in groups:
        for dep in group:
            key = id(dep)
            if key in seen:
                continue
            seen.add(key)
            merged.append(dep)
    return merged

class CalculatorBase(Generic[TRaw, TPublic], ABC):
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
    Subclasses should make :meth:`instance_signature` stable for calculators
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
    @dataclass_transform(field_specifiers=(dynamic, static))
    def dataclass(
        cls,
        target: type[TCalc] | None = None,
        **dataclass_kwargs: Any,
    ) -> type[TCalc] | Callable[[type[TCalc]], type[TCalc]]:
        """Decorate an explicit subclass with dataclass-style calculator fields."""
        from .declarative import dataclass_calc

        def wrap(raw_cls: type[TCalc]) -> type[TCalc]:
            if not issubclass(raw_cls, cls):
                raise TypeError(f"{cls.__name__}.dataclass can only decorate {cls.__name__} subclasses.")
            decorator: Any = dataclass_calc
            return decorator(raw_cls, **dataclass_kwargs)

        if target is None:
            return wrap
        return wrap(target)

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

    @property
    def log_label(self) -> str:
        """Name used in reports, progress logs, and graph displays."""
        return self.name or self.__class__.__name__

    @property
    def tree_label(self) -> str:
        """Label used by format_tree()/format_flow()."""
        return self.log_label

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = []
        signature = self.instance_signature()
        if signature and isinstance(signature[0], str):
            fields.extend((None, value) for value in signature[1:])
        elif signature:
            fields.append(("signature", signature))
        if self.name is not None:
            fields.append(("name", self.name))
        if self.record_policy is not None:
            fields.append(("record", display_value(self.record_policy)))
        if not self.scope.is_empty:
            fields.append(("scope", self.scope.short_label()))
        return fields

    def _repr_summary_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("type", self.__class__.__name__),
            ("kind", display_value(self.kind)),
            ("label", self.log_label),
            ("signature", self.signature_hash()),
            ("cache", display_value(self.cache_policy)),
        ]
        if not self.scope.is_empty:
            rows.append(("scope", self.scope.short_label()))
        deps = self.dependencies()
        if deps:
            rows.append(("dependencies", len(deps)))
        return rows

    def __repr__(self) -> str:
        parts: list[str] = []
        for key, value in self._repr_fields():
            text = compact_repr(value)
            parts.append(text if key is None else f"{key}={text}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(f"{self.__class__.__name__}(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        return html_card(
            self.__class__.__name__,
            self._repr_summary_rows(),
            body=html_pre(self.format_tree()),
        )

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())

    def instance_signature(self) -> tuple[Any, ...]:
        """Return the instance-specific part of the node signature.

        Returns
        -------
        tuple
            Hashable or JSON-normalizable values describing this instance.
        """
        return ("id", id(self))

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        """Return explicitly declared calculator dependencies for this node."""
        return []

    def dynamic_param_names(self) -> tuple[str, ...]:
        """Return declared dynamic parameter names."""
        return tuple(type(self).dynamic_param_specs)

    def dynamic_param_spec(self, name: str) -> DynamicParamSpec:
        """Return normalized metadata for one dynamic parameter."""
        spec = type(self).dynamic_param_specs.get(name)
        if isinstance(spec, DynamicParamSpec):
            return spec
        if isinstance(spec, str):
            return DynamicParamSpec(field_name=spec)
        return DynamicParamSpec()

    def dynamic_param_signature(self) -> tuple[Any, ...]:
        """Return signature fragments for declared dynamic parameters."""
        return tuple(
            (name, dynamic_value_signature(getattr(self, name)))
            for name in self.dynamic_param_names()
        )

    def dynamic_param_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        """Return calculator dependencies nested inside dynamic parameters."""
        deps: list[CalculatorBase[Any, Any]] = []
        for name in self.dynamic_param_names():
            deps.extend(dynamic_value_dependencies(getattr(self, name)))
        return deps

    def resolve_param_with(
        self,
        resolver: ValueResolver,
        name: str,
        *,
        field_name: str | None = None,
        target_units: Any | None = None,
        optional_units: bool | None = None,
        allow_calculator: bool = True,
        allow_callable: bool = True,
        coerce_unit_string: bool = False,
    ) -> Any:
        """Resolve one named dynamic parameter using a resolver strategy."""
        spec = self.dynamic_param_spec(name)
        return resolve_value_for(
            resolver,
            getattr(self, name),
            field_name=spec.field_name if field_name is None else field_name,
            target_units=spec.target_units if target_units is None else target_units,
            optional_units=spec.optional_units if optional_units is None else optional_units,
            allow_calculator=allow_calculator,
            allow_callable=allow_callable,
            coerce_unit_string=coerce_unit_string,
        )

    def resolve_dynamic_param(
        self,
        ctx: ExecutionContext,
        input: NodeInput,
        name: str,
        **kwargs: Any,
    ) -> Any:
        """Resolve one named dynamic parameter inside an active run."""
        return self.resolve_param_with(RuntimeValueResolver(ctx, input), name, **kwargs)

    def resolve_dynamic_params(self, ctx: ExecutionContext, input: NodeInput) -> dict[str, Any]:
        """Resolve all declared dynamic parameters inside an active run."""
        resolver = RuntimeValueResolver(ctx, input)
        return {name: self.resolve_param_with(resolver, name) for name in self.dynamic_param_names()}

    def resolve_param_for_sim(
        self,
        sim: Any | None,
        name: str,
        *,
        options: RunOptions | None = None,
        **kwargs: Any,
    ) -> Any:
        """Resolve one named dynamic parameter outside an active run."""
        return self.resolve_param_with(StandaloneValueResolver(sim, options=options), name, **kwargs)

    def resolve_params_for_sim(
        self,
        sim: Any | None,
        *,
        options: RunOptions | None = None,
    ) -> dict[str, Any]:
        """Resolve all declared dynamic parameters outside an active run."""
        resolver = StandaloneValueResolver(sim, options=options)
        return {name: self.resolve_param_with(resolver, name) for name in self.dynamic_param_names()}

    @classmethod
    def is_dynamic_value(cls, value: Any) -> bool:
        """Whether a constructor value should be resolved at run time."""
        return bool(dynamic_value_dependencies(value)) or callable(value)

    def has_dynamic_param(self, name: str) -> bool:
        """Whether one declared dynamic parameter needs runtime resolution."""
        return self.is_dynamic_value(getattr(self, name))

    def dependencies(self) -> list[CalculatorBase[Any, Any]]:
        """Return all calculator dependencies, including dynamic parameters."""
        return _merge_dependencies(
            self.declared_dependencies(),
            self.dynamic_param_dependencies(),
        )

    def children(self) -> list[CalculatorBase[Any, Any]]:
        """Return child nodes shown in graph displays."""
        return self.dependencies()

    def signature(self) -> tuple[Any, ...]:
        """Return the full structural signature used for caching."""
        return (
            self.kind,
            self.__class__.__name__,
            self.instance_signature(),
            ("dynamic", self.dynamic_param_signature()),
            tuple(dep.signature() for dep in self.dependencies()),
        )

    @classmethod
    def _signature_jsonable(cls, value: Any) -> Any:
        fallback = {
            "__non_portable__": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "id": id(value),
        }

        result: Any
        if value is None or isinstance(value, (str, bool, int, float)):
            result = value
        elif isinstance(value, Enum):
            result = value.value
        elif isinstance(value, np.generic):
            result = value.item()
        elif isinstance(value, (tuple, list)):
            result = [cls._signature_jsonable(item) for item in value]
        elif isinstance(value, dict):
            result = {
                "__dict__": [
                    [cls._signature_jsonable(key), cls._signature_jsonable(item)]
                    for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
                ]
            }
        else:
            signature = getattr(value, "signature", None)
            if callable(signature):
                try:
                    result = {"__signature__": cls._signature_jsonable(signature())}
                except Exception:
                    result = fallback
            else:
                result = fallback

        return result

    def signature_text(self) -> str:
        """Return a stable JSON representation of :meth:`signature`."""
        return json.dumps(
            self._signature_jsonable(self.signature()),
            sort_keys=True,
            separators=(",", ":"),
        )

    def signature_hash(self, *, length: int = 12) -> str:
        """Return a short hash of :meth:`signature_text`.

        Parameters
        ----------
        length : int, default: 12
            Number of hexadecimal characters to return.
        """
        return hashlib.sha256(self.signature_text().encode("utf-8")).hexdigest()[:length]

    @abstractmethod
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

    def __call__(
        self,
        sim: Any,
        options: RunOptions | None = None,
        *,
        cache: bool | None = None,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None,
        perf_time: bool | None = None,
        perf_memory: bool | None = None,
        backend: str | None = None,
        default_record_policy: RecordPolicy | None = None,
        errors: ErrorPolicy | str | None = None,
        cache_small_value_bytes: int | None = None,
    ) -> TPublic:
        """Run the calculator and return a public value.

        This is a convenience alias for :meth:`run` that also accepts common
        execution overrides as keyword-only arguments.

        Parameters
        ----------
        sim : object
            pynbody snapshot or compatible simulation object.
        options : RunOptions, optional
            Base execution options to merge with keyword overrides.
        cache, progress, perf_time, perf_memory,
        backend, default_record_policy, errors, cache_small_value_bytes : optional
            Per-run overrides for :class:`RunOptions`.

        Returns
        -------
        TPublic
            Public value returned by this calculator after execution and materialization.
        """
        return self.run(
            sim,
            options=options,
            cache=cache,
            progress=progress,
            perf_time=perf_time,
            perf_memory=perf_memory,
            backend=backend,
            default_record_policy=default_record_policy,
            errors=errors,
            cache_small_value_bytes=cache_small_value_bytes,
        ).value


    def _resolve_run_options(
        self,
        options: RunOptions | None = None,
        *,
        cache: bool | None = None,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None,
        perf_time: bool | None = None,
        perf_memory: bool | None = None,
        backend: str | None = None,
        default_record_policy: RecordPolicy | None = None,
        errors: ErrorPolicy | str | None = None,
        cache_small_value_bytes: int | None = None,
    ) -> RunOptions:
        merged = copy.copy(options) if options is not None else copy.copy(self.default_options)

        if cache is not None:
            merged.cache = cache
        if progress is not None:
            merged.progress = progress
        if perf_time is not None:
            merged.perf_time = perf_time
        if perf_memory is not None:
            merged.perf_memory = perf_memory
        if backend is not None:
            merged.backend = backend
        if default_record_policy is not None:
            merged.default_record_policy = default_record_policy
        if errors is not None:
            merged.errors = errors
        if cache_small_value_bytes is not None:
            merged.cache_small_value_bytes = cache_small_value_bytes

        merged.errors = ErrorPolicy(merged.errors)

        return merged

    def run(
        self,
        sim: Any,
        options: RunOptions | None = None,
        *,
        cache: bool | None = None,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None,
        perf_time: bool | None = None,
        perf_memory: bool | None = None,
        backend: str | None = None,
        default_record_policy: RecordPolicy | None = None,
        errors: ErrorPolicy | str | None = None,
        cache_small_value_bytes: int | None = None,
    ) -> Result[TPublic]:
        """Evaluate this calculator on a simulation object.

        See :meth:`__call__` for parameter details.
        """
        engine = EvalEngine()
        merged = self._resolve_run_options(
            options=options,
            cache=cache,
            progress=progress,
            perf_time=perf_time,
            perf_memory=perf_memory,
            backend=backend,
            default_record_policy=default_record_policy,
            errors=errors,
            cache_small_value_bytes=cache_small_value_bytes,
        )
        return engine.run(self, sim, merged)

    def value(
        self,
        sim: Any,
        options: RunOptions | None = None,
        **overrides: Any,
    ) -> TPublic:
        """Evaluate the calculator and return only the public value."""
        return self.run(sim, options=options, **overrides).value

    def named(self, name: str) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy that records this node under ``name``."""
        return self._clone(name=name)

    def record(self, policy: RecordPolicy) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with a different result recording policy."""
        return self._clone(record_policy=policy)

    def with_filter(self, filt: FilterBase) -> BoundCalculator[Any, TRaw, TPublic]:
        """Return a calculator evaluated on the subset selected by ``filt``."""
        return BoundCalculator(base=self, scope=self.scope.with_filter(filt))

    def filter(self, filt: FilterBase) -> BoundCalculator[Any, TRaw, TPublic]:
        """Alias for :meth:`with_filter`."""
        return self.with_filter(filt)

    def with_transformation(
        self,
        transform: TransformBase[Any],
        *,
        revert: bool = True,
    ) -> BoundCalculator[Any, TRaw, TPublic]:
        """Return a calculator evaluated after a pre-transform."""
        return BoundCalculator(base=self, scope=self.scope.with_transform(transform, revert=revert))

    def transform(
        self,
        transform: TransformBase[Any],
        *,
        revert: bool = True,
    ) -> BoundCalculator[Any, TRaw, TPublic]:
        """Return a calculator evaluated after applying ``transform``."""
        return self.with_transformation(transform, revert=revert)

    def keep(self, name: str, policy: RecordPolicy = RecordPolicy.FULL) -> CalculatorBase[TRaw, TPublic]:
        """Name the node and retain its value in the returned result."""
        return self._clone(name=name, record_policy=policy)

    def _with_options(self, **changes: Any) -> CalculatorBase[TRaw, TPublic]:
        opts = copy.copy(self.default_options)
        for key, value in changes.items():
            setattr(opts, key, value)
        return self._clone(default_options=opts)

    def with_cache(self, enabled: bool = True) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with a default cache override."""
        return self._with_options(cache=enabled)

    def with_perf(
        self,
        *,
        time: bool = True,
        memory: bool = False,
    ) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with performance collection defaults."""
        return self._with_options(perf_time=time, perf_memory=memory)

    def with_progress(
        self,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] = True,
    ) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with a default progress reporting option."""
        return self._with_options(progress=progress)

    def with_backend(self, name: str) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with a default backend label."""
        return self._with_options(backend=name)

    def with_record_policy(self, policy: RecordPolicy) -> CalculatorBase[TRaw, TPublic]:
        """Alias for :meth:`record`."""
        return self.record(policy)

    def _as_value_property(self) -> PropertyBase[Any]:
        from .expr import as_property

        return as_property(self)

    def __add__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property() + other

    def __radd__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property().__radd__(other)

    def __sub__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property() - other

    def __rsub__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property().__rsub__(other)

    def __mul__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property() * other

    def __rmul__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property().__rmul__(other)

    def __truediv__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property() / other

    def __rtruediv__(self, other: object) -> CalculatorBase[Any, Any]:
        from .expr import as_property, make_binary_op

        left = as_property(other)
        right = self._as_value_property()
        return make_binary_op("truediv", left, right)

    def __pow__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property() ** other

    def __rpow__(self, other: object) -> CalculatorBase[Any, Any]:
        return self._as_value_property().__rpow__(other)



    def format_tree(self) -> str:
        """Return a text tree of this calculator and its dependencies."""
        def render(node: CalculatorBase[Any, Any], prefix: str, is_last: bool) -> list[str]:
            branch = "└─" if is_last else "├─"
            label = f"{node.tree_label}<{node.kind}>"
            lines = [f"{prefix}{branch} {label}"]
            next_prefix = prefix + ("   " if is_last else "│  ")
            kids = node.children()
            for idx, child in enumerate(kids):
                lines.extend(render(child, next_prefix, idx == len(kids) - 1))
            return lines

        return "\n".join(render(self, "", True))

    def format_flow(self) -> str:
        """Return a text flow representation of the calculator graph."""
        return self.format_tree()

    def _clone(self, **changes: Any) -> CalculatorBase[TRaw, TPublic]:
        clone = copy.copy(self)
        for key, value in changes.items():
            setattr(clone, key, value)
        return clone

    def _in_sim_units(
        self,
        value: UnitLike | float | int | SingleElementArray,
        sim_parameter: str,
        sim: Any,
        target_units: UnitLike | None = None,
    ) -> float:
        target_unit = _coerce_unit(target_units) if target_units is not None else sim[sim_parameter].units

        if isinstance(value, str):
            value = _coerce_unit(value)

        if isinstance(value, units.UnitBase):
            value = float(value.in_units(target_unit, **sim.conversion_context()))

        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 1:
                if isinstance(value, SimArray):
                    value = value.in_units(target_unit, **sim.conversion_context()).item()
                else:
                    value = value.item()
            else:
                raise TypeError(f"value must be scalar-like, got shape {value.shape}")

        if isinstance(value, (int, float)):
            return float(value)

        raise TypeError(f"unsupported value type: {type(value)!r}")

    @overload
    def __and__(
        self,
        other: CombinedCalculator[Unpack[Us]],
    ) -> CombinedCalculator[TPublic, Unpack[Us]]: ...

    @overload
    def __and__(
        self,
        other: CalculatorBase[Any, U],
    ) -> CombinedCalculator[TPublic, U]: ...

    def __and__(self, other: object) -> CombinedCalculator[Any, Any]:
        if not isinstance(other, CalculatorBase):
            raise TypeError(f"unsupported operand for &: {type(other)!r}")

        if self.kind == BuiltinKinds.FILTER and other.kind == BuiltinKinds.FILTER:
            from .filters import AndFilter
            and_filter = cast("Any", AndFilter)
            return and_filter(self, other)

        return CombinedCalculator(self, other)


class BoundCalculator(CalculatorBase[TRaw, TPublic], Generic[TBase, TRaw, TPublic]):
    """Calculator wrapper that applies a scope before running a base node.

    ``BoundCalculator`` is created by methods such as
    :meth:`CalculatorBase.filter`, :meth:`CalculatorBase.transform`, and
    :meth:`Scope.apply`.  The concrete wrapped calculator type is preserved on
    :attr:`base`, so scoped calculators can still expose original dataclass
    fields through ``scoped.base`` in static analysis.
    """

    node_kind = BuiltinKinds.CALCULATOR

    def __init__(
        self,
        *,
        base: TBase,
        pre_filter: FilterBase | None = None,
        pre_transform: TransformBase[Any] | None = None,
        revert_transform: bool = True,
        scope: ScopeSpec | None = None,
        name: str | None = None,
        record_policy: RecordPolicy | None = None,
        default_options: RunOptions | None = None,
    ) -> None:
        super().__init__(
            name=name or base.name,
            record_policy=record_policy or base.record_policy,
            default_options=default_options or base.default_options,
        )
        self.base = base
        if scope is None:
            scope = ScopeSpec(filter=pre_filter)
            if pre_transform is not None:
                scope = scope.with_transform(pre_transform, revert=revert_transform)
        self.scope = scope
        self.pre_filter = scope.filter
        self.pre_transform = scope.as_transform()
        self.revert_transform = scope.should_revert
        if self.pre_transform is not None:
            self.cacheable = False

    @property
    def kind(self) -> NodeKind:
        """Kind inherited from the wrapped base calculator."""
        return self.base.kind

    @property
    def log_label(self) -> str:
        """Use the wrapped calculator label instead of the wrapper class name."""
        return self.name or self.base.log_label

    @property
    def tree_label(self) -> str:
        return self.log_label

    def instance_signature(self) -> tuple[Any, ...]:
        return (
            "bound",
            self.base.signature(),
            self.scope.signature(),
        )

    def children(self) -> list[CalculatorBase[Any, Any]]:
        """Display children for graph views.

        The wrapper node already represents ``base`` itself, so tree displays
        should expand the base children directly instead of showing an extra
        nested copy of the base node.
        """
        children = list(self.base.children())
        if self.pre_filter is not None:
            children.append(self.pre_filter)
        if self.pre_transform is not None:
            children.append(self.pre_transform)
        return children

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps: list[CalculatorBase[Any, Any]] = [self.base]
        if self.pre_filter is not None:
            deps.append(self.pre_filter)
        if self.pre_transform is not None:
            deps.append(self.pre_transform)
        return deps

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = [("base", self.base)]
        if self.pre_filter is not None:
            fields.append(("filter", self.pre_filter))
        if self.pre_transform is not None:
            fields.append(("transform", self.pre_transform))
            fields.append(("revert", self.revert_transform))
        if self.name is not None and self.name != self.base.name:
            fields.append(("name", self.name))
        if self.record_policy is not None and self.record_policy != self.base.record_policy:
            fields.append(("record", display_value(self.record_policy)))
        return fields

    def materialize(self, ctx: ExecutionContext, value: TRaw) -> TRaw:
        return self.base.materialize(ctx, value)

    def public_value(self, value: TRaw) -> TPublic:
        return self.base.public_value(value)

    def materialize_public(self, ctx: ExecutionContext, value: TPublic) -> TPublic:
        return self.base.materialize_public(ctx, value)

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> TRaw:
        work = input
        transform_result: TransformResult[Any] | None = None

        if self.pre_transform is not None:
            with ctx.phase(self, "transform"):
                transform_result = ctx.raw_value(self.pre_transform, work)
                if not isinstance(transform_result, TransformResult):
                    raise TypeError("transform nodes must return TransformResult")
                work = work.with_transform(transform_result)

        if self.pre_filter is not None:
            with ctx.phase(self, "filter"):
                filter_result = ctx.raw_value(self.pre_filter, work)
                if not isinstance(filter_result, FilterResult):
                    raise TypeError("filter nodes must return FilterResult")
                work = work.with_selection(filter_result)

        try:
            with ctx.phase(self, "calculate"):
                return ctx.raw_value(self.base, work)
        finally:
            if transform_result is not None and self.revert_transform and transform_result.revertible:
                with ctx.phase(self, "revert"):
                    assert self.pre_transform is not None
                    cleanup = getattr(self.pre_transform, "cleanup", None)
                    if cleanup is None:
                        raise TypeError("transform nodes must provide cleanup()")
                    cleanup(ctx, transform_result.handle)

    def with_filter(self, filt: FilterBase) -> BoundCalculator[TBase, TRaw, TPublic]:
        """Compose another filter into this bound calculator."""
        return BoundCalculator(
            base=self.base,
            scope=self.scope.with_filter(filt),
            name=self.name,
            record_policy=self.record_policy,
            default_options=self.default_options,
        )

    def with_transformation(
        self,
        transform: TransformBase[Any],
        *,
        revert: bool = True,
    ) -> BoundCalculator[TBase, TRaw, TPublic]:
        """Compose another transform into this bound calculator."""
        return BoundCalculator(
            base=self.base,
            scope=self.scope.with_transform(transform, revert=revert),
            name=self.name,
            record_policy=self.record_policy,
            default_options=self.default_options,
        )

    def cleanup(self, ctx: ExecutionContext, handle: Any) -> None:
        """Delegate transform cleanup to the wrapped base calculator when available."""
        cleanup = getattr(self.base, "cleanup", None)
        if cleanup is not None:
            cleanup(ctx, handle)

    def is_revertible(self, handle: Any) -> bool:
        """Delegate transform revertibility checks to the wrapped base calculator."""
        is_revertible = getattr(self.base, "is_revertible", None)
        if is_revertible is not None:
            return bool(is_revertible(handle))
        return hasattr(handle, "revert")


class CombinedCalculator(
    CalculatorBase[tuple[Unpack[Ts]], tuple[Unpack[Ts]]],
    Generic[Unpack[Ts]],
):
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
    def __and__(self, other: CalculatorBase[Any,U]) -> CombinedCalculator[Unpack[Ts], U]: ...

    @overload
    def __and__(self, other: CombinedCalculator[Any]) -> CombinedCalculator[Any]: ...

    def __and__(self, other: object) -> CombinedCalculator[Any]:
        if not isinstance(other, CalculatorBase):
            raise TypeError(f"unsupported operand for &: {type(other)!r}")
        return CombinedCalculator(self, other)

    def instance_signature(self) -> tuple[Any, ...]:
        return ("combined", tuple(item.signature() for item in self.items))

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

