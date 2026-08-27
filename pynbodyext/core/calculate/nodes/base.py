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

import copy
from abc import ABC
from contextlib import contextmanager
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

from pynbodyext.core.calculate.display import (
    compact_repr,
    display_value,
    html_badge,
    html_card,
    html_details,
    html_pre,
    html_scroll_x,
    html_table,
    mimebundle,
)
from pynbodyext.core.calculate.params import (
    DynamicParamSpec,
    RuntimeValueResolver,
    StandaloneValueResolver,
    ValueResolver,
    dynamic_value_dependencies,
    resolve_value_for,
)
from pynbodyext.core.calculate.params.fields import Param
from pynbodyext.core.calculate.result.enums import (
    BuiltinKinds,
    CachePolicy,
    EffectPolicy,
    ErrorPolicy,
    NodeKind,
    RecordPolicy,
    normalize_kind,
)
from pynbodyext.core.calculate.runtime.input import FilterResult, NodeInput, TransformResult
from pynbodyext.core.calculate.runtime.options import RunOptions
from pynbodyext.core.calculate.runtime.scopes import ScopeSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pynbodyext.core.calculate.result.result import Result
    from pynbodyext.core.calculate.result.signature import CalculatorSignature
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.runtime.progress import ProgressSink, ProgressVerbosity
    from pynbodyext.core.calculate.runtime.sim_identity import SimIdentityProvider
    from pynbodyext.core.calculate.store.base import ResultStore
    from pynbodyext.util._type import SingleElementArray, UnitLike

    from .filters import FilterBase
    from .properties import PropertyBase
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


def _tree_kind_label(kind: str, *, compact: bool) -> str:
    if not compact:
        return kind
    return {"property": "prop", "filter": "filt", "transform": "trans", "calculator": "calc", "combined": "comb"}.get(
        kind, kind
    )


def _tree_input_node(node: CalculatorBase[Any, Any]) -> CalculatorBase[Any, Any]:
    if isinstance(node, BoundCalculator):
        return node.base
    return node


def _tree_label_for(node: CalculatorBase[Any, Any], *, show_inputs: bool, compact_kinds: bool) -> str:
    from pynbodyext.core.calculate.result.signature import calculator_pretty_init_args

    label = node.tree_label
    input_node = _tree_input_node(node)

    if show_inputs:
        init_text = calculator_pretty_init_args(input_node)
        if init_text:
            label = f"{label}({init_text})"

    kind = _tree_kind_label(node.kind, compact=compact_kinds)
    return f"{label}<{kind}>"


def _tree_hidden_label(children: list[CalculatorBase[Any, Any]]) -> str:
    if not children:
        return "..."

    descendants = 0
    stack = list(children)
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        key = id(current)
        if key in seen:
            continue
        seen.add(key)
        descendants += 1
        stack.extend(current.children())

    suffix = "node" if descendants == 1 else "nodes"
    return f"... {descendants} {suffix} hidden"


def _tree_render_children(
    node: CalculatorBase[Any, Any],
    *,
    prefix: str,
    depth: int,
    max_depth: int | None,
    max_children: int | None,
    show_inputs: bool,
    compact_kinds: bool,
) -> list[str]:
    children = node.children()
    if not children:
        return []

    if max_depth is not None and depth >= max_depth:
        return [f"{prefix}└─ {_tree_hidden_label(children)}"]

    visible_children = children if max_children is None else children[:max_children]
    hidden_children = [] if max_children is None else children[max_children:]

    lines: list[str] = []
    for index, child in enumerate(visible_children):
        is_last = index == len(visible_children) - 1 and not hidden_children
        branch = "└─" if is_last else "├─"
        lines.append(f"{prefix}{branch} {_tree_label_for(child, show_inputs=show_inputs, compact_kinds=compact_kinds)}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        lines.extend(
            _tree_render_children(
                child,
                prefix=child_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                show_inputs=show_inputs,
                compact_kinds=compact_kinds,
            )
        )

    if hidden_children:
        lines.append(f"{prefix}└─ {_tree_hidden_label(hidden_children)}")

    return lines


# ---------------------------------------------------------------------------
# Focused mixins
#
# ``CalculatorBase`` grew to span several unrelated concerns.  These mixins
# group its methods by concern without changing behaviour: every method below is
# relocated verbatim from ``CalculatorBase``, and no mixin defines ``__init__``
# or ``__slots__`` (so the dataclass-style subclass machinery and the existing
# ``__dict__``/slot layout are untouched).  The code left in ``CalculatorBase``
# is the contract: class-kind metadata, ``__init__``/``_init_dataclass_base``,
# the ``dataclass`` classmethod, the ``kind`` property, the abstract
# ``execute``, and the shared unit-conversion helper.
# ---------------------------------------------------------------------------


class _CalculatorSignatureMixin:
    """Identity, hashing, and reconstruction of a calculator.

    Kept separate so the save/load system can depend on the structured-signature
    path without touching the rest of the calculator surface.
    """

    def signature_payload(self) -> Mapping[str, Any] | None:
        """Return calculator state used by CalculatorSignature generic fallback.

        Dataclass calculators and special calculator nodes do not need this.
        Non-dataclass custom calculators should override it when their behavior
        depends on constructor state beyond declared dependencies.

        Returns
        -------
        Mapping[str, Any] | None
            A JSON-encodable-or-encodable-by-signature mapping describing this
            calculator's identity state, or None to fall back to opaque identity.
        """
        return None

    def signature(self) -> tuple[Any, ...]:
        """Return the canonical cache key for this calculator."""
        return self.to_signature().cache_key()

    def signature_text(self) -> str:
        """Return the canonical JSON representation of this calculator."""
        return self.to_signature().to_json()

    def signature_hash(self, *, length: int = 12) -> str:
        """Return a short hash of the canonical calculator signature."""
        return self.to_signature().short_hash(length=length)

    def to_signature(self, *, inline_array_bytes: int = 128) -> CalculatorSignature:
        """Return a structured signature that can reconstruct this calculator when possible."""
        from pynbodyext.core.calculate.result.signature import calculator_to_signature

        return calculator_to_signature(self, inline_array_bytes=inline_array_bytes)

    @classmethod
    def from_signature(cls, signature: Any) -> CalculatorBase[Any, Any]:
        """Reconstruct a calculator from a structured signature."""
        from pynbodyext.core.calculate.result.signature import calculator_from_signature

        calculator = calculator_from_signature(signature)
        if not isinstance(calculator, CalculatorBase):
            raise TypeError(f"signature did not reconstruct a CalculatorBase: {type(calculator)!r}")
        return calculator


class _CalculatorGraphMixin:
    """Parameter declaration and dependency-traversal for a calculator node."""

    # Declared here (not only on ``CalculatorBase``) so this mixin is
    # self-contained for static analysis.  ``CalculatorBase`` provides the
    # concrete value.
    dynamic_param_specs: ClassVar[Mapping[str, DynamicParamSpec | str | None]] = {}

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

    def resolve_dynamic_param(self, ctx: ExecutionContext, input: NodeInput, name: str, **kwargs: Any) -> Any:
        """Resolve one named dynamic parameter inside an active run."""
        return self.resolve_param_with(RuntimeValueResolver(ctx, input), name, **kwargs)

    def resolve_dynamic_params(self, ctx: ExecutionContext, input: NodeInput) -> dict[str, Any]:
        """Resolve all declared dynamic parameters inside an active run."""
        resolver = RuntimeValueResolver(ctx, input)
        return {name: self.resolve_param_with(resolver, name) for name in self.dynamic_param_names()}

    def resolve_param_for_sim(
        self, sim: Any | None, name: str, *, options: RunOptions | None = None, **kwargs: Any
    ) -> Any:
        """Resolve one named dynamic parameter outside an active run."""
        return self.resolve_param_with(StandaloneValueResolver(sim, options=options), name, **kwargs)

    def resolve_params_for_sim(self, sim: Any | None, *, options: RunOptions | None = None) -> dict[str, Any]:
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
        return _merge_dependencies(self.declared_dependencies(), self.dynamic_param_dependencies())

    def children(self) -> list[CalculatorBase[Any, Any]]:
        """Return child nodes shown in graph displays."""
        return self.dependencies()


class _CalculatorLoggingMixin:
    """Runtime-aware logging helpers for calculator hooks."""

    def current_runtime(self) -> Any | None:
        """Return the active runtime when this node is inside a run."""
        from pynbodyext.core.calculate.runtime import current_runtime

        return current_runtime()

    def log(self, level: str, message: str, *, phase: str | None = None) -> None:
        """Record a runtime log event from simple subclass hooks.

        This is primarily for ``calculate(self, sim, params=None)`` and
        ``build_handle(...)`` hooks that do not accept ``ctx`` directly. When
        called outside a calculator run, it falls back to the package logger.
        """
        runtime = self.current_runtime()
        if runtime is not None:
            runtime.log(level, message, phase=phase)
            return

        from pynbodyext.log import logger

        log_fn = getattr(logger, level, logger.debug)
        log_fn(message)

    def debug(self, message: str, *, phase: str | None = None) -> None:
        self.log("debug", message, phase=phase)

    def info(self, message: str, *, phase: str | None = None) -> None:
        self.log("info", message, phase=phase)

    def warning(self, message: str, *, phase: str | None = None) -> None:
        self.log("warning", message, phase=phase)

    def error(self, message: str, *, phase: str | None = None) -> None:
        self.log("error", message, phase=phase)


class _CalculatorDisplayMixin:
    """Human-readable and notebook-friendly representation of a calculator."""

    name: str | None
    record_policy: RecordPolicy | None
    scope: ScopeSpec
    cache_policy: CachePolicy

    # Method stubs (overridden by the earlier ``_CalculatorSignatureMixin`` /
    # ``_CalculatorGraphMixin`` bases of ``CalculatorBase``) so this mixin is
    # self-contained for static analysis without changing runtime MRO.
    @property
    def kind(self) -> NodeKind:
        """Placeholder overridden by ``CalculatorBase.kind``."""
        raise NotImplementedError

    def signature_payload(self) -> Mapping[str, Any] | None:
        raise NotImplementedError

    def signature_hash(self, *, length: int = 12) -> str:
        raise NotImplementedError

    def dependencies(self) -> list[CalculatorBase[Any, Any]]:
        raise NotImplementedError

    def children(self) -> list[CalculatorBase[Any, Any]]:
        raise NotImplementedError

    @property
    def log_label(self) -> str:
        """Name used in reports, progress logs, and graph displays."""
        return self.name or self.__class__.__name__

    @property
    def tree_label(self) -> str:
        """Label used by format_tree()."""
        return self.log_label

    def _repr_init_text(self) -> str:
        from pynbodyext.core.calculate.result.signature import calculator_pretty_init_args

        init_text = calculator_pretty_init_args(self)
        if init_text:
            return init_text

        payload = self.signature_payload()
        if not payload:
            return ""

        parts = [f"{key}={compact_repr(value)}" for key, value in payload.items()]
        return ", ".join(parts)

    def _repr_fields(self) -> list[tuple[str | None, Any]]:
        fields: list[tuple[str | None, Any]] = []
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

        init_text = self._repr_init_text()
        if init_text:
            parts.append(init_text)

        for key, value in self._repr_fields():
            text = compact_repr(value)
            parts.append(text if key is None else f"{key}={text}")

        return f"{self.__class__.__name__}({'; '.join(parts)})"

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(f"{self.__class__.__name__}(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        rows: list[tuple[str, Any]] = []
        for key, value in self._repr_summary_rows():
            if key == "kind":
                rows.append((key, html_badge(display_value(value), tone="info")))
            elif key == "cache":
                rows.append((key, html_badge(display_value(value), tone="neutral")))
            elif key == "scope":
                rows.append((key, html_badge(value, tone="neutral")))
            else:
                rows.append((key, value))

        detail_rows: list[tuple[str, Any]] = []
        positional_index = 0
        for key, value in self._repr_fields():  # type: ignore
            if key is None:
                positional_index += 1  # type: ignore
                label = f"arg{positional_index}"
            else:
                label = key
            detail_rows.append((label, compact_repr(value, max_length=220)))

        body_parts: list[str] = []
        if detail_rows:
            body_parts.append(
                html_details(
                    "Configuration",
                    html_scroll_x(
                        html_table(
                            detail_rows,
                            class_name="pynbodyext-calc-table pynbodyext-calc-table-nowrap pynbodyext-calc-monospace",
                        ),
                        min_width="56rem",
                    ),
                )
            )

        body_parts.append(html_details("Dependency tree", html_pre(self.format_tree()), open=False))

        return html_card(self.__class__.__name__, rows, body="".join(body_parts), escape_values=False)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())

    def format_tree(
        self,
        max_depth: int | None = None,
        show_inputs: bool = True,
        *,
        max_children: int | None = None,
        compact_kinds: bool = True,
    ) -> str:
        """Return a text tree of this calculator and its dependencies."""
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if max_children is not None and max_children < 0:
            raise ValueError("max_children must be non-negative or None")

        calculator = cast("CalculatorBase[Any, Any]", self)
        lines = [_tree_label_for(calculator, show_inputs=show_inputs, compact_kinds=compact_kinds)]

        if max_depth == 0 and calculator.children():
            lines.append(f"└─ {_tree_hidden_label(calculator.children())}")
        else:
            lines.extend(
                _tree_render_children(
                    calculator,
                    prefix="",
                    depth=1,
                    max_depth=max_depth,
                    max_children=max_children,
                    show_inputs=show_inputs,
                    compact_kinds=compact_kinds,
                )
            )

        return "\n" + "\n".join(lines)


class _CalculatorRunMixin(Generic[TRaw, TPublic]):
    """Public execution entry points and run-option resolution."""

    default_options: RunOptions

    # Method stubs (overridden by the earlier ``_CalculatorSignatureMixin``
    # base of ``CalculatorBase``) so this mixin is self-contained for static
    # analysis without changing runtime MRO.
    def signature(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def to_signature(self, *, inline_array_bytes: int = 128) -> CalculatorSignature:
        raise NotImplementedError

    def __call__(
        self,
        sim: Any,
        options: RunOptions | None = None,
        *,
        cache: bool | None = None,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None,
        perf_time: bool | None = None,
        perf_memory: bool | None = None,
        observe: bool | None = None,
        backend: str | None = None,
        default_record_policy: RecordPolicy | None = None,
        errors: ErrorPolicy | str | None = None,
        cache_small_value_bytes: int | None = None,
        auto_record_cached_values: bool | None = None,
        auto_record_small_value_bytes: int | None = None,
        sim_identity: SimIdentityProvider | None = None,
        store: ResultStore | None = None,
    ) -> TPublic:
        """Evaluate the calculator and return a public value.

        This is a convenience alias for :meth:`run` that also accepts common
        execution overrides as keyword-only arguments.

        Parameters
        ----------
        sim : object
            pynbody snapshot or compatible simulation object.
        options : RunOptions, optional
            Base execution options to merge with keyword overrides.
        cache: bool, optional
            Whether to cache results of this run.
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None, optional
            Whether and how to report progress for this run. e.g. ``True`` or ``"node"`` to log each node,
            ``"run"``, ``"phase"``, or ``"debug"`` for coarser logs;
            add "bar:" prefix for progress bars (e.g., ``"bar:node"``); or pass custom ProgressSink instances.
        perf_time: bool, optional
            Whether to measure and report execution time for this run.
        perf_memory: bool, optional
            Whether to measure and report memory usage for this run.
        observe: bool, optional
            Whether to observe read, dirty, and delete operations for this simulation.
        backend: str, optional
            Backend to use for this run.
        default_record_policy: RecordPolicy, optional
            Default record policy for this run.
        errors: ErrorPolicy | str, optional
            Error policy for this run.
        cache_small_value_bytes: int, optional
            Cache small value bytes for this run.
        auto_record_cached_values: bool, optional
            Whether cached SUMMARY nodes may retain their public value.
        auto_record_small_value_bytes: int, optional
            Maximum public-value size eligible for automatic SUMMARY recording.
        sim_identity : SimIdentityProvider, optional
            Identity provider for a restart-stable simulation address (see
            :meth:`run`).
        store : ResultStore, optional
            Persistence hook (see :meth:`run`).

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
            observe=observe,
            backend=backend,
            default_record_policy=default_record_policy,
            errors=errors,
            cache_small_value_bytes=cache_small_value_bytes,
            auto_record_cached_values=auto_record_cached_values,
            auto_record_small_value_bytes=auto_record_small_value_bytes,
            sim_identity=sim_identity,
            store=store,
        ).value

    def _resolve_run_options(
        self,
        options: RunOptions | None = None,
        *,
        cache: bool | None = None,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None,
        perf_time: bool | None = None,
        perf_memory: bool | None = None,
        observe: bool | None = None,
        backend: str | None = None,
        default_record_policy: RecordPolicy | None = None,
        errors: ErrorPolicy | str | None = None,
        cache_small_value_bytes: int | None = None,
        auto_record_cached_values: bool | None = None,
        auto_record_small_value_bytes: int | None = None,
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
        if observe is not None:
            merged.observe = observe
        if backend is not None:
            merged.backend = backend
        if default_record_policy is not None:
            merged.default_record_policy = default_record_policy
        if errors is not None:
            merged.errors = errors
        if cache_small_value_bytes is not None:
            merged.cache_small_value_bytes = cache_small_value_bytes
        if auto_record_cached_values is not None:
            merged.auto_record_cached_values = auto_record_cached_values
        if auto_record_small_value_bytes is not None:
            merged.auto_record_small_value_bytes = auto_record_small_value_bytes
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
        observe: bool | None = None,
        backend: str | None = None,
        default_record_policy: RecordPolicy | None = None,
        errors: ErrorPolicy | str | None = None,
        cache_small_value_bytes: int | None = None,
        auto_record_cached_values: bool | None = None,
        auto_record_small_value_bytes: int | None = None,
        sim_identity: SimIdentityProvider | None = None,
        store: ResultStore | None = None,
    ) -> Result[TPublic]:
        """Run the calculator and return a Result.

        Parameters
        ----------
        sim : object
            pynbody snapshot or compatible simulation object.
        options : RunOptions, optional
            Base execution options to merge with keyword overrides.
        cache: bool, optional
            Whether to cache results of this run.
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None, optional
            Whether and how to report progress for this run. e.g. ``True`` or ``"node"`` to log each node,
            ``"run"``, ``"phase"``, or ``"debug"`` for coarser logs;
            add "bar:" prefix for progress bars (e.g., ``"bar:node"``); or pass custom ProgressSink instances.
        perf_time: bool, optional
            Whether to measure and report execution time for this run.
        perf_memory: bool, optional
            Whether to measure and report memory usage for this run.
        observe: bool, optional
            Whether to observe read, dirty, and delete operations for this simulation.
        backend: str, optional
            Backend to use for this run.
        default_record_policy: RecordPolicy, optional
            Default record policy for this run.
        errors: ErrorPolicy | str, optional
            Error policy for this run.
        cache_small_value_bytes: int, optional
            Cache small value bytes for this run.
        auto_record_cached_values: bool, optional
            Whether cached SUMMARY nodes may retain their public value.
        auto_record_small_value_bytes: int, optional
            Maximum public-value size eligible for automatic SUMMARY recording.
        sim_identity : SimIdentityProvider, optional
            Identity provider mapping the simulation object to a stable address
            tuple (``("sim", "snapshot_103", "halo_0")``).  When given, the engine
            keys provenance and the store on this identity instead of the default
            id-based one, so a stored result survives restarts.
        store : ResultStore, optional
            When given, the assembled result is persisted on completion keyed by
            ``(sim_identity(sim), root CalculatorSignature)``.  Only written on a
            clean run (see :meth:`EvalEngine.run`).

        Returns
        -------
        Result[TPublic]
            Result object containing the public value and diagnostics from this run.
        """
        from pynbodyext.core.calculate.runtime.engine import EvalEngine

        engine = EvalEngine(sim_identity=sim_identity) if sim_identity is not None else EvalEngine()
        merged = self._resolve_run_options(
            options=options,
            cache=cache,
            progress=progress,
            perf_time=perf_time,
            perf_memory=perf_memory,
            observe=observe,
            backend=backend,
            default_record_policy=default_record_policy,
            errors=errors,
            cache_small_value_bytes=cache_small_value_bytes,
            auto_record_cached_values=auto_record_cached_values,
            auto_record_small_value_bytes=auto_record_small_value_bytes,
        )
        return engine.run(cast("CalculatorBase[Any, TPublic]", self), sim, merged, store=store)

    def value(self, sim: Any, options: RunOptions | None = None, **overrides: Any) -> TPublic:
        """Evaluate the calculator and return only the public value."""
        return self.run(sim, options=options, **overrides).value

    @contextmanager
    def batch(self, options: RunOptions | None = None, **overrides: Any) -> Any:  # yields _BatchCaller[TPublic]
        """Context manager for efficient repeated execution on many sims.

        Pre-computes the node signature once and creates a single
        :class:`~pynbodyext.core.calculate.runtime.engine.EvalEngine`, then
        yields a callable ``run_one(sim) -> TPublic`` that uses the lightweight
        :meth:`~pynbodyext.core.calculate.runtime.engine.EvalEngine.run_light`
        path for each invocation.

        This avoids the per-call overhead of ``uuid.uuid4()``,
        ``_estimate_total_nodes``, ``_assemble_result``, and duplicate
        ``to_signature()`` calls, giving a **~4–10×** speedup over calling
        ``__call__`` or ``run()`` in a tight loop.

        Example::

            with calc.batch(cache=False, progress=False) as run_one:
                for sub_sim in bin_subs:
                    value = run_one(sub_sim)

        Parameters
        ----------
        options:
            Base :class:`~pynbodyext.core.calculate.runtime.options.RunOptions`
            to use for all iterations.  Keyword *overrides* are merged on top.
        **overrides:
            Keyword overrides forwarded to :meth:`_resolve_run_options` (e.g.
            ``cache=False``, ``progress=False``).
        """
        from pynbodyext.core.calculate.runtime.engine import EvalEngine

        opts = self._resolve_run_options(options, **overrides)
        engine = EvalEngine()
        node_sig = self.signature()
        structured_sig = self.to_signature()
        yield _BatchCaller(cast("CalculatorBase[Any, TPublic]", self), engine, opts, node_sig, structured_sig)

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


class _CalculatorComposeMixin(Generic[TRaw, TPublic]):
    """Scoped composition, cloning, and arithmetic operators for a calculator."""

    default_options: RunOptions
    scope: ScopeSpec

    @property
    def kind(self) -> NodeKind:
        """Placeholder overridden by ``CalculatorBase.kind``."""
        raise NotImplementedError

    def named(self, name: str) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy that records this node under ``name``."""
        return self._clone(name=name)

    def record(self, policy: RecordPolicy) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with a different result recording policy."""
        return self._clone(record_policy=policy)

    def with_filter(self, filt: FilterBase) -> BoundCalculator[Any, TRaw, TPublic]:
        """Return a calculator evaluated on the subset selected by ``filt``."""
        return BoundCalculator(base=cast("CalculatorBase[Any, Any]", self), scope=self.scope.with_filter(filt))

    def filter(self, filt: FilterBase) -> BoundCalculator[Any, TRaw, TPublic]:
        """Alias for :meth:`with_filter`."""
        return self.with_filter(filt)

    def with_transformation(
        self, transform: TransformBase[Any], *, revert: bool = True
    ) -> BoundCalculator[Any, TRaw, TPublic]:
        """Return a calculator evaluated after a pre-transform."""
        return BoundCalculator(
            base=cast("CalculatorBase[Any, Any]", self), scope=self.scope.with_transform(transform, revert=revert)
        )

    def transform(self, transform: TransformBase[Any], *, revert: bool = True) -> BoundCalculator[Any, TRaw, TPublic]:
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

    def with_perf(self, *, time: bool = True, memory: bool = False) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with performance collection defaults."""
        return self._with_options(perf_time=time, perf_memory=memory)

    def with_progress(
        self, progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] = True
    ) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with a default progress reporting option."""
        return self._with_options(progress=progress)

    def with_observer(self, enabled: bool = True) -> CalculatorBase[TRaw, TPublic]:
        """Return a copy with diagnostic field-access observation enabled or disabled."""
        return self._with_options(observe=enabled)

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

    def _clone(self, **changes: Any) -> CalculatorBase[TRaw, TPublic]:
        clone = copy.copy(self)
        for key, value in changes.items():
            setattr(clone, key, value)
        return cast("CalculatorBase[TRaw, TPublic]", clone)

    @overload
    def __and__(self, other: CombinedCalculator[Unpack[Us]]) -> CombinedCalculator[TPublic, Unpack[Us]]: ...

    @overload
    def __and__(self, other: CalculatorBase[Any, U]) -> CombinedCalculator[TPublic, U]: ...

    def __and__(self, other: object) -> CombinedCalculator[Any, Any]:
        if not isinstance(other, CalculatorBase):
            raise TypeError(f"unsupported operand for &: {type(other)!r}")

        if self.kind == BuiltinKinds.FILTER and other.kind == BuiltinKinds.FILTER:
            from .filters import AndFilter

            and_filter = cast("Any", AndFilter)
            return and_filter(self, other)

        return CombinedCalculator(cast("CalculatorBase[Any, Any]", self), other)


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

    def filter(self, filt: FilterBase) -> BoundCalculator[TBase, TRaw, TPublic]:
        return self.with_filter(filt)

    def transform(self, transform: TransformBase[Any], *, revert: bool = True) -> BoundCalculator[TBase, TRaw, TPublic]:
        return self.with_transformation(transform, revert=revert)

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
        self, transform: TransformBase[Any], *, revert: bool = True
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
