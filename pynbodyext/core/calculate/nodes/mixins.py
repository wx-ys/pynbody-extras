"""Focused mixins composing :class:`~pynbodyext.core.calculate.nodes.base.CalculatorBase`.

``CalculatorBase`` grew to span several unrelated concerns.  Each mixin below
groups a single concern without changing behaviour: every method is relocated
verbatim from ``CalculatorBase``, and no mixin defines ``__init__`` or
``__slots__`` (so the dataclass-style subclass machinery and the existing
``__dict__``/slot layout are untouched).  ``nodes/base.py`` keeps the contract:
class-kind metadata, ``__init__``, the ``dataclass`` classmethod, the ``kind``
property, the abstract ``execute``, and the shared unit-conversion helper.

Mixins call each other only through instance attributes, resolved by the MRO of
``CalculatorBase``, so they stay import-order independent.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, TypeVarTuple, Unpack, cast, overload

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.core.calculate.display import (
    ViewObject,
    _style,
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
from pynbodyext.core.calculate.nodes._tree import hidden_label, label_for, render_children
from pynbodyext.core.calculate.params import (
    DynamicParamSpec,
    RuntimeValueResolver,
    StandaloneValueResolver,
    ValueResolver,
    dynamic_value_dependencies,
    resolve_value_for,
)
from pynbodyext.core.calculate.result.enums import (
    BuiltinKinds,
    ErrorPolicy,
    RecordPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbodyext.core.calculate.result.enums import CachePolicy, NodeKind
    from pynbodyext.core.calculate.result.result import Result
    from pynbodyext.core.calculate.result.signature import CalculatorSignature
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput
    from pynbodyext.core.calculate.runtime.options import RunOptions
    from pynbodyext.core.calculate.runtime.progress import ProgressSink, ProgressVerbosity
    from pynbodyext.core.calculate.runtime.scopes import ScopeSpec
    from pynbodyext.core.calculate.runtime.sim_identity import SimIdentityProvider
    from pynbodyext.core.calculate.store.base import ResultStore
    from pynbodyext.util._type import SingleElementArray, UnitLike

    from .base import CalculatorBase, CombinedCalculator
    from .filters import FilterBase
    from .properties import PropertyBase
    from .transforms import TransformBase

TRaw = TypeVar("TRaw")
TPublic = TypeVar("TPublic")
U = TypeVar("U")
Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")


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
        from pynbodyext.core.calculate.nodes.base import CalculatorBase
        from pynbodyext.core.calculate.result.signature import calculator_from_signature

        calculator = calculator_from_signature(signature)
        if not isinstance(calculator, CalculatorBase):
            raise TypeError(f"signature did not reconstruct a CalculatorBase: {type(calculator)!r}")
        return calculator


class _CalculatorGraphMixin:
    """Parameter declaration and dependency-traversal for a calculator node."""

    scope: ScopeSpec

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
        return _merge_dependencies(
            self.declared_dependencies(), self.dynamic_param_dependencies(), self.scope.dependencies()
        )

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

    @property
    def config(self) -> ViewObject:
        """A view of this calculator's configuration (name/record/scope args)."""
        rows = [(k, v) for k, v in self._repr_fields() if k != "scope"]

        class _Config(ViewObject):
            def _title(self) -> str:
                return "Configuration"

            def _summary(self) -> str:
                if not rows:
                    return "config()"
                return f"config({compact_repr(rows)})"

            def _sections(self) -> list[tuple[str | None, str]]:
                return [(k or "arg", compact_repr(v, max_length=120)) for k, v in rows]

        return _Config()

    @property
    def dependency_tree(self) -> ViewObject:
        """A view of this calculator's dependency tree."""
        tree = self.format_tree()

        class _Tree(ViewObject):
            def _title(self) -> str:
                return "Dependency tree"

            def _summary(self) -> str:
                first = tree.strip().splitlines()
                return first[0] if first else tree.strip()

            def _sections(self) -> list[tuple[str | None, str]]:
                return [(None, tree)]

        return _Tree()

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

        body_parts: list[str] = []
        if _style() == "rich":
            detail_rows: list[tuple[str, Any]] = []
            positional_index = 0
            for key, value in self._repr_fields():  # type: ignore
                if key is None:
                    positional_index += 1  # type: ignore
                    label = f"arg{positional_index}"
                else:
                    label = key
                detail_rows.append((label, compact_repr(value, max_length=220)))
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
        else:
            body_parts.append(html_pre("Use .config and .dependency_tree for details"))

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
        lines = [label_for(calculator, show_inputs=show_inputs, compact_kinds=compact_kinds)]

        if max_depth == 0 and calculator.children():
            lines.append(f"└─ {hidden_label(calculator.children())}")
        else:
            lines.extend(
                render_children(
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
        from pynbodyext.core.calculate.nodes.base import _BatchCaller
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

    def named(self: Self, name: str) -> Self:
        """Return a copy that records this node under ``name``."""
        return self._clone(name=name)

    def record(self: Self, policy: RecordPolicy) -> Self:
        """Return a copy with a different result recording policy."""
        return self._clone(record_policy=policy)

    def with_filter(self: Self, filt: FilterBase) -> Self:
        """Return a calculator evaluated on the subset selected by ``filt``."""
        return self._clone(scope=self.scope.with_filter(filt))

    def filter(self: Self, filt: FilterBase) -> Self:
        """Alias for :meth:`with_filter`."""
        return self.with_filter(filt)

    def with_transformation(self: Self, transform: TransformBase[Any], *, revert: bool = True) -> Self:
        """Return a calculator evaluated after a pre-transform."""
        return self._clone(scope=self.scope.with_transform(transform, revert=revert))

    def transform(self: Self, transform: TransformBase[Any], *, revert: bool = True) -> Self:
        """Return a calculator evaluated after applying ``transform``."""
        return self.with_transformation(transform, revert=revert)

    def keep(self: Self, name: str, policy: RecordPolicy = RecordPolicy.FULL) -> Self:
        """Name the node and retain its value in the returned result."""
        return self._clone(name=name, record_policy=policy)

    def _with_options(self: Self, **changes: Any) -> Self:
        opts = copy.copy(self.default_options)
        for key, value in changes.items():
            setattr(opts, key, value)
        return self._clone(default_options=opts)

    def with_cache(self: Self, enabled: bool = True) -> Self:
        """Return a copy with a default cache override."""
        return self._with_options(cache=enabled)

    def with_perf(self: Self, *, time: bool = True, memory: bool = False) -> Self:
        """Return a copy with performance collection defaults."""
        return self._with_options(perf_time=time, perf_memory=memory)

    def with_progress(
        self: Self,
        progress: bool | ProgressVerbosity | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] = True,
    ) -> Self:
        """Return a copy with a default progress reporting option."""
        return self._with_options(progress=progress)

    def with_observer(self: Self, enabled: bool = True) -> Self:
        """Return a copy with diagnostic field-access observation enabled or disabled."""
        return self._with_options(observe=enabled)

    def with_backend(self: Self, name: str) -> Self:
        """Return a copy with a default backend label."""
        return self._with_options(backend=name)

    def with_record_policy(self: Self, policy: RecordPolicy) -> Self:
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

    def _clone(self: Self, **changes: Any) -> Self:
        clone = copy.copy(self)
        for key, value in changes.items():
            setattr(clone, key, value)
        return clone

    @overload
    def __and__(self, other: CombinedCalculator[Unpack[Us]]) -> CombinedCalculator[TPublic, Unpack[Us]]: ...

    @overload
    def __and__(self, other: CalculatorBase[Any, U]) -> CombinedCalculator[TPublic, U]: ...

    def __and__(self, other: object) -> CombinedCalculator[Any, Any]:
        from pynbodyext.core.calculate.nodes.base import CalculatorBase, CombinedCalculator

        if not isinstance(other, CalculatorBase):
            raise TypeError(f"unsupported operand for &: {type(other)!r}")

        if self.kind == BuiltinKinds.FILTER and other.kind == BuiltinKinds.FILTER:
            from .filters import AndFilter

            and_filter = cast("Any", AndFilter)
            return and_filter(self, other)

        return CombinedCalculator(cast("CalculatorBase[Any, Any]", self), other)
