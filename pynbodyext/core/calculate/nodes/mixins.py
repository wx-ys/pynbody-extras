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
import dataclasses
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.core.calculate.display import (
    Badge,
    Block,
    Card,
    Code,
    Fields,
    Pre,
    Section,
    TextReport,
    compact_repr,
    display_value,
)
from pynbodyext.core.calculate.nodes._tree import render_tree
from pynbodyext.core.calculate.params import (
    DynamicParamSpec,
    RuntimeValueResolver,
    StandaloneValueResolver,
    ValueResolver,
    dynamic_value_dependencies,
    resolve_value_for,
)
from pynbodyext.core.calculate.params.fields import record_init_value
from pynbodyext.core.calculate.result.enums import (
    BuiltinKinds,
    ErrorPolicy,
    RecordPolicy,
)
from pynbodyext.util._type import Self, TypeVarTuple, Unpack

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pynbodyext.core.calculate.result.enums import CachePolicy, NodeKind
    from pynbodyext.core.calculate.result.result import Result
    from pynbodyext.core.calculate.result.signature import CalculatorSignature
    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput
    from pynbodyext.core.calculate.runtime.options import RunOptionKwargs, RunOptions
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


def _refuse_unknown_options(names: Mapping[str, Any], options: RunOptions) -> None:
    """Refuse an override that is not a field of *options*, naming it.

    ``**overrides`` would otherwise accept a typo and either ignore it or fail
    deep inside the dataclass; every entry point that takes run options by keyword
    (``run``, ``__call__``, ``batch``, ``options``) reports the same way.
    """
    option_names = {field.name for field in dataclasses.fields(options)}
    unknown = sorted(set(names) - option_names)
    if unknown:
        raise TypeError(
            f"unknown run option(s): {', '.join(unknown)}. "
            f"Valid options are: {', '.join(sorted(option_names))}."
        )


class _CalculatorContract:
    """Type-only declaration of the ``CalculatorBase`` surface that mixins use.

    The focused mixins below reference members such as ``kind``,
    ``dependencies``, ``children``, ``signature`` and ``to_signature`` that are
    implemented by *other* mixins or by :class:`CalculatorBase` itself and are
    resolved through the composed MRO.  Declaring that contract once here keeps
    each mixin self-contained for static analysis without scattering runtime
    ``raise NotImplementedError`` stubs that must never actually run.

    Nothing is defined at runtime: every member lives under ``TYPE_CHECKING``,
    so the concrete implementation always wins via the MRO.
    """

    if TYPE_CHECKING:
        name: str | None

        @property
        def kind(self) -> NodeKind: ...
        def signature_payload(self) -> Mapping[str, Any] | None: ...
        def signature_hash(self, *, length: int = 12) -> str: ...
        def signature(self) -> tuple[Any, ...]: ...
        def to_signature(self, *, inline_array_bytes: int = 128) -> CalculatorSignature: ...
        def dependencies(self) -> list[CalculatorBase[Any, Any]]: ...
        def children(self) -> list[CalculatorBase[Any, Any]]: ...


class _CalculatorSignatureMixin(_CalculatorContract):
    """Identity, hashing, and reconstruction of a calculator.

    Kept separate so the save/load system can depend on the structured-signature
    path without touching the rest of the calculator surface.
    """

    def signature_payload(self) -> Mapping[str, Any] | None:
        """Return calculator state used by CalculatorSignature generic fallback.

        Dataclass calculators and special calculator nodes do not need this.
        Non-dataclass custom calculators should override it when their behavior
        depends on constructor state beyond declared dependencies.

        To make such a node rebuildable from its signature as well, define the
        inverse as a classmethod, ``from_signature_payload(cls, payload)``, which
        receives this mapping (values decoded) and returns an equal calculator::

            def signature_payload(self):
                return {"hot_mass": self.hot_mass, "total_mass": self.total_mass}

            @classmethod
            def from_signature_payload(cls, payload):
                return cls(payload["hot_mass"], payload["total_mass"])

        Returns
        -------
        Mapping[str, Any] | None
            A JSON-encodable-or-encodable-by-signature mapping describing this
            calculator's identity state, or None to fall back to opaque identity
            (which is neither rebuildable nor stable across sessions).
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

        if inline_array_bytes == 128:
            cached = getattr(self, "_signature_cache", None)
            if cached is not None:
                return cached
            cached = calculator_to_signature(self, inline_array_bytes=inline_array_bytes)
            self._signature_cache = cached
            return cached
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


class _CalculatorGraphMixin(_CalculatorContract):
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


class _CalculatorLoggingMixin(_CalculatorContract):
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


class _CalculatorDisplayMixin(_CalculatorContract):
    """Human-readable and notebook-friendly representation of a calculator."""

    name: str | None
    record_policy: RecordPolicy | None
    scope: ScopeSpec
    cache_policy: CachePolicy

    @property
    def log_label(self) -> str:
        """Name used in reports, progress logs, and graph displays."""
        return self.name or self.__class__.__name__

    @property
    def tree_label(self) -> str:
        """Label used by the dependency tree."""
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
        if self.record_policy is not None:
            rows.append(("record", display_value(self.record_policy)))
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

    #: Summary rows rendered as badges, and their tone.
    _REPR_BADGE_TONES: ClassVar[dict[str, str]] = {"kind": "info", "cache": "neutral", "record": "neutral", "scope": "neutral"}

    def _repr_parameter_blocks(self) -> list[Block]:
        """The *Parameters* section: every signature parameter and its value."""
        from pynbodyext.core.calculate.result.signature import calculator_parameter_rows

        rows = [
            (name, (Code(value), Badge("default")) if is_default else Code(value))
            for name, value, is_default in calculator_parameter_rows(self)
        ]
        if not rows:
            return []
        return [Section("Parameters", [Fields(rows, monospace=True)], collapsible=False)]

    def _repr_signature_blocks(self) -> list[Block]:
        """The *Signature* section: the identity a result is stored under."""
        signature = self.to_signature()
        rows: list[tuple[str, Any]] = [
            ("expression", Code(signature.pretty())),
            ("hash", Code(signature.hash)),
            (
                "reconstructible",
                Badge("yes", "ok") if signature.constructible else Badge("no", "warn"),
            ),
        ]
        if signature.non_constructible_paths:
            rows.append(("not reconstructible at", ", ".join(signature.non_constructible_paths)))
        return [Section("Signature", [Fields(rows, nowrap=True, monospace=True)])]

    def _repr_card(self) -> Card:
        """What this calculator shows in a notebook, in every repr style."""
        rows: list[tuple[str, Any]] = []
        for key, value in self._repr_summary_rows():
            if key == "signature":
                rows.append((key, Code(value)))
            elif key in self._REPR_BADGE_TONES:
                rows.append((key, Badge(display_value(value), self._REPR_BADGE_TONES[key])))
            else:
                rows.append((key, value))
        blocks: list[Block] = [
            *self._repr_parameter_blocks(),
            *self._repr_signature_blocks(),
            Section("Dependency tree", [Pre(self.dependency_tree)], attr="dependency_tree"),
        ]
        return Card(self.__class__.__name__, rows, blocks)

    def _repr_html_(self) -> str:
        return self._repr_card().html()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return self._repr_card().mimebundle(repr(self))

    @property
    def dependency_tree(self) -> TextReport:
        """This calculator's dependency tree, as a renderable text report.

        A :class:`~pynbodyext.core.calculate.display.TextReport`, so it reads like
        the ``str`` it always was (``print``, ``in``, slicing, equality) while
        also rendering as a card in notebooks — no call needed.  The name matches
        the card's ``Dependency tree`` section.  For a truncated tree, use
        :func:`pynbodyext.core.calculate.nodes._tree.render_tree`.
        """
        return TextReport("Dependency tree", render_tree(cast("CalculatorBase[Any, Any]", self)))


class _CalculatorRunMixin(_CalculatorContract, Generic[TRaw, TPublic]):
    """Public execution entry points and run-option resolution."""

    default_options: RunOptions

    def __call__(
        self,
        sim: Any,
        options: RunOptions | None = None,
        *,
        store: ResultStore | None = None,
        sim_identity: SimIdentityProvider | None = None,
        reuse: bool | None = None,
        database: Any = None,
        mode: str | None = None,
        **overrides: Unpack[RunOptionKwargs],
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
        store : ResultStore, optional
            Persistence hook (see :meth:`run`).
        sim_identity : SimIdentityProvider, optional
            Identity provider for a restart-stable simulation address (see
            :meth:`run`).
        reuse : bool, default True
            Return a stored result instead of computing (see :meth:`run`).
        database : Database, optional
            Save and reuse through a local Database. Cannot combine with store/reuse.
        mode : {"manual", "auto", "refresh"}, optional
            Compute only, reuse-or-save, or recompute-and-save. Defaults to the
            applicable db.use context, otherwise auto with a database and manual without.
        **overrides
            Any :class:`RunOptions` field, by name — ``cache``, ``progress``,
            ``perf_time``, ``perf_memory``, ``observe``, ``backend``,
            ``default_record_policy``, ``errors``, ``cache_small_value_bytes``,
            ``auto_record_cached_values``, ``auto_record_small_value_bytes``.  Each
            option is documented on :class:`RunOptions`.

        Returns
        -------
        TPublic
            Public value returned by this calculator after execution and materialization.
        """
        return self.run(sim, options=options, store=store, sim_identity=sim_identity, reuse=reuse,
                        database=database, mode=mode, **overrides).value

    def _resolve_run_options(
        self,
        options: RunOptions | None = None,
        **overrides: Unpack[RunOptionKwargs],
    ) -> RunOptions:
        """Lay keyword *overrides* onto *options* (or this node's own options).

        An override passed as ``None`` keeps the value the base options already
        had, so a call site can forward a partially-filled argument without
        erasing the defaults.  An unknown name is refused by name rather than
        silently ignored, since ``**overrides`` would otherwise swallow a typo.
        """
        merged = copy.copy(options) if options is not None else copy.copy(self.default_options)
        _refuse_unknown_options(overrides, merged)
        for name, value in overrides.items():
            if value is not None:
                setattr(merged, name, value)
        merged.errors = ErrorPolicy(merged.errors)

        return merged

    def run(
        self,
        sim: Any,
        options: RunOptions | None = None,
        *,
        sim_identity: SimIdentityProvider | None = None,
        store: ResultStore | None = None,
        reuse: bool | None = None,
        database: Any = None,
        mode: str | None = None,
        **overrides: Unpack[RunOptionKwargs],
    ) -> Result[TPublic]:
        """Run the calculator and return a Result.

        Parameters
        ----------
        sim : object
            pynbody snapshot or compatible simulation object.
        options : RunOptions, optional
            Base execution options to merge with keyword overrides.
        sim_identity : SimIdentityProvider, optional
            Identity provider mapping the simulation object to an address tuple
            (``("sim", "/data/snap_103", "halo", 0, ...)``).  When given, the engine
            keys provenance and the store on this identity. High-level runs default
            to snapshot_identity (file path plus halo/particle selection); explicit
            transient stores retain session-only defaults. Database identities are
            configured through Database.register instead of this argument.
        store : ResultStore, optional
            A result store keyed by ``(sim_identity(sim), root CalculatorSignature)``.
            A stored result is returned instead of computing (``result.from_store``);
            otherwise the result is stored on a clean run (see :meth:`EvalEngine.run`).
        reuse : bool, default True
            Look the result up in *store* first.  ``False`` always computes and
            replaces the stored result.
        database : Database, optional
            Save and reuse through a local Database. Cannot combine with store/reuse.
        mode : {"manual", "auto", "refresh"}, optional
            Compute only, reuse-or-save, or recompute-and-save. Defaults to the
            applicable db.use context, otherwise auto with a database and manual without.
        **overrides
            Any :class:`RunOptions` field, by name; each option is documented on
            :class:`RunOptions`.  A name passed as ``None`` keeps the value the
            base options already had.

        Returns
        -------
        Result[TPublic]
            Result object containing the public value and diagnostics from this run.
        """
        from pynbodyext.core.calculate.interfaces import DATABASE_CONTEXT
        from pynbodyext.core.calculate.runtime.persistence import execute
        from pynbodyext.core.calculate.runtime.sim_identity import snapshot_identity

        owner = getattr(sim, "_database", None)
        ambient = DATABASE_CONTEXT.get()
        legacy = store is not None or reuse is not None
        if legacy and (database is not None or mode is not None or owner is not None):
            raise ValueError("store/reuse cannot be combined with database/mode or database inputs")
        if owner is not None and database is not None and owner is not database:
            raise ValueError("input belongs to another database")
        selected = None if legacy else (database if database is not None else owner)
        if selected is None and not legacy and ambient is not None:
            selected = ambient[0]
        resolved_mode = mode
        if resolved_mode is None:
            resolved_mode = ambient[1] if ambient is not None and selected is ambient[0] else (
                "auto" if selected is not None else "manual"
            )
        if resolved_mode not in ("manual", "auto", "refresh"):
            raise ValueError("mode must be manual, auto or refresh")
        merged = self._resolve_run_options(options=options, **overrides)
        if selected is not None:
            if sim_identity is not None:
                raise ValueError("register the input identity with Database instead of passing sim_identity")
            return selected.run_calculator(self, sim, options=merged, mode=resolved_mode)
        if not legacy and resolved_mode != "manual":
            raise ValueError("auto/refresh requires a database")
        if sim_identity is None and (store is None or getattr(store, "persistent", True)):
            sim_identity = snapshot_identity
        return execute(cast("CalculatorBase[Any, TPublic]", self), sim, merged, provider=sim_identity, store=store,
                       reuse=True if reuse is None else reuse)

    def value(self, sim: Any, options: RunOptions | None = None, **overrides: Unpack[RunOptionKwargs]) -> TPublic:
        """Evaluate the calculator and return only the public value.

        Parameters
        ----------
        sim :
            A pynbody snapshot or compatible simulation object.
        options : RunOptions, optional
            Base run options.  Keyword *overrides* are merged on top.
        **overrides :
            RunOptions fields to override for this run (see :meth:`options` for
            the recognised keys), e.g. ``value(sim, cache=False)``.

        Returns
        -------
        TPublic
            The public value produced by the calculator.
        """
        return self.run(sim, options=options, **overrides).value

    @contextmanager
    def batch(self, options: RunOptions | None = None, **overrides: Unpack[RunOptionKwargs]) -> Any:  # yields _BatchCaller[TPublic]
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


class _CalculatorComposeMixin(_CalculatorContract, Generic[TRaw, TPublic]):
    """Scoped composition, cloning, and arithmetic operators for a calculator."""

    default_options: RunOptions
    scope: ScopeSpec

    def named(self: Self, name: str) -> Self:
        """Return a copy that records this node under ``name``.

        Parameters
        ----------
        name : str
            Name used to retrieve the node from :attr:`Result.named`.

        Returns
        -------
        Self
            A new calculator carrying the name.
        """
        return self._clone(name=name)

    def record(self: Self, policy: RecordPolicy) -> Self:
        """Return a copy with a different result recording policy.

        Parameters
        ----------
        policy : RecordPolicy
            One of :class:`RecordPolicy` (``FULL``, ``SUMMARY``, ``ERROR_ONLY``,
            ``NONE``).

        Returns
        -------
        Self
            A new calculator with the recording policy set.
        """
        return self._clone(record_policy=policy)

    def filter(self: Self, filt: FilterBase) -> Self:
        """Return a copy evaluated only on the subset selected by ``filt``.

        Parameters
        ----------
        filt : FilterBase
            A filter node producing a boolean mask over the snapshot.

        Returns
        -------
        Self
            A new calculator scoped to the filtered selection.
        """
        return self._clone(scope=self.scope.with_filter(filt))

    def transform(self: Self, transform: TransformBase[Any], *, revert: bool = True) -> Self:
        """Return a copy evaluated after applying ``transform``.

        Parameters
        ----------
        transform : TransformBase
            A transform node applied to the target before the run.
        revert : bool, default: True
            If True, the transform is reverted after the run.

        Returns
        -------
        Self
            A new calculator with the transform composed into its scope.
        """
        return self._clone(scope=self.scope.with_transform(transform, revert=revert))

    def keep(self: Self, name: str, policy: RecordPolicy = RecordPolicy.FULL) -> Self:
        """Name the node and retain its value in the returned result.

        Parameters
        ----------
        name : str
            Name used to retrieve the node from :attr:`Result.named`.
        policy : RecordPolicy, default: RecordPolicy.FULL
            Recording policy: retain the value (``FULL``) or a summary (``SUMMARY``).

        Returns
        -------
        Self
            A new calculator with the name and recording policy set.
        """
        return self._clone(name=name, record_policy=policy)

    def _with_options(self: Self, **changes: Unpack[RunOptionKwargs]) -> Self:
        opts = copy.copy(self.default_options)
        _refuse_unknown_options(changes, opts)
        for key, value in changes.items():
            setattr(opts, key, value)
        return self._clone(default_options=opts)

    def options(self: Self, **changes: Unpack[RunOptionKwargs]) -> Self:
        """Return a copy with run options overridden.

        Accepts any ``RunOptions`` field as a keyword argument; unchanged fields
        keep the calculator's defaults.

        Parameters
        ----------
        **changes :
            Any :class:`~pynbodyext.core.calculate.runtime.options.RunOptions` field,
            by name; each is documented on :class:`RunOptions`.

        Returns
        -------
        Self
            A copy of the calculator with the selected run options overridden.

        Examples
        --------
        >>> from pynbodyext.calculate import Param, PropertyBase
        >>> @PropertyBase.dataclass
        ... class MassSum(PropertyBase[float]):
        ...     qty: Param[str] = Param("mass")
        ...     def calculate(self, sim, params=None):
        ...         return 0.0
        >>> MassSum().options(cache=False).default_options.cache
        False
        >>> MassSum().options(errors="collect").default_options.errors
        'collect'
        """
        return self._with_options(**changes)

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
        clone.__dict__.pop("_signature_cache", None)
        for key, value in changes.items():
            setattr(clone, key, value)
            # The clone changed this field after construction, so the recorded
            # constructor value (used to decide explicit-vs-default) follows.
            record_init_value(clone, key, value)
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
