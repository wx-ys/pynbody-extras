"""Scope helpers for applying filters and transforms to calculators.

Scopes are immutable descriptions of preprocessing applied before a calculator
runs.  They can be built explicitly with :class:`Scope` or created implicitly
through fluent helpers such as ``calculator.filter(...)`` and
``calculator.transform(...)``.

Why Scope Exists
----------------
A scope is the framework's way to describe "run this calculator, but first
apply these filters and transforms".  Scopes make that behavior explicit,
composable, and reusable.

This lets the same base calculator be reused across many filtered or transformed
views without mutating the calculator itself.

Basic Example
-------------
Create a reusable scope and apply it to several calculators::

    from pynbodyext.core.calculate import Scope

    aperture = Scope(filter=TemperatureAbove(1.0e5))

    hot_mass = aperture.apply(StellarMass())
    hot_temp = aperture.apply(MeanTemperature())

    print(hot_mass.run(sim).value)
    print(hot_temp.run(sim).value)

Fluent Style
------------
In normal user code, scopes are usually created indirectly through calculator
methods::

    hot_mass = StellarMass().filter(TemperatureAbove(1.0e5))
    shifted_temp = MeanTemperature().transform(XShift(1.0))

This fluent style is the most common way to work with scope in day-to-day use.

Explicit Scope Builder
----------------------
The explicit :class:`Scope` API is useful when the same preprocessing should be
applied to many calculators or pipelines::

    summary_scope = Scope(filter=TemperatureAbove(1.0e5))
    pipe = summary_scope.pipeline(
        {
            "mass": StellarMass(),
            "temp_mean": MeanTemperature(),
        }
    )

    result = pipe.run(sim)
    print(result.value["mass"])

ScopeSpec
---------
:class:`ScopeSpec` is the immutable internal representation used by bound
calculators.  It stores:

- the ordered transform chain
- the composed filter
- the selected revert policy

Most end users do not need to construct :class:`ScopeSpec` directly, but it is
helpful to understand when debugging composed scopes.

Composition Example
-------------------
Scopes compose naturally.  A filter can be merged with another filter, and a
transform chain can be extended by appending more transforms::

    scoped = (
        StellarMass()
        .filter(TemperatureAbove(1.0e5))
        .transform(XShift(1.0))
    )

    result = scoped.run(sim)

When This Module Matters
------------------------
This module is most relevant when you are:

- building reusable analysis presets
- applying one scope to many calculators
- debugging how filter/transform composition works
- writing custom tooling on top of calculator graphs

Notes
-----
Prefer the fluent calculator API for one-off usage.  Reach for :class:`Scope`
when you want to share the same preprocessing across many calculators or build
higher-level analysis presets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .display import compact_repr, display_value, html_card, mimebundle
from .enums import BuiltinKinds, RevertPolicy, normalize_revert_policy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .base import CalculatorBase
    from .filters import FilterBase
    from .transforms import TransformBase


@dataclass(frozen=True, slots=True)
class ScopeSpec:
    """Immutable scope specification used internally by bound calculators."""

    transforms: tuple[TransformBase[Any], ...] = ()
    filter: FilterBase | None = None
    revert_policy: RevertPolicy = RevertPolicy.ALWAYS

    def with_filter(self, filt: FilterBase) -> ScopeSpec:
        """Return a copy with ``filt`` composed into the scope."""
        merged = filt if self.filter is None else self.filter & filt
        return ScopeSpec(
            transforms=self.transforms,
            filter=merged,
            revert_policy=self.revert_policy,
        )

    def with_transform(
        self,
        transform: TransformBase[Any],
        *,
        revert: RevertPolicy | str | bool | None = None,
    ) -> ScopeSpec:
        """Return a copy with a transform appended to the scope."""
        if transform.kind != BuiltinKinds.TRANSFORM:
            raise TypeError(f"ScopeSpec transforms must be transform nodes, got {type(transform)!r}")
        if revert is None:
            policy = self.revert_policy
        elif self.revert_policy == RevertPolicy.NEVER:
            policy = RevertPolicy.NEVER
        else:
            policy = normalize_revert_policy(revert)
        return ScopeSpec(
            transforms=(*self.transforms, transform),
            filter=self.filter,
            revert_policy=policy,
        )

    def compose(self, child: ScopeSpec) -> ScopeSpec:
        """Compose this scope with a child scope."""
        merged_filter = self.filter
        if child.filter is not None:
            merged_filter = child.filter if merged_filter is None else merged_filter & child.filter
        policy = RevertPolicy.NEVER if RevertPolicy.NEVER in (self.revert_policy, child.revert_policy) else RevertPolicy.ALWAYS
        return ScopeSpec(
            transforms=(*self.transforms, *child.transforms),
            filter=merged_filter,
            revert_policy=policy,
        )

    def as_transform(self) -> TransformBase[Any] | None:
        """Return the transform component as a single transform node."""
        if not self.transforms:
            return None
        if len(self.transforms) == 1:
            return self.transforms[0]
        from .transforms import chain_transforms

        return chain_transforms(*self.transforms)

    def dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps: list[CalculatorBase[Any, Any]]
        deps = list(self.transforms)
        if self.filter is not None:
            deps.append(self.filter)
        return deps

    def signature(self) -> tuple[Any, ...]:
        """Return the structural signature for this scope."""
        return (
            "scope",
            tuple(transform.signature() for transform in self.transforms),
            self.filter.signature() if self.filter is not None else None,
            self.revert_policy.value,
        )

    @property
    def is_empty(self) -> bool:
        """Whether this scope contains no transforms, filters, or policy changes."""
        return not self.transforms and self.filter is None and self.revert_policy == RevertPolicy.ALWAYS

    @property
    def should_revert(self) -> bool:
        """Whether scoped transforms should be cleaned up automatically."""
        return self.revert_policy == RevertPolicy.ALWAYS

    def short_label(self) -> str:
        """Return a compact human-readable scope label."""
        parts: list[str] = []
        if self.transforms:
            names = " -> ".join(transform.log_label for transform in self.transforms)
            parts.append(f"transform={names}")
        if self.filter is not None:
            parts.append(f"filter={self.filter.log_label}")
        if self.revert_policy != RevertPolicy.ALWAYS:
            parts.append(f"revert={self.revert_policy.value}")
        return ", ".join(parts) if parts else "empty"

    def __repr__(self) -> str:
        fields: list[str] = []
        if self.transforms:
            fields.append(f"transforms={compact_repr(tuple(transform.log_label for transform in self.transforms))}")
        if self.filter is not None:
            fields.append(f"filter={compact_repr(self.filter)}")
        if self.revert_policy != RevertPolicy.ALWAYS:
            fields.append(f"revert_policy={display_value(self.revert_policy)!r}")
        return f"ScopeSpec({', '.join(fields)})"

    def _repr_html_(self) -> str:
        return html_card(
            "ScopeSpec",
            [
                ("transforms", len(self.transforms)),
                ("filter", self.filter.log_label if self.filter is not None else "-"),
                ("revert", display_value(self.revert_policy)),
            ],
        )

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())


class Scope:
    """User-facing scope builder for filters and transforms.

    Parameters
    ----------
    transform, transforms : CalculatorBase, optional
        Transform node or sequence of transform nodes to apply before the
        target calculator.
    filter : FilterBase, optional
        Filter applied to the active simulation view.
    revert_policy : RevertPolicy, str, or bool, default: RevertPolicy.ALWAYS
        Whether transform cleanup should run after the scoped calculation.
    """

    def __init__(
        self,
        *,
        transform: TransformBase[Any] | None = None,
        transforms: Sequence[TransformBase[Any]] | None = None,
        filter: FilterBase | None = None,
        revert_policy: RevertPolicy | str | bool = RevertPolicy.ALWAYS,
    ) -> None:
        if transform is not None and transforms is not None:
            raise ValueError("Use either transform= or transforms=, not both.")
        spec = ScopeSpec(revert_policy=normalize_revert_policy(revert_policy))
        for item in (() if transforms is None else transforms):
            spec = spec.with_transform(item)
        if transform is not None:
            spec = spec.with_transform(transform)
        if filter is not None:
            spec = spec.with_filter(filter)
        self.spec = spec

    @classmethod
    def from_spec(cls, spec: ScopeSpec) -> Scope:
        """Create a user-facing scope from an existing specification."""
        obj = cls()
        obj.spec = spec
        return obj

    def filter(self, filt: FilterBase) -> Scope:
        """Return a copy with ``filt`` added to the scope."""
        return Scope.from_spec(self.spec.with_filter(filt))

    def transform(
        self,
        transform: TransformBase[Any],
        *,
        revert: RevertPolicy | str | bool | None = None,
    ) -> Scope:
        """Return a copy with ``transform`` appended to the scope."""
        return Scope.from_spec(self.spec.with_transform(transform, revert=revert))

    def compose(self, child: Scope | ScopeSpec) -> Scope:
        """Compose this scope with another scope or specification."""
        child_spec = child.spec if isinstance(child, Scope) else child
        return Scope.from_spec(self.spec.compose(child_spec))

    def apply(self, calculator: CalculatorBase[Any, Any]) -> CalculatorBase[Any, Any]:
        """Apply this scope to ``calculator`` and return a bound calculator."""
        from .base import BoundCalculator

        if isinstance(calculator, BoundCalculator):
            return BoundCalculator(
                base=calculator.base,
                scope=self.spec.compose(calculator.scope),
                name=calculator.name,
                record_policy=calculator.record_policy,
                default_options=calculator.default_options,
            )
        return BoundCalculator(base=calculator, scope=self.spec.compose(getattr(calculator, "scope", ScopeSpec())))

    def pipeline(self, outputs: Mapping[str, CalculatorBase[Any, Any]], *, name: str | None = None) -> CalculatorBase[dict[str, Any], dict[str, Any]]:
        """Build a :class:`Pipeline` and apply this scope to it."""
        from .pipeline import Pipeline

        return self.apply(Pipeline(outputs, name=name))

    def signature(self) -> tuple[Any, ...]:
        """Return the structural signature for this scope object."""
        return ("scope_object", self.spec.signature())

    def __repr__(self) -> str:
        return f"Scope({self.spec.short_label()})" if not self.spec.is_empty else "Scope()"

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("Scope(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        return html_card(
            "Scope",
            [
                ("transforms", len(self.spec.transforms)),
                ("filter", self.spec.filter.log_label if self.spec.filter is not None else "-"),
                ("revert", display_value(self.spec.revert_policy)),
            ],
        )

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())


TransformScope = Scope
