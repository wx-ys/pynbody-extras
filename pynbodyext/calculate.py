"""Compatibility facade for the new calculator framework.

This module re-exports the primary calculator base classes from
:mod:`pynbodyext.core.calculate` so that users can import them from the shorter
module path::

    from pynbodyext.calculate import (
        CalculatorBase,
        PropertyBase,
        FilterBase,
        TransformBase,
        Pipeline,
    )

The new implementation lives in :mod:`pynbodyext.core.calculate`.  Prefer
importing from that package directly when you need the full public API,
including results, progress/reporting types, concrete calculators, and runtime
utilities.

What This Module Exports
------------------------
This facade exposes the main node abstractions used to define and compose
calculator graphs:

- :class:`CalculatorBase` for custom execution nodes
- :class:`PropertyBase` for read-only value calculators
- :class:`FilterBase` for boolean selection calculators
- :class:`TransformBase` for temporary mutating calculators
- :class:`Pipeline` for grouped multi-output evaluation
- :class:`CombinedCalculator` for tuple-style grouped evaluation

Quick Start
-----------
Define a simple property-like calculator::

    from pynbodyext.calculate import CalculatorBase

    class MassMean(CalculatorBase[float]):
        def instance_signature(self):
            return ("mass_mean",)

        def execute(self, ctx, input):
            sim = input.active_sim
            return float(sim["mass"].mean())

    value = MassMean().value(sim)

Use the narrower specialized bases when possible::

    from pynbodyext.calculate import PropertyBase, FilterBase, TransformBase

    class StellarMass(PropertyBase[float]):
        def instance_signature(self):
            return ("stellar_mass",)

        def calculate(self, sim):
            return float(sim["mass"].sum())

Compose calculators with filters, transforms, and pipelines::

    from pynbodyext.calculate import Pipeline
    from pynbodyext.core.calculate import ParamContain, KappaRot, FamilyFilter, Sphere

    pipe = (
        Pipeline(
            {
                "re": ParamContain(),
                "krot": KappaRot(),
            }
        )
        .filter(FamilyFilter("star") & Sphere("30 kpc"))
    )

    result = pipe(sim)
    print(result.get("re"))
    print(result.get("krot"))

Compatibility Notes
-------------------
This module is a facade over the new framework.  It does not preserve the old
``pynbodyext.calculate`` execution contract based on ``calculate(sim)``,
``with_filter(...)`` mutating the same instance, or legacy profiling/cache
helpers such as ``enable_perf()`` and ``enable_cache()``.

As a result:

- new code may safely import these base classes from :mod:`pynbodyext.calculate`
- old modules that still depend on the legacy calculator implementation must be
  migrated before they can inherit from these re-exported classes

If you are updating older code, prefer migrating imports directly to
:mod:`pynbodyext.core.calculate` and updating subclasses to the new execution
contract.
"""

from .core.calculate import (
    CalculatorBase,
    CombinedCalculator,
    FilterBase,
    Param,
    Pipeline,
    PropertyBase,
    RuntimeCalculatorBase,
    TransformBase,
)

__all__ = [
    "CalculatorBase",
    "RuntimeCalculatorBase",
    "PropertyBase",
    "FilterBase",
    "TransformBase",
    "Pipeline",
    "CombinedCalculator",
    "Param",
]
