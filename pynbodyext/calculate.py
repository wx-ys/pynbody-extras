"""Public calculator entry point for most pynbodyext users.

This module provides the short import path for the calculator framework::

    from pynbodyext.calculate import (
        CalculatorBase,
        RuntimeCalculatorBase,
        PropertyBase,
        FilterBase,
        TransformBase,
        Pipeline,
        Param,
    )

Use this module when you want the common public building blocks for defining
and composing calculator graphs without importing from the longer
:mod:`pynbodyext.core.calculate` path.

Most users only need:

- :class:`PropertyBase`
- :class:`FilterBase`
- :class:`TransformBase`
- :class:`Param`
- :class:`Pipeline`

Advanced users may also need:

- :class:`RuntimeCalculatorBase` for custom nodes that still follow the
  standard runtime lifecycle
- :class:`CalculatorBase` for the lowest-level custom execution model

Choosing A Base Class
---------------------
For new calculators, prefer the narrowest base class that matches the node.

1. Start with :class:`PropertyBase` for a read-only derived value.
2. Use :class:`FilterBase` for a boolean mask.
3. Use :class:`TransformBase` for a temporary mutation.
4. If none of those fit, but the node still follows the standard runtime
   lifecycle, use :class:`RuntimeCalculatorBase`.
5. Only use :class:`CalculatorBase` directly when you need to implement
   :meth:`CalculatorBase.execute` yourself.

This matches the internal layering of the framework:
:class:`PropertyBase`, :class:`FilterBase`, and :class:`TransformBase` are all
specialized layers built on top of :class:`RuntimeCalculatorBase`.

Quick Start
-----------
Define a simple property calculator::

    from pynbodyext.calculate import PropertyBase


    @PropertyBase.dataclass
    class StellarMass(PropertyBase[float]):
        def calculate(self, sim, params=None):
            return float(sim["mass"].sum())


    result = StellarMass().run(sim)
    print(result.value)

Compose a property with a custom filter::

    import numpy as np

    from pynbodyext.calculate import FilterBase, PropertyBase


    @FilterBase.dataclass
    class TemperatureAbove(FilterBase):
        threshold: float

        def calculate(self, sim, params=None):
            return sim["temp"] > self.threshold


    @PropertyBase.dataclass
    class MeanTemperature(PropertyBase[float]):
        def calculate(self, sim, params=None):
            return float(np.asarray(sim["temp"]).mean())


    result = MeanTemperature().filter(TemperatureAbove(1.0e5)).run(sim)
    print(result.value)

Apply a temporary transform::

    from pynbodyext.calculate import PropertyBase, TransformBase


    @TransformBase.dataclass
    class XShift(TransformBase[dict[str, object]]):
        dx: float

        def build_handle(self, sim, target, params=None):
            original = target["x"].copy()
            target["x"] = target["x"] + self.dx
            return {"target": target, "original_x": original}

        def cleanup(self, ctx, handle):
            handle["target"]["x"] = handle["original_x"]

        def is_revertible(self, handle):
            return True


    @PropertyBase.dataclass
    class XMean(PropertyBase[float]):
        def calculate(self, sim, params=None):
            return float(sim["x"].mean())


    result = XMean().transform(XShift(1.0)).run(sim)
    print(result.value)

Evaluate several outputs in one run::

    from pynbodyext.calculate import Pipeline

    pipe = Pipeline(
        {"mass": StellarMass(), "hot_temp": MeanTemperature().filter(TemperatureAbove(1.0e5))}, name="basic_summary"
    )

    result = pipe.run(sim, progress="phase")
    print(result.value["mass"])
    print(result.value["hot_temp"])

When To Import From Core
------------------------
Stay on :mod:`pynbodyext.calculate` when you only need the common public
building blocks.

Switch to :mod:`pynbodyext.core.calculate` when you also need:

- :class:`Result` and :class:`ResultNode`
- :class:`RunOptions`
- trace, cache, and performance helpers
- low-level runtime and debugging types

Compatibility Note
------------------
This module is a public facade over the newer calculator framework. It is the
recommended short import path for new code, but it does not restore the legacy
historical execution contract from older implementations.

If you are migrating older subclasses, review the role-specific modules in
:mod:`pynbodyext.core.calculate` and prefer the modern dataclass-based style
already used in :mod:`pynbodyext.properties`, :mod:`pynbodyext.filters`, and
:mod:`pynbodyext.transforms`.
"""

from .core.calculate import (
    Bin1D,
    CalculatorBase,
    FilterBase,
    Param,
    Pipeline,
    PropertyBase,
    RuntimeCalculatorBase,
    TransformBase,
    get_repr_style,
    reset_repr_style,
    set_repr_style,
)

__all__ = [
    "CalculatorBase",
    "Bin1D",
    "RuntimeCalculatorBase",
    "PropertyBase",
    "FilterBase",
    "TransformBase",
    "Pipeline",
    "Param",
    "get_repr_style",
    "set_repr_style",
    "reset_repr_style",
]
