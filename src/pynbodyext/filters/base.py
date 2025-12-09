"""
Filter calculators for pynbody-extras.

This module defines :class:`~pynbodyext.filters.base.FilterBase`, which
bridges pynbody-style filters and the :class:`~pynbodyext.calculate.CalculatorBase`
framework. It also provides logical combination helpers:

* :class:`And` – logical AND of two filters
* :class:`Or`  – logical OR of two filters
* :class:`Not` – logical NOT of a filter

A :class:`FilterBase` is a calculator with signature::

    SimSnap -> np.ndarray[bool]

and can be attached to any other calculator via
:meth:`pynbodyext.calculate.CalculatorBase.with_filter`.

See also
--------
* :mod:`pynbodyext.calculate` for the core calculator base class.
* :mod:`pynbodyext.filters.filt` for high-level domain filters such as
  :class:`~pynbodyext.filters.filt.FamilyFilter` and spatial selections.

Basic usage
-----------

A filter can be directly used to select particles from a :class:`pynbody.snapshot.SimSnap`:

.. code-block:: python
    from pynbody.snapshot import SimSnap
    from pynbodyext.filters import FamilyFilter

    sim = ...  # a SimSnap
    star_filter = FamilyFilter("stars")
    stars = sim[star_filter]        # view with only star particles


A filter can also be used to restrict the particles considered by another calculator.
For example, to compute the total stellar mass within 30 kpc:

.. code-block:: python

   import pynbody.filt as pyn_filt
   from pynbodyext.filters import FamilyFilter
   from pynbodyext.properties.base import ParamSum

   # sum of stellar mass within 30 kpc
   mstar = (
       ParamSum("mass")
       .with_filter(pyn_filt.Sphere("30 kpc") & FamilyFilter("stars"))
   )

   value = mstar(sim)

Here,

* :class:`FamilyFilter` is defined in :mod:`pynbodyext.filters.filt`
  but inherits from :class:`FilterBase`, so it participates in caching
  and diagnostics.
* The bitwise operators ``&``, ``|`` and ``~`` create :class:`And`,
  :class:`Or` and :class:`Not` filters.

Custom filters
--------------

To write a custom filter calculator, subclass :class:`FilterBase` and
implement :meth:`calculate`:

.. code-block:: python

   import numpy as np
   from pynbody.snapshot import SimSnap
   from pynbodyext.filters.base import FilterBase

   class HighSFR(FilterBase):
       \"\"\"Select particles with SFR above a threshold.\"\"\"
       def __init__(self, thr: float):
           self.thr = thr

       def instance_signature(self):
           return (self.__class__.__name__, float(self.thr))

       def calculate(self, sim: SimSnap) -> np.ndarray:
           sfr = sim["sfr"]
           return sfr > self.thr

This filter can be combined and used like built-ins:

.. code-block:: python

   from pynbodyext.properties.base import ParamSum

   sfr_high_mass = (
       ParamSum("mass")
       .with_filter(HighSFR(1e-2) & FamilyFilter("gas"))
   )

   val = sfr_high_mass(sim)

Because :class:`FilterBase` inherits from :class:`CalculatorBase`, it
integrates with:

* profiling (:meth:`enable_perf`)
* run-local caching (:meth:`enable_cache`)
* :meth:`format_flow` and :meth:`format_tree` for inspection.

"""

from typing import Any

import numpy as np
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import CalculatorBase
from pynbodyext.log import logger
from pynbodyext.util._type import get_signature_safe

from .pynfilt import _And, _Filter, _Not, _Or

__all__ = ["FilterBase", "And", "Or", "Not"]

class FilterBase(CalculatorBase[np.ndarray],_Filter):
    """
    Base class for filters.

    Connect pynbody filters with CalculatorBase.
    """
    def children(self) -> list[CalculatorBase[Any]]:
        return super().children()

    def calculate(self, sim: SimSnap)-> np.ndarray:
        return np.ones(len(sim), dtype=bool)

    def __and__(self, f2):
        return And(self, f2)

    def __invert__(self):
        return Not(self)

    def __repr__(self):
        return f"<Filter {self.__class__.__name__}>"

    def __or__(self, f2):
        return Or(self, f2)

    def with_filter(self, filter: Any) -> "FilterBase":
        logger.warning("You'd typically want to use bitwise operators (&, |, ~) to combine filters. \
                    For example: filter1 & filter2, filter1 | filter2, ~filter1")
        return self

    def _apply_filter(self, sim):
        if self._filter is not None:
            logger.warning("FilterBase should not have a pre-filter applied. Ignoring pre-filter.")

        return sim


class And(FilterBase,_And):
    calculate = _And.__call__

    def __init__(self, f1: FilterBase, f2: FilterBase):
        _And.__init__(self, f1, f2)

    def instance_signature(self):
        f1 = get_signature_safe(self.f1, fallback_to_id=True)
        f2 = get_signature_safe(self.f2, fallback_to_id=True)
        return (self.__class__.__name__, f1, f2)

    def children(self) -> list[CalculatorBase[Any]]:
        # Combine child nodes + own filter/transformation
        return [self.f1,self.f2,*super().children()]
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        return [self.f1,self.f2]
class Or(FilterBase,_Or):
    calculate = _Or.__call__

    def __init__(self, f1: FilterBase, f2: FilterBase):
        _Or.__init__(self, f1, f2)

    def instance_signature(self):
        f1 = get_signature_safe(self.f1, fallback_to_id=True)
        f2 = get_signature_safe(self.f2, fallback_to_id=True)
        return (self.__class__.__name__, f1, f2)
    def children(self) -> list[CalculatorBase[Any]]:
        return [self.f1,self.f2,*super().children()]
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        return [self.f1,self.f2]

class Not(FilterBase,_Not):
    calculate = _Not.__call__

    def __init__(self, f: FilterBase):
        _Not.__init__(self, f)

    def instance_signature(self):
        f = get_signature_safe(self.f, fallback_to_id=True)
        return (self.__class__.__name__, f)

    def children(self) -> list[CalculatorBase[Any]]:
        return [self.f,*super().children()]
