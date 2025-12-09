"""
Transformation calculators for pynbody-extras.

This module defines :class:`~pynbodyext.transforms.base.TransformBase`, the
base class for all *transformation calculators*, and the helper
:func:`~pynbodyext.transforms.base.chain_transforms` for building readable
transformation chains.

A transformation calculator is a small object that, given a
:class:`pynbody.snapshot.SimSnap`, returns a
:class:`pynbody.transformation.Transformation` (or subclass), and applies
it to the appropriate target (either the full ancestor snapshot or the
current view/halo). Additionally, these transformations can be attached to other
calculators via :meth:`pynbodyext.calculate.CalculatorBase.with_transformation`.

See also
--------
* :mod:`pynbodyext.calculate` for the generic calculator abstraction.
* :mod:`pynbodyext.transforms.shift` and :mod:`pynbodyext.transforms.rotate`
  for concrete transformations (centering, rotations, etc.).

Basic usage
-----------

TransformBase subclasses implement:

.. code-block:: python

   from pynbody.snapshot import SimSnap
   from pynbody.transformation import Transformation
   from pynbodyext.transforms.base import TransformBase

   class MyTransform(TransformBase[Transformation]):
       def calculate(
           self,
           sim: SimSnap,
           previous: Transformation | None = None,
       ) -> Transformation:
           target = self.get_target(sim, previous)
           # construct a new Transformation acting on target
           trans = Transformation(target)   # placeholder example
           return trans

Transformations not only can be used directly:

.. code-block:: python
    sim = ...  # a SimSnap
    transform_calc = MyTransform()
    trans = transform_calc(sim)

But they can also be attached to other calculators via
:meth:`pynbodyext.calculate.CalculatorBase.with_transformation`:

.. code-block:: python

   from pynbodyext.calculate import CalculatorBase

   class MyCalculator(CalculatorBase[float]):
       def calculate(self, sim: SimSnap) -> float:
           # compute something on sim
           return float(len(sim))

   sim = ...  # a SimSnap
   my_calc = MyCalculator().with_transformation(MyTransform())
   value = my_calc(sim)  # MyTransform is applied to sim before MyCalculator.calculate is called.

Chaining transformations
------------------------

To build readable, linear transformation chains, use
:func:`chain_transforms` or :meth:`TransformBase.chain`:

.. code-block:: python

   from pynbodyext.transforms.shift import PosToCenter, VelToCenter
   from pynbodyext.transforms.rotate import AlignAngMomVec
   from pynbodyext.transforms.base import chain_transforms

   # center in position, center in velocity, then align face-on
   faceon = chain_transforms(
       PosToCenter("ssc"),
       VelToCenter("com"),
       AlignAngMomVec(),
   )

   # attach to a calculator
   from pynbodyext.properties.kappa import KappaRot
   from pynbodyext.filters.filt import Sphere
   from pynbodyext.filters import FamilyFilter

   re_faceon = (
       KappaRot()
       .with_filter(Sphere("30 kpc") & FamilyFilter("stars"))
       .with_transformation(faceon)
   )

   value = re_faceon(sim.halos()[0])

Each transform in the chain is applied in the order given; the final
return value is the last transform instance, which has previous steps
attached via :meth:`with_transformation`.

Implementation notes
--------------------

* :class:`TransformBase` itself inherits from
  :class:`~pynbodyext.calculate.CalculatorBase` and reuses its profiling,
  caching and chunking facilities.
* :meth:`TransformBase.get_target` helps decide whether a transform should
  operate on ``sim.ancestor`` (global) or the current view/halo.

To add a new transformation, subclass :class:`TransformBase` and implement
:meth:`calculate(sim, previous)`. See the existing implementations in
:mod:`pynbodyext.transforms.shift` and :mod:`pynbodyext.transforms.rotate`
for patterns.

"""


from typing import Any, Generic, TypeVar, cast

from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation

from pynbodyext.calculate import CalculatorBase
from pynbodyext.log import logger

TTrans = TypeVar("TTrans",bound=Transformation, covariant=True)

_LastT = TypeVar("_LastT", bound="TransformBase[Any]")

class TransformBase(CalculatorBase[TTrans], Generic[TTrans]):
    """ Simsnap -> TTrans """

    move_all: bool

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.move_all = True
        return instance

    def _apply_calculation(self, sim, trans_obj, sim_masked):
        target_obj = self.get_target(sim_masked, trans_obj)
        if trans_obj is not None and self._revert_transformation:
            logger.debug("[%s] merging pre-transformation: %s", self, self._transformation)
        return self.calculate(sim_masked, target_obj)

    def __repr__(self):
        return f"<Transform {self.__class__.__name__}>"

    def get_target(self, sim: SimSnap, previous: Transformation | None = None) -> Transformation | SimSnap:
        target: Transformation | SimSnap
        if previous is not None:
            target = previous
        elif self.move_all:
            target = sim.ancestor
        else:
            target = sim
        return target

    def with_transformation(self, transformation, revert = True):
        logger.info("You'd typically want to use chain_transforms() to build transformation chains. For example: chain_transforms(transform1, transform2, ...)")
        return super().with_transformation(transformation, revert)

    def calculate(self, sim: SimSnap, apply_to: SimSnap | Transformation | None = None) -> TTrans:
        raise NotImplementedError

    @classmethod
    def chain(self, *transforms: *tuple[*tuple["TransformBase[Any]", ...], _LastT]) -> _LastT:
        return chain_transforms(*transforms)

# ---------- typed TransformChain helper ----------

def chain_transforms(
    *transforms: *tuple[*tuple[TransformBase[Any], ...], _LastT]
) -> _LastT:
    """ Chain multiple TransformBase instances together.

    Parameters
    ----------
    *transforms : TransformBase
        The transformations to chain together.

    Returns
    -------
    TransformBase
        The final transformation in the chain.

    Notes
    -----
    Each transformation will be applied in the order they are provided.
    """
    if not transforms:
        raise ValueError("At least one transform must be provided.")
    prev: TransformBase[Any] | _LastT | None = None
    for transform in transforms:
        cur = transform
        if prev is not None:
            cur = cur.with_transformation(prev)
        prev = cur

    assert prev is not None
    return cast("_LastT", prev)
