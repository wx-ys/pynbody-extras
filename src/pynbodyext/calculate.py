"""

Core calculator abstraction for pynbody-extras.

This module defines :class:`~pynbodyext.calculate.CalculatorBase`, the base
class for all calculations, and :class:`~pynbodyext.calculate.CombinedCalculator`
for evaluating several calculators together.

A *calculator* is a small object with a :meth:`calculate` method that turns a
:class:`pynbody.snapshot.SimSnap` into some result. Optional **filters** and
**transformations** can be attached; profiling, caching and (optional)
chunked/Dask evaluation are handled by the base class.

See also
--------
* :mod:`pynbodyext.filters.base` for filter calculators.
* :mod:`pynbodyext.transforms.base` for transformation calculators.
* :mod:`pynbodyext.properties.base` for property calculators.
* :mod:`pynbodyext.profiles.base` for profile builders.

Basic usage
-----------

To define a new calculator, subclass :class:`CalculatorBase` and implement
:meth:`calculate`:

.. code-block:: python

   import numpy as np
   from pynbody.snapshot import SimSnap
   from pynbodyext.calculate import CalculatorBase

   class SumMass(CalculatorBase[float]):
       \"\"\"Return the mean particle mass.\"\"\"
       def calculate(self, sim: SimSnap) -> float:
           return float(np.sum(sim["mass"]))

   sim = ...  # a SimSnap
   calc = SumMass()
   value = calc(sim)   # calls SumMass.calculate(sim) internally

You never call :meth:`calculate` directly; always call the instance:
``calc(sim)``.


Filters and transformations
---------------------------

Any calculator can be used with a pre-filter and/or pre-transformation.

Filters
^^^^^^
Attach a filter with :meth:`with_filter`:

.. code-block:: python

   from pynbody import filt as pyn_filt
   from pynbodyext.filters import FamilyFilter

   mass_sum = (
       SumMass()
       .with_filter(pyn_filt.Sphere("30 kpc") & FamilyFilter("stars"))
   )

   value = mass_sum(sim)

See :mod:`pynbodyext.filters.base` for how to write calculator-style filters.

Transformations
^^^^^^^^^^^^^^^
Attach a transformation with :meth:`with_transformation`:

.. code-block:: python

   from pynbodyext.transforms.shift import PosToCenter, VelToCenter
   from pynbodyext.transforms.rotate import AlignAngMomVec
   from pynbodyext.transforms.base import chain_transforms
   from pynbodyext.filters.filt import Sphere
   from pynbodyext.properties.base import ParameterContain
   from pynbodyext.filters import FamilyFilter

   # half-mass radius of stars
   re = ParameterContain("r", 0.5, "mass").with_filter(FamilyFilter("stars"))

   # readable transform chain: center -> velocity center -> face-on
   faceon = chain_transforms(
       PosToCenter("ssc"),
       VelToCenter().with_filter(Sphere(0.5 * re) & FamilyFilter("stars")),
       AlignAngMomVec().with_filter(Sphere(2 * re) & FamilyFilter("stars")),
   )

   from pynbodyext.properties.kappa import KappaRot

   krot = (
       KappaRot()
       .with_filter(Sphere("30 kpc") & FamilyFilter("stars"))
       .with_transformation(faceon)
   )

   value = krot(sim)

See :mod:`pynbodyext.transforms.base` for more details on :class:`TransformBase`
and transformation chaining.


Combining calculators
---------------------

Use the ``&`` operator to evaluate several calculators together:

.. code-block:: python

   sum_mass = SumMass().with_filter(FamilyFilter("stars"))
   combined = krot & sum_mass   # CombinedCalculator[(float, float)]

   krot_val, sum_mass_val = combined(sim)


Profiling, caching and chunking
-------------------------------

Profiling
^^^^^^^^^
Enable simple per-node profiling via :meth:`enable_perf`:

.. code-block:: python

   calc = krot.enable_perf(time=True, memory=True)
   result = calc(sim)

This records time (and optionally memory) spent in the main phases
(transform, filter, calculate, revert, dask compute). See
:mod:`pynbodyext.util.perf` for details.

Caching
^^^^^^^
Within a single top-level call (a "run"), results can be reused based on a
structural signature:

.. code-block:: python

   calc = krot.enable_cache(True)
   v1 = calc(sim)    # computed
   v2 = calc(sim)    # reused from cache

For custom calculators, override :meth:`instance_signature` to describe how
the calculator was constructed. See :mod:`pynbodyext.util.tracecache` for
the caching internals.

Chunked / Dask-backed evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For large simulations, you can let :class:`CalculatorBase` wrap the input
snapshot in a :class:`~pynbodyext.chunk.chunksnap.ChunkSimSnap` so particle
arrays are loaded lazily as Dask graphs:

.. code-block:: python

   calc = krot.enable_chunk(True, chunk_size=4_000_000)
result = calc(sim)   # uses ChunkSimSnap internally

If a calculator returns a Dask-backed array, :class:`CalculatorBase` will
call ``.compute()`` on the top-level result so user code always receives
concrete arrays/scalars. When you implement your own heavy calculators,
you may either:

* work on in-memory arrays (simple but eager), or
* accept Dask arrays and push vectorized reductions down to Dask, only
  calling ``compute()`` at the end.

See :mod:`pynbodyext.chunk` for chunk/Dask utilities.


Inspecting flows and trees
--------------------------

For debugging, two helpers are available on any calculator:

* :meth:`format_flow` – semantic view (transform → filter → calculate)
* :meth:`format_tree` – structural tree of calculators

Example:

.. code-block:: python

   print(krot.format_flow())
   print()
   print(krot.format_tree())


Extending the system
--------------------

To add **new calculations**, subclass :class:`CalculatorBase` (or one of the
domain-specific bases, such as :class:`~pynbodyext.properties.base.PropertyBase`
or :class:`~pynbodyext.profiles.base.ProfileBuilderBase`):

.. code-block:: python

   class MyCalc(CalculatorBase[float]):
       def __init__(self, param: float):
           self.param = param

       def instance_signature(self) -> tuple:
           return (self.__class__.__name__, float(self.param))

       def calculate(self, sim: SimSnap) -> float:
           # use self._filter / self._transformation transparently
           return float(sim["mass"].mean() * self.param)

This new calculator can now:

* be filtered via :meth:`with_filter`
* be transformed via :meth:`with_transformation`
* be composed via ``&`` with other calculators
* participate in profiling, caching and (if enabled) chunked evaluation.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload, runtime_checkable

import numpy as np
from pynbody import units
from pynbody.array import SimArray

from pynbodyext.chunk import DASK_AVAILABLE, is_dask_array
from pynbodyext.log import logger
from pynbodyext.util.perf import PerfStats
from pynbodyext.util.tracecache import EvalCacheManager, TraceManager

from .util._type import (
    FilterLike,
    Self,
    SimCallable,
    SingleElementArray,
    TransformLike,
    TypeVarTuple,
    UnitLike,
    Unpack,
    get_signature_safe,
)

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap
    from pynbody.transformation import Transformation

    from pynbodyext.chunk.chunksnap import ChunkSimSnap


ReturnT = TypeVar("ReturnT",covariant=True)

@runtime_checkable
class ReturnTr(Protocol[ReturnT]):
    """Protocol for return type with .compute() method."""
    def compute(self, **kwargs: Any) -> ReturnT:
        ...

Ts = TypeVarTuple("Ts")
Us = TypeVarTuple("Us")
U = TypeVar("U")
T = TypeVar("T")
TCalculator = TypeVar("TCalculator", bound="CalculatorBase[Any]")

class CalculatorBase(SimCallable[ReturnT], Generic[ReturnT], ABC):
    """
    Abstract base class for performing calculations on particle data.

    This class provides a unified interface for calculations on pynbody
    simulation snapshots, supporting composable filters, transformations,
    performance profiling, and evaluation caching.

    Subclasses should implement the `calculate` method.

    Attributes
    ----------
    _filter : FilterLike or None
        Optional filter to apply before calculation.
    _transformation : TransformLike or None
        Optional transformation to apply before calculation.
    _revert_transformation : bool
        Whether to revert the transformation after calculation.
    _perf_stats : PerfStats
        Performance statistics collector.
    _enable_eval_cache : bool
        Whether to enable evaluation caching for this instance.

    Methods
    -------
    calculate(sim, *args, **kwargs)
        Abstract method to perform the calculation.
    with_filter(filt)
        Returns a new instance with the given filter applied.
    with_transformation(transformation, revert=True)
        Returns a new instance with the given transformation applied.
    enable_perf(time=True, memory=True)
        Enables/disables timing and memory profiling.
    enable_cache(enable=True)
        Enables/disables evaluation caching.
    format_flow(indent=0, short=True)
        Returns a semantic flow rendering of the calculation.
    format_tree(indent=0)
        Returns a pretty-printed tree of the calculator and its children.
    """

    _filter: FilterLike | None
    _transformation: TransformLike | None
    _revert_transformation: bool
    _perf_stats: PerfStats
    _enable_eval_cache: bool
    _enable_chunk: bool
    _chunk_size: int
    _cached_signature: tuple | None
    def __new__(cls: type[Self], *args: Any, **kwargs: Any) -> Self:
        """
        Create a new instance, initializing filter, transformation, and performance stats.
        """
        # initialize per-instance filter and transformation settings
        self = super().__new__(cls)
        self._filter = None
        self._transformation = None
        self._revert_transformation = True
        self._perf_stats = PerfStats(time=False, memory=False)
        self._enable_eval_cache = True
        self._enable_chunk = False
        self._chunk_size = 4_000_000

        self._cached_signature = None
        return self

    def instance_signature(self) -> tuple:
        """
        Returns a signature describing how this calculator was constructed.

        Default is identity-based. Subclasses should override to return a stable
        representation of constructor arguments if possible.
        """
        return ("id", id(self))

    def signature(self) -> tuple:
        """
        Returns the full structural signature for this calculator node.

        Includes instance signature, filter signature, transformation signature,
        and revert flag. Used for run-level caching and common subexpression elimination.
        """
        # if cached, return it
        if self._cached_signature:
            return self._cached_signature

        # instance/init signature
        try:
            init_sig = self.instance_signature()
        except Exception:
            init_sig = ("id", id(self))

        # filter signature
        flt = getattr(self, "_filter", None)
        if flt is None:
            filter_sig = None
        else:
            filter_sig = get_signature_safe(flt, fallback_to_id=True)

        # transformation signature
        trans = getattr(self, "_transformation", None)
        if trans is None:
            trans_sig = None
        else:
            trans_sig = get_signature_safe(trans, fallback_to_id=True)

        self._cached_signature = (
            self.__class__.__name__,
            ("init", init_sig),
            ("filter", filter_sig),
            ("trans", trans_sig),
            ("rev", bool(getattr(self, "_revert_transformation", True))),
        )

        return self._cached_signature

    # Calculation phase "child calculation" hook: default is none, subclasses may override
    def calculate_children(self) -> list[CalculatorBase[Any]]:
        """
        Returns a list of child calculations for this node. Used for flow/tree rendering.

        Default is an empty list. Subclasses may override to provide child calculators.
        """
        return []

    def format_flow(self, indent: int = 0, short: bool = True) -> str:
        """
        Returns a semantic flow rendering of the calculation node.

        Parameters
        ----------
        indent : int, optional
            Indentation level for pretty-printing, by default 0
        short : bool, optional
            Whether to use a short form omitting empty transform/filter, by default True

        Rendering
        ---------
        Node
        ├─ 1) transform: <transform-node-or-repr>
        ├─ 2) filter:    <filter-node-or-repr>
        └─ 3) calculate:
             └─ <child-calculation(s)...>

        Notes
        -----
        If transform/filter is CalculatorBase, recursively expand with format_flow.
        """
        pad = "  " * indent
        lines: list[str] = []
        title = f"{pad}{TraceManager._node_label(self)}"
        lines.append(title)

        # 1) transform
        if getattr(self, "_transformation", None) is not None:
            tobj = self._transformation
            if isinstance(tobj, CalculatorBase):
                lines.append(f"{pad}├─ 1) transform:")
                lines.append(tobj.format_flow(indent + 2, short=short))
            else:
                lines.append(f"{pad}├─ 1) transform: {repr(tobj)}")
        elif not short:
            lines.append(f"{pad}├─ 1) transform: <none>")

        # 2) filter
        if getattr(self, "_filter", None) is not None:
            fobj = self._filter
            if isinstance(fobj, CalculatorBase):
                lines.append(f"{pad}├─ 2) filter:")
                lines.append(fobj.format_flow(indent + 2))
            else:
                lines.append(f"{pad}├─ 2) filter: {repr(fobj)}")
        elif not short:
            lines.append(f"{pad}├─ 2) filter: <none>")

        # 3) calculate
        kids = self.calculate_children()
        if kids:
            lines.append(f"{pad}└─ 3) calculate:")
            last = len(kids) - 1
            for i, ch in enumerate(kids):
                # Child calculation also uses format_flow (they may have their own three steps)
                rendered = ch.format_flow(indent + 2)
                # Visual alignment: add a branch line before the first child block
                if i < last:
                    # Middle block
                    lines.append(f"{pad}   ├─")
                else:
                    # Last block
                    lines.append(f"{pad}   └─")
                lines.append(rendered)
        else:
            lines.append(f"{pad}└─ 3) calculate: {self.__class__.__name__}")

        return "\n".join(lines)

    # children hook for tree formatting and optional traversal
    def children(self) -> list[CalculatorBase[Any]]:
        """
        Returns a list of calculator-like children (filter and transformation if present).

        Used for tree formatting and traversal.
        """
        kids: list[CalculatorBase[Any]] = []
        if isinstance(getattr(self, "_filter", None), CalculatorBase):
            kids.append(self._filter)   # type: ignore[arg-type]
        if isinstance(getattr(self, "_transformation", None), CalculatorBase):
            kids.append(self._transformation)  # type: ignore[arg-type]
        return kids

    def format_tree(self, indent: int = 0) -> str:
        """
        Returns a pretty-printed string representation of the calculator tree.

        Parameters
        ----------
        indent : int, optional
            Indentation level for pretty-printing, by default 0

        Pretty-print the calculator tree using children() to discover subnodes.
        Falls back to showing filter/transformation if they don't support children().
        """
        # box-drawing ASCII
        def render(node: CalculatorBase[Any], prefix: str, is_last: bool) -> list[str]:
            branch = "└─" if is_last else "├─"
            name = TraceManager._node_label(node)
            line = f"{prefix}{branch} {name}"
            out = [line]
            new_prefix = f"{prefix}{'  ' if is_last else '│ '}"
            # Use children() to traverse deeper
            kids = []
            if hasattr(node, "children"):
                try:
                    kids = list(node.children())
                except Exception:
                    kids = []
            # For non-Calculator filter/transformation, still display a one-liner
            fobj = getattr(node, "_filter", None)
            tobj = getattr(node, "_transformation", None)
            if fobj is not None and not isinstance(fobj, CalculatorBase):
                out.append(f"{new_prefix}├─ filter: {repr(fobj)}")
            if tobj is not None and not isinstance(tobj, CalculatorBase):
                out.append(f"{new_prefix}├─ transformation: {repr(tobj)}")
            # Recurse into calculator-like children
            for i, ch in enumerate(kids):
                out.extend(render(ch, new_prefix, i == len(kids)-1))
            return out

        lines = render(self, "" if indent == 0 else "  " * indent, True)
        return "\n".join(lines)

    def __call__(self, sim: SimSnap) -> ReturnT:
        """
        Unified top-level call for calculator evaluation.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot to evaluate the calculator on.

        Examples
        --------
        >>> # If enable performance profiling
        >>> calculator.enable_perf()(sim)
        >>>
        >>> # If enable evaluation caching
        >>> calculator.enable_cache()(sim)
        """
        logger.debug("")
        use_sim: SimSnap | ChunkSimSnap = sim

        created_chunk = False
        # Wrap with ChunkSnap if chunking enabled and not already chunked
        if self._enable_chunk and DASK_AVAILABLE:
            try:
                from .chunk.chunksnap import ChunkSimSnap
                if not isinstance(sim, ChunkSimSnap):
                    use_sim = ChunkSimSnap(sim, self._chunk_size)
                    created_chunk = True
                    logger.debug("[%s] chunk mode enabled with chunk_size=%s", self, self._chunk_size)
                    logger.debug("%d chunks created with %d particles", use_sim.chunks.num_chunks, use_sim.chunks.num_particles)
            except Exception as e:
                logger.warning("Chunk mode requested but failed to enable: %s", e)
        elif self._enable_chunk and not DASK_AVAILABLE:
            logger.info("Chunk mode requested but chunk/dask not available; running without chunking")

        token = id(sim)

        # detect existing ctx for this sim (fast path)
        prev_ctx = EvalCacheManager._CTX.get()
        prev_tok = EvalCacheManager._SIM_TOKEN.get()
        is_top = not (prev_ctx is not None and prev_tok == token)

        title = repr(self) + " : " + repr(sim)

        # Convenience local refs
        eval_mgr = EvalCacheManager
        trace_mgr = TraceManager

        # Helper to handle cached lookup/store when cache is enabled
        def _run_with_cache() -> ReturnT:
            # compute node signature once (fallback to struct_signature on error)
            node_sig = self.signature()
            # build key only when needed
            key = (token, ("calc", ("sig", node_sig)))
            cached = eval_mgr.get(key)
            if cached is not None:
                trace_mgr.trace_cache_event(self, "hit", {"sig": node_sig})
                return cached
            res = self._evaluate(use_sim)
            eval_mgr.set(key, res)
            return res

        # Top-level: create trace run and evaluation context (decide enable_cache once)
        if is_top:
            with trace_mgr.trace_run(self):
                # pass instance preference once to use(); returns to previous ctx on exit
                with eval_mgr.use(sim, enable_cache=bool(getattr(self, "_enable_eval_cache", True))):
                    if eval_mgr.cache_enabled():
                        result =  _run_with_cache()
                    else:
                        # cache disabled for this run -> direct call
                        result = self._evaluate(use_sim)
        else:
            # nested call: reuse existing ctx and its cache_enabled decision
            if eval_mgr.cache_enabled():
                return _run_with_cache()
            result = self._evaluate(use_sim)
        if created_chunk:
            from .chunk.chunksnap import ChunkSimSnap
            if isinstance(use_sim, ChunkSimSnap):
                use_sim.clear_cache()
                del use_sim

        self._perf_stats.report(logger, title=title)
        return result

    def _evaluate(self, sim: SimSnap) -> ReturnT:
        """
        Perform the full evaluation pipeline under :class:`PerfStats`.

        The pipeline is:
        transform → filter → calculate → optional dask compute → revert.

        Note
        ----
        :class:`TraceManager` and :class:`EvalCacheManager` should already
        be active when this is called (handled by :meth:`__call__`).
        """
        # At this point TraceManager.run and EvalCacheManager.use(sim) should already be active.
        with self._perf_stats as stats:
            trans_obj = self._run_transform(sim, stats)
            sim_masked = self._run_filter(sim, stats)
            result = self._run_calculation(sim, stats, trans_obj, sim_masked)
            result = self._run_dask_compute(result, stats)
            self._run_revert(trans_obj, stats)

        return result

    def _run_transform(self, sim: SimSnap, stats: PerfStats) -> Transformation | None:
        """
        Runs the transformation on the simulation snapshot if present.

        Returns the transformation object, or None if no transformation is set.
        """
        with stats.step("transform"):
            if self._transformation is not None:
                with TraceManager.trace_phase(self, "transform"):
                    try:
                        logger.debug("[%s] transformation: %s", self, self._transformation)
                        trans_obj = self._apply_transform(sim)
                    except Exception as e:
                        logger.error("Error applying pre-transformation: %s, when calculating %s", e, self)
                        raise RuntimeError(f"Error applying pre-transformation {self._transformation}") from e
                    return trans_obj
        return None


    def _apply_transform(self, sim: SimSnap) -> Transformation | None:
        """
        Applies the transformation to the simulation snapshot if present.

        Returns the transformation object.
        """
        return self._transformation(sim) if self._transformation is not None else None

    def _run_filter(self, sim: SimSnap, stats: PerfStats) -> SimSnap:
        """
        Run the filter phase on the simulation snapshot if present.

        Parameters
        ----------
        sim : SimSnap
            The (possibly transformed) simulation snapshot to be filtered.
        stats : PerfStats
            Profiler for the "filter" phase.

        Returns
        -------
        SimSnap
            The filtered snapshot if a filter is set, otherwise the original
            ``sim`` unchanged.
        """
        with stats.step("filter"):
            if self._filter is not None:
                with TraceManager.trace_phase(self, "filter"):
                    try:
                        logger.debug("[%s] filter: %s", self, self._filter)
                        sim_masked = self._apply_filter(sim)
                    except Exception as e:
                        logger.error("Error applying pre-filter: %s, when calculating %s", e, self)
                        raise RuntimeError(f"Error applying pre-filter {self._filter}") from e
                return sim_masked
        return sim

    def _apply_filter(self, sim: SimSnap) -> SimSnap:
        """
        Apply the configured filter to the simulation snapshot if present.

        Parameters
        ----------
        sim : SimSnap
            The (possibly transformed) snapshot to filter.

        Returns
        -------
        SimSnap
            The filtered snapshot if a filter is set, otherwise ``sim``.
        """
        return sim if self._filter is None else sim[self._filter]

    def _run_calculation(self, sim: SimSnap, stats: PerfStats, trans_obj: Transformation | None, sim_masked: SimSnap) -> ReturnT:
        """
        Execute the main calculation phase.

        Parameters
        ----------
        sim : SimSnap
            Original (possibly transformed) snapshot.
        stats : PerfStats
            Profiler for the "calculate" phase.
        trans_obj : Transformation or None
            Transformation object returned by the transform phase, if any.
        sim_masked : SimSnap
            Snapshot after applying any active filter.

        Returns
        -------
        ReturnT
            Result of the calculation.
        """
        try:
            with TraceManager.trace_phase(self, "calculate"):
                with stats.step("calculate"):
                    logger.debug("[%s] performing calculation ...", self)
                    result = self._apply_calculation(sim, trans_obj, sim_masked)
        except Exception as e:
            if trans_obj is not None:
                logger.debug("[%s] reverting transformation due to error: %s", self, self._transformation)
                trans_obj.revert()
            logger.error("Error during calculation: %s, when calculating %s", e, self)
            raise RuntimeError(f"Error during calculation in {self}") from e
        return result

    def _apply_calculation(self, sim: SimSnap, trans_obj: Transformation | None, sim_masked: SimSnap) -> ReturnT:
        """
        Hook for subclasses to customize the actual calculation.

        Parameters
        ----------
        sim : SimSnap
            Original (possibly transformed) snapshot.
        trans_obj : Transformation or None
            Transformation object returned by the transform phase, if any.
        sim_masked : SimSnap
            Snapshot after applying any active filter.

        Notes
        -----
        Default implementation ignores ``sim`` and ``trans_obj`` and just
        calls ``calculate(sim_masked)``. Subclasses may override this method
        if they need access to the unfiltered snapshot or the transformation.
        """
        return self.calculate(sim_masked)


    def _run_dask_compute(self, result: ReturnT | ReturnTr[ReturnT], stats: PerfStats) -> ReturnT:
        """
        Optionally compute a Dask-backed result.

        If Dask is available and ``result`` looks like a Dask collection
        (has a ``.compute`` method and is recognised by ``is_dask_array``),
        run ``.compute()`` under a profiling step. Otherwise return
        ``result`` unchanged.

        Parameters
        ----------
        result : ReturnT or ReturnTr[ReturnT]
            The calculation result, possibly Dask-backed.
        stats : PerfStats
            Profiler for the "dask compute" phase.

        Returns
        -------
        ReturnT
            The concrete (computed) result.
        """
        if DASK_AVAILABLE and is_dask_array(result) and hasattr(result, "compute"):
            with stats.step("dask compute"):
                from dask.diagnostics.progress import ProgressBar

                from .chunk.chunksnap import ChunkSimSnap  # local import to avoid cycles
                with TraceManager.trace_phase(self, "dask compute"):
                    logger.debug("[%s] computing dask result ...", self)
                    with ProgressBar(minimum=1):
                        result = self._apply_compute(result)
                if hasattr(result, "sim") and isinstance(result, SimArray):
                    if isinstance(result.sim, ChunkSimSnap):
                        result.sim = result.sim.base
        logger.debug("Calculator %s, return type %s", self, result)
        return result   # type: ignore[return-value]

    def _apply_compute(self, result: ReturnTr[ReturnT]) -> ReturnT:
        """
        Compute a Dask-backed result.

        This is the low-level hook used by :meth:`_run_dask_compute`
        to call ``result.compute(...)``.

        Parameters
        ----------
        result : ReturnTr[ReturnT]
            An object exposing a ``compute`` method (e.g. Dask array).

        Returns
        -------
        ReturnT
            The computed concrete result.
        """
        return result.compute(scheduler="single-threaded")

    def _run_revert(self, trans_obj: Transformation | None, stats: PerfStats) -> None:
        """
        Run the revert phase if a transformation was applied.

        Parameters
        ----------
        trans_obj : Transformation or None
            The transformation instance returned by :meth:`_run_transform`,
            or ``None`` if no transformation was used.
        stats : PerfStats
            Profiler for the "revert" phase.
        """
        with stats.step("revert"):
            if trans_obj is not None and self._revert_transformation:
                with TraceManager.trace_phase(self, "revert"):
                    logger.debug("[%s] reverting transformation: %s", self, self._transformation)
                    self._apply_revert(trans_obj)

    def _apply_revert(self, trans_obj: Transformation) -> None:
        """
        Apply the revert operation on the transformation object.

        Parameters
        ----------
        trans_obj : Transformation
            The transformation object whose :meth:`revert` method will be called.
        """
        trans_obj.revert()

    @abstractmethod
    def calculate(self, sim: SimSnap, *args: Any, **kwargs: Any) -> ReturnT:
        """
        Abstract method to perform the calculation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate"
        )

    def enable_perf(self: Self, time: bool = True, memory: bool = True) -> Self:
        """
        Enable or disable timing and memory profiling for this calculator.

        Parameters
        ----------
        time : bool
            Enable timing profiling.
        memory : bool
            Enable memory profiling.

        Returns
        -------
        Self
            The calculator instance.
        """
        self._perf_stats.time_enabled = time
        self._perf_stats.memory_enabled = memory
        return self

    def enable_cache(self: Self, enable: bool = True) -> Self:
        """
        Enable or disable evaluation caching for this calculator instance.

        Parameters
        ----------
        enable : bool
            Whether to enable caching.

        Returns
        -------
        Self
            The calculator instance.
        """
        self._enable_eval_cache = bool(enable)
        return self

    def enable_chunk(self: Self, enable: bool = True, chunk_size: int = 4_000_000) -> Self:
        """
        Enable or disable chunked evaluation for this calculator instance.

        Parameters
        ----------
        enable : bool
            Whether to enable chunked evaluation.
        chunk_size : int
            The size of each chunk.

        Returns
        -------
        Self
            The calculator instance.
        """
        self._enable_chunk = bool(enable)
        self._chunk_size = int(chunk_size)
        return self

    def with_filter(self: Self, filt: FilterLike) -> Self:
        """
        Return self with the given filter applied.

        Parameters
        ----------
        filt : FilterLike
            The filter to apply.

        Returns
        -------
        Self
            The calculator instance.
        """
        logger.debug("Applying filter: %s to %s", filt, self)
        self._filter = filt
        self._cached_signature = None
        return self

    def with_transformation(self: Self, transformation: TransformLike, revert: bool = True) -> Self:
        """
        Return self with the given transformation applied.

        Parameters
        ----------
        transformation : TransformLike
            The transformation to apply.
        revert : bool
            Whether to revert the transformation after calculation.

        Returns
        -------
        Self
            The calculator instance.
        """
        logger.debug("Applying transformation: %s with revert=%s to %s", transformation, revert, self)
        self._transformation = transformation
        self._revert_transformation = revert
        self._cached_signature = None
        return self

    def __getitem__(self, key: FilterLike) -> Self:
        """
        Shorthand for applying a filter using indexing syntax.

        Returns
        -------
        Self
            The calculator instance with the filter applied.
        """
        return self.with_filter(key)

    def _in_sim_units(
        self,
        value:  UnitLike | float | int | SingleElementArray,
        sim_parameter: str,
        sim: SimSnap,
    ) -> float:
        """
        Convert a value into the simulation's native units for a given array.

        Converts `value` (str, Unit, number, or single-element array) to a float
        in the units of `sim[sim_parameter]`.

        Parameters
        ----------
        value : str or pynbody.units.UnitBase or float or int or single-element SimArray or NDArray
            The value to be converted.
            - str: parsed by `pynbody.units.Unit`, e.g., "10 kpc", "200 km s**-1".
            - UnitBase: converted using `.in_units(target_units, **context)`.
            - number: treated as already in the target native units.
        sim_parameter : str
            The name of the simulation array (e.g., 'r', 'rxy', 'mass') whose units
            define the conversion target.
        sim : pynbody.snapshot.SimSnap
            The simulation snapshot.

        Returns
        -------
        float
            The numeric value in the native units of `sim[sim_parameter]`.
        """
        if isinstance(value, str):
            value = units.Unit(value)
        if isinstance(value, units.UnitBase):
            value = float(value.in_units(sim[sim_parameter].units,
                                          **sim[sim_parameter].conversion_context()))
        if isinstance(value, np.ndarray):
            if value.ndim == 0 or value.size == 1:
                if isinstance(value, SimArray):
                    value = value.in_units(sim[sim_parameter].units,
                                       **sim[sim_parameter].conversion_context()).item()
                else:
                    value = value.item()
            else:
                raise TypeError(
                    "value must be a single-element array if it is an ndarray, got array with shape "
                    f"{value.shape}"
                )
        if isinstance(value, (int, float)):
            return float(value)

        raise TypeError(
            "value must be str, pynbody.units.UnitBase (or unit-like), or a number, or a single-element SimArray or NDArray, got "
            f"{type(value)}"
        )

    # ------- combinators --------

    @overload
    def __and__(                # type: ignore
        self: CombinedCalculator[Unpack[Ts]],
        other: CombinedCalculator[Unpack[Us]],
    ) -> CombinedCalculator[Unpack[Ts], Unpack[Us]]: ...  # type: ignore

    @overload
    def __and__(                # type: ignore
        self: CombinedCalculator[Unpack[Ts]],
        other: CalculatorBase[U],
    ) -> CombinedCalculator[Unpack[Ts], U]: ...

    @overload
    def __and__(
        self: CalculatorBase[T],
        other: CombinedCalculator[Unpack[Us]],
    ) -> CombinedCalculator[T, Unpack[Us]]: ...
    @overload
    def __and__(self: CalculatorBase[T], other: CalculatorBase[U]) -> CombinedCalculator[T, U]: ...

    def __and__(self: TCalculator, other: object) -> CombinedCalculator[Unpack[tuple[Any, ...]]]:
        """
        Combine this calculator with another using the & operator.

        Returns a CombinedCalculator containing both calculators.

        Parameters
        ----------
        other : CalculatorBase or CombinedCalculator
            The other calculator to combine.

        Returns
        -------
        CombinedCalculator
            A new CombinedCalculator instance.
        """
        if not isinstance(other, CalculatorBase):
            raise TypeError(f"Unsupported operand type(s) for &: '{type(self).__name__}' and '{type(other).__name__}'")

        left_props: tuple[CalculatorBase[Any], ...]
        if isinstance(self, CombinedCalculator):
            left_props = self.props
        else:
            left_props = (self,)

        right_props: tuple[CalculatorBase[Any], ...]
        if isinstance(other, CombinedCalculator):
            right_props = other.props
        else:
            right_props = (other,)
        return CombinedCalculator(*left_props, *right_props)

    def __repr__(self):
        if hasattr(self,"name"):
            return f"<Calculator {self.name}()>"
        return f"<Calculator {self.__class__.__name__}()>"


class CombinedCalculator(CalculatorBase[tuple[Unpack[Ts]]], Generic[Unpack[Ts]]):
    """
    Combines multiple CalculatorBase instances for joint evaluation.

    Evaluates all contained calculators and returns a tuple of their results.

    Attributes
    ----------
    props : tuple of CalculatorBase
        The calculators to combine.

    Methods
    -------
    calculate(sim)
        Evaluates all calculators and returns a tuple of results.
    children()
        Returns the list of child calculators.
    calculate_children()
        Returns the list of calculators for semantic flow rendering.
    """

    props: tuple[CalculatorBase[Any], ...]
    def __init__(self, *props: CalculatorBase[Unpack[Ts]]) -> None: # type: ignore
        """
        Initialize with the given calculators.

        Parameters
        ----------
        *props : CalculatorBase
            The calculators to combine.
        """
        self.props = props


    def instance_signature(self):
        """
        Returns a signature describing how this CombinedCalculator was constructed.
        """
        return (self.__class__.__name__, tuple(p.instance_signature() for p in self.props))

    def calculate(self, sim: SimSnap) -> tuple[Unpack[Ts]]:
        """
        Evaluates all contained calculators and returns a tuple of their results.

        Parameters
        ----------
        sim : SimSnap
            The simulation snapshot.

        Returns
        -------
        tuple
            Tuple of results from each calculator.
        """
        return tuple(p(sim) for p in self.props)

    def children(self) -> list[CalculatorBase[Any]]:
        """
        Returns the list of child calculators, including parent filter/transformation.
        """
        # Expand own props and also keep parent filter/transformation expansion
        return [*self.props, *super().children()]

    def calculate_children(self) -> list[CalculatorBase[Any]]:
        """
        Returns the list of calculators for semantic flow rendering.
        """
        # Expand all child calculations under "3) calculate"
        return list(self.props)


    def __repr__(self) -> str:
        return f"CombinedCalculator({', '.join(repr(p) for p in self.props)})"
