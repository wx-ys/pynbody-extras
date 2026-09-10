"""Calculator nodes for 1-D / N-D binning.

This module defines :class:`Bin1D` (a single binning axis) and :class:`BinND`
(a Cartesian product of :class:`Bin1D` axes).  They are :class:`CalculatorBase`
subclasses, so they compose with the rest of the calculator framework via
``@`` (join axes), ``with_active``, ``.run``/``.__call__``, and the scoped
transform/filter helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.params.fields import Param, declarative_dependencies
from pynbodyext.core.calculate.result.enums import BuiltinKinds, NodeKind

from .axes import AxisPropertyFunc, BinAxis, register_bin_algorithm
from .executor import BinExecutor
from .result import BinNDResult, SubBinNDResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput
    from pynbodyext.util._type import UnitLike


TBinNode = TypeVar("TBinNode", bound="_BinNodeBase")


class _BinNodeBase(CalculatorBase[BinNDResult, BinNDResult]):
    kind: ClassVar[NodeKind] = BuiltinKinds.BINND

    def _own_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        raise NotImplementedError

    def _axes_for_concat(self) -> tuple[Bin1D, ...]:
        raise NotImplementedError

    @staticmethod
    def register_bin_algorithm(name: str, func: Any = None, *, overwrite: bool = False) -> Any:
        """Register a custom bin-edge algorithm for this node family.

        Resolves to :func:`pynbodyext.core.calculate.bins.axes.register_bin_algorithm`.
        The algorithm is ``f(values, nbins, vmin, vmax) -> ndarray`` returning
        ``nbins + 1`` monotonic edges.

        Parameters
        ----------
        name : str
            Key used by ``Bin1D(mode=...)``.
        func : callable, optional
            The algorithm; omit to use as a decorator.
        overwrite : bool, default: False
            Whether to replace an existing algorithm.

        Returns
        -------
        callable
            The algorithm (direct call) or a decorator.

        Examples
        --------
        >>> import numpy as np
        >>> @Bin1D.register_bin_algorithm("centroid_edges", overwrite=True)
        ... def centroid_edges(values, nbins, vmin, vmax):
        ...     return np.linspace(vmin, vmax, nbins + 1)
        >>> bins = Bin1D("r", vmin=0, vmax=6, nbins=3, mode="centroid_edges")(sim)
        """
        return register_bin_algorithm(name, func, overwrite=overwrite)

    register_algorithm = register_bin_algorithm

    @staticmethod
    def register_axis_property(
        name: str | AxisPropertyFunc, func: AxisPropertyFunc | None = None, *, overwrite: bool = False
    ) -> AxisPropertyFunc | Callable[[AxisPropertyFunc], AxisPropertyFunc]:
        """Register a dynamic axis property accessible as ``bins.axis.<name>``.

        The property function receives a :class:`BinAxis` and returns an array
        shaped like the axis (``len == nbins``).  Resolves to
        :meth:`BinAxis.register_property`.

        Parameters
        ----------
        name : str
            Property name (attribute/``dir`` key on the axis accessor).
        func : callable, optional
            ``f(axis) -> array``; omit to use as a decorator.
        overwrite : bool, default: False
            Whether to replace an existing property.

        Returns
        -------
        callable
            The property function or a decorator.

        Examples
        --------
        >>> @Bin1D.register_axis_property("midpoint", overwrite=True)
        ... def midpoint(axis):
        ...     return axis.mins + 0.5 * axis.widths
        >>> bins.axis.r.midpoint.tolist()
        [1.0, 3.0, 5.0]
        """
        return BinAxis.register_property(cast("Any", name), cast("Any", func), overwrite=overwrite)

    axis_property = register_axis_property

    @staticmethod
    def register_derived(
        fn: Callable[[Any], Any] | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: Callable[[Any], bool] | None = None,
        overwrite: bool = False,
    ) -> Callable[[Any], Any] | Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Register a derived per-bin property on this node's results.

        Equivalent to ``BinNDResult.derived``.  The decorated function receives
        the :class:`BinNDResult` (or its data model) and returns a per-bin array.
        A ``condition`` (e.g. ``has_axis({"x"})``) gates availability.

        Parameters
        ----------
        fn : callable or str, optional
            The function (or name when used as a factory).
        name : str, optional
            Registration name (defaults to ``fn.__name__``).
        scope : {"derived", "geometry", "particles"}, default: "derived"
            Query scope.
        condition : callable, optional
            ``lambda result -> bool`` gating availability.
        overwrite : bool, default: False
            Whether to replace an existing property.

        Examples
        --------
        >>> import numpy as np
        >>> @Bin1D.derived("x_span", condition=has_axis({"x"}), overwrite=True)
        ... def x_span(result):
        ...     axis = result.find_axis({"x"})
        ...     return np.full(result.nbins, float(axis.maxs[-1] - axis.mins[0]))
        >>> "x_span" in Bin1D("x", vmin=0, vmax=6, nbins=3)(sim).keys()
        True
        """
        return BinNDResult.derived(
            cast("Any", fn), name=cast("Any", name), scope=scope, condition=condition, overwrite=overwrite
        )

    derived = register_derived

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps = self._own_dependencies()
        for key in getattr(self, "active", ()):
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        """Compose axes into an :class:`BinND` product (``@`` operator).

        Examples
        --------
        >>> bins = Bin1D("x", vmin=0, vmax=6, nbins=3) @ Bin1D("y", vmin=0, vmax=3, nbins=3)
        >>> bins(sim).shape_bins
        (3, 3)
        """
        if isinstance(other, BinND):
            return BinND((*self._axes_for_concat(), *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((*self._axes_for_concat(), other))
        return NotImplemented

    def with_active(self: TBinNode, keys: Iterable[Any]) -> TBinNode:
        """Mark queries that should be resolved (and cached) during the run.

        Parameters
        ----------
        keys : iterable of str or callable
            Query keys (e.g. ``"mass.sum"``) or callables to pre-resolve.

        Returns
        -------
        Bin1D or BinND
            A copy of this node with ``active`` queries set.

        Examples
        --------
        >>> calc = Bin1D("x", vmin=0, vmax=6, nbins=3).with_active(["count", "mass.sum"])
        >>> result = calc(sim)
        >>> result.cache_report()["queries"] >= 1
        True
        """
        cl = self._clone()
        cl.active = tuple(keys)  # type: ignore[attr-defined]
        return cl

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value

    def with_transformation(self, transform, revert=True):
        if revert:
            self.warning(
                f"{type(self).__name__} applies transform {transform} with revert=True; "
                "the subsequent binned result will be derived from the original untransformed data. "
                "Do you really want this? If so, consider setting revert=False to avoid such behavior."
            )
        return super().transform(transform, revert=revert)


@CalculatorBase.dataclass
class Bin1D(_BinNodeBase):
    """A single binning axis.

    Parameters
    ----------
    prop : str, callable, or CalculatorBase
        Field name (or callable / calculator) whose values are binned.
    vmin, vmax : float or str, optional
        Lower/upper bound, optionally a unit string such as ``"30 kpc"``.
    nbins : int, optional
        Number of bins.  Ignored when ``edges`` or ``lows``/``highs`` are given.
    mode : {"linear", "lin", "log", "equaln", "quantile"}, default: "linear"
        Edge-generation algorithm for the ``vmin``/``vmax``/``nbins`` path.
    edges : array-like, optional
        Explicit bin edges (overrides ``vmin``/``vmax``/``nbins``).
    lows, highs : array-like, optional
        Explicit lower/upper bounds per bin (overrides ``vmin``/``vmax``/``nbins``).
    alias : str, optional
        Short alias used by ``bins.axis.<alias>`` and dictionaries.
    include_rightmost : bool, default: True
        Whether the rightmost edge is included in the last bin.
    out_of_range : {"drop"}, default: "drop"
        How out-of-range particles are handled.
    units : str or UnitBase, optional
        Units attached to the axis values.

    Examples
    --------
    >>> import pynbody
    >>> sim = pynbody.new(dm=6)
    >>> sim["r"] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    >>> sim["mass"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> bins = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)
    >>> bins["count"].tolist()
    [2, 2, 2]
    >>> bins["mass.sum"].tolist()
    [3.0, 7.0, 11.0]

    Bins can also be constructed from explicit edges or unit ranges::

    >>> bins = Bin1D("r", edges=[0, 2, 4, 6])(sim)
    >>> bins = Bin1D("r", vmin=0, vmax="30 kpc", nbins=10)(sim)
    """

    prop: Param[Any]
    vmin: Param[float | UnitLike | None] = Param(default=None)
    vmax: Param[float | UnitLike | None] = Param(default=None)
    nbins: Param[int | None] = Param(default=None)

    mode: str = Param.static(default="linear", kw_only=False)
    edges: Param[Any | None] = Param(default=None, kw_only=True)
    lows: Param[Any | None] = Param(default=None, kw_only=True)
    highs: Param[Any | None] = Param(default=None, kw_only=True)
    alias: str | None = Param.static(default=None, kw_only=True)
    include_rightmost: bool = Param.static(default=True, kw_only=True)
    out_of_range: str = Param.static(default="drop", kw_only=True)
    units: Any | None = Param.static(default=None, kw_only=True)

    def __post_init__(self) -> None:
        self.active = ()

    def _own_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return declarative_dependencies(self)

    def _axes_for_concat(self) -> tuple[Bin1D, ...]:
        return (self,)

    def _as_binnd(self) -> BinND:
        wrapper = BinND((self,))
        wrapper.active = self.active
        return wrapper

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return self._as_binnd().execute(ctx, input)


@CalculatorBase.dataclass
class BinND(_BinNodeBase):
    """An N-dimensional binned result from a product of :class:`Bin1D` axes.

    Usually created with the ``@`` operator rather than directly::

        axes = Bin1D("x", vmin=0, vmax=6, nbins=3) @ Bin1D("y", vmin=0, vmax=3, nbins=3)

    Parameters
    ----------
    axes_specs : tuple of Bin1D
        The per-axis specifications, in C order (last axis varies fastest).

    Examples
    --------
    >>> import pynbody
    >>> sim = pynbody.new(dm=6)
    >>> sim["x"] = [0, 1, 2, 3, 4, 5]
    >>> sim["y"] = [0, 0, 1, 1, 2, 2]
    >>> bins = Bin1D("x", vmin=0, vmax=6, nbins=3) @ Bin1D("y", vmin=0, vmax=3, nbins=3)
    >>> result = bins(sim)
    >>> result.shape_bins
    (3, 3)
    """

    axes_specs: tuple[Bin1D, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes_specs, tuple):
            self.axes_specs = tuple(self.axes_specs)  # type: ignore[unreachable]
        if not self.axes_specs:
            raise ValueError("BinND requires at least one axis.")
        inherited: list[Any] = []
        for axis in self.axes_specs:
            inherited.extend(axis.active)
        self.active = tuple(inherited)

    def _own_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        return list(self.axes_specs)

    def _axes_for_concat(self) -> tuple[Bin1D, ...]:
        return self.axes_specs

    def _executor(self) -> BinExecutor:
        return BinExecutor(self)

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return self._executor().execute(ctx, input)

    def _spawn_result(self, parent: BinNDResult, subset: Any) -> SubBinNDResult:
        return self._executor().spawn_result(parent, subset)
