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


TBinNode = TypeVar("TBinNode", bound="_BinNodeBase")

class _BinNodeBase(CalculatorBase[BinNDResult, BinNDResult]):
    kind: ClassVar[NodeKind] = BuiltinKinds.BINND


    def _own_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        raise NotImplementedError

    def _axes_for_concat(self) -> tuple[Bin1D, ...]:
        raise NotImplementedError

    @staticmethod
    def register_bin_algorithm(
        name: str,
        func: Any = None,
        *,
        overwrite: bool = False,
    ) -> Any:
        return register_bin_algorithm(name, func, overwrite=overwrite)

    register_algorithm = register_bin_algorithm

    @staticmethod
    def register_axis_property(
        name: str | AxisPropertyFunc,
        func: AxisPropertyFunc | None = None,
        *,
        overwrite: bool = False,
    ) -> AxisPropertyFunc | Callable[[AxisPropertyFunc], AxisPropertyFunc]:
        return BinAxis.register_property(
            cast("Any", name),
            cast("Any", func),
            overwrite=overwrite,
        )

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
        return BinNDResult.derived(
            cast("Any", fn),
            name=cast("Any", name),
            scope=scope,
            condition=condition,
            overwrite=overwrite,
        )

    derived = register_derived

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps = self._own_dependencies()
        for key in getattr(self, "active", ()):
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((*self._axes_for_concat(), *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((*self._axes_for_concat(), other))
        return NotImplemented

    def with_active(self: TBinNode, keys: Iterable[Any]) -> TBinNode:
        cl = cast("TBinNode", self._clone())
        cl.active = tuple(keys) # type: ignore[attr-defined]
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
        return super().with_transformation(transform, revert=revert)


@CalculatorBase.dataclass
class Bin1D(_BinNodeBase):
    prop: Param[Any]
    vmin: Param[float | None] = Param(default=None)
    vmax: Param[float | None] = Param(default=None)
    nbins: Param[int | None] = Param(default=None)

    mode: str = Param.static(default="linear", kw_only=False)
    edges: Param[Any] = Param(default=None, kw_only=True)
    lows: Param[Any] = Param(default=None, kw_only=True)
    highs: Param[Any] = Param(default=None, kw_only=True)
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
