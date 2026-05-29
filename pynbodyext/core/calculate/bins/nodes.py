from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from pynbodyext.core.calculate.nodes.base import CalculatorBase
from pynbodyext.core.calculate.params.fields import Param, declarative_dependencies
from pynbodyext.core.calculate.result.enums import BuiltinKinds, NodeKind

from .axes import AxisPropertyFunc, BinAxis, register_bin_algorithm
from .executor import BinExecutor
from .result import BinNDResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pynbodyext.core.calculate.runtime.context import ExecutionContext
    from pynbodyext.core.calculate.runtime.input import NodeInput


@CalculatorBase.dataclass
class Bin1D(CalculatorBase[BinNDResult, BinNDResult]):
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

    kind: ClassVar[NodeKind] = BuiltinKinds.BINND

    def __post_init__(self) -> None:
        self.active: tuple[Any, ...] = ()

    def _as_binnd(self) -> BinND:
        wrapper = BinND((self,))
        wrapper.active = self.active
        return wrapper

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps = declarative_dependencies(self)
        for key in self.active:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    @staticmethod
    def register_bin_algorithm(
        name: str,
        func: Any = None,
        *,
        overwrite: bool = False,
    ) -> Any:
        return register_bin_algorithm(name, func, overwrite=overwrite)

    @staticmethod
    def register_axis_property(
        name: str | AxisPropertyFunc,
        func: AxisPropertyFunc | None = None,
        *,
        overwrite: bool = False,
    ) -> AxisPropertyFunc | Callable[[AxisPropertyFunc], AxisPropertyFunc]:
        return BinAxis.register_property(cast("Any", name), cast("Any", func), overwrite=overwrite)

    @staticmethod
    def register_derived(
        fn: Callable[[Any], Any] | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: Callable[[Any], bool] | None = None,
        overwrite: bool = False,
    ) -> Callable[[Any], Any] | Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        return BinNDResult.derived(cast("Any", fn), name=cast("Any", name), scope=scope, condition=condition, overwrite=overwrite)

    derived = register_derived

    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((self, *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((self, other))
        return NotImplemented

    def with_active(self, keys: Iterable[Any]) -> Bin1D:
        cl = cast("Bin1D", self._clone())
        cl.active = tuple(keys)
        return cl

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return self._as_binnd().execute(ctx, input)

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value

    def with_transformation(self, transform, revert = True):
        if revert:
            self.warning(f"Bin1D applies transform {transform} with revert=True; "
                         "the subsequent binned result will be derived from the original untransformed data. "
                         "Do you really want this? If so, consider setting revert=False to avoid such behavior.")
        return super().with_transformation(transform, revert=revert)


@CalculatorBase.dataclass
class BinND(CalculatorBase[BinNDResult, BinNDResult]):
    axes_specs: tuple[Bin1D, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes_specs, tuple):
            self.axes_specs = tuple(self.axes_specs)    # type: ignore[unreachable]
        if not self.axes_specs:
            raise ValueError("BinND requires at least one axis.")
        # Inherit active keys from constituent Bin1D specs
        inherited: list[Any] = []
        for axis in self.axes_specs:
            inherited.extend(axis.active)
        self.active = tuple(inherited)

    def _executor(self) -> BinExecutor:
        return BinExecutor(self)


    def __matmul__(self, other: Bin1D | BinND) -> BinND:
        if isinstance(other, BinND):
            return BinND((*self.axes_specs, *other.axes_specs))
        if isinstance(other, Bin1D):
            return BinND((*self.axes_specs, other))
        return NotImplemented

    def with_active(self, keys: Iterable[Any]) -> BinND:
        cl= cast("BinND", self._clone())
        cl.active = tuple(keys)
        return cl

    def with_transformation(self, transform, revert = True):
        if revert:
            self.warning(f"BinND applies transform {transform} with revert=True; "
                         "the subsequent binned result will be derived from the original untransformed data. "
                         "Do you really want this? If so, consider setting revert=False to avoid such behavior.")
        return super().with_transformation(transform, revert=revert)

    def declared_dependencies(self) -> list[CalculatorBase[Any, Any]]:
        deps: list[CalculatorBase[Any, Any]] = []
        for axis in self.axes_specs:
            deps.append(axis)
        for key in self.active:
            if isinstance(key, CalculatorBase):
                deps.append(key)
        return deps

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> BinNDResult:
        return self._executor().execute(ctx, input)

    def public_value(self, value: BinNDResult) -> BinNDResult:
        return value
