"""Property adapters that consume one calculator's resolved public value.

Authors override ``process(value)`` and read static configuration from ``self``::

    @PostprocessorBase.dataclass
    class ScaleValues(PostprocessorBase[object]):
        factor: float = 2.0

        def process(self, value):
            return value * self.factor


    scaled = ScaleValues(3.0, source=upstream)

The required keyword-only source is inherited through the existing declarative
field machinery. The ordinary property runtime resolves it inside the same DAG;
the adapter neither starts another run nor converts the source value's units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pynbodyext.core.calculate.params.fields import Param, collect_param_specs
from pynbodyext.core.calculate.result.enums import RevertPolicy

from .base import CalculatorBase
from .properties import PropertyBase

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap

    from pynbodyext.core.calculate.runtime import ExecutionContext, NodeInput

T = TypeVar("T")


class PostprocessorBase(PropertyBase[T], Generic[T], ABC):
    """Consume one Calculator source via ``process(value)`` and static fields.

    Use ``@PostprocessorBase.dataclass`` on scientific subclasses. ``source`` is
    the only dynamic parameter; raw Result and RecordRef sources are unsupported.
    The adapter owns ``calculate`` so subclasses implement ``process`` only.
    """

    source: Param[CalculatorBase[Any, Any]] = Param(kw_only=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Validate before the base hook registers this class's signature type.
        if cls.calculate is not PostprocessorBase.calculate:
            raise TypeError("PostprocessorBase owns calculate(); override process(value) instead")
        super().__init_subclass__(**kwargs)

    def _init_dataclass_base(self) -> None:
        super()._init_dataclass_base()
        if not isinstance(self.source, CalculatorBase):
            raise TypeError("source must be a CalculatorBase; Result and RecordRef are not supported")
        specs = {spec.name: spec for spec in collect_param_specs(type(self))}
        source_spec = specs.get("source")
        if source_spec is None or source_spec.kind != "dynamic" or not source_spec.signature:
            raise TypeError("source must be a signature-bearing dynamic Param")
        if set(self.dynamic_param_names()) != {"source"}:
            raise TypeError("source is the only supported dynamic postprocessor parameter")
        if source_spec.field_name is not None or source_spec.target_units is not None or not source_spec.optional_units:
            raise TypeError("source must pass its value without unit conversion")

    def calculate(self, sim: SimSnap, params: Any = None) -> T:
        """Pass the resolved source value to the scientific authoring hook."""
        return self.process(params.source)

    def execute(self, ctx: ExecutionContext, input: NodeInput) -> T:
        """Use the property lifecycle, cleaning our bound transform on failure."""
        try:
            return super().execute(ctx, input)
        except Exception:
            transform = self.scope.as_transform()
            if (
                self.scope.should_revert
                and transform is not None
                and transform.revert_policy == RevertPolicy.ALWAYS
                and input.transform is not None
                and input.transform.revertible
            ):
                try:
                    with ctx.phase(self, "revert"):
                        transform.cleanup(ctx, input.transform.handle)
                    ctx.log("debug", f"cleaned up transform after error in {self.log_label}")
                except Exception as cleanup_error:
                    ctx.log("error", f"transform cleanup failed after error: {cleanup_error}")
            raise

    @abstractmethod
    def process(self, value: Any) -> T:
        """Process the public source value using static configuration on self."""
        raise NotImplementedError
