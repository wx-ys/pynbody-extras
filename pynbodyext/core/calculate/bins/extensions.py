from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .axes import BinDerivedCondition, BinDerivedFunc, BinDerivedSpec
from .statistics import get_statistic, parse_pipeline_key, register_pipeline_transform

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from .result import BinNDResult
    from .statistics import BinStatisticBase


class BinExtensionRegistry:
    """Registry for extensions to BinNDResult, including derived properties and pipeline transforms."""
    def __init__(self) -> None:
        self._derived_specs: defaultdict[type, dict[str, BinDerivedSpec]] = defaultdict(dict)

    def register_transform(
        self,
        name: str,
        func: Callable[[np.ndarray], np.ndarray],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a new pipeline transform that can be used in BinNDResult statistic definitions."""
        register_pipeline_transform(name, func, overwrite=overwrite)

    def register_derived(
        self,
        owner_cls: type[BinNDResult],
        fn: BinDerivedFunc | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: BinDerivedCondition | None = None,
        overwrite: bool = False,
    ) -> Any:
        if isinstance(fn, str):
            return self.register_derived(
                owner_cls,
                name=fn,
                scope=scope,
                condition=condition,
                overwrite=overwrite,
            )

        def decorator(func: BinDerivedFunc) -> BinDerivedFunc:
            query_name = name or func.__name__
            bucket = self._derived_specs[owner_cls]
            if not overwrite and query_name in bucket:
                raise KeyError(f"BinNDResult derived property {query_name!r} is already registered.")
            bucket[query_name] = BinDerivedSpec(
                name=query_name,
                func=func,
                scope=scope,
                condition=condition,
            )
            return func

        if fn is None:
            return decorator
        return decorator(fn)

    def get_derived_spec(self, owner: BinNDResult, key: str) -> BinDerivedSpec | None:
        for cls in type(owner).mro():
            bucket = self._derived_specs.get(cls)
            if bucket and key in bucket:
                spec = bucket[key]
                if spec.is_available(owner):
                    return spec
        return None

    def property_keys(self, owner: BinNDResult) -> list[str]:
        keys: set[str] = set()
        for cls in type(owner).mro():
            bucket = self._derived_specs.get(cls, {})
            keys.update(name for name, spec in bucket.items() if spec.is_available(owner))
        return sorted(keys)

    def query_scope(self, owner: BinNDResult, key: str) -> str:
        spec = self.get_derived_spec(owner, key)
        if spec is not None:
            return spec.scope
        return "particles"

    @staticmethod
    def parse_pipeline_key(key: str) -> tuple[str, list[str], BinStatisticBase, str | None] | None:
        return parse_pipeline_key(key)

    @staticmethod
    def get_statistic(key: str) -> BinStatisticBase | None:
        return get_statistic(key)


BIN_RESULT_EXTENSIONS = BinExtensionRegistry()
