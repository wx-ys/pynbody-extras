"""Explicit, injectable registry bundle for the bins extension system."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .axes import BinDerivedSpec
    from .result import BinNDResult


class DerivedRegistry:
    """Per-class derived-property registry used by ``BinNDResult.derived``."""

    def __init__(self) -> None:
        self._specs: dict[type, dict[str, BinDerivedSpec]] = {}

    def register(
        self,
        owner_cls: type,
        fn: Callable[[Any], Any] | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: Any = None,
        overwrite: bool = False,
    ) -> Any:
        if isinstance(fn, str):
            return self.register(owner_cls, name=fn, scope=scope, condition=condition, overwrite=overwrite)

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            query_name = name or func.__name__
            self._register(owner_cls, query_name, func, scope=scope, condition=condition, overwrite=overwrite)
            return func

        if fn is None:
            return decorator
        return decorator(fn)

    def _register(
        self, owner_cls: type, name: str, func: Callable[[Any], Any], *, scope: str, condition: Any, overwrite: bool
    ) -> None:
        bucket = self._specs.setdefault(owner_cls, {})
        if not overwrite and name in bucket:
            raise KeyError(f"BinNDResult derived property {name!r} is already registered.")
        from .axes import BinDerivedSpec

        bucket[name] = BinDerivedSpec(name=name, func=func, scope=scope, condition=condition)

    def get(self, owner: BinNDResult | type[BinNDResult], key: str) -> BinDerivedSpec | None:
        for cls in self._mro(owner):
            spec = self._specs.get(cls, {}).get(key)
            if spec is not None and spec.is_available(owner):
                return spec
        return None

    def keys(self, owner: BinNDResult | type[BinNDResult]) -> list[str]:
        out: set[str] = set()
        for cls in self._mro(owner):
            out.update(n for n, spec in self._specs.get(cls, {}).items() if spec.is_available(owner))
        return sorted(out)

    @staticmethod
    def _mro(owner: Any) -> list[type]:
        return owner.mro() if isinstance(owner, type) else type(owner).mro()


class BinsRegistry:
    """Bundle of the bins extension registries (single source of truth)."""

    def __init__(self) -> None:
        self.derived = DerivedRegistry()

    def register_derived(
        self,
        fn: Callable[[Any], Any] | str | None = None,
        *,
        name: str | None = None,
        scope: str = "derived",
        condition: Any = None,
        overwrite: bool = False,
    ) -> Any:
        from .result import BinNDResult

        return self.derived.register(BinNDResult, fn, name=name, scope=scope, condition=condition, overwrite=overwrite)

    def derived_keys(self) -> list[str]:
        from .result import BinNDResult

        return self.derived.keys(BinNDResult)


# Default instance used by exported classmethods/decorators for back-compat.
DEFAULT_REGISTRY = BinsRegistry()
