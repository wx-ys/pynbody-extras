"""Runtime observation of pynbody snapshot field access.

The observer records what a calculator node actually reads or invalidates while
it runs. It is intentionally diagnostic-only: cache keys and cache invalidation
continue to use the existing runtime behavior.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Literal

FieldOperation = Literal["read", "dirty", "delete"]

_MAX_REPR_LENGTH = 120


def _short_repr(value: Any) -> str:
    text = repr(value)
    if len(text) > _MAX_REPR_LENGTH:
        return text[: _MAX_REPR_LENGTH - 3] + "..."
    return text


def _field_names(value: Any, *, fallback_to_string: bool = False) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)) and value and all(isinstance(item, str) for item in value):
        return tuple(value)
    if fallback_to_string and value is not None:
        return (str(value),)
    return ()


@dataclass(slots=True)
class AccessEvent:
    """One observed pynbody snapshot access event."""

    timestamp: float
    operation: FieldOperation
    field: str
    sim_id: int
    sim_type: str
    key_repr: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "field": self.field,
            "sim_id": self.sim_id,
            "sim_type": self.sim_type,
            "key_repr": self.key_repr,
        }


@dataclass(slots=True)
class AccessObservation:
    """Observed snapshot fields for one calculator node execution."""

    node_id: str
    node_label: str
    reads: set[str] = field(default_factory=set)
    dirty_fields: set[str] = field(default_factory=set)
    deletes: set[str] = field(default_factory=set)
    events: list[AccessEvent] = field(default_factory=list)

    @property
    def writes(self) -> set[str]:
        """Alias for fields observed through ``SimSnap._dirty``."""
        return self.dirty_fields

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def is_empty(self) -> bool:
        return not self.events

    def record_read(self, sim: Any, key: Any) -> None:
        self._record("read", sim, key, fallback_to_string=False)

    def record_dirty(self, sim: Any, key: Any) -> None:
        self._record("dirty", sim, key, fallback_to_string=True)

    def record_delete(self, sim: Any, key: Any) -> None:
        self._record("delete", sim, key, fallback_to_string=True)

    def _record(
        self,
        operation: FieldOperation,
        sim: Any,
        key: Any,
        *,
        fallback_to_string: bool,
    ) -> None:
        for name in _field_names(key, fallback_to_string=fallback_to_string):
            if operation == "read":
                self.reads.add(name)
            elif operation == "dirty":
                self.dirty_fields.add(name)
            elif operation == "delete":
                self.deletes.add(name)

            self.events.append(
                AccessEvent(
                    timestamp=time.perf_counter(),
                    operation=operation,
                    field=name,
                    sim_id=id(sim),
                    sim_type=type(sim).__name__,
                    key_repr=_short_repr(key),
                )
            )

    def summary(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
            "reads": sorted(self.reads),
            "dirty_fields": sorted(self.dirty_fields),
            "writes": sorted(self.writes),
            "deletes": sorted(self.deletes),
            "event_count": self.event_count,
        }

    def as_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = self.summary()
        if include_events:
            payload["events"] = [event.as_dict() for event in self.events]
        return payload


_OBSERVATION_STACK: ContextVar[tuple[AccessObservation, ...]] = ContextVar(
    "pynbodyext_calculate_observation_stack",
    default=(),
)


def current_observation() -> AccessObservation | None:
    stack = _OBSERVATION_STACK.get()
    return stack[-1] if stack else None


def _all_subclasses(cls: type[Any]) -> list[type[Any]]:
    out: list[type[Any]] = []
    for child in cls.__subclasses__():
        out.append(child)
        out.extend(_all_subclasses(child))
    return out


def _candidate_snapshot_classes() -> tuple[type[Any], ...]:
    classes: list[type[Any]] = []
    modules: list[Any] = []

    for module_name in ("pynbody.snapshot", "pynbody.snapshot.simsnap"):
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        modules.append(module)

    for module in modules:
        for name in (
            "SimSnap",
            "SubSnapBase",
            "SubSnap",
            "IndexedSubSnap",
            "FamilySubSnap",
        ):
            cls = getattr(module, name, None)
            if isinstance(cls, type):
                classes.append(cls)
                classes.extend(_all_subclasses(cls))

    seen: set[type[Any]] = set()
    unique: list[type[Any]] = []
    for cls in classes:
        if cls in seen:
            continue
        seen.add(cls)
        unique.append(cls)
    return tuple(unique)


class _SnapshotPatchManager:
    _lock = threading.RLock()
    _active_count = 0
    _originals: dict[tuple[type[Any], str], Any] = {}

    @classmethod
    def acquire(cls) -> None:
        with cls._lock:
            if cls._active_count == 0:
                cls._install()
            cls._active_count += 1

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._active_count <= 0:
                return
            cls._active_count -= 1
            if cls._active_count == 0:
                cls._restore()

    @classmethod
    def _install(cls) -> None:
        for target_cls in _candidate_snapshot_classes():
            cls._patch_method(target_cls, "__getitem__", _wrap_getitem)
            cls._patch_method(target_cls, "_dirty", _wrap_dirty)
            cls._patch_method(target_cls, "__delitem__", _wrap_delitem)

    @classmethod
    def _patch_method(cls, target_cls: type[Any], method_name: str, factory: Any) -> None:
        original = target_cls.__dict__.get(method_name)
        if original is None or not callable(original):
            return
        key = (target_cls, method_name)
        if key in cls._originals:
            return
        cls._originals[key] = original
        setattr(target_cls, method_name, factory(original))

    @classmethod
    def _restore(cls) -> None:
        for (target_cls, method_name), original in reversed(list(cls._originals.items())):
            setattr(target_cls, method_name, original)
        cls._originals.clear()


def _wrap_getitem(original: Any) -> Any:
    @wraps(original)
    def wrapped(self: Any, key: Any) -> Any:
        value = original(self, key)
        observation = current_observation()
        if observation is not None:
            observation.record_read(self, key)
        return value

    return wrapped


def _wrap_dirty(original: Any) -> Any:
    @wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        value = original(self, *args, **kwargs)
        observation = current_observation()
        if observation is not None:
            key = args[0] if args else kwargs.get("name")
            observation.record_dirty(self, key)
        return value

    return wrapped


def _wrap_delitem(original: Any) -> Any:
    @wraps(original)
    def wrapped(self: Any, key: Any) -> Any:
        value = original(self, key)
        observation = current_observation()
        if observation is not None:
            observation.record_delete(self, key)
        return value

    return wrapped



@contextmanager
def observe_sim_access(node_id: str, node_label: str) -> Any:
    """Observe pynbody snapshot reads, dirty calls, and deletes in this context."""
    observation = AccessObservation(node_id=node_id, node_label=node_label)
    _SnapshotPatchManager.acquire()
    stack = _OBSERVATION_STACK.get()
    token = _OBSERVATION_STACK.set(stack + (observation,))
    try:
        yield observation
    finally:
        _OBSERVATION_STACK.reset(token)
        _SnapshotPatchManager.release()


def _format_fields(values: set[str], *, max_items: int = 8) -> str:
    if not values:
        return "-"
    items = sorted(values)
    if len(items) > max_items:
        shown = items[:max_items]
        return ", ".join(shown) + f", ... (+{len(items) - max_items})"
    return ", ".join(items)


def render_observer_report(
    observations: Any,
    *,
    include_empty: bool = False,
) -> str:
    """Render a compact text report for access observations."""
    items = list(observations)
    if not include_empty:
        items = [item for item in items if not item.is_empty]

    if not items:
        return "No observed pynbody field access events."

    header = "Node                           | Reads                    | Dirty                    | Deletes"
    lines = ["Observer", "-" * len(header), header, "-" * len(header)]
    for observation in items:
        label = observation.node_label[:30]
        lines.append(
            f"{label:<30} | "
            f"{_format_fields(observation.reads):<24} | "
            f"{_format_fields(observation.dirty_fields):<24} | "
            f"{_format_fields(observation.deletes)}"
        )
    lines.append("-" * len(header))
    return "\n".join(lines)
