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

FieldOperation = Literal["read", "dirty", "delete", "derive"]

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
    phase: str | None = None
    key_repr: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "operation": self.operation,
            "field": self.field,
            "sim_id": self.sim_id,
            "sim_type": self.sim_type,
            "phase": self.phase,
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
    derived_fields: set[str] = field(default_factory=set)
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

    def record_derive(self, sim: Any, key: Any) -> None:
        self._record("derive", sim, key, fallback_to_string=True)

    def _record(self, operation: FieldOperation, sim: Any, key: Any, *, fallback_to_string: bool) -> None:
        phase = current_observation_phase()
        for name in _field_names(key, fallback_to_string=fallback_to_string):
            if operation == "read":
                self.reads.add(name)
            elif operation == "dirty":
                self.dirty_fields.add(name)
            elif operation == "delete":
                self.deletes.add(name)
            elif operation == "derive":
                self.derived_fields.add(name)

            self.events.append(
                AccessEvent(
                    timestamp=time.perf_counter(),
                    operation=operation,
                    field=name,
                    sim_id=id(sim),
                    sim_type=type(sim).__name__,
                    phase=phase,
                    key_repr=_short_repr(key),
                )
            )

    def phases(self) -> tuple[str | None, ...]:
        """Return observed phases in first-seen order."""
        seen: set[str | None] = set()
        phases: list[str | None] = []
        for event in self.events:
            if event.phase in seen:
                continue
            seen.add(event.phase)
            phases.append(event.phase)
        return tuple(phases)

    def fields_for(self, operation: FieldOperation, *, phase: str | None | object = Ellipsis) -> set[str]:
        """Return observed fields for an operation, optionally limited to one phase."""
        return {
            event.field
            for event in self.events
            if event.operation == operation and (phase is Ellipsis or event.phase == phase)
        }

    def phase_summary(self, phase: str | None | object = Ellipsis) -> dict[str, Any]:
        """Return read/dirty/delete fields for one phase or for the whole node."""
        reads = self.fields_for("read", phase=phase)
        dirty_fields = self.fields_for("dirty", phase=phase)
        deletes = self.fields_for("delete", phase=phase)
        derived_fields = self.fields_for("derive", phase=phase)
        event_count = sum(1 for event in self.events if phase is Ellipsis or event.phase == phase)
        return {
            "reads": sorted(reads),
            "dirty_fields": sorted(dirty_fields),
            "writes": sorted(dirty_fields),
            "deletes": sorted(deletes),
            "derived_fields": sorted(derived_fields),
            "event_count": event_count,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_label": self.node_label,
            "reads": sorted(self.reads),
            "dirty_fields": sorted(self.dirty_fields),
            "writes": sorted(self.writes),
            "deletes": sorted(self.deletes),
            "derived_fields": sorted(self.derived_fields),
            "event_count": self.event_count,
            "phases": [phase if phase is not None else "node" for phase in self.phases()],
        }

    def as_dict(self, *, include_events: bool = True) -> dict[str, Any]:
        payload = self.summary()
        if include_events:
            payload["events"] = [event.as_dict() for event in self.events]
        return payload


_OBSERVATION_STACK: ContextVar[tuple[AccessObservation, ...]] = ContextVar(
    "pynbodyext_calculate_observation_stack", default=()
)

_OBSERVATION_PHASE_STACK: ContextVar[tuple[str | None, ...]] = ContextVar(
    "pynbodyext_calculate_observation_phase_stack", default=()
)

_DERIVATION_STACK: ContextVar[tuple[str, ...]] = ContextVar("pynbodyext_calculate_derivation_stack", default=())


def current_observation() -> AccessObservation | None:
    stack = _OBSERVATION_STACK.get()
    return stack[-1] if stack else None


def current_observation_phase() -> str | None:
    stack = _OBSERVATION_PHASE_STACK.get()
    return stack[-1] if stack else None


def current_derivation_field() -> str | None:
    stack = _DERIVATION_STACK.get()
    return stack[-1] if stack else None


@contextmanager
def observation_phase(phase: str | None) -> Any:
    """Attach a phase label to observed access events in this context."""
    stack = _OBSERVATION_PHASE_STACK.get()
    token = _OBSERVATION_PHASE_STACK.set(stack + (phase,))
    try:
        yield
    finally:
        _OBSERVATION_PHASE_STACK.reset(token)


@contextmanager
def derivation_field(field: str) -> Any:
    """Mark writes caused by materializing one derived array."""
    stack = _DERIVATION_STACK.get()
    token = _DERIVATION_STACK.set(stack + (field,))
    try:
        yield
    finally:
        _DERIVATION_STACK.reset(token)


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
        for name in ("SimSnap", "SubSnapBase", "SubSnap", "IndexedSubSnap", "FamilySubSnap"):
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
            cls._patch_method(target_cls, "_derive_array", _wrap_derive_array)
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
            if key == current_derivation_field():
                observation.record_derive(self, key)
            else:
                observation.record_dirty(self, key)
        return value

    return wrapped


def _wrap_derive_array(original: Any) -> Any:
    @wraps(original)
    def wrapped(self: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(name, str):
            with derivation_field(name):
                return original(self, name, *args, **kwargs)
        return original(self, name, *args, **kwargs)

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


def _truncate_text(text: str, max_length: int | None) -> str:
    if max_length is None or len(text) <= max_length:
        return text
    if max_length <= 0:
        return ""
    if max_length <= 3:
        return "." * max_length
    return text[: max_length - 3] + "..."


def _join_fields(
    values: set[str] | list[str] | tuple[str, ...], *, max_items: int | None = None, max_length: int | None = None
) -> str:
    if not values:
        return "-"
    items = sorted(values)
    if max_items is not None and len(items) > max_items:
        hidden = len(items) - max_items
        shown = ", ".join(items[:max_items])
        text = f"{shown} ..(+{hidden})"
        if max_length is not None and len(text) > max_length:
            suffix = f"..(+{hidden})"
            keep = max_length - len(suffix)
            if keep <= 0:
                return suffix[:max_length]
            return shown[:keep].rstrip(", ") + suffix
        return text
    return _truncate_text(", ".join(items), max_length)


def _phase_label(phase: str | None) -> str:
    return phase if phase is not None else "node"


def format_observation_access(
    observation: AccessObservation | None,
    *,
    phase: str | None | object = Ellipsis,
    include_reads: bool = True,
    read_items: int = 4,
    dirty_items: int | None = None,
    delete_items: int | None = None,
    derive_items: int | None = None,
) -> str:
    """Return a compact access suffix such as ``read=pos dirty=vel del=r``."""
    if observation is None or observation.is_empty:
        return ""

    reads = observation.fields_for("read", phase=phase)
    dirty_fields = observation.fields_for("dirty", phase=phase)
    deletes = observation.fields_for("delete", phase=phase)
    derived_fields = observation.fields_for("derive", phase=phase)

    parts: list[str] = []
    if include_reads and reads:
        parts.append(f"read[{_join_fields(reads, max_items=read_items)}]")
    if dirty_fields:
        parts.append(f"dirty[{_join_fields(dirty_fields, max_items=dirty_items)}]")
    if deletes:
        parts.append(f"del[{_join_fields(deletes, max_items=delete_items)}]")
    if derived_fields:
        parts.append(f"derive[{_join_fields(derived_fields, max_items=derive_items)}]")
    return "; ".join(parts)


def render_observer_report(observations: Any, *, include_empty: bool = False, show_ids: bool = False) -> str:
    """Render a compact text report for access observations."""
    items = list(observations)
    if not include_empty:
        items = [item for item in items if not item.is_empty]

    if not items:
        return "No observed pynbody field access events."

    widths = {"node": 16 if not show_ids else 25, "phase": 14, "reads": 34, "dirty": 34, "deletes": 28, "derived": 28}
    header = (
        f"{'Node':<{widths['node']}} | "
        f"{'Phase':<{widths['phase']}} | "
        f"{'Reads':<{widths['reads']}} | "
        f"{'Dirty':<{widths['dirty']}} | "
        f"{'Deletes':<{widths['deletes']}} | "
        f"{'Derived':<{widths['derived']}}"
    )
    lines = ["Observer", "-" * len(header), header, "-" * len(header)]
    for observation in items:
        phases = observation.phases() or (None,)
        for phase in phases:
            reads = observation.fields_for("read", phase=phase)
            dirty_fields = observation.fields_for("dirty", phase=phase)
            deletes = observation.fields_for("delete", phase=phase)
            derived_fields = observation.fields_for("derive", phase=phase)
            if not include_empty and not (reads or dirty_fields or deletes or derived_fields):
                continue

            label = observation.node_label
            if show_ids:
                label = f"[{observation.node_id.rsplit(':', 1)[-1]}] {label}"
            label = _truncate_text(label, widths["node"])
            phase_text = _truncate_text(_phase_label(phase), widths["phase"])
            reads_text = _join_fields(reads, max_items=4, max_length=widths["reads"])
            dirty_text = _join_fields(dirty_fields, max_length=widths["dirty"])
            delete_text = _join_fields(deletes, max_length=widths["deletes"])
            derived_text = _join_fields(derived_fields, max_length=widths["derived"])

            lines.append(
                f"{label:<{widths['node']}} | "
                f"{phase_text:<{widths['phase']}} | "
                f"{reads_text:<{widths['reads']}} | "
                f"{dirty_text:<{widths['dirty']}} | "
                f"{delete_text:<{widths['deletes']}} | "
                f"{derived_text:<{widths['derived']}}"
            )
    lines.append("-" * len(header))
    return "\n".join(lines)
