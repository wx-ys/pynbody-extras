"""Observe native applications inside science hooks, including handles reverted before return."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from typing import TYPE_CHECKING, Any
from weakref import ref

from pynbody.transformation import Transformable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from weakref import ReferenceType

    from .selection_observations import SelectionRecorder


_active: ContextVar[ReferenceType[SelectionRecorder] | None] = ContextVar("pynbodyext_native_frame_observer", default=None)


class _NativeFrameHook:
    def __init__(self) -> None:
        self.lock = RLock()
        self.by_root: dict[int, list[ReferenceType[SelectionRecorder]]] = {}
        self.saved: Any = None

    @contextmanager
    def observe(self, recorder: SelectionRecorder) -> Iterator[None]:
        """Observe registrations on this physical root for the duration of a full run."""
        root_id = id(recorder.root)
        observer = ref(recorder)
        with self.lock:
            if not self.by_root:
                original = Transformable._register_transformation
                self.saved = original

                @wraps(original)
                def registered(target: Any, handle: Any) -> Any:
                    value = original(target, handle)
                    with self.lock:
                        listeners = tuple(self.by_root.get(id(getattr(target, "ancestor", None)), ()))
                    owner = _active.get()
                    for listener in listeners:
                        current = listener()
                        if current is not None:
                            current.native_application(target, handle, own_run=listener is owner)
                    return value

                Transformable._register_transformation = registered
            self.by_root.setdefault(root_id, []).append(observer)
        token = _active.set(observer)
        try:
            yield
        finally:
            _active.reset(token)
            with self.lock:
                listeners = self.by_root[root_id]
                listeners.remove(observer)
                if not listeners:
                    del self.by_root[root_id]
                if not self.by_root:
                    Transformable._register_transformation = self.saved
                    self.saved = None


observe_native_frames = _NativeFrameHook().observe
