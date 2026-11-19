"""Observe native applications inside science hooks, including handles reverted before return."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from typing import TYPE_CHECKING, Any

from pynbody.transformation import Transformable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .selection_observations import SelectionRecorder


_active: ContextVar[SelectionRecorder | None] = ContextVar("pynbodyext_native_frame_observer", default=None)


class _NativeFrameHook:
    def __init__(self) -> None:
        self.lock = RLock()
        self.users = 0
        self.saved: Any = None

    @contextmanager
    def observe(self, recorder: SelectionRecorder) -> Iterator[None]:
        """Install for overlapping full runs, delivering events only to the active context."""
        with self.lock:
            if not self.users:
                original = Transformable._register_transformation
                self.saved = original

                @wraps(original)
                def registered(target: Any, handle: Any) -> Any:
                    value = original(target, handle)
                    current = _active.get()
                    if current is not None:
                        current.native_application(target, handle)
                    return value

                Transformable._register_transformation = registered
            self.users += 1
        token = _active.set(recorder)
        try:
            yield
        finally:
            _active.reset(token)
            with self.lock:
                self.users -= 1
                if not self.users:
                    Transformable._register_transformation = self.saved
                    self.saved = None


observe_native_frames = _NativeFrameHook().observe
