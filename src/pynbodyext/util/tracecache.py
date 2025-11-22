import itertools
import json
import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pynbody.snapshot import SimSnap

from pynbodyext.log import logger


class EvalCacheManager:
    """Manage ephemeral per-top-level-call evaluation cache (sim-local)."""
    _CTX: ContextVar[dict[tuple[int, tuple[Any, ...]], Any] | None] = ContextVar("_EVAL_CTX", default=None)
    _SIM_TOKEN: ContextVar[int | None] = ContextVar("_EVAL_SIM_TOKEN", default=None)
    _CACHE_ENABLED: ContextVar[bool | None] = ContextVar("_EVAL_CACHE_ENABLED", default=None)

    @classmethod
    @contextmanager
    def use(cls, sim: SimSnap, enable_cache: bool = True) -> Generator[None, None, None]:
        """
        Establish a per-top-level-call evaluation cache for `sim`.
        If a context for this `sim` already exists, this is a no-op (and will
        not override the previously established cache-enabled flag).
        When this call actually creates the context, it will set the run-level
        cache-enabled flag to the provided `enable_cache` value.
        """
        prev_ctx = cls._CTX.get()
        prev_tok = cls._SIM_TOKEN.get()
        prev_ce = cls._CACHE_ENABLED.get()
        token = id(sim)

        # If there is already a context for the same sim, treat use() as a no-op
        # so callers that pre-created a context are not overwritten by an inner use().
        if prev_ctx is not None and prev_tok == token:
            yield
            return

        try:
            cls._CTX.set({})
            cls._SIM_TOKEN.set(token)
            cls._CACHE_ENABLED.set(bool(enable_cache))
            yield
        finally:
            cls._CTX.set(prev_ctx)
            cls._SIM_TOKEN.set(prev_tok)
            cls._CACHE_ENABLED.set(prev_ce)
    @classmethod
    def cache_enabled(cls) -> bool:
        """Return whether evaluation caching is enabled for the current ctx."""
        ce = cls._CACHE_ENABLED.get()
        return bool(ce)

    @classmethod
    def ensure_ctx_for_sim(cls, sim: SimSnap) -> tuple[dict, bool]:
        """Ensure a cache dict exists for given sim; return (ctx, owner)."""
        token = id(sim)
        ctx = cls._CTX.get()
        cur_tok = cls._SIM_TOKEN.get()
        owner = False
        if ctx is None or cur_tok != token:
            ctx = {}
            cls._CTX.set(ctx)
            cls._SIM_TOKEN.set(token)
            # If owner created the ctx and no cache-enabled flag previously set,
            # default to enabling cache.
            cls._CACHE_ENABLED.set(True)
            owner = True
        return ctx, owner

    @classmethod
    def clear_if_owner(cls, owner: bool) -> None:
        if owner:
            cls._CTX.set(None)
            cls._SIM_TOKEN.set(None)

    @classmethod
    def get(cls, key):
        ctx = cls._CTX.get()
        if ctx is None:
            return None
        return ctx.get(key)

    @classmethod
    def set(cls, key, val):
        """Store cached value if cache enabled; otherwise no-op."""
        if not cls.cache_enabled():
            return
        ctx = cls._CTX.get()
        if ctx is None:
            ctx = {}
            cls._CTX.set(ctx)
        ctx[key] = val

@contextmanager
def eval_cache(sim: SimSnap) -> Generator[None, None, None]:
    """Compatibility wrapper: manage a temporary evaluation cache for multiple top-level calls."""
    with EvalCacheManager.use(sim):
        yield

# --- runtime trace context (run-level id + call path) ---
class TraceManager:
    """Centralize trace context and provide a contextmanager for phases."""
    _CALC_PATH: ContextVar[tuple[str, ...]] = ContextVar("_CALC_PATH", default=())
    _CALC_RUN_ID: ContextVar[int | None] = ContextVar("_CALC_RUN_ID", default=None)
    _RUN_COUNTER = itertools.count(1)
    _TRACE_JSON = bool(int(os.getenv("PYNBODYEXT_TRACE_JSON", "0")))

    @staticmethod
    def _node_label(node: Any) -> str:
        return getattr(node, "name", node.__class__.__name__)

    @classmethod
    @contextmanager
    def trace_run(cls, node: Any) -> Iterator[int]:
        """
        Establish a run-level id for the duration of a call/operation.

        Use this at the start of a top-level call so subsequent trace_phase
        invocations reuse the same run id instead of allocating new ones.
        """
        run_id = cls._CALC_RUN_ID.get()
        owner = False
        if run_id is None:
            run_id = next(cls._RUN_COUNTER)
            cls._CALC_RUN_ID.set(run_id)
            owner = True
        try:
            yield run_id
        finally:
            if owner:
                cls._CALC_RUN_ID.set(None)
    @classmethod
    def trace_cache_event(cls, node: Any, event: str, payload: dict | None = None) -> None:
        """
        Log a cache-related event (e.g. 'hit' / 'miss') for `node`.
        Payload may contain extra info (e.g. signature).
        """
        path = cls._CALC_PATH.get()
        run_id = cls._CALC_RUN_ID.get()
        if cls._TRACE_JSON:
            data = {
                "event": "cache",
                "sub": event,
                "run": run_id,
                "node": cls._node_label(node),
                "class": node.__class__.__name__,
                "id": id(node),
                "path": list(path),
                "payload": payload,
            }
            logger.debug("TRACE %s", json.dumps(data, ensure_ascii=False))
        else:
            prefix = f"calc#{run_id}[cache] {' > '.join(path)}"
            msg = f"{prefix} | {event}"
            if payload is not None:
                msg = f"{msg}: {payload}"
            logger.debug(msg)

    @classmethod
    @contextmanager
    def trace_phase(cls, node: Any, phase: str) -> Iterator[None]:
        path = cls._CALC_PATH.get()
        run_id = cls._CALC_RUN_ID.get()
        owner = False
        if run_id is None:
            run_id = next(cls._RUN_COUNTER)
            cls._CALC_RUN_ID.set(run_id)
            owner = True

        new_path = (*path, cls._node_label(node))
        token = cls._CALC_PATH.set(new_path)
        prefix = f"calc#{run_id}[{phase}] {' > '.join(new_path)}"

        if cls._TRACE_JSON:
            logger.debug("TRACE %s", json.dumps({
                "event": "enter",
                "run": run_id,
                "phase": phase,
                "node": cls._node_label(node),
                "class": node.__class__.__name__,
                "id": id(node),
                "depth": len(new_path)-1
            }, ensure_ascii=False))
        else:
            logger.debug("%s | enter: %s", prefix, phase)

        try:
            yield
        finally:
            if cls._TRACE_JSON:
                logger.debug("TRACE %s", json.dumps({
                    "event": "leave",
                    "run": run_id,
                    "phase": phase,
                    "node": cls._node_label(node),
                    "class": node.__class__.__name__,
                    "id": id(node),
                    "depth": len(new_path)-1
                }, ensure_ascii=False))
            else:
                logger.debug("%s | leave: %s", prefix, phase)

            cls._CALC_PATH.reset(token)
            if owner:
                cls._CALC_RUN_ID.set(None)
