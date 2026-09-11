"""Internal helper types for the evaluation engine.

These small, near-pure data holders and the ultra-minimal batch context live
apart from :class:`~pynbodyext.core.calculate.runtime.engine.EvalEngine` so the
engine module stays focused on orchestration.  They have no business logic and
no round-trip through the engine's collaborators; ``EvalEngine`` stores and
consumes them directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from pynbodyext.core.calculate.result.enums import CachePolicy
    from pynbodyext.core.calculate.runtime.engine import EvalEngine
    from pynbodyext.core.calculate.runtime.input import NodeInput
    from pynbodyext.core.calculate.runtime.options import RunOptions

from .context import _MutationState


@dataclass(slots=True)
class _EvaluationPlan:
    work: NodeInput
    stack_key: tuple[Any, ...]
    cache_key: tuple[Any, ...]
    observed_cache_prefix: tuple[Any, ...] | None
    cacheable: bool
    cache_policy: CachePolicy
    #: Structured signature computed once for this node evaluation.  Shared with
    #: the engine's result-node creation (and the root provenance) so a node's
    #: signature is serialized once per run instead of once per consumer.
    structured_signature: Any | None = None


@dataclass(slots=True)
class _NodeExecutionState:
    raw_value: Any = None
    public_value: Any = None


class _NodeExecutionFailure(Exception):
    def __init__(self, cause: Exception, state: _NodeExecutionState) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.state = state


class _DummyNodeResult:
    """Minimal node placeholder used by :class:`_MinimalBatchContext`."""

    __slots__ = ("phases",)

    def __init__(self) -> None:
        self.phases: list[Any] = []


_DUMMY_NODE_RESULT = _DummyNodeResult()


class _MinimalBatchContext:
    """Ultra-minimal execution context for per-batch evaluation.

    Used by the engine for calculator nodes that have **no** child calculator
    dependencies — i.e. no nested calculators in ``Param`` fields and no
    filter/transform wrappers.

    Eliminates the overhead of the full execution context: runtime cache,
    trace/perf collectors, result nodes, log events, and node registry
    allocation.  Compared with :class:`ExecutionContext`, this class creates
    **zero** sub-objects beyond itself.
    """

    __slots__ = ("sim", "sim_signature", "options", "engine", "mutation", "_node_stack", "_evaluation_stack")

    def __init__(self, sim: Any, options: RunOptions, engine: EvalEngine) -> None:
        self.sim = sim
        self.sim_signature: tuple[()] = ()
        self.options = options
        self.engine = engine
        self.mutation = _MutationState()
        self._node_stack: list[Any] = [_DUMMY_NODE_RESULT]
        self._evaluation_stack: list[Any] = []

    @property
    def current_node(self) -> Any:
        return self._node_stack[-1] if self._node_stack else None

    @contextmanager
    def phase(self, node: Any, phase_name: str) -> Generator[None, None, None]:
        yield

    def log(self, level: str, message: str, *, node_id: str | None = None, phase: str | None = None) -> None:
        pass

    @contextmanager
    def observe_node_access(self, node_result: Any, node: Any) -> Generator[None, None, None]:
        yield None
