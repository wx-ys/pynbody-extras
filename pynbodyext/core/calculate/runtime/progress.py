"""Progress event and sink support for calculator execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pynbodyext.core.calculate.display import format_time
from pynbodyext.log import logger

ProgressVerbosity = Literal["run", "node", "phase", "debug"]


if TYPE_CHECKING:
    import logging

def _node_ref(node_id: str) -> str:
    return f"n{node_id.rsplit(':', 1)[-1]}"


@dataclass(slots=True)
class RunProgressEvent:
    """Progress event emitted at the start and end of a calculator run."""

    run_id: str
    root_name: str
    status: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    total_nodes: int | None = None
    node_count: int | None = None
    warning_count: int | None = None
    error_count: int | None = None


@dataclass(slots=True)
class NodeProgressEvent:
    """Progress event emitted for one calculator node."""

    run_id: str
    node_id: str
    node_name: str
    kind: str
    depth: int
    status: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    from_cache: bool = False
    access_summary: str | None = None


@dataclass(slots=True)
class PhaseProgressEvent:
    """Progress event emitted for one named node phase."""

    run_id: str
    node_id: str
    node_name: str
    phase: str
    depth: int
    status: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    access_summary: str | None = None


def _format_count(value: int | None) -> str:
    return str(value) if value is not None else "-"


def _status_text(value: str | None) -> str:
    return value or "-"


progress_logger = logger.getChild("progress")


class ProgressSink(Protocol):
    """Protocol for receiving run, node, and phase progress events."""

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Handle the start of a calculator run."""
        ...

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Handle the end of a calculator run."""
        ...

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Handle the start of a node evaluation."""
        ...

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Handle the end of a node evaluation."""
        ...

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Handle the start of a node phase."""
        ...

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Handle the end of a node phase."""
        ...


@dataclass(slots=True)
class LoggerProgressSink:
    """Progress sink that writes compact events to the package logger."""

    verbosity: ProgressVerbosity = "node"
    indent: str = "  "
    show_ids: bool = False
    log: logging.Logger = field(default_factory=lambda: progress_logger)

    def _shows_node_summary(self) -> bool:
        return self.verbosity in {"node", "phase", "debug"}

    def _shows_node_start(self) -> bool:
        return self.verbosity in {"node", "phase", "debug"}

    def _shows_phase(self) -> bool:
        return self.verbosity in {"phase", "debug"}

    def _shows_phase_start(self) -> bool:
        return self.verbosity == "debug"

    def _node_prefix(self, depth: int) -> str:
        return ("│  " * max(depth, 0)) + "├─ "

    def _phase_prefix(self, depth: int) -> str:
        return ("│  " * max(depth + 1, 0)) + "├─ "

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Log a run start event."""
        id_suffix = f" [{event.run_id}]" if self.show_ids or self.verbosity == "debug" else ""
        self.log.info("run start %s%s", event.root_name, id_suffix)

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Log a run end event."""
        self.log.info(
            "run end %s status=%s total=%s nodes=%s warnings=%s errors=%s",
            event.root_name,
            _status_text(event.status),
            format_time(event.elapsed_s),
            _format_count(event.node_count),
            _format_count(event.warning_count),
            _format_count(event.error_count),
        )

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Log a node start event when verbosity includes debug starts."""
        if not self._shows_node_start():
            return
        self.log.info(
            "%s[%s] %s <%s> start",
            self._node_prefix(event.depth),
            _node_ref(event.node_id),
            event.node_name,
            event.kind,
        )

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Log a node end event when verbosity includes node summaries."""
        if not self._shows_node_summary():
            return

        cache_suffix = " cache-hit" if event.from_cache else ""
        access_suffix = f" {event.access_summary}" if event.access_summary else ""
        self.log.info(
            "%s[%s] %s <%s> %s %s%s%s",
            self._node_prefix(event.depth),
            _node_ref(event.node_id),
            event.node_name,
            event.kind,
            _status_text(event.status),
            format_time(event.elapsed_s),
            cache_suffix,
            access_suffix,
        )

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Log a phase start event when verbosity includes debug starts."""
        if not self._shows_phase_start():
            return
        self.log.info(
            "%s[%s] phase %s start",
            self._phase_prefix(event.depth),
            _node_ref(event.node_id),
            event.phase,
        )

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Log a phase summary event."""
        if not self._shows_phase():
            return
        access_suffix = f" {event.access_summary}" if event.access_summary else ""
        self.log.info(
            "%s[%s] phase %s %s %s%s",
            self._phase_prefix(event.depth),
            _node_ref(event.node_id),
            event.phase,
            _status_text(event.status),
            format_time(event.elapsed_s),
            access_suffix,
        )


@dataclass(slots=True)
class TqdmProgressSink:
    """Optional tqdm-backed progress bar sink.

    Falls back to LoggerProgressSink when tqdm is unavailable.
    """

    verbosity: ProgressVerbosity = "node"
    leave: bool = True
    mininterval: float = 0.1
    show_current: bool = True

    _bar: Any = field(init=False, default=None)
    _fallback: LoggerProgressSink | None = field(init=False, default=None)
    _warned_fallback: bool = field(init=False, default=False)

    def _ensure_backend(self, event: RunProgressEvent) -> None:
        if self._bar is not None or self._fallback is not None:
            return

        try:
            from tqdm.auto import tqdm
        except Exception:
            self._fallback = LoggerProgressSink(verbosity=self.verbosity)
            if not self._warned_fallback:
                progress_logger.warning("tqdm unavailable, falling back to LoggerProgressSink")
                self._warned_fallback = True
            return

        self._bar = tqdm(
            total=event.total_nodes if event.total_nodes and event.total_nodes > 0 else None,
            desc=event.root_name,
            leave=self.leave,
            dynamic_ncols=True,
            mininterval=self.mininterval,
        )

    def _set_postfix(self, text: str) -> None:
        if self._bar is not None and self.show_current:
            self._bar.set_postfix_str(text[:96], refresh=False)

    def on_run_start(self, event: RunProgressEvent) -> None:
        self._ensure_backend(event)
        if self._fallback is not None:
            self._fallback.on_run_start(event)
            return

        if self._bar is not None and self._bar.total is None and event.total_nodes:
            self._bar.total = event.total_nodes
            self._bar.refresh()

    def on_run_end(self, event: RunProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_run_end(event)
            return

        if self._bar is None:
            return

        if event.node_count is not None and self._bar.n < event.node_count:
            self._bar.n = event.node_count
            self._bar.refresh()

        status = _status_text(event.status)
        total = format_time(event.elapsed_s)
        self._set_postfix(f"status={status} total={total}")

        self._bar.close()
        self._bar = None

    def on_node_start(self, event: NodeProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_node_start(event)
            return
        if self.verbosity == "debug":
            self._set_postfix(f"{event.node_name} start")

    def on_node_end(self, event: NodeProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_node_end(event)
            return

        if self._bar is None:
            return

        if not event.from_cache:
            self._bar.update(1)

        if self.verbosity in {"node", "phase", "debug"}:
            cache_suffix = " cache-hit" if event.from_cache else ""
            access_suffix = f" {event.access_summary}" if event.access_summary else ""
            self._set_postfix(
                f"{event.node_name} {_status_text(event.status)} {format_time(event.elapsed_s)}{cache_suffix}{access_suffix}"
            )

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_phase_start(event)
            return
        if self.verbosity == "debug":
            self._set_postfix(f"{event.node_name}:{event.phase} start")

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        if self._fallback is not None:
            self._fallback.on_phase_end(event)
            return
        if self.verbosity in {"phase", "debug"}:
            access_suffix = f" {event.access_summary}" if event.access_summary else ""
            self._set_postfix(
                f"{event.node_name}:{event.phase} {_status_text(event.status)} {format_time(event.elapsed_s)}{access_suffix}"
            )


BasicProgressSink = LoggerProgressSink


@dataclass(slots=True)
class CompositeProgressSink:
    """Progress sink that forwards events to multiple sinks."""

    sinks: tuple[ProgressSink, ...]

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Forward a run start event to all sinks."""
        for sink in self.sinks:
            sink.on_run_start(event)

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Forward a run end event to all sinks."""
        for sink in self.sinks:
            sink.on_run_end(event)

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Forward a node start event to all sinks."""
        for sink in self.sinks:
            sink.on_node_start(event)

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Forward a node end event to all sinks."""
        for sink in self.sinks:
            sink.on_node_end(event)

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Forward a phase start event to all sinks."""
        for sink in self.sinks:
            sink.on_phase_start(event)

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Forward a phase end event to all sinks."""
        for sink in self.sinks:
            sink.on_phase_end(event)


@dataclass(slots=True)
class NullProgressSink:
    """Progress sink that ignores all events."""

    def on_run_start(self, event: RunProgressEvent) -> None:
        """Ignore a run start event."""
        return None

    def on_run_end(self, event: RunProgressEvent) -> None:
        """Ignore a run end event."""
        return None

    def on_node_start(self, event: NodeProgressEvent) -> None:
        """Ignore a node start event."""
        return None

    def on_node_end(self, event: NodeProgressEvent) -> None:
        """Ignore a node end event."""
        return None

    def on_phase_start(self, event: PhaseProgressEvent) -> None:
        """Ignore a phase start event."""
        return None

    def on_phase_end(self, event: PhaseProgressEvent) -> None:
        """Ignore a phase end event."""
        return None


def resolve_progress_sink(
    progress: bool | str | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None,
) -> ProgressSink:
    """Normalize a progress option into a ProgressSink."""
    sink: ProgressSink

    if progress is None or progress is False:
        sink = NullProgressSink()
    elif progress is True:
        sink = LoggerProgressSink(verbosity="node")
    elif isinstance(progress, str):
        progress_name = progress.lower()
        if progress_name == "bar":
            sink = CompositeProgressSink(
                (
                    LoggerProgressSink(verbosity="node"),
                    TqdmProgressSink(verbosity="node"),
                )
            )
        elif progress_name.startswith("bar:"):
            verbosity = cast("ProgressVerbosity", progress_name.split(":", 1)[1])
            sink = CompositeProgressSink(
                (
                    LoggerProgressSink(verbosity=verbosity),
                    TqdmProgressSink(verbosity=verbosity),
                )
            )
        elif progress_name == "bar-only":
            sink = TqdmProgressSink(verbosity="node")
        else:
            sink = LoggerProgressSink(verbosity=cast("ProgressVerbosity", progress_name))
    elif isinstance(progress, list):
        sink = CompositeProgressSink(tuple(progress))
    elif isinstance(progress, tuple):
        sink = CompositeProgressSink(progress)
    else:
        sink = progress

    return sink
