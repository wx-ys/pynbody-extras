"""Run-time execution options for calculator evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.display import display_value, html_card, mimebundle
from pynbodyext.core.calculate.result.enums import ErrorPolicy, RecordPolicy, normalize_error_policy

if TYPE_CHECKING:
    from .progress import ProgressSink

@dataclass(slots=True)
class RunOptions:
    """Execution options for a calculator run.

    Parameters
    ----------
    cache : bool, default: True
        Enable the per-run runtime cache.
    progress : bool, str, ProgressSink, or sequence, optional
        Progress reporting configuration. Strings may be "run", "node",
        "phase", "debug", "bar", or "bar:<verbosity>", or "bar-only".
    perf_time, perf_memory : bool
        Enable time and memory collection.
    observe : bool, default: True
        Enable diagnostic observation of pynbody snapshot field reads and
        invalidations. Disable this to avoid installing observer patches.
    backend : str, default: "serial"
        Backend label reserved for future execution backends.
    default_record_policy : RecordPolicy, default: RecordPolicy.SUMMARY
        Recording policy for nodes without an explicit policy.
    errors : ErrorPolicy or str, default: ErrorPolicy.RAISE
        Error handling policy.
    cache_small_value_bytes : int, default: 10_000_000
        Maximum public-value size for automatic cache storage.
    auto_record_cached_values : bool, default: True
        When true, cached SUMMARY nodes may retain their public value in the
        returned result.
    auto_record_small_value_bytes : int or None, default: 1_000_000
        Maximum public-value size for automatic SUMMARY recording of cached
        values. ``None`` disables auto-recording.
    """

    cache: bool = True
    progress: bool | str | ProgressSink | list[ProgressSink] | tuple[ProgressSink, ...] | None = None
    perf_time: bool = True
    perf_memory: bool = False
    observe: bool = True
    backend: str = "serial"
    default_record_policy: RecordPolicy = RecordPolicy.SUMMARY
    errors: ErrorPolicy | str = ErrorPolicy.RAISE
    cache_small_value_bytes: int = 10_000_000
    auto_record_cached_values: bool = True
    auto_record_small_value_bytes: int | None = 1_000_000


    def __post_init__(self) -> None:
        self.errors = normalize_error_policy(self.errors)

    def __repr__(self) -> str:
        return (
            "RunOptions("
            f"cache={self.cache!r}, progress={self.progress!r}, "
            f"perf_time={self.perf_time!r}, perf_memory={self.perf_memory!r}, "
            f"observe={self.observe!r}, backend={self.backend!r}, "
            f"errors={display_value(self.errors)!r}, "
            f"auto_record_cached_values={self.auto_record_cached_values!r}, "
            f"auto_record_small_value_bytes={self.auto_record_small_value_bytes!r}"
            ")"
        )

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text("RunOptions(...)" if cycle else repr(self))

    def _repr_html_(self) -> str:
        return html_card(
            "RunOptions",
            [
                ("cache", self.cache),
                ("progress", self.progress),
                ("perf time", self.perf_time),
                ("perf memory", self.perf_memory),
                ("observe", self.observe),
                ("backend", self.backend),
                ("record policy", display_value(self.default_record_policy)),
                ("errors", display_value(self.errors)),
                ("small cache bytes", self.cache_small_value_bytes),
                ("auto record cached values", self.auto_record_cached_values),
                ("auto record small bytes", self.auto_record_small_value_bytes),
            ],
        )

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, str]:
        return mimebundle(repr(self), self._repr_html_())
