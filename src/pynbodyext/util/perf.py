"""
perf.py
=======

Profiling utilities for measuring execution time and memory usage in Python code blocks.

This module provides context managers and classes for profiling code, including wall-clock time and memory usage statistics using `tracemalloc`.
It is designed to be used in scientific and performance-sensitive applications.

Classes
-------
StatsTool
    Base class for formatting time and memory statistics.
ProfileInfo
    Stores profiling results for a code block.
PerfStats
    Profiles multiple steps within a code block and reports statistics.

Functions
---------
profile_block
    Context manager for profiling execution time and memory usage.
"""

import contextlib
import logging
import time
import tracemalloc


class StatsTool:
    """
    Base class for formatting time and memory statistics.

    Provides static methods for formatting time and memory values for display.
    """

    @staticmethod
    def _format_time(t):
        """
        Format a time value for display.

        Parameters
        ----------
        t : float or None
            Time in seconds.

        Returns
        -------
        str
            Formatted time string.
        """
        if t is None:
            return "-"
        if t < 1e-3:
            return f"{t*1e6:.1f} μs"
        elif t < 1:
            return f"{t*1e3:.2f} ms"
        elif t < 60:
            return f"{t:.3f} s"
        else:
            return f"{t/60:.2f} min"

    @staticmethod
    def _format_mem(m):
        """
        Format a memory value for display.

        Parameters
        ----------
        m : int or None
            Memory in bytes.

        Returns
        -------
        str
            Formatted memory string.
        """
        if m is None:
            return "-"
        mabs = abs(m)
        if mabs < 1024:
            return f"{m:.1f} B"
        elif mabs < 1048576:        # 1024**2
            return f"{m/1024:.1f} KiB"
        elif mabs < 1073741824:      # 1024**3
            return f"{m/1048576:.2f} MiB"
        else:
            return f"{m/1073741824:.2f} GiB"



class ProfileInfo(StatsTool):
    """
    Stores profiling results for a code block.

    Attributes
    ----------
    time : float or None
        Wall-clock time elapsed during the profiled block, in seconds.
    memory_peak : int or None
        Peak memory usage (in bytes) during the block.
    memory_start : int or None
        Memory usage (in bytes) at the start of the block.
    memory_end : int or None
        Memory usage (in bytes) at the end of the block.
    """
    time: float | None
    memory_peak: int | None
    memory_start: int | None
    memory_end: int | None

    def __init__(self):
        """
        Initialize a ProfileInfo object.
        """
        self.time = None
        self.memory_peak = None
        self.memory_start = None
        self.memory_end = None

    def __repr__(self):
        """
        Return a string representation of the profiling results.

        Returns
        -------
        str
            Formatted profiling information.
        """
        return (f"ProfileInfo(Time={self._format_time(self.time)}, "
                f"Mem={self._format_mem(self.memory_used)}, "
                f"Peak={self._format_mem(self.max_memory_used)})")

    @property
    def memory_used(self):
        """
        Memory used during the block.

        Returns
        -------
        int or None
            Difference between end and start memory usage.
        """
        if self.memory_start is not None and self.memory_end is not None:
            return self.memory_end - self.memory_start
        return None
    @property
    def max_memory_used(self):
        """
        Maximum memory used during the block.

        Returns
        -------
        int or None
            Difference between peak and start memory usage.
        """
        if self.memory_peak is not None and self.memory_start is not None:
            return self.memory_peak - self.memory_start
        return None


@contextlib.contextmanager
def profile_block(measure_time=True, measure_memory=True, tracemalloc_nframe=1):
    """
    Context manager for profiling execution time and memory usage.

    Parameters
    ----------
    measure_time : bool, optional
        Whether to measure wall-clock time (default True).
    measure_memory : bool, optional
        Whether to measure memory usage with tracemalloc (default True).
    tracemalloc_nframe : int, optional
        Number of frames to store in tracemalloc (default 1).

    Yields
    ------
    info : ProfileInfo
        Object with timing and memory usage statistics.

    Examples
    --------
    >>> with profile_block(measure_time=True, measure_memory=True) as info:
    ...     do_work()
    >>> print(info.time, info.memory_peak)
    """
    info = ProfileInfo()
    _tracemalloc_started = False
    if not measure_time and not measure_memory:
        yield info
        return

    if measure_memory:
        if not tracemalloc.is_tracing():
            tracemalloc.start(tracemalloc_nframe)
            _tracemalloc_started = True
        info.memory_start, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    if measure_time:
        t0 = time.perf_counter()
    try:
        yield info
    finally:
        if measure_time:
            t1 = time.perf_counter()
            info.time = t1 - t0
        if measure_memory:
            info.memory_end, info.memory_peak = tracemalloc.get_traced_memory()
            if _tracemalloc_started:
                tracemalloc.stop()


class PerfStats(StatsTool):
    """
    Profiles multiple steps within a code block and reports statistics.

    Attributes
    ----------
    time_enabled : bool
        Whether time profiling is enabled.
    memory_enabled : bool
        Whether memory profiling is enabled.
    tracemalloc_nframe : int
        Number of frames to store in tracemalloc.
    steps : list
        List of profiling results for each step.
    _mem_start : int or None
        Memory usage at the start of profiling.
    _mem_end : int or None
        Memory usage at the end of profiling.
    _step_peaks : list
        List of peak memory usages for each step.
    _total_time : float or None
        Total time elapsed during profiling.
    """
    def __init__(self, time=True, memory=True, tracemalloc_nframe=1):
        """
        Initialize a PerfStats object.

        Parameters
        ----------
        time : bool, optional
            Enable time profiling (default True).
        memory : bool, optional
            Enable memory profiling (default True).
        tracemalloc_nframe : int, optional
            Number of frames to store in tracemalloc (default 1).
        """
        self.time_enabled = time
        self.memory_enabled = memory
        self.tracemalloc_nframe = tracemalloc_nframe
        self.steps = []
        self._mem_start = None
        self._mem_end = None
        self._step_peaks = []
        self._total_time = None

    def reset(self):
        self.steps.clear()
        self._step_peaks.clear()
        self._mem_start = None
        self._mem_end = None
        self._total_time = None

    def __enter__(self):
        """
        Enter the context manager and start profiling.

        Returns
        -------
        self : PerfStats
            The PerfStats object.
        """
        self.reset()
        self._start_time = time.perf_counter() if self.time_enabled else None
        if self.memory_enabled:
            if not tracemalloc.is_tracing():
                tracemalloc.start(self.tracemalloc_nframe)
                self._tracemalloc_started = True
            else:
                self._tracemalloc_started = False
            self._mem_start, _ = tracemalloc.get_traced_memory()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and finalize profiling.

        Parameters
        ----------
        exc_type : type
            Exception type.
        exc_val : Exception
            Exception value.
        exc_tb : traceback
            Exception traceback.
        """
        if self.time_enabled and self._start_time is not None:
            self._total_time = time.perf_counter() - self._start_time
        if self.memory_enabled:
            self._mem_end, peak = tracemalloc.get_traced_memory()
            self._step_peaks.append(peak)
            if self._tracemalloc_started:
                tracemalloc.stop()

    @contextlib.contextmanager
    def step(self, name):
        """
        Context manager for profiling a single step.

        Parameters
        ----------
        name : str
            Name of the step.

        Yields
        ------
        info : ProfileInfo
            Profiling information for the step.

        Raises
        ------
        RuntimeError
            If PerfStats is not used as a context manager.
        """
        if (self.time_enabled and self._start_time is None) or \
           (self.memory_enabled and self._mem_start is None):
            raise RuntimeError("PerfStats must be used as a context manager (with ... as ...) before calling step.")
        info = ProfileInfo()
        if self.memory_enabled:
            info.memory_start, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
        if self.time_enabled:
            t0 = time.perf_counter()
        try:
            yield info
        finally:
            if self.time_enabled:
                t1 = time.perf_counter()
                info.time = t1 - t0
            if self.memory_enabled:
                info.memory_end, info.memory_peak = tracemalloc.get_traced_memory()
                self._step_peaks.append(info.memory_peak)
            self.steps.append((name, info))


    def report(self, logger: logging.Logger | None = None, title: str = "") -> str:
        """
        Print or log a report of profiling statistics.

        Parameters
        ----------
        logger : logging.Logger or None, optional
            Logger to use for output. If None, prints to stdout.
        """
        if not self.time_enabled and not self.memory_enabled:
            return ""
        header = (
            f"{'Step':<15} | {'Time':>12} | {'Mem Used':>15} | {'Peak Mem':>15}"
        )
        lines = [title] if title else []
        lines.extend(["-" * len(header), header, "-" * len(header)])

        for name, info in self.steps:
            time_str = self._format_time(info.time)
            mem_used = self._format_mem(info.memory_used)
            peak_mem = self._format_mem(info.max_memory_used)
            lines.append(f"{name:<15} | {time_str:>12} | {mem_used:>15} | {peak_mem:>15}")
        lines.append("-" * len(header))

        # total memory/peak
        total_mem_used = (self._mem_end - self._mem_start) if (self._mem_start is not None and self._mem_end is not None) else None
        total_peak = (max(self._step_peaks) - self._mem_start) if self._step_peaks and self._mem_start is not None else None
        total_time = self._format_time(self._total_time)
        total_mem = self._format_mem(total_mem_used)
        total_peak_str = self._format_mem(total_peak)

        lines.append(f"{'Total':<15} | {total_time:>12} | {total_mem:>15} | {total_peak_str:>15}")
        lines.append("-" * len(header))

        msg = "\n" + "\n".join(lines)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
        return msg
