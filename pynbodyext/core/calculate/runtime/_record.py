"""Value and record-policy helpers for the evaluation engine.

These helpers decide what a :class:`ResultNode` should retain (raw / public
value, value summary) and whether a value is small enough to cache.  They are
pure functions of their inputs, so they were extracted from
:class:`~pynbodyext.core.calculate.runtime.engine.EvalEngine` to keep the engine
module focused on orchestration.

``EvalEngine`` keeps a thin :meth:`~EvalEngine.summarize_value` wrapper because
that method is reached externally via ``ctx.engine.summarize_value`` (see
``nodes/filters.py``).  It delegates here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pynbody.array import SimArray

from pynbodyext.core.calculate.result.enums import CachePolicy, RecordPolicy
from pynbodyext.core.calculate.result.result import ValueSummary
from pynbodyext.core.calculate.runtime.input import FilterResult

if TYPE_CHECKING:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.result.result import ResultNode
    from pynbodyext.core.calculate.runtime.options import RunOptions


class ValueRecorder:
    """Decide what a result node records and what is cacheable."""

    @staticmethod
    def should_store_runtime_cache(
        node: CalculatorBase[Any, Any],
        raw_value: Any,
        public_value: Any,
        policy: CachePolicy,
        options: RunOptions,
    ) -> bool:
        if policy == CachePolicy.NONE:
            return False
        if policy == CachePolicy.FULL:
            return True
        if getattr(node, "effect", None) is not None and str(node.effect) == "mutating":
            return False
        size = ValueRecorder.estimate_cache_bytes(public_value)
        if isinstance(raw_value, FilterResult):
            if raw_value.mask is not public_value:
                size += ValueRecorder.estimate_cache_bytes(raw_value.mask)
        elif raw_value is not public_value:
            size += ValueRecorder.estimate_cache_bytes(raw_value)
        return size <= options.cache_small_value_bytes

    @staticmethod
    def should_auto_record_public_value(
        *, public_value: Any, record_policy: RecordPolicy | None, options: RunOptions
    ) -> bool:
        if not options.auto_record_cached_values:
            return False
        if record_policy != RecordPolicy.SUMMARY:
            return False
        limit = options.auto_record_small_value_bytes
        if limit is None:
            return False
        return ValueRecorder.estimate_cache_bytes(public_value) <= limit

    @staticmethod
    def estimate_cache_bytes(value: Any) -> int:
        size = 2 * 1_000_000

        if value is None or isinstance(value, (bool, int, float, np.generic)):
            size = 64
        elif isinstance(value, str):
            size = len(value.encode("utf-8"))
        elif isinstance(value, np.ndarray):
            size = int(value.nbytes)
        elif isinstance(value, FilterResult):
            size = ValueRecorder.estimate_cache_bytes(value.mask)
        elif isinstance(value, dict):
            size = sum(
                ValueRecorder.estimate_cache_bytes(key) + ValueRecorder.estimate_cache_bytes(item)
                for key, item in value.items()
            )
        elif isinstance(value, (tuple, list)):
            size = sum(ValueRecorder.estimate_cache_bytes(item) for item in value)

        return size

    @staticmethod
    def store_recorded_values(
        node_result: ResultNode,
        raw_value: Any,
        public_value: Any,
        *,
        is_root: bool,
        had_error: bool = False,
        auto_record_public_value: bool = False,
    ) -> None:
        policy = node_result.record.policy or RecordPolicy.SUMMARY

        node_result.record.raw_value = None
        node_result.record.value = None
        node_result.record.stored_raw = False
        node_result.record.stored_value = False

        if is_root:
            if public_value is not None:
                node_result.record.value = public_value
                node_result.record.stored_value = True
            if policy == RecordPolicy.FULL or (
                had_error and policy == RecordPolicy.ERROR_ONLY and raw_value is not None
            ):
                node_result.record.raw_value = raw_value
                node_result.record.stored_raw = True
            return

        if policy == RecordPolicy.FULL:
            node_result.record.raw_value = raw_value
            node_result.record.value = public_value
            node_result.record.stored_raw = True
            node_result.record.stored_value = True

        elif policy == RecordPolicy.ERROR_ONLY:
            if had_error:
                if raw_value is not None:
                    node_result.record.raw_value = raw_value
                    node_result.record.stored_raw = True
                if public_value is not None:
                    node_result.record.value = public_value
                    node_result.record.stored_value = True

        elif policy == RecordPolicy.SUMMARY:
            if auto_record_public_value and public_value is not None:
                node_result.record.value = public_value
                node_result.record.stored_value = True

        elif policy == RecordPolicy.NONE:
            node_result.record.raw_value = None
            node_result.record.value = None

    @staticmethod
    def summarize_value(value: Any) -> ValueSummary | None:
        """Create a compact summary used in reports and result nodes."""
        if value is None:
            return ValueSummary(python_type="NoneType", preview="None")

        units = None
        if hasattr(value, "units"):
            try:
                units = str(value.units)
            except Exception:
                units = None

        shape = None
        if hasattr(value, "shape"):
            try:
                shape = tuple(value.shape)
            except Exception:
                shape = None

        dtype = None
        if hasattr(value, "dtype"):
            try:
                dtype = str(value.dtype)
            except Exception:
                dtype = None

        if isinstance(value, (int, float, bool, str)):
            preview = repr(value)
        elif isinstance(value, np.ndarray):
            preview = f"ndarray(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, SimArray):
            preview = f"SimArray(shape={value.shape}, units={units})"
        else:
            preview = value.__class__.__name__

        return ValueSummary(
            python_type=value.__class__.__name__, shape=shape, dtype=dtype, units=units, preview=preview
        )
