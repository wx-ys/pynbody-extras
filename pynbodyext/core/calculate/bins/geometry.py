"""Axis geometry and measure computation over a bin provider.

A *provider* is anything that exposes ``axes``, ``shape_bins``, and ``nbins`` —
for now the :class:`~.result.BinNDResult` façade, later the ``BinResultModel``.
This isolates geometry logic so it can be unit-tested independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .axes import BinMeasureResolver, axis_matches


class BinGeometry:
    """Per-bin geometry (centers, widths, measure) and axis lookup."""

    def __init__(self, provider: Any, measure_resolver: BinMeasureResolver) -> None:
        self._provider = provider
        self._measure = measure_resolver
        self._multi_index: np.ndarray | None = None

    def _require_1d(self, name: str) -> None:
        if self._provider.ndim != 1:
            raise AttributeError(f"{name} is ambiguous for ND bins; use bins.axis[alias].{name}.")

    @property
    def centers(self) -> Any:
        self._require_1d("centers")
        return self._provider.axes[0].centers

    @property
    def mins(self) -> Any:
        self._require_1d("mins")
        return self._provider.axes[0].mins

    @property
    def maxs(self) -> Any:
        self._require_1d("maxs")
        return self._provider.axes[0].maxs

    @property
    def widths(self) -> Any:
        self._require_1d("widths")
        return self._provider.axes[0].widths

    @property
    def edges(self) -> Any:
        self._require_1d("edges")
        return self._provider.axes[0].edges

    def axis_measure(self, axis: Any) -> np.ndarray:
        return self._measure.resolve(axis)

    @property
    def measure(self) -> np.ndarray:
        """Per-bin combined physical measure (product of each axis's measure)."""
        axes = self._provider.axes
        if len(axes) == 1:
            return self.axis_measure(axes[0])
        multi = self.multi_index_array()
        result = self.axis_measure(axes[0])[multi[:, 0]].copy()
        for i in range(1, len(axes)):
            result *= self.axis_measure(axes[i])[multi[:, i]]
        return result

    def multi_index_array(self) -> np.ndarray:
        if self._multi_index is None:
            self._multi_index = np.column_stack(
                np.unravel_index(np.arange(self._provider.nbins), self._provider.shape_bins, order="C")
            )
        return self._multi_index

    def find_axis(self, aliases: set[str]) -> Any:
        for axis in self._provider.axes:
            if axis_matches(axis, aliases):
                return axis
        raise KeyError(f"No axis matching {sorted(aliases)!r}.")
