from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def is_int_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, (int, np.integer)) for item in value)


def is_bool_array(value: Any) -> bool:
    arr = np.asarray(value)
    return arr.ndim == 1 and arr.dtype == np.dtype(np.bool_)


def normalize_flat_bin_selector(selector: Any, nbins: int) -> np.ndarray:
    if isinstance(selector, (int, np.integer)):
        index = int(selector)
        if index < 0:
            index += nbins
        if index < 0 or index >= nbins:
            raise IndexError("bin index out of range.")
        return np.asarray([index], dtype=int)

    if isinstance(selector, slice):
        return np.asarray(range(*selector.indices(nbins)), dtype=int)

    if is_bool_array(selector):
        arr = np.asarray(selector, dtype=bool)
        if len(arr) != nbins:
            raise ValueError("Boolean bin mask length must match total_nbins.")
        return np.nonzero(arr)[0].astype(int)

    if is_int_sequence(selector):
        arr = np.asarray(selector, dtype=int)
        arr[arr < 0] += nbins
        if np.any((arr < 0) | (arr >= nbins)):
            raise IndexError("bin index out of range.")
        if len(np.unique(arr)) != len(arr):
            raise ValueError("Repeated bin indices are not allowed.")
        return arr

    raise TypeError("Unsupported bin selector.")


def normalize_axis_selector(selector: Any, axis_size: int) -> np.ndarray:
    if isinstance(selector, (int, np.integer)):
        index = int(selector)
        if index < 0:
            index += axis_size
        if index < 0 or index >= axis_size:
            raise IndexError("axis bin index out of range.")
        return np.asarray([index], dtype=int)
    if isinstance(selector, slice):
        return np.asarray(range(*selector.indices(axis_size)), dtype=int)
    if is_bool_array(selector):
        arr = np.asarray(selector, dtype=bool)
        if len(arr) != axis_size:
            raise ValueError("Axis boolean selector length mismatch.")
        return np.nonzero(arr)[0].astype(int)
    if is_int_sequence(selector):
        arr = np.asarray(selector, dtype=int)
        arr[arr < 0] += axis_size
        if np.any((arr < 0) | (arr >= axis_size)):
            raise IndexError("axis bin index out of range.")
        if len(np.unique(arr)) != len(arr):
            raise ValueError("Repeated axis bin indices are not allowed.")
        return arr
    raise TypeError("Unsupported axis selector.")


def normalize_nd_bin_selector(selector: tuple[Any, ...], shape_bins: tuple[int, ...]) -> np.ndarray:
    ellipsis_positions = [index for index, part in enumerate(selector) if part is Ellipsis]
    if ellipsis_positions:
        if len(ellipsis_positions) > 1:
            raise IndexError("Only one ellipsis is allowed in a bin selector.")
        pos = ellipsis_positions[0]
        fill = len(shape_bins) - (len(selector) - 1)
        selector = selector[:pos] + (slice(None),) * fill + selector[pos + 1 :]

    if len(selector) != len(shape_bins):
        raise IndexError("ND bin selector length must match ndim.")

    per_axis = [normalize_axis_selector(part, size) for part, size in zip(selector, shape_bins, strict=True)]
    if any(len(axis_indices) == 0 for axis_indices in per_axis):
        return np.asarray([], dtype=int)
    mesh = np.meshgrid(*per_axis, indexing="ij")
    flat = np.ravel_multi_index(tuple(item.ravel() for item in mesh), shape_bins, order="C")
    if len(np.unique(flat)) != len(flat):
        raise ValueError("Bin selector produced repeated flat bin indices.")
    return np.asarray(flat, dtype=int)
