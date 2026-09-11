"""``get_array`` / ``has_array`` must see derived arrays that ``keys()`` hides."""

from __future__ import annotations

import numpy as np

from pynbodyext.util.arrays import get_array, has_array


class _FakeSim:
    """Snapshot stand-in where every array is 'derived' (absent from keys())."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def keys(self) -> list[str]:
        return []  # stored arrays only — derived ones are invisible here

    def __getitem__(self, name: str) -> object:
        return self._data[name]


def test_get_array_returns_derived_array_despite_empty_keys() -> None:
    sim = _FakeSim({"smooth": np.array([1.0, 2.0])})
    assert list(sim.keys()) == []
    arr = get_array(sim, "smooth")
    assert arr is not None
    assert list(arr) == [1.0, 2.0]


def test_get_array_returns_default_when_missing() -> None:
    sim = _FakeSim({})
    assert get_array(sim, "smooth") is None
    assert get_array(sim, "smooth", default=0.0) == 0.0


def test_has_array_reflects_derived_arrays() -> None:
    sim = _FakeSim({"smooth": np.array([1.0])})
    assert has_array(sim, "smooth") is True
    assert has_array(sim, "missing") is False
