"""BinND transform safety warning must fire on the recommended ``transform`` API."""

from __future__ import annotations

import logging

from pynbodyext.core.calculate import TransformBase
from pynbodyext.core.calculate.bins.nodes import Bin1D


@TransformBase.dataclass
class _Shift(TransformBase[object]):
    def build_handle(self, sim, target, params=None):
        return None


def _bin():
    return Bin1D("x", vmin=0, vmax=6, nbins=3)


def test_bin_transform_revert_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _bin().transform(_Shift())
    assert any("original untransformed data" in rec.getMessage() for rec in caplog.records)


def test_bin_transform_revert_false_does_not_warn(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _bin().transform(_Shift(), revert=False)
    assert not any("original untransformed data" in rec.getMessage() for rec in caplog.records)
