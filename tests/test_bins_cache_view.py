"""BinNDResult groups cache/query diagnostics under ``.cache``."""

from __future__ import annotations

import pynbody

from pynbodyext.core.calculate.bins.nodes import Bin1D
from pynbodyext.core.calculate.bins.result import BinCacheView, BinNDResult


def _bins():
    sim = pynbody.new(dm=6)
    sim["r"] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    sim["mass"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)


def test_cache_view_replaces_flat_members() -> None:
    for name in ("cache_report", "query_report", "num_cached_arr", "total_cached_arr"):
        assert not hasattr(BinNDResult, name), f"BinNDResult.{name} should be folded into .cache"


def test_cache_view_diagnostics() -> None:
    bins = _bins()
    assert isinstance(bins.cache, BinCacheView)
    report = bins.cache.report()
    assert {"queries", "total_queries"} <= set(report)
    assert isinstance(bins.cache.queries(), list)
    assert bins.cache.num_cached >= 0
    assert bins.cache.total_cached >= bins.cache.num_cached
