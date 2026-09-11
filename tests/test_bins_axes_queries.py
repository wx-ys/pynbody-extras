"""BinNDResult exposes axes as one accessor and queries as one catalog."""

from __future__ import annotations

import pynbody

from pynbodyext.core.calculate.bins.axes import BinAxisAccessor
from pynbodyext.core.calculate.bins.nodes import Bin1D
from pynbodyext.core.calculate.bins.result import BinNDResult, BinQueriesView


def _bins():
    sim = pynbody.new(dm=6)
    sim["r"] = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    sim["mass"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)


def test_consolidated_members_removed() -> None:
    for name in (
        "axis",
        "keys",
        "property_keys",
        "all_keys",
        "stat_explicit",
        "apply",
        "bin_indices",
        "find_axis",
        "set_axis_measure_type",
    ):
        assert not hasattr(BinNDResult, name), f"BinNDResult.{name} should be consolidated"


def test_axes_is_a_single_accessor() -> None:
    bins = _bins()
    assert isinstance(bins.axes, BinAxisAccessor)
    assert len(bins.axes) == 1
    assert [ax.alias for ax in bins.axes] == ["r"]
    assert bins.axes.r is bins.axes["r"] is bins.axes[0]
    assert bins.axes.find({"r"}).alias == "r"


def test_axes_set_measure_type_is_per_instance() -> None:
    bins = _bins()
    bins.axes.set_measure_type("r", "linear")  # should not raise


def test_queries_view_catalog_and_explicit() -> None:
    bins = _bins()
    assert isinstance(bins.queries, BinQueriesView)
    assert "count" in bins.queries.names()
    assert isinstance(bins.queries.properties(), list)
    assert list(bins.queries.explicit("mass", "sum").grid) == [2.0, 2.0, 2.0]
