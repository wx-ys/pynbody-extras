"""Tests for the bins system (``Bin1D`` / ``BinNDResult``) end to end.

Covers counts/stats/geometry, tuple and grid selectors, sub-result locality,
query and cache wiring, ``BinsArray`` metadata and plotting, family sub-results,
and the custom bin-algorithm / derived-property registration paths.
"""

from __future__ import annotations

import numpy as np
import pynbody
import pytest

from pynbodyext.core.calculate import (
    Bin1D,
    BinNDResult,
    BinsArray,
    SubBinNDResult,
    has_axis,
    register_bin_algorithm,
)


def make_sim() -> pynbody.SimSnap:
    sim = pynbody.new(dm=6)
    sim["x"] = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    sim["y"] = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    sim["r"] = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)
    sim["mass"] = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    sim["vz"] = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    return sim


def test_bin1d_counts_stats_and_geometry() -> None:
    sim = make_sim()

    bins = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)

    assert isinstance(bins, BinNDResult)
    assert bins.shape_bins == (3,)
    np.testing.assert_array_equal(bins["count"], [2, 2, 2])
    np.testing.assert_allclose(bins["mass.sum"], [3, 7, 11])
    np.testing.assert_allclose(bins.centers, [1, 3, 5])
    np.testing.assert_allclose(bins["density"], np.asarray(bins["mass.sum"]) / np.asarray(bins["measure"]))
    assert isinstance(bins["mass.sum"], BinsArray)


def test_binnd_tuple_particles_and_grid_order() -> None:
    sim = make_sim()

    bins = Bin1D("x", vmin=0, vmax=6, nbins=3, alias="x") @ Bin1D("y", vmin=0, vmax=3, nbins=3, alias="y")
    result = bins(sim)

    assert result.shape_bins == (3, 3)
    np.testing.assert_array_equal(result["count"].grid, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert len(result.particles_at_bin[:, 2]) == 2

    mask = np.array([True, False, True])
    assert len(result.particles_at_bin[..., mask]) == 4


def test_getitem_callable_and_illegal_bin_selectors() -> None:
    sim = make_sim()
    bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)

    vz_mean = bins[lambda sub: sub["vz"].mean()]

    assert isinstance(vz_mean, BinsArray)
    np.testing.assert_allclose(vz_mean, [15, 35, 55])
    with pytest.raises(TypeError, match="particles_at_bin"):
        bins[0]
    with pytest.raises(TypeError, match="particles_at_bin"):
        bins[0:2]
    with pytest.raises(TypeError, match="particles_at_bin"):
        bins[np.array([True, False, True])]


def test_subresult_cache_and_particle_stats_are_local() -> None:
    sim = make_sim()
    bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)
    mask = np.array([True, True, False, False, False, False])

    first = bins[mask]
    second = bins[mask]

    assert isinstance(first, SubBinNDResult)
    assert first is second
    assert first.parent is bins
    np.testing.assert_array_equal(first["count"], [2, 0, 0])
    np.testing.assert_array_equal(bins["count"], [2, 2, 2])
    assert first["mass.sum"] is not bins["mass.sum"]


def test_run_active_query_populates_result_cache() -> None:
    sim = make_sim()
    calculator = Bin1D("x", vmin=0, vmax=6, nbins=3).with_active(["mass.sum"])

    run = calculator.run(sim)
    bins = run.value

    assert isinstance(bins, BinNDResult)
    np.testing.assert_allclose(bins["mass.sum"], [3, 7, 11])
    assert bins.cache.report()["queries"] >= 1


def test_profile_like_keys_registered_derived_and_cache_summary() -> None:
    sim = make_sim()
    bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)

    @BinNDResult.derived_property(name="double_mass_sum_for_test")
    def double_mass_sum_for_test(result) -> np.ndarray:
        return result["mass.sum"] * 2

    keys = bins.queries.names()

    assert "count" in keys
    assert "double_mass_sum_for_test" in keys
    np.testing.assert_allclose(bins["double_mass_sum_for_test"], [6, 14, 22])
    assert "double_mass_sum_for_test" in bins.queries.properties()
    assert bins.queries.names() == bins.queries.names()
    assert bins._ipython_key_completions_() == bins.queries.names()

    first = bins.stat("mass", "mean", weight="mass")
    second = bins.stat("mass", "mean", weight="mass")
    assert first is second
    np.testing.assert_allclose(first, [5 / 3, 25 / 7, 61 / 11])
    assert bins.cache.report()["total_queries"] >= bins.cache.report()["queries"]
    assert "shape=(3,)" in repr(bins)


def test_binsarray_metadata_and_1d_plot() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sim = make_sim()
    bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)
    mass_sum = bins["mass.sum"]

    assert mass_sum.name == "mass.sum"
    assert mass_sum.field == "mass"
    assert mass_sum.mode == "sum"
    assert mass_sum.axis_aliases == ("x",)
    assert mass_sum.shape_bins == (3,)
    assert mass_sum.provenance == {}

    fig, ax = plt.subplots()
    lines = mass_sum.plot(ax=ax)
    assert len(lines) == 1
    plt.close(fig)


def test_family_subresult_and_gas_fraction() -> None:
    sim = pynbody.new(gas=2, dm=4)
    sim["x"] = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    sim["mass"] = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)

    gas_bins = bins.gas

    assert isinstance(gas_bins, SubBinNDResult)
    np.testing.assert_array_equal(gas_bins["count"], [2, 0, 0])
    np.testing.assert_allclose(gas_bins["mass.sum"], [3, np.nan, np.nan])
    np.testing.assert_allclose(bins["gas_fraction"], [1, 0, 0])


def test_register_custom_bin_algorithm() -> None:
    sim = make_sim()

    @register_bin_algorithm("half_split_for_test", overwrite=True)
    def half_split(values, nbins, vmin, vmax) -> np.ndarray:
        assert nbins == 2
        return np.array([vmin, 0.5 * (vmin + vmax), vmax])

    bins = Bin1D("x", vmin=0, vmax=6, nbins=2, mode="half_split_for_test")(sim)

    np.testing.assert_array_equal(bins["count"], [3, 3])
    with pytest.raises(KeyError):
        register_bin_algorithm("half_split_for_test", half_split)


def test_register_bin_derived_with_condition() -> None:
    sim = make_sim()

    @Bin1D.derived("x_span_for_test", condition=has_axis({"x"}), overwrite=True)
    def x_span(result) -> np.ndarray:
        axis = result.find_axis({"x"})
        return np.full(result.nbins, float(np.asarray(axis.maxs)[-1] - np.asarray(axis.mins)[0]))

    x_bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)
    r_bins = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)

    assert "x_span_for_test" in x_bins.queries.names()
    assert "x_span_for_test" not in r_bins.queries.names()
    np.testing.assert_allclose(x_bins["x_span_for_test"], [6, 6, 6])
    with pytest.raises(KeyError):
        r_bins["x_span_for_test"]
