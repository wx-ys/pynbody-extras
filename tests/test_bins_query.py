from __future__ import annotations

import numpy as np


def test_model_derives_shape_and_root() -> None:
    from pynbodyext.core.calculate.bins.model import BinResultModel

    class FakeCalc:
        pass

    model = BinResultModel(
        sim="s",
        source_sim="ss",
        axes=(),
        bin_data=None,
        bin_indptr=None,
        particle_bin=None,
        valid_mask=None,
        calculator=FakeCalc(),
        scope_signature=None,
        parent=None,
    )
    assert model.ndim == 0
    assert model.shape_bins == ()
    assert model.nbins == 1
    assert model.is_root is True
    assert model.root is model


def test_registry_shell_registers_and_gets_derived() -> None:
    from pynbodyext.core.calculate.bins.registries import BinsRegistry

    reg = BinsRegistry()

    @reg.register_derived("double")
    def double(bins):
        return [1, 2, 3]

    assert "double" in reg.derived_keys()


def test_query_cache_typed_keys_and_invalidation() -> None:
    from pynbodyext.core.calculate.bins.query import DerivedKey, QueryCache

    cache = QueryCache()
    cache.put(DerivedKey("derived", "mass.sum.density"), "arr")
    assert cache.get(DerivedKey("derived", "mass.sum.density")) == "arr"
    assert cache.num_cached == 1

    cleared, names = cache.invalidate_measure_dependent()
    assert cleared == 1
    assert names == ["mass.sum.density"]
    assert cache.num_cached == 0


def test_a_weighted_percentile_of_one_particle_is_that_particle() -> None:
    """A one-point distribution's percentile is the point, not a NaN."""
    from pynbodyext.core.calculate.bins.statistics import Median, Percentile

    values = np.array([42.0])
    weights = np.array([0.25])

    assert Percentile("p16", 16.0)(values, weights) == 42.0
    assert Median("median")(values, weights) == 42.0
    assert np.isnan(Percentile("p16", 16.0)(values, np.zeros(1)))


def test_vectorised_percentiles_agree_with_the_statistic_row_by_row() -> None:
    """``weighted_percentiles`` is the statistics' one definition, done in bulk."""
    import pytest

    from pynbodyext.core.calculate.bins.statistics import Percentile, weighted_percentiles

    rng = np.random.default_rng(3)
    values = rng.normal(size=(200, 17))
    weights = rng.uniform(0.0, 2.0, size=(200, 17))
    weights[rng.random(weights.shape) < 0.3] = 0.0  # particles outside the support
    values[0] = np.nan  # nothing usable
    values[1, 1:] = np.nan  # one usable point
    weights[2, 1:] = 0.0  # one positive weight

    for percentile in (0.0, 16.0, 50.0, 84.0, 100.0):
        statistic = Percentile(f"p{percentile:g}", percentile)
        bulk = weighted_percentiles(values, weights, percentile)
        row_by_row = np.array([statistic(values[i], weights[i]) for i in range(len(values))])

        np.testing.assert_allclose(bulk, row_by_row, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert isinstance(weighted_percentiles(values[3], weights[3], 50.0), float)
    with pytest.raises(ValueError, match="same shape"):
        weighted_percentiles(np.ones(3), np.ones(4), 50.0)
