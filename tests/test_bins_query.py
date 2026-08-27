from __future__ import annotations


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
