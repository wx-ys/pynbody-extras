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
