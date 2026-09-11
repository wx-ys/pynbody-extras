import numpy as np
import pynbody

from pynbodyext.core.calculate import Bin1D


def make_sim():
    sim = pynbody.new(dm=6)
    sim["r"] = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=float)
    sim["mass"] = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    return sim


def test_geometry_centers_and_measure():
    sim = make_sim()
    bins = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)

    geo = bins._geometry
    np.testing.assert_allclose(geo.centers, [1, 3, 5])
    np.testing.assert_allclose(geo.measure, bins["measure"])


def test_measure_override_is_per_instance():
    sim = make_sim()
    a = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)
    b = Bin1D("r", vmin=0, vmax=6, nbins=3)(sim)
    b.axes.set_measure_type("r", "linear")
    assert not np.allclose(a["measure"], b["measure"])


def test_model_subresult_tree():
    from pynbodyext.core.calculate.bins.model import BinResultModel

    sim = pynbody.new(gas=2, dm=4)
    sim["x"] = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    sim["mass"] = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)

    root = bins.model
    gas = root.gas
    assert isinstance(gas, BinResultModel)
    assert gas.parent is root
    assert gas.owner is bins.gas
    np.testing.assert_allclose(gas["count"], [2, 0, 0])

    mask = np.array([True, True, False, False, False, False])
    sub = root[mask]
    assert isinstance(sub, BinResultModel)
    assert sub is bins[mask]._model
    assert sub.parent is root
