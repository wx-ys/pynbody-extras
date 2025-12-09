
import numpy.testing as npt
import numpy as np
import pytest
from pynbody.array import SimArray
from pynbody.filt import FamilyFilter
from pynbody.transformation import GenericTranslation

from pynbodyext.chunk import DASK_AVAILABLE
from pynbodyext.calculate import CalculatorBase
from pynbodyext.util.perf import PerfStats


class SumMass(CalculatorBase[SimArray]):
    def calculate(self, sim):
        return sim['mass'].sum()

def test_calculate_init(snap):
    snap
    calc = SumMass()

    assert calc._filter is None
    assert calc._transformation is None
    assert calc._revert_transformation is True
    
    assert isinstance(calc._perf_stats, PerfStats)
    
    assert calc._enable_eval_cache is True
    
    assert calc._enable_chunk is False
    assert isinstance(calc._chunk_size, int)
    
    assert calc._cached_signature is None
    

def test_calculate_enable():
    calc = SumMass()

    calc.enable_cache(False)
    assert calc._enable_eval_cache is False
    
    calc.enable_chunk(chunk_size=2_000_000)
    assert calc._enable_chunk is True
    assert calc._chunk_size == 2_000_000
    
    calc.enable_perf(time=True, memory=True)
    
    assert calc._perf_stats.memory_enabled is True
    assert calc._perf_stats.time_enabled is True


def test_calculate_signature():
    calc = SumMass()
    calc.instance_signature()
    calc.signature()
    
    assert calc._cached_signature is not None
    calc.with_filter(None)
    assert calc._cached_signature is None
    
    calc.signature()
    assert calc._cached_signature is not None
    calc.with_transformation(None)
    assert calc._cached_signature is None



def test_calculate_formatting():
    calc = SumMass()
    calc.children()
    calc.calculate_children()
    
    calc.format_flow()
    calc.format_tree()
    

def test_calculate_basic_use(snap):
    
    # basic use
    calc = SumMass()
    total_mass = calc(snap)
    npt.assert_allclose(total_mass, 7632.60595703125)
    assert isinstance(total_mass, SimArray)
    assert total_mass.units is snap['mass'].units
    assert total_mass.sim is snap
    total_mass.in_units("Msol")
    
    
    # with filter
    calc.with_filter(FamilyFilter("star"))
    star_mass = calc(snap)
    npt.assert_allclose(star_mass, 9.930469512939453)
    assert isinstance(star_mass, SimArray)
    assert star_mass.units is snap['mass'].units
    assert star_mass.sim is snap
    
    # with transformation
    calc.with_transformation(lambda s: GenericTranslation(s, "pos", [10, -10, 0]))
    pre_pos = snap['pos'].view(np.ndarray)
    star_mass = calc(snap)
    assert isinstance(star_mass, SimArray)
    assert star_mass.units is snap['mass'].units
    assert star_mass.sim is snap
    post_pos = snap['pos'].view(np.ndarray)
    npt.assert_allclose(pre_pos, post_pos)


def test_combined_calculate(snap):
    calc1 = SumMass()
    calc2 = SumMass().with_filter(FamilyFilter("star")).with_transformation(
        lambda s: GenericTranslation(s, "pos", [10, -10, 0])
    )
    
    total_mass, star_mass = (calc1 & calc2)(snap)
    npt.assert_allclose(total_mass, 7632.60595703125)
    npt.assert_allclose(star_mass, 9.930469512939453)
    
    (calc1 & calc2).format_tree()
    (calc1 & calc2).format_flow()
    
    m1, m2, m3 = (calc1 & (calc2 & calc1))(snap)
    npt.assert_allclose(m1, 7632.60595703125)
    npt.assert_allclose(m2, 9.930469512939453)
    npt.assert_allclose(m3, 7632.60595703125)


@pytest.mark.skipif(not DASK_AVAILABLE, reason="Dask not available")
def test_chunked_calculate(snap):
    calc = SumMass().enable_chunk(chunk_size=1_000_000)
    star_mass = calc(snap)
    npt.assert_allclose(star_mass, 7632.6, rtol=1e-4)
    assert isinstance(star_mass, SimArray)
    assert star_mass.units.__repr__() == snap['mass'].units.__repr__()
    assert star_mass.sim is snap
