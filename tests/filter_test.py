import numpy as np
#from pynbody.snapshot import SimSnap

import numpy as np
import numpy.testing as npt


from pynbodyext.filters import FilterBase, FamilyFilter, BandPass, HighPass, LowPass
from pynbodyext.properties.base import ParamSum


def test_filter_base(snap):
    f = FilterBase()
    assert len(snap[f]) == len(snap)


def test_family_filter(snap):
    stars = FamilyFilter("stars")
    gas = FamilyFilter("gas")

    either = stars | gas
    not_stars = ~stars
    both = stars & gas

    m_both = both(snap)
    snap[either]
    m_either = either(snap)
    m_not = not_stars(snap)

    # sanity: intersection subset of union
    assert np.all(m_both <= m_either)
    # star & ~star should be empty
    assert np.all(~(m_not & stars(snap)))

def test_band_pass(snap):

    bp1 = BandPass("x",0, 5)
    bp2 = BandPass("x",5,10)
    hp = HighPass("x",6)
    lp = LowPass("x",6)
    

    assert (len(snap[bp1]) + len(snap[bp2]))== len(snap) 
    
    assert (len(snap[hp]) + len(snap[lp]))== len(snap) 
    
    assert len(snap[bp1 & bp2]) == 0
    assert len(snap[bp1 | bp2]) == len(snap)
    
    assert len(snap[~bp1]) == len(snap[bp2])
    assert len(snap[~bp2]) == len(snap[bp1])
    
    assert len(snap[~hp]) == len(snap[lp])
    assert len(snap[lp & hp]) == 0
    assert len(snap[lp | hp]) == len(snap)


def test_band_pass_diff_inits(snap):

    bp1 = BandPass("x", 0, 10)
    bp2 = BandPass("x", 0, "10 Mpc")
    bp3 = BandPass("x", 0, lambda sim: sim['x'].max()+0.1)
    
    assert len(snap[bp1]) == len(snap[bp2]) == len(snap[bp3])


def test_filter_with_filter(snap):
    bp = BandPass("x", 0, 2)
    ff = FamilyFilter("star")
    
    combined = bp.with_filter(ff)
    assert combined._filter is None
    




