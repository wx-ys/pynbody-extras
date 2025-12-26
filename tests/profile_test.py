
import numpy as np
import numpy.testing as npt

from pynbodyext.profiles import RadialProfileBuilder, StarAgeProfile



def test_radial_profile_builder(snap):
    radial = RadialProfileBuilder(weight="mass")
    
    pr = radial(snap)
    pr.s
    pr.dm
    pr.gas
    pr[:10000]
    pr[snap["x"]<5]
    pr.s['density']
    pr.g["temp"]["med"]
    npt.assert_allclose(pr.g["temp"]["med"] , pr.g["temp_p50"])
    pr.particles_at_bin[:2]
    
    assert len(pr.particles_at_bin[3]) == (len(pr.s.particles_at_bin[3]) + len(pr.g.particles_at_bin[3]) + len(pr.dm.particles_at_bin[3]))
    assert len(pr.particles_at_bin[:3]) == len(pr.particles_at_bin[0,1,2]) 
    assert len(pr.particles_at_bin[:3]) == len(pr.particles_at_bin[np.array([True,True,True]+[False]*(pr.nbins-3))])


def test_time_profile(snap):
    pr = StarAgeProfile(snap,bins_type="equaln")
    pr['sfr']