
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


def test_time_profile(snap):
    pr = StarAgeProfile(snap,bins_type="equaln")
    pr['sfr']