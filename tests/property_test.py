

import numpy as np
import numpy.testing as npt

from pynbodyext.properties import (
    ParameterContain, ParamSum, KappaRot, VolumeDensity, SurfaceDensity
)


def test_param_sum(snap):
    ps = ParamSum("mass")
    total_mass = ps(snap)
    assert total_mass.sim is snap
    

def test_contain(snap):
    re = ParameterContain("r", 0.5, "mass")
    
    re_val = re(snap)
    assert re_val.sim is snap
    npt.assert_allclose(re_val, 9.8005476)
    
    # mutiple fractions
    re_multi = ParameterContain("r", [0.1, 0.5, 0.9], "mass")
    re_vals = re_multi(snap)
    assert re_vals.sim is snap
    assert np.gradient(re_vals).all() > 0


def test_kappa_rot(snap):
    kappa = KappaRot()
    kappa_val = kappa(snap)
    npt.assert_allclose(kappa_val, 0.39817017)

def test_volume_density(snap):
    rho = VolumeDensity(10, "mass")
    rho_val = rho(snap)
    npt.assert_allclose(rho_val, 1.8221505)
    assert rho_val.units == snap["mass"].units / snap["pos"].units**3

def test_surface_density(snap):
    sigma = SurfaceDensity(10, "mass")
    sigma_val = sigma(snap)
    npt.assert_allclose(sigma_val, 19.229359)
    assert sigma_val.units == snap["mass"].units / snap["pos"].units**2




def test_operations(snap):
    re = ParameterContain("r", 0.5, "mass")
    re_val = re(snap)

    re2_1 = 2*re
    re2_2 = re+re
    npt.assert_allclose(re2_1(snap), re2_2(snap))
    npt.assert_allclose(re2_1(snap), 2*re_val)
    
    re3 = re**3
    re3_val = re3(snap)
    npt.assert_allclose(re3_val, re_val**3)
    assert re3_val.units == re_val.units**3
    
    zero = re - re 
    zero_val = zero(snap)
    npt.assert_allclose(zero_val, 0)
    assert zero_val.units == re_val.units
    
    re_clip = re.clip(0, 1)
    re_clip_val = re_clip(snap)
    npt.assert_allclose(re_clip_val, min(1, re_val))
    assert re_clip_val.units == re_val.units
    














