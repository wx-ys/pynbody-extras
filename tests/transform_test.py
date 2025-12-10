

import numpy as np
import numpy.testing as npt

from pynbodyext.transforms import (
    PosToCenter, VelToCenter, AlignAngMomVec, WrapBox
)

def test_pos_to_center(snap):
    transform = PosToCenter(mode="ssc")
    trans = transform(snap)
    
def test_wrap_box(subfind):
    halo = subfind.halos()[0]
    wrap = WrapBox()
    wrapped = wrap(halo)

def test_vel_to_center(subfind):
    halo = subfind.halos()[0]
    transform = VelToCenter(mode="com").with_transformation(PosToCenter(mode="ssc"))
    tran = transform(halo)
    
def test_align_angmom_vec(subfind):
    halo = subfind.halos()[0]
    transform = AlignAngMomVec.chain(
        PosToCenter(mode="ssc"),
        VelToCenter(mode="com"),
        AlignAngMomVec()
    )
    tran = transform.enable_perf()(halo)