import os
import pathlib
import sys
import time
import traceback
from functools import lru_cache

import numpy as np
import pynbody
import pynbody.test_utils

from pynbodyext.gravity import Gravity

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

testdata_dir = (
    _REPO_ROOT
    / "testdata"
    / "gadget3"
    / "data"
    / "subhalos_103"
    / "subhalo_103"
)
def _ensure_testdata_in_repo_root():
    """
    ASV may run benchmarks from a temp working directory.
    Ensure pynbody testdata is prepared under the repo root (./testdata).
    """
    old_cwd = os.getcwd()
    try:
        os.chdir(_REPO_ROOT)
        pynbody.test_utils.ensure_test_data_available("gadget", "arepo")
    finally:
        os.chdir(old_cwd)

@lru_cache(maxsize=2)
def _load_pos_mass(only_halo: bool = False):
    """
    Load the test snapshot once, and prepare BOTH:
      - full snapshot (pos, mass)
      - first halo (pos, mass)
    This avoids re-loading/re-slicing when ASV varies only_halo.
    """
    try:
        _ensure_testdata_in_repo_root()

        if not testdata_dir.parent.exists():
            raise FileNotFoundError(
                f"Testdata path does not exist: {testdata_dir}\n"
                f"Repo root: {_REPO_ROOT}\n"
                f"Current cwd: {os.getcwd()}\n"
                "Hint: run from repo root:\n"
                "  python -c \"import pynbody.test_utils as t; t.ensure_test_data_available('gadget','arepo')\""
            )

        data = pynbody.load(str(testdata_dir))  # Path -> str for compatibility
        if only_halo:
            data = data.halos()[0].load_copy()
        pos = np.asarray(data["pos"], dtype=np.float64, order="C")
        mass = np.asarray(data["mass"], dtype=np.float64, order="C")

        return pos, mass

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError("Failed to load pynbody testdata for ASV benchmark") from exc




# -----------------------
# ASV benchmark entrypoint
# -----------------------
class TimeGravityRealBaseData:
    """
    baenchmark pynbodyext.gravity.Gravity.tree_potentials() on real data from pynbody testdata.
    """
    pos, mass = _load_pos_mass(only_halo=True)
        
    def _construct_tree(self, **kwargs):
        grav = Gravity(
            self.pos,
            self.mass,
            softening=kwargs.get("softening", None),
            kernel=kwargs.get("kernel", None),
            leaf_capacity=kwargs.get("leaf_capacity", 8),
            multipole_order=kwargs.get("multipole_order", 0),
        )
        grav.tree

    def _construct_and_tree_potentials(self, **kwargs):
        grav = Gravity(
            self.pos,
            self.mass,
            softening=kwargs.get("softening", None),
            kernel=kwargs.get("kernel", None),
            leaf_capacity=kwargs.get("leaf_capacity", 8),
            multipole_order=kwargs.get("multipole_order", 0),
        )
        grav.tree_potentials(theta=kwargs.get("theta", 0.7), 
                             leaf_capacity=kwargs.get("leaf_capacity", 8),
                             multipole_order=kwargs.get("multipole_order", 0), 
                             kernel = kwargs.get("kernel", None))

class TimeTreeConstruct(TimeGravityRealBaseData):
    """
    baenchmark pynbodyext.gravity.Gravity._construct_tree() on real data from pynbody testdata.
    """
    params = [
              [8,32,128], #leaf_capacity
              [None, 0.288],  # softening
              [0,3,5],    # multipole order
    ]
    param_names = ["leaf_capacity","softening","multipole_order"]
    
    def time_construct_tree(self, leaf_capacity, softening, multipole_order):
        self._construct_tree(
            leaf_capacity=leaf_capacity,
            softening=softening,
            kernel=1,
            multipole_order=multipole_order,
        )

class TimeTreeGravityTheta(TimeGravityRealBaseData):
    """
    baenchmark pynbodyext.gravity.Gravity.tree_potentials() on real data from pynbody testdata.
    """
    params = [0.5, 0.7, 1.0]
    param_names = ["theta"]
    
    def time_construct_and_tree_potentials(self, theta):
        self._construct_and_tree_potentials(
            theta=theta,
        )

class TimeTreeGravityOrder(TimeGravityRealBaseData):
    """
    baenchmark pynbodyext.gravity.Gravity.tree_potentials() on real data from pynbody testdata.
    """
    params = [2, 3, 4, 5]
    param_names = ["multipole_order"]
    
    def time_construct_and_tree_potentials(self, multipole_order):
        self._construct_and_tree_potentials(
            multipole_order=multipole_order,
        )

class TimeTreeGravityFull(TimeGravityRealBaseData):
    """
    baenchmark pynbodyext.gravity.Gravity.tree_potentials() on real data from pynbody testdata.
    """
    
    def time_construct_and_tree_potentials(self):
        self._construct_and_tree_potentials(
            theta=0.7,
            softening=0.001,
            kernel=1,
            multipole_order=3,
        )


class TimeGravityRealSnapData(TimeGravityRealBaseData):
    """
    baenchmark pynbodyext.gravity.Gravity.tree_potentials() on real data from pynbody testdata.
    """
    pos, mass = _load_pos_mass(only_halo=False)



def main():
    bench = TimeGravityRealBaseData()
    theta = 0.7
    multipole_order = 3
    softening = 0.001
    kernel = 1
    
    n_repeat = 10
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        bench._construct_and_tree_potentials(
            theta=theta,
            softening=softening,
            kernel=kernel,
            multipole_order=multipole_order,
        )
    t1 = time.perf_counter()

    print(f"N={len(bench.pos)} repeat={n_repeat} avg={(t1 - t0) / n_repeat:.6f}s")


if __name__ == "__main__":
    main()
