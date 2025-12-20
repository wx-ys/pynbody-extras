from importlib.metadata import version
from importlib.util import find_spec

__all__ = ["GRAVITY_RUST_AVAILABLE", "DASK_AVAILABLE", "module_available", "PYNBODY_VERSION"]




def module_available(name: str) -> bool:
    """Return True if a module is importable."""
    return find_spec(name) is not None


GRAVITY_RUST_AVAILABLE: bool = module_available("pynbodyext._rust")

DASK_AVAILABLE: bool = module_available("dask")


PYNBODY_VERSION = version("pynbody")
