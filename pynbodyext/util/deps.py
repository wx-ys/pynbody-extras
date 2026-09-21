from importlib.metadata import version
from importlib.util import find_spec

__all__ = ["GRAVITY_NATIVE_AVAILABLE", "POWERBIN_AVAILABLE", "module_available", "PYNBODY_VERSION"]


def module_available(name: str) -> bool:
    """Return True if a module is importable."""
    return find_spec(name) is not None


GRAVITY_NATIVE_AVAILABLE: bool = module_available("pynbodyext._native")

# Adaptive image binning (`pynbodyext.plot.image.adaptive`).
POWERBIN_AVAILABLE: bool = module_available("powerbin")


PYNBODY_VERSION = version("pynbody")
