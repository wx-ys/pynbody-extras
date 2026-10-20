import importlib
from functools import lru_cache
from importlib.metadata import version
from importlib.util import find_spec

__all__ = [
    "GRAVITY_NATIVE_AVAILABLE",
    "NATIVE_MODULE",
    "NativeExtensionWarning",
    "POWERBIN_AVAILABLE",
    "PYNBODY_VERSION",
    "module_available",
    "native_import_error",
    "native_unavailable_reason",
]


def module_available(name: str) -> bool:
    """Return True if a module can be found (it may still fail to import)."""
    return find_spec(name) is not None


#: The compiled C++ extension: gravity solver and SPH scatter engine.
NATIVE_MODULE = "pynbodyext._native"


class NativeExtensionWarning(UserWarning):
    """The compiled ``pynbodyext._native`` extension is missing or cannot be loaded.

    A :class:`UserWarning`, so it is shown by default.  Silence it with
    ``warnings.filterwarnings("ignore", category=NativeExtensionWarning)``.
    """


@lru_cache(maxsize=1)
def native_import_error() -> ImportError | None:
    """The error raised by ``import pynbodyext._native``, or ``None`` if it imports.

    Availability means "imports", not "a file exists": a ``_native*.so`` that is
    present but cannot be loaded (wrong architecture, a missing OpenMP runtime,
    an ABI mismatch) counts as unavailable.  The result is cached.
    """
    try:
        importlib.import_module(NATIVE_MODULE)
    except ImportError as exc:
        return exc
    return None


def native_unavailable_reason() -> str:
    """Why ``pynbodyext._native`` is unavailable, with how to fix it."""
    exc = native_import_error()
    rebuild = (
        "Rebuild it with `python setup.py build_ext --inplace` (or reinstall pynbodyext) "
        "and check the build output; see the README for compiler requirements."
    )
    if exc is None:
        return f"{NATIVE_MODULE} is reported unavailable."
    if isinstance(exc, ModuleNotFoundError) and exc.name == NATIVE_MODULE:
        return f"{NATIVE_MODULE} is not built. {rebuild}"
    return f"{NATIVE_MODULE} is present but failed to load ({type(exc).__name__}: {exc}). {rebuild}"


GRAVITY_NATIVE_AVAILABLE: bool = native_import_error() is None

# Adaptive image binning (`pynbodyext.plot.image.adaptive`).
POWERBIN_AVAILABLE: bool = module_available("powerbin")


PYNBODY_VERSION = version("pynbody")
