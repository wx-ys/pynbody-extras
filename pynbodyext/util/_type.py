""" Some type utilities for pynbodyext """

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, TypeAlias, TypeVar, Union, runtime_checkable

#Protocol need python version >3.8
import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.family import Family
from pynbody.filt import Filter
from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation
from pynbody.units import UnitBase

try:
    from typing import TypeVarTuple, Unpack  # python >= 3.11
except ImportError:
    from typing_extensions import TypeVarTuple, Unpack  # type: ignore

if TYPE_CHECKING:
    from pynbodyext.profiles.bins import BinsSet
    from pynbodyext.profiles.proarray import ProfileArray
    from pynbodyext.profiles.profile import ProfileBase

__all__ = ["Self","TypeVarTuple","Unpack"]


__all__ += ["UnitLike","SingleElementArray","SimNpArray","FilterLike","TransformLike","SimCallable","SimNpArrayFunc","SignatureProvider"]
# ---------------- General---------------------------------

# can be convert to Unit
UnitLike: TypeAlias = UnitBase | str
"""A type alias representing values that can be converted to a pynbody Unit."""

# Means this can be converted into scarler
SingleElementArray: TypeAlias = np.ndarray | SimArray | IndexedSimArray
"""A type alias representing arrays that contain a single element."""


SimNpArray: TypeAlias = SimArray | IndexedSimArray | np.ndarray
"""A type alias representing simulation or numpy arrays."""

# This type, SimSnap[FilterLike] -> SimSnap
FilterLike: TypeAlias = Filter | np.ndarray | Family | slice | int | np.int32 | np.int64
"""A type alias representing valid filters for simulation snapshots."""


# TransformLike(sim) -> Transformation
TransformLike: TypeAlias = Callable[[SimSnap], Transformation]
"""A type alias representing a callable that takes a simulation snapshot and returns a Transformation."""


_ReturnT = TypeVar("_ReturnT",covariant=True)
class SimCallable(Protocol[_ReturnT]):
    """A protocol representing a callable that takes a simulation snapshot and returns a value of type _ReturnT."""
    def __call__(self, sim: SimSnap, *args: Any, **kwargs: Any) -> _ReturnT: ...


SimNpArrayFunc: TypeAlias = SimCallable[SimNpArray]
"""A type alias representing a callable that takes a simulation snapshot and returns a simulation or numpy array."""

@runtime_checkable
class SignatureProvider(Protocol):
    """
    Duck-typed protocol: any object with a callable .signature() matches this.
    Use runtime_checkable so isinstance(obj, SignatureProvider) works.
    """
    def signature(self) -> object: ...



__all__ += ["BinByFunc","BinsAreaFunc","BinsAlgorithmFunc",
            "RegistBinByString","RegistBinAreaString","RegistBinAlgorithmString"]
# ------------------Profile---------------------------------
BinByFunc: TypeAlias = Callable[[SimSnap], SimNpArray]
"""Callable extracting a 1D array from a simulation."""

BinsAreaFunc: TypeAlias = Callable[["BinsSet", SimNpArray], SimNpArray]
"""Callable computing per-bin area or volume given bin edges."""

BinsAlgorithmFunc: TypeAlias = Callable[["BinsSet", SimNpArray], SimNpArray]
"""Callable constructing bin edges from raw data (returns length nbins+1)."""

RegistBinByString: TypeAlias = str | Literal["r","rxy"]
"""A type alias representing valid strings for binning by."""

RegistBinAreaString: TypeAlias = str | Literal["length","annulus","cylindrical_shell","spherical_shell"]
"""A type alias representing valid strings for binning area."""

RegistBinAlgorithmString: TypeAlias = str | Literal["lin","log","equaln"]
"""A type alias representing valid strings for binning algorithms."""


SimNpPrArray: TypeAlias = Union[SimNpArray, "ProfileArray"]
"""A type alias representing simulation arrays, numpy arrays, or profile arrays."""

SimNpPrArrayFunc: TypeAlias = SimCallable[SimNpPrArray]

ProfileType = TypeVar("ProfileType", bound = "ProfileBase")
SimNpPrArrayPrFunc: TypeAlias = Callable[[ProfileType], SimNpPrArray]



__all__ += ["has_signature","get_signature_safe"]
# ---------------helpful functions---------------------

def has_signature(obj: object) -> bool:
    """
    Return True if obj appears to provide a signature() call.
    Uses runtime Protocol check or presence of a callable attribute.
    """
    if isinstance(obj, SignatureProvider):
        return True
    # fallback: attribute exists and is callable
    return callable(getattr(obj, "signature", None))

def get_signature_safe(obj: object, *, fallback_to_id: bool = True) -> object | None:
    """
    Try to obtain obj.signature() in a safe way.
    - If obj has a signature() callable, call it and return the result.
    - If calling raises, return ("id", id(obj)) when fallback_to_id True, else return None.
    - If obj has no signature(), return None or id fallback.
    """
    # None -> None (no signature)
    if obj is None:
        return None

    # Immutable built-ins: return as-is (value semantics)
    if isinstance(obj, (str, bytes, bool, int, float)):
        return obj

    try:
        if isinstance(obj, np.generic):
            return obj.item()

        sig_fn = getattr(obj, "signature", None)
        if callable(sig_fn):
            return sig_fn()

    except Exception:
        return ("id", id(obj)) if fallback_to_id else None
    return ("id", id(obj)) if fallback_to_id else None
