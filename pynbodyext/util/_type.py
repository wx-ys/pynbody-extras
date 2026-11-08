"""Some type utilities for pynbodyext"""

import importlib
from collections.abc import Callable
from typing import Any, Protocol, Self, TypeAlias, TypeVar, TypeVarTuple, Unpack, dataclass_transform, runtime_checkable

# Protocol need python version >3.8
import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.family import Family
from pynbody.filt import Filter
from pynbody.snapshot import SimSnap
from pynbody.transformation import Transformation
from pynbody.units import UnitBase

__all__ = ["Self", "TypeVarTuple", "Unpack", "dataclass_transform"]


__all__ += [
    "UnitLike",
    "SingleElementArray",
    "SimNpArray",
    "FilterLike",
    "TransformLike",
    "SimCallable",
    "SimNpArrayFunc",
    "SignatureProvider",
]
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


_ReturnT = TypeVar("_ReturnT", covariant=True)


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


# ---------------annotation name registry----------------

_ANNOTATION_NAMES: dict[str, object] = {}


def register_annotation_names(**names: object) -> None:
    """Make *names* resolvable in this package's deferred annotations.

    Modules here annotate with types they deliberately do not import (PEP 563 keeps
    the annotation a string, which is what keeps the import graph light).
    ``typing.get_type_hints`` then needs a namespace to resolve those strings
    against, and the module that *owns* a name is the one that supplies it: the
    binning layer registers its result types here so the image layer can resolve
    annotations that mention them without importing the calculator layer
    (ADR-0004).
    """
    _ANNOTATION_NAMES.update(names)


def annotation_names() -> dict[str, object]:
    """The names registered by :func:`register_annotation_names`."""
    return dict(_ANNOTATION_NAMES)


__all__ += ["register_annotation_names", "annotation_names"]


__all__ += ["has_signature", "get_signature_safe"]
# ---------------helpful functions---------------------


def class_path(obj: object) -> str:
    """The dotted path of a class, or of the class of an instance.

    ``class_path(SomeCalculator)`` and ``class_path(SomeCalculator())`` are both
    ``"pynbodyext....SomeCalculator"``.  Signatures are keyed on this, so it is
    written once: a payload that names a class must name it the way every other
    module spells it.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def import_object(path: str) -> Any:
    """Import and return an object by its fully-qualified dotted path.

    The inverse of :func:`class_path`, for the modules that rebuild a calculator
    from a stored signature.
    """
    module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"invalid object path: {path!r}")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


__all__ += ["class_path", "import_object"]


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
