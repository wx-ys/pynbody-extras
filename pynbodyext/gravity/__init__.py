"""
pynbodyext.gravity — public exports for gravity helpers.

This package exposes the core Gravity helper (TreeBH + direct-summation)
and two high-level convenience functions that operate on pynbody Snapshots.

It also supports optional gravitational softening (scalar or per-particle)
with selectable softening kernels via ``KernelKind``.

Without a working ``pynbodyext._native`` extension, only
``GRAVITY_NATIVE_AVAILABLE`` is exported and importing this package emits a
:class:`~pynbodyext.util.deps.NativeExtensionWarning` that says why.

Example
-------
>>> from pynbodyext.gravity import calculate_acceleration, KernelKind
>>> a = calculate_acceleration(sim, method="tree", theta=0.7, softening=0.01, kernel=KernelKind.Plummer)
"""

from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE

__all__ = ["GRAVITY_NATIVE_AVAILABLE"]

if GRAVITY_NATIVE_AVAILABLE:
    from .base import Gravity, KernelKind
    from .pyn_gravity import calculate_acceleration, calculate_potential

    __all__ += ["Gravity", "KernelKind", "calculate_potential", "calculate_acceleration"]
else:
    import warnings

    from pynbodyext.util.deps import NativeExtensionWarning, native_unavailable_reason

    warnings.warn(
        f"pynbodyext.gravity is unavailable: {native_unavailable_reason()}", NativeExtensionWarning, stacklevel=2
    )
