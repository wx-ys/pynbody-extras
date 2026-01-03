"""
pynbodyext.gravity — public exports for gravity helpers.

This package exposes the core Gravity helper (TreeBH + direct-summation)
and two high-level convenience functions that operate on pynbody Snapshots.

It also supports optional gravitational softening (scalar or per-particle)
with selectable softening kernels via ``KernelKind``.

Example
-------
>>> from pynbodyext.gravity import calculate_acceleration, KernelKind
>>> a = calculate_acceleration(sim, method="tree", theta=0.7, softening=0.01, kernel=KernelKind.Plummer)
"""
from pynbodyext.util.deps import GRAVITY_RUST_AVAILABLE

__all__ = ["GRAVITY_RUST_AVAILABLE"]

if GRAVITY_RUST_AVAILABLE:
    from .base import Gravity, KernelKind
    from .pyn_gravity import calculate_acceleration, calculate_potential

    __all__ += ["Gravity", "KernelKind", "calculate_potential", "calculate_acceleration"]
else:
    warning_msg = (
        "pynbodyext.gravity: Rust extension not available; "
        "gravity calculations will be unavailable."
    )
    import warnings
    warnings.warn(warning_msg, ImportWarning, stacklevel=2)
