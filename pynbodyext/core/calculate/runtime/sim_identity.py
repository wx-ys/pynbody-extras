"""Pluggable simulation identity for provenance and persistence.

The calculator engine keys in-run cache entries and stored provenance on a
simulation identity tuple.  The default provider is object-identity based,
which is correct for the transient in-run cache but useless across processes or
notebook sessions: ``id(sim)`` changes every time a snapshot is reloaded.

A persistence-aware caller (e.g. a database-backed result store) injects a
stable identity provider that maps a snapshot to an address-based reference such
as ``("sim", "/path/snap_103", "halo_0")``.  This module defines the provider
``Protocol`` together with the default id-based implementation.

Threading
---------
The engine receives a provider at construction time
(:meth:`~pynbodyext.core.calculate.runtime.engine.EvalEngine`); ``calculator.run``
keeps the default.  The stored store factory that wants a stable identity should
construct its own ``EvalEngine`` with its provider (or set the provider on an
engine it owns).
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

__all__ = ["SimIdentity", "SimIdentityProvider", "id_based_sim_identity"]


class SimIdentityProvider(Protocol):
    """Maps a simulation object to a stable identity tuple."""

    def __call__(self, sim: Any) -> tuple[Any, ...]: ...


#: Concrete identity tuple produced by a provider.  The first element is
#: conventionally the provider tag (e.g. ``"sim"``); trailing elements carry the
#: address used to re-fetch the snapshot/object.
SimIdentity: TypeAlias = tuple[Any, ...]


def id_based_sim_identity(sim: Any) -> tuple[Any, ...]:
    """Default provider: identity based on the object's in-process id.

    Unique within a process and cheap to compute, so it is correct for the
    transient per-run cache.  Not stable across loads, so it is unsuitable on
    its own for persistence.
    """
    return ("sim", id(sim))
