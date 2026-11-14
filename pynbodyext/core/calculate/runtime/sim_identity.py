"""Pluggable simulation identity for provenance, caching and persistence.

The calculator engine keys in-run cache entries, provenance and stored results on
a simulation identity tuple.  Its first element says how far the identity can be
trusted:

``("sim", ...)``
    A *stable* address: the same simulation object, loaded again in another
    session, gets the same tuple.  Only stable identities may key a persistent
    result store.
``("session", id(obj))``
    Valid in this process only: the default :func:`id_based_sim_identity`, and the
    answer of :func:`snapshot_identity` for anything it cannot address durably.

:func:`snapshot_identity` is the stable provider for pynbody snapshots:

- a snapshot loaded from a file → ``("sim", <absolute path>)``;
- a halo of it → ``("sim", <path>, "halo", <number>, <digest>)``;
- any other sub-snapshot (a filter, a slice) → ``("sim", <path>, "subset", <digest>)``.

The *digest* is a hash of the particle indices the sub-snapshot selects from its
file, so a halo is matched by its particles and not only by a catalogue number
that a re-run halo finder may reuse.

What the identity cannot see, and therefore makes session-only: a snapshot with no
file (``pynbody.new``), and a snapshot with a pynbody transformation still applied
(``sim.translate``, ``analysis.center``, a rotation) — its positions are no longer
the file's.  Apply such transformations inside the calculator graph instead
(``ShiftPosTo``, ``AlignVec`` …), where they are part of the calculator signature.
Editing arrays in place (``sim["mass"] *= 2``) is not tracked at all.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Protocol, TypeAlias

import numpy as np

from pynbodyext.core.calculate.interfaces import SESSION_TAG, STABLE_TAG, is_stable_sim_identity

__all__ = [
    "SimIdentity",
    "SimIdentityProvider",
    "SESSION_TAG",
    "STABLE_TAG",
    "id_based_sim_identity",
    "snapshot_identity",
    "is_stable_sim_identity",
]

class SimIdentityProvider(Protocol):
    """Maps a simulation object to an identity tuple.

    Stores and SampleTable accept built-in str, bool, int, finite float, None
    and nested tuples. Convert NumPy scalar IDs explicitly. Identity component
    types are significant, unlike calculator-parameter numerical equivalence.
    """

    def __call__(self, sim: Any) -> tuple[Any, ...]: ...


#: Concrete identity tuple produced by a provider: a tag (:data:`STABLE_TAG` or
#: :data:`SESSION_TAG`) followed by the address.
SimIdentity: TypeAlias = tuple[Any, ...]


def id_based_sim_identity(sim: Any) -> tuple[Any, ...]:
    """Default provider: the object's in-process ``id``.

    Unique within a process and free to compute, so it is right for the transient
    per-run cache; tagged :data:`SESSION_TAG` because it means nothing after a
    reload.
    """
    return (SESSION_TAG, id(sim))


def _is_file_name(filename: Any) -> bool:
    """Whether *filename* names a file, not a placeholder such as ``"<created>"``.

    ``pynbody.new`` names its snapshots ``"<created>"``.  Existence is not checked:
    a multi-file snapshot is named by a prefix that is not itself a file.
    """
    if not filename:
        return False
    text = os.fspath(filename)
    return not (text.startswith("<") and text.endswith(">"))


def _has_transformations(obj: Any) -> bool:
    return bool(getattr(obj, "_transformations", None))


def _index_digest(sim: Any, ancestor: Any) -> str:
    """A short hash of the particle indices *sim* selects from *ancestor*."""
    return particle_index_digest(sim.get_index_list(ancestor))


def particle_index_digest(indices: Any) -> str:
    """Canonical digest shared by live views and captured replay definitions."""
    array = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def snapshot_identity(sim: Any) -> tuple[Any, ...]:
    """Stable provider for pynbody snapshots: file path, plus halo or particle subset.

    Falls back to :func:`id_based_sim_identity` (a session-only identity) for a
    snapshot without a file or with a pynbody transformation applied; see the
    module documentation.
    """
    ancestor = getattr(sim, "ancestor", sim)
    filename: Any = getattr(ancestor, "filename", None)
    if not _is_file_name(filename):
        return id_based_sim_identity(sim)
    if _has_transformations(sim) or _has_transformations(ancestor):
        return id_based_sim_identity(sim)

    path = os.path.abspath(os.fspath(filename))
    return _snapshot_view_identity(sim, ancestor, (STABLE_TAG, path))


def _snapshot_view_identity(sim: Any, ancestor: Any, base: tuple[Any, ...]) -> tuple[Any, ...]:
    """Append the existing halo/subset address to a path or logical root identity."""
    if sim is ancestor:
        return base

    digest = _index_digest(sim, ancestor)
    halo_number = getattr(sim, "_halo_number", None)
    if halo_number is not None:
        number = halo_number.item() if isinstance(halo_number, np.generic) else halo_number
        return base + ("halo", number, digest)
    return base + ("subset", digest)
