"""Typed scientific catalogue identities, independent of snapshot-view identities."""

from __future__ import annotations

from typing import Any

from pynbodyext.core.calculate.store.identity import sim_identity_key

CatalogueObjectKey = str | int | bool | float


def catalogue_identity_key(
    root_signature: tuple[Any, ...], catalogue_id: str, finder: str, revision: str | None, loader_id: str
) -> str:
    """Include the exact root and all labels, including an absent revision."""
    for name, value in (("catalogue_id", catalogue_id), ("finder", finder), ("loader_id", loader_id)):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if revision is not None and (type(revision) is not str or not revision.strip()):
        raise ValueError("revision must be None or a non-empty string")
    return sim_identity_key(("pynbodyext.orm.catalogue/v1", root_signature, catalogue_id, finder, revision, loader_id))


def typed_object_key(value: CatalogueObjectKey) -> str:
    """Preserve built-in scalar types; signed floating zeros share one key."""
    if type(value) not in (str, int, bool, float):
        raise TypeError("catalogue object keys require built-in str, int, bool or finite float")
    return sim_identity_key((value,))


def validate_kind(kind: str) -> None:
    if type(kind) is not str or kind not in ("halo", "subhalo"):
        raise ValueError("catalogue object kind must be halo or subhalo")
