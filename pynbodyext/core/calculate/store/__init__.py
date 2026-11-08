"""Result persistence seam for the calculator system.

This subpackage defines the store boundary that a database-backed adapter
(the tangos-like :mod:`~pynbodyext.core.calculate.store.sqlalchemy_store`)
implements.  The public surface is small:

- :class:`~pynbodyext.core.calculate.store.codec.ValueCodec` — portable value
  serialization (numpy, pynbody arrays, scalars, nested structures).
- :class:`~pynbodyext.core.calculate.store.base.ResultStore` — the abstract
  persistence interface keyed on simulation + calculator identity. Saves append
  immutable versions; current-cache lookup selects the latest saved version.
- :class:`~pynbodyext.core.calculate.store.base.RecordRef` — a current-cache
  address or a pinned version returned by ``store()`` / ``Result.store_ref``.
- :class:`~pynbodyext.core.calculate.store.base.InMemoryResultStore` — the
  reference in-memory adapter.
- :class:`~pynbodyext.core.calculate.store.sqlalchemy_store.SQLAlchemyResultStore`
  — a relational (SQLite-default) adapter; SQLAlchemy is an optional dependency
  imported lazily, so the rest of the package works without it.
"""

from .base import (
    RESULT_STORE_SCHEMA,
    InMemoryResultStore,
    RecordRef,
    ResultRecord,
    ResultStore,
    SignatureMismatchError,
    UnstableSignatureError,
    UnstableSimIdentityError,
    compute_calculator_key,
)
from .catalog import CatalogEntry, SignatureCatalog
from .codec import ValueCodec
from .sqlalchemy_store import SQLAlchemyResultStore

__all__ = [
    "RESULT_STORE_SCHEMA",
    "ValueCodec",
    "RecordRef",
    "ResultRecord",
    "ResultStore",
    "InMemoryResultStore",
    "SQLAlchemyResultStore",
    "compute_calculator_key",
    "UnstableSignatureError",
    "UnstableSimIdentityError",
    "SignatureMismatchError",
    "SignatureCatalog",
    "CatalogEntry",
]
