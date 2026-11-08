"""SQLAlchemy-backed :class:`ResultStore` adapter (tangos-like relational store).

Implements the six abstract primitives from
:mod:`~pynbodyext.core.calculate.store.base` against a SQLAlchemy relational
database.  The full codec-encoded :class:`ResultRecord` is stored as a JSON
payload in a single column; the simulation and calculator identities are
additionally promoted to indexed columns so the store can be listed and filtered
by simulation. A separate current-cache table points to immutable version rows.
Each save appends a version and replaces the cache pointer in one transaction;
invalidation deletes only the pointer. No historical garbage collection is provided.

SQLAlchemy is an *optional* dependency: it is imported lazily inside the
constructor so that importing this module never requires it.  Install the
``store`` extra (``pip install pynbodyext[store]`` / ``uv add --extra store
pynbodyext``) before constructing an instance.

The :class:`ResultRecord.to_json`/``from_json`` helpers keep the adapter a
shallow *storage* layer: identity question are answered by the columns, the
record content round-trips through the same codec used everywhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .base import RecordRef, ResultRecord, ResultStore, SignatureMismatchError, _record_key, _sig_text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

_IMPORT_ERROR = (
    "SQLAlchemy is required for SQLAlchemyResultStore. "
    "Install it with `pip install pynbodyext[store]` or `uv add --extra store pynbodyext`."
)


class SQLAlchemyResultStore(ResultStore):
    """A :class:`ResultStore` persisted as SQLAlchemy relational rows.

    Parameters
    ----------
    url_or_engine : str | Engine
        A SQLAlchemy database URL (e.g. ``"sqlite:///results.sqlite"``) or an
        already-constructed :class:`~sqlalchemy.engine.Engine`.  SQLite is the
        tested target; other SQLAlchemy dialects need their own validation.
    codec : ValueCodec | None
        Value codec used to encode/decode result values (defaults to a shared
        :class:`ValueCodec`).
    create : bool, default: True
        Create ``result_versions`` and ``stored_results`` tables if absent.
    table_prefix : str, default: ``"pynbodyext_"``
        Prefix prepended to both table names, to avoid clashing with other tables
        in the same database.
    """

    def __init__(
        self, url_or_engine: str | Engine, *, codec: Any = None, create: bool = True, table_prefix: str = "pynbodyext_"
    ) -> None:
        super().__init__(codec)

        try:
            import sqlalchemy
            from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text
        except ImportError as exc:  # pragma: no cover - exercised only without sqlalchemy
            raise ImportError(_IMPORT_ERROR) from exc

        self._sa = sqlalchemy

        if hasattr(url_or_engine, "connect"):
            self.engine: Engine = cast("Engine", url_or_engine)
        else:
            self.engine = sqlalchemy.create_engine(str(url_or_engine))

        self._metadata = MetaData()
        self._versions = Table(
            table_prefix + "result_versions",
            self._metadata,
            Column("id", Integer, primary_key=True),
            Column("version_id", String(36), unique=True, nullable=False, index=True),
            Column("record_key", String(1024), nullable=False, index=True),
            Column("sim_signature", String(1024), nullable=False, index=True),
            Column("calculator_signature_hash", String(64), nullable=False, index=True),
            Column("record_json", Text, nullable=False),
            Column("created_at", Float, nullable=False),
        )
        self._table = Table(
            table_prefix + "stored_results",
            self._metadata,
            Column("record_key", String(1024), primary_key=True),
            Column("version_id", String(36), nullable=False),
        )
        if create:
            self._metadata.create_all(self.engine)

    def _put(self, record: ResultRecord) -> RecordRef:
        # Save the immutable version and replace the current cache pointer in
        # one transaction. Failure at either step leaves no partial save.
        key = _record_key(record.sim_signature, record.calculator_signature_hash)
        encoded = record.to_json()
        with self.engine.begin() as conn:
            conn.execute(
                self._versions.insert().values(
                    version_id=record.version_id,
                    record_key=key,
                    sim_signature=_sig_text(record.sim_signature),
                    calculator_signature_hash=record.calculator_signature_hash,
                    record_json=encoded,
                    created_at=record.created_at,
                )
            )
            conn.execute(self._table.delete().where(self._table.c.record_key == key))
            conn.execute(self._table.insert().values(record_key=key, version_id=record.version_id))
        return record.ref

    @staticmethod
    def _decode_row(row: Any) -> ResultRecord | None:
        if row is None:
            return None
        record = ResultRecord.from_json(row["record_json"])
        if record.version_id != row["version_id"] or record.ref.cache_key != row["record_key"]:
            raise SignatureMismatchError("stored version metadata does not match its record")
        return record

    def _get_by_key(self, sim_signature: tuple[Any, ...], calculator_signature_hash: str) -> ResultRecord | None:
        key = _record_key(tuple(sim_signature), calculator_signature_hash)
        stmt = (
            self._versions.select()
            .join(self._table, self._versions.c.version_id == self._table.c.version_id)
            .where(self._table.c.record_key == key)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._decode_row(row)

    def _get_version(self, version_id: str) -> ResultRecord | None:
        with self.engine.connect() as conn:
            row = (
                conn.execute(self._versions.select().where(self._versions.c.version_id == version_id))
                .mappings()
                .first()
            )
        return self._decode_row(row)

    def _delete(self, ref: RecordRef) -> bool:
        stmt = self._table.delete().where(self._table.c.record_key == ref.cache_key)
        if ref.version_id is not None:
            stmt = stmt.where(self._table.c.version_id == ref.version_id)
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        return result.rowcount > 0

    def _list(self, sim_signature: tuple[Any, ...] | None) -> list[RecordRef]:
        stmt = (
            self._versions.select()
            .join(self._table, self._versions.c.version_id == self._table.c.version_id)
            .order_by(self._versions.c.id)
        )
        if sim_signature is not None:
            stmt = stmt.where(self._versions.c.sim_signature == _sig_text(tuple(sim_signature)))
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        refs: list[RecordRef] = []
        for row in rows:
            record = self._decode_row(row)
            if record is not None:
                refs.append(RecordRef(record.sim_signature, record.calculator_signature_hash))
        return refs

    def _list_versions(self, ref: RecordRef) -> list[RecordRef]:
        stmt = self._versions.select().where(self._versions.c.record_key == ref.cache_key).order_by(self._versions.c.id)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        refs: list[RecordRef] = []
        for row in rows:
            record = self._decode_row(row)
            if record is not None:
                refs.append(record.ref)
        return refs
