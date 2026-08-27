"""SQLAlchemy-backed :class:`ResultStore` adapter (tangos-like relational store).

Implements the four abstract primitives from
:mod:`~pynbodyext.core.calculate.store.base` against a SQLAlchemy relational
database.  The full codec-encoded :class:`ResultRecord` is stored as a JSON
payload in a single column; the simulation and calculator identities are
additionally promoted to indexed columns so the store can be listed and filtered
by simulation without deserialising every record.

SQLAlchemy is an *optional* dependency: it is imported lazily inside the
constructor so that importing this module never requires it.  Install the
``store`` extra (``pip install pynbodyext[store]`` / ``uv add --extra store
pynbodyext``) before constructing an instance.

The :class:`ResultRecord.to_json`/``from_json`` helpers keep the adapter a
shallow *storage* layer: identity question are answered by the columns, the
record content round-trips through the same codec used everywhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import RecordRef, ResultRecord, ResultStore, _record_key, _sig_text

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
        default target but any SQLAlchemy dialect works.
    codec : ValueCodec | None
        Value codec used to encode/decode result values (defaults to a shared
        :class:`ValueCodec`).
    create : bool, default: True
        Create the ``stored_results`` table if it does not exist.
    table_prefix : str, default: ``"pynbodyext_"``
        Prefix prepended to the table name, to avoid clashing with other tables
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
            self.engine: Engine = url_or_engine
        else:
            self.engine = sqlalchemy.create_engine(str(url_or_engine))

        self._table = Table(
            table_prefix + "stored_results",
            MetaData(),
            Column("id", Integer, primary_key=True),
            Column("record_key", String(1024), unique=True, index=True),
            Column("sim_signature", String(1024), index=True),
            Column("calculator_signature_hash", String(64), index=True),
            Column("pretty_key", String(512), nullable=True),
            Column("record_json", Text),
            Column("created_at", Float),
        )
        self._metadata = self._table.metadata
        if create:
            self._metadata.create_all(self.engine)

    # ------------------------------------------------------------------
    # ResultStore primitives
    # ------------------------------------------------------------------

    def _put(self, record: ResultRecord) -> RecordRef:
        with self.engine.begin() as conn:
            result = conn.execute(
                self._table.insert().values(
                    record_key=_record_key(record.sim_signature, record.calculator_signature_hash),
                    sim_signature=_sig_text(record.sim_signature),
                    calculator_signature_hash=record.calculator_signature_hash,
                    pretty_key=record.pretty_key,
                    record_json=record.to_json(),
                    created_at=record.created_at,
                )
            )
            row_id = result.inserted_primary_key[0]
        return RecordRef(record.sim_signature, record.calculator_signature_hash, id=row_id)

    def _get_by_key(self, sim_signature: tuple[Any, ...], calculator_signature_hash: str) -> ResultRecord | None:
        key = _record_key(tuple(sim_signature), calculator_signature_hash)
        with self.engine.connect() as conn:
            row = conn.execute(self._table.select().where(self._table.c.record_key == key)).mappings().first()
        return None if row is None else ResultRecord.from_json(row["record_json"])

    def _delete(self, ref: RecordRef) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(self._table.delete().where(self._table.c.record_key == ref.key))
        return result.rowcount > 0

    def _list(self, sim_signature: tuple[Any, ...] | None) -> list[RecordRef]:
        with self.engine.connect() as conn:
            stmt = self._table.select().with_only_columns(self._table.c.record_json)
            if sim_signature is not None:
                stmt = stmt.where(self._table.c.sim_signature == _sig_text(tuple(sim_signature)))
            rows = conn.execute(stmt).mappings().all()

        refs: list[RecordRef] = []
        for row in rows:
            record = ResultRecord.from_json(row["record_json"])
            refs.append(RecordRef(record.sim_signature, record.calculator_signature_hash))
        return refs
