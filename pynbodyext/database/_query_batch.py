"""Bound SQLite metadata lookups on an existing read transaction."""

from __future__ import annotations

import sqlite3
from itertools import islice
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from sqlalchemy import Row, Select, SQLColumnExpression
    from sqlalchemy.orm import Session


def matching_rows(
    session: Session, stmt: Select[Any], column: SQLColumnExpression[Any], keys: Collection[Any], *, width: int = 1
) -> Iterator[Row[Any]]:
    """Select unique scalar or composite keys within this connection's bind limit."""
    if not keys:
        return
    connection = cast("sqlite3.Connection", session.connection().connection.driver_connection)
    room = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) - len(stmt.compile().params or {})
    batch_size = min(500, room // width)
    if batch_size < 1:
        raise ValueError("SQLite's variable limit is too small for this metadata query")
    keys_iter = iter(keys)
    while batch := list(islice(keys_iter, batch_size)):
        yield from session.execute(stmt.where(column.in_(batch)))
