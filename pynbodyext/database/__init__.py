"""User-facing persistence; imported only when Database is requested."""

from .database import Database, DatabasePersistenceError

__all__ = ["Database", "DatabasePersistenceError"]
