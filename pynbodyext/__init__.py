"""
Extensions and utilities for pynbody
"""

__version__ = "0.0.1"


from typing import Any

from .log import logger as logger


def __getattr__(name: str) -> Any:
    if name == "Database":
        from .database import Database
        return Database
    if name == "StudyTable":
        from .core.calculate.study.table import StudyTable
        return StudyTable
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
