"""SQLAlchemy metadata associations; explicitly import with the ``store`` extra.

The calculator core does not import this package. Result payloads stay in a
named ResultStore; AnalysisRepository manages relationships and run lifecycle.
"""

from .loading import LoadedInput, SnapshotLoader
from .models import (
    Base,
    Catalogue,
    CatalogueObject,
    DataLocation,
    DataObject,
    Recipe,
    ResultLink,
    Run,
    Simulation,
    Timestep,
)
from .records import AssociationIssue, RecordedRun, RunExecutionError, RunPersistenceError
from .repository import AnalysisRepository

__all__ = [
    "AnalysisRepository",
    "AssociationIssue",
    "Base",
    "Catalogue",
    "CatalogueObject",
    "DataLocation",
    "DataObject",
    "LoadedInput",
    "Recipe",
    "RecordedRun",
    "ResultLink",
    "Run",
    "RunExecutionError",
    "RunPersistenceError",
    "Simulation",
    "SnapshotLoader",
    "Timestep",
]
