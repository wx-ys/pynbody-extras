"""SQLAlchemy metadata associations; explicitly import with the ``store`` extra.

The calculator core does not import this package. Result payloads stay in a
named ResultStore; AnalysisRepository manages relationships and run lifecycle.
"""

from .jobs import JobSpec
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
    Selection,
    Simulation,
    Timestep,
)
from .records import AssociationIssue, RecordedRun, RunExecutionError, RunPersistenceError
from .repository import AnalysisRepository
from .selections import SelectionSpec
from .study import StudyCell, StudyPersistenceError, StudyReport, run_study

__all__ = [
    "AnalysisRepository",
    "AssociationIssue",
    "Base",
    "Catalogue",
    "CatalogueObject",
    "DataLocation",
    "DataObject",
    "JobSpec",
    "LoadedInput",
    "Recipe",
    "RecordedRun",
    "ResultLink",
    "Run",
    "RunExecutionError",
    "RunPersistenceError",
    "Selection",
    "SelectionSpec",
    "Simulation",
    "SnapshotLoader",
    "StudyCell",
    "StudyPersistenceError",
    "StudyReport",
    "Timestep",
    "run_study",
]
