"""Run receipts and explicit failure/recovery information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynbodyext.core.calculate.result.result import Result
    from pynbodyext.core.calculate.store import RecordRef

    from .models import Run


@dataclass(frozen=True)
class RecordedRun:
    run: Run
    result: Result


@dataclass(frozen=True)
class AssociationIssue:
    run_id: str
    code: str
    detail: str


class RunExecutionError(RuntimeError):
    """Execution failed; the original exception is chained and the run is recorded."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"execution failed for run {run_id}")


class RunPersistenceError(RuntimeError):
    """Metadata finalization failed; keep the receipt for explicit recovery."""

    def __init__(self, run_id: str, *, result_ref: RecordRef | None = None, result: Result | None = None) -> None:
        self.run_id = run_id
        self.result_ref = result_ref
        self.result = result
        super().__init__(f"could not finalize run {run_id}; inspect its status before explicit recovery")
