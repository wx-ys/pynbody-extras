"""Framework-specific exceptions for calculator execution."""

from __future__ import annotations


class CalculatorError(RuntimeError):
    """Base class for calculator framework runtime errors."""


class CycleError(CalculatorError):
    """Raised when calculator evaluation encounters a dependency cycle."""
