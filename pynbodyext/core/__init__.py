"""Core calculator subsystem for pynbodyext.

This namespace package hosts :mod:`pynbodyext.core.calculate`, the calculator
framework that powers the short public facade :mod:`pynbodyext.calculate`.

Most users never import directly from here. Prefer the public facade::

    from pynbodyext.calculate import PropertyBase

See :mod:`pynbodyext.core.calculate` for the full framework and
``CONTEXT.md`` in the repository root for the shared domain vocabulary.
"""

from . import calculate as calculate

__all__ = ["calculate"]
