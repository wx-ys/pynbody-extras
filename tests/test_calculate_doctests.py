"""Run doctests for the ``core/calculate`` modules whose examples are runnable.

Many docstrings under ``bins/*`` carry illustrative pseudo-code
(``>>> bins = Bin1D(...)(sim)``) that references names a doctest does not have
and therefore cannot be executed; those examples are documentation-only.  This
test guards the modules whose examples *are* runnable so they cannot silently
drift as the API evolves.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

DOCTYPE_MODULES = (
    "pynbodyext.core.calculate.display",
    "pynbodyext.core.calculate.nodes.mixins",
)


@pytest.mark.parametrize("module_name", DOCTYPE_MODULES)
def test_doctests(module_name: str) -> None:
    module = importlib.import_module(module_name)
    result = doctest.testmod(module, verbose=False, optionflags=doctest.ELLIPSIS)
    assert result.failed == 0, f"{result.failed} doctest(s) failed in {module_name}"
