"""Availability check for tests that exercise the compiled gravity extension."""

import pytest

from pynbodyext.util import deps


def require_native() -> None:
    """Skip an unbuilt extension and surface loader errors from an installed one."""
    error = deps.native_import_error()
    if isinstance(error, ModuleNotFoundError) and error.name == deps.NATIVE_MODULE:
        pytest.skip(f"needs the compiled {deps.NATIVE_MODULE} extension", allow_module_level=True)
    if error is not None:
        raise error
