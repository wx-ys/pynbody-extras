"""How ``pynbodyext`` decides that the compiled ``_native`` extension is usable (BUG-0005).

"Available" means ``import pynbodyext._native`` succeeds, not that a file exists.
A present-but-unloadable extension (wrong architecture, missing OpenMP runtime)
must degrade like a missing one: ``import pynbodyext.gravity`` still works and
emits a visible :class:`~pynbodyext.util.deps.NativeExtensionWarning` with the
reason.

The broken/missing cases run in a subprocess with a meta-path finder that fakes
the extension (found but failing to load, or not found), so they run on every
machine and leave this process untouched.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap
import pytest

from pynbodyext.util import deps

ROOT = pathlib.Path(__file__).resolve().parents[1]

_PROBE = textwrap.dedent(
    """
    import importlib.abc, importlib.util, json, sys, warnings

    MODE = {mode!r}

    class FailLoader(importlib.abc.Loader):
        # A present-but-unloadable extension: the file is found, loading fails.
        def create_module(self, spec):
            raise ImportError("dlopen(_native.so): Library not loaded: libomp.dylib")

        def exec_module(self, module):
            pass

    class FakeNative(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "pynbodyext._native":
                return None
            if MODE == "missing":
                raise ModuleNotFoundError("No module named 'pynbodyext._native'", name=name)
            return importlib.util.spec_from_loader(name, FailLoader())

    if MODE != "real":
        sys.meta_path.insert(0, FakeNative())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import pynbodyext.gravity as gravity

    from pynbodyext.util import deps

    import numpy as np
    import pynbody
    from pynbody.array import SimArray
    from pynbodyext.core.calculate import Bin1D

    sim = pynbody.new(dm=1)
    sim["pos"] = SimArray(np.zeros((1, 3)), "kpc")
    sim["mass"] = SimArray([1e10], "Msol")
    sim["smooth"] = SimArray([0.0], "kpc")
    bins = Bin1D("rxy", vmin="1 kpc", vmax="3 kpc", nbins=2)(sim)
    try:
        bins["vcirc"]
        vcirc_error = None
    except ImportError as exc:
        vcirc_error = [type(exc).__name__, exc.name, str(exc)]

    print(json.dumps({{
        "available": deps.GRAVITY_NATIVE_AVAILABLE,
        "exports": sorted(gravity.__all__),
        "reason": deps.native_unavailable_reason(),
        "warnings": [[w.category.__name__, str(w.message)] for w in caught],
        "vcirc_error": vcirc_error,
    }}))

    # Default filters: the warning must be visible without opting in.
    for name in [m for m in sys.modules if m.startswith("pynbodyext.gravity")]:
        del sys.modules[name]
    import pynbodyext.gravity
    """
)


def run_probe(mode: str) -> tuple[dict, str]:
    """Import ``pynbodyext.gravity`` in a fresh interpreter.

    *mode* is ``"broken"`` (found, fails to load), ``"missing"`` (not found) or
    ``"real"`` (whatever this environment has).
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(mode=mode)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def test_warning_is_shown_by_default():
    """A UserWarning subclass: not hidden like ImportWarning / DeprecationWarning."""
    assert issubclass(deps.NativeExtensionWarning, UserWarning)
    assert not issubclass(deps.NativeExtensionWarning, (ImportWarning, DeprecationWarning))


def test_present_but_unloadable_extension_degrades_with_a_visible_warning():
    result, stderr = run_probe("broken")

    assert result["available"] is False
    assert result["exports"] == ["GRAVITY_NATIVE_AVAILABLE"]
    assert "failed to load" in result["reason"]
    assert "libomp.dylib" in result["reason"]
    categories = [category for category, _ in result["warnings"]]
    assert categories == ["NativeExtensionWarning"], result["warnings"]
    assert "failed to load" in result["warnings"][0][1]
    assert "NativeExtensionWarning" in stderr, "the warning must be visible under default filters"
    # bins["vcirc"] raises the clear error, with the load failure in the message.
    name, module, message = result["vcirc_error"]
    assert (name, module) == ("ImportError", "pynbodyext._native")
    assert "failed to load" in message and "libomp.dylib" in message


def test_missing_extension_says_it_is_not_built():
    result, stderr = run_probe("missing")

    assert result["available"] is False
    assert "is not built" in result["reason"]
    assert [category for category, _ in result["warnings"]] == ["NativeExtensionWarning"]
    assert "NativeExtensionWarning" in stderr


def test_availability_matches_a_real_import():
    """In this process, the flag agrees with actually importing the extension."""
    try:
        import pynbodyext._native  # noqa: F401
    except ImportError:
        imports = False
    else:
        imports = True
    assert deps.GRAVITY_NATIVE_AVAILABLE is imports
    assert (deps.native_import_error() is None) is imports


def test_gravity_imports_without_warning_when_the_extension_works():
    pytest.importorskip("pynbodyext._native", reason="needs the compiled extension")
    result, stderr = run_probe("real")

    assert result["available"] is True
    assert "Gravity" in result["exports"]
    assert result["warnings"] == []
    assert result["vcirc_error"] is None
    assert "NativeExtensionWarning" not in stderr
