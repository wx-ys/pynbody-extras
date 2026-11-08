"""Pin the private pynbody renderer surface the SPH view is built on.

``pynbodyext.core.calculate.bins.sph_render`` renders through
``pynbody.sph._render``, which is private and positional-argument only.  These tests
exist so that an upstream change to that surface fails here, with the parameter list
in the failure message, instead of silently changing the numbers a user sees.
``_pynbody_render`` is the only module allowed to call it.

If upstream renames or reorders a parameter, update
``pynbodyext/core/calculate/bins/_pynbody_render.py`` and this file together.
"""

from __future__ import annotations

import inspect


def test_pynbody_floor_excludes_the_broken_3d_grid_renderer() -> None:
    """BUG-0002: both installation metadata and the environment require the fix."""
    import importlib.metadata
    import pathlib
    import tomllib

    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    project = tomllib.loads((pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    requirement = next(Requirement(dep) for dep in project["dependencies"] if Requirement(dep).name == "pynbody")
    assert requirement.specifier.contains("2.8.0")
    assert not requirement.specifier.contains("2.3.3")
    assert Version(importlib.metadata.version("pynbody")) >= Version("2.8.0")
    python_versions = SpecifierSet(project["requires-python"])
    assert python_versions.contains("3.11")
    assert not python_versions.contains("3.10")


def test_render_image_signature_is_pinned() -> None:
    from pynbody.sph import _render

    from pynbodyext.core.calculate.bins._pynbody_render import RENDER_IMAGE_PARAMS

    assert tuple(inspect.signature(_render.render_image).parameters) == RENDER_IMAGE_PARAMS


def test_to_3d_grid_signature_is_pinned() -> None:
    from pynbody.sph import _render

    from pynbodyext.core.calculate.bins._pynbody_render import TO_3D_GRID_PARAMS

    assert tuple(inspect.signature(_render.to_3d_grid).parameters) == TO_3D_GRID_PARAMS


def test_only_the_adapter_calls_pynbody_private_renderers() -> None:
    """No module but the adapter may import ``pynbody.sph._render``."""
    import pathlib

    import pynbodyext

    package = pathlib.Path(pynbodyext.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        if path.name == "_pynbody_render.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "from pynbody.sph import _render" in text or "import pynbody.sph._render" in text:
            offenders.append(path.relative_to(package).as_posix())
    assert offenders == []
