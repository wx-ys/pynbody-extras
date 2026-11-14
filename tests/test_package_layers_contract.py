"""The import direction across the whole package, as a test.

``core/calculate`` has its own, stricter order (``test_layering_contract.py``).
This file covers everything else — the layers above the core, the two independent
subsystems beside it, and the package root — so "may A import B?" has one answer
for the whole repository.

The order, lowest first::

    log  <  root  <  util  <  plot  <  gravity  <  core  <  calculate  <  orm  <  filters  <  properties  <  transforms

Notes a reader will want:

- **``root`` is low, not high.**  ``pynbodyext/__init__.py`` only exports the logger
  (and hosts the compiled ``_native``/``_rust`` modules), and ``gravity`` and
  ``bins`` reach into it for the extension — so it belongs underneath them.  The
  *facade* is ``pynbodyext.calculate``, the short public import path, and that is at
  the top of the core.
- **``plot`` is low too.**  The image layer is generic 2-D-array work: it
  takes plain arrays, and the binning layer is one of its callers.  It names the
  calculator layer only under ``TYPE_CHECKING``, which this test does not count.
- **A layer may import itself.**  Intra-layer cycles are cohesion, not disorder:
  ``plot.image``'s ten modules and the core's own knot are inspected by their
  respective tests, and neither is a layering problem.
- **Function-local imports are still imports.**  Reaching upward "only when called"
  is how the layers got tangled in the first place.  If a lower layer needs an
  upper one, move the module down (that is why ``plot`` sits below ``core``) or
  register a hook the way ``core/calculate/interfaces.py`` does.

When this test fails, the answer is almost never an entry in ``_ALLOWED_UPWARD``.
"""

from __future__ import annotations

import pathlib

from import_graph import runtime_importers, runtime_imports

import pynbodyext

#: Package layers, lowest first.  A module may import from any layer at or below
#: its own — never from one above.
_ORDER = (
    "log",
    "root",
    "util",
    "plot",
    "gravity",
    "core",
    "calculate",
    "orm",  # ADR-009: associations depend on calculators; core never imports ORM
    "filters",
    "properties",
    "transforms",
    "database",  # ADR-020: user interface above ORM and calculators
)

#: ``(importer layer, imported layer)`` pairs that break the order and are known.
#: Empty, and meant to stay that way: an entry here needs a reason and an ADR.
_ALLOWED_UPWARD: frozenset[tuple[str, str]] = frozenset()

#: Layer of a module, by its dotted name.  No default: a module that is not
#: classified here fails ``test_every_module_belongs_to_a_declared_layer``, so a
#: new subsystem cannot slip past the rule by being unknown.
_LAYERS: dict[str, str] = {
    "pynbodyext.log": "log",
    "pynbodyext": "root",
    "pynbodyext._mypy_plugin": "root",
    "pynbodyext.util": "util",
    "pynbodyext.plot": "plot",
    "pynbodyext.gravity": "gravity",
    "pynbodyext.core": "core",
    "pynbodyext.core.calculate": "core",
    "pynbodyext.calculate": "calculate",
    "pynbodyext.orm": "orm",
    "pynbodyext.database": "database",
    "pynbodyext.filters": "filters",
    "pynbodyext.properties": "properties",
    "pynbodyext.transforms": "transforms",
}


def _layer_of(module: str) -> str | None:
    """The layer *module* belongs to, or None when it is not this package's."""
    if module != "pynbodyext" and not module.startswith("pynbodyext."):
        return None
    best: tuple[int, str] | None = None
    for prefix, layer in _LAYERS.items():
        if module == prefix or module.startswith(f"{prefix}."):
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), layer)
    return best[1] if best is not None else None


def _package_root() -> pathlib.Path:
    return pathlib.Path(pynbodyext.__file__).parent.parent


def _importers() -> dict[str, list[str]]:
    package = pathlib.Path(pynbodyext.__file__).parent
    return runtime_importers(package, package_root=_package_root())


def test_every_import_in_the_package_points_downwards() -> None:
    violations: list[str] = []
    for importer, modules in _importers().items():
        source = _layer_of(importer)
        if source is None:
            continue
        for module in modules:
            target = _layer_of(module)
            if target is None or target == source:
                continue  # outside the package, or inside this layer
            if source == "core" and target == "core":
                continue  # the core's own order is test_layering_contract.py's job
            # ADR-020 permits exactly this lazy public export, not general upward imports.
            if importer == "pynbodyext" and module == "pynbodyext.database":
                continue
            if _ORDER.index(target) > _ORDER.index(source):
                if (source, target) in _ALLOWED_UPWARD:
                    continue
                violations.append(f"{importer} ({source}) -> {module} ({target})")

    assert violations == []


def test_every_module_belongs_to_a_declared_layer() -> None:
    """A new subsystem must be placed in the order deliberately, not by default."""
    unclassified = sorted(module for module in _importers() if _layer_of(module) is None)
    assert unclassified == [], "add these to _LAYERS (and to the order) on purpose"


def test_the_layers_are_the_ones_the_rule_names() -> None:
    """A renamed or new top-level name must be added to ``_ORDER`` deliberately."""
    package = pathlib.Path(pynbodyext.__file__).parent
    present = {
        path.stem if path.is_file() else path.name
        for path in package.iterdir()
        if (path.is_file() and path.suffix == ".py" and path.name != "__init__.py")
        or (path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__")
    }
    unmapped = sorted(str(name) for name in present if _layer_of(f"pynbodyext.{name}") is None)
    assert unmapped == [], "add these to _LAYERS, and to the order, on purpose"
    layers = {_layer_of(f"pynbodyext.{name}") for name in present} | {"root"}  # + __init__.py
    assert layers == set(_ORDER)


def test_the_scan_actually_looked_at_something() -> None:
    """A scan that finds nothing must not be a scan that looks at nothing."""
    importers = _importers()
    edges = sum(len(modules) for modules in importers.values())
    assert len(importers) > 80, f"only {len(importers)} modules scanned"
    assert edges > 200, f"only {edges} imports collected"


def test_the_scanner_resolves_relative_and_absolute_imports(tmp_path: pathlib.Path) -> None:
    """The one self-test for ``tests/import_graph.py``, used by three test files."""
    package_root = tmp_path
    sample = tmp_path / "pkg" / "a" / "b" / "sample.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "from __future__ import annotations\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.hidden import Deferred\n"
        "from pkg.other import Absolute\n"
        "from .sibling import OneDot\n"
        "from ...top import ThreeDots\n"
        "def inside():\n"
        "    from pkg.local import AlsoSeen\n",
        encoding="utf-8",
    )
    from import_graph import module_name, package_name

    assert module_name(sample, package_root) == "pkg.a.b.sample"
    assert package_name(sample, package_root) == "pkg.a.b"

    found = runtime_imports(sample, package_name(sample, package_root))
    # standard-library imports are collected too; the deferred one is not
    assert found == ["__future__", "typing", "pkg.other", "pkg.a.b.sibling", "pkg.top", "pkg.local"]

    # and an __init__.py is its own package, not one level up
    init = tmp_path / "pkg" / "a" / "__init__.py"
    init.write_text("from .b import something\n", encoding="utf-8")
    assert module_name(init, package_root) == "pkg.a"
    assert package_name(init, package_root) == "pkg.a"
    assert runtime_imports(init, package_name(init, package_root)) == ["pkg.a.b"]
