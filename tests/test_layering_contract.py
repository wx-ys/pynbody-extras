"""The import direction inside ``core.calculate``, as a test.

``interfaces.py`` states the rule: a module imports from the layers below it, never
from the ones above.  ``TYPE_CHECKING`` imports are exempt, because they cost nothing
at run time and the annotations name types on purpose.

The framework used to be one cycle of 29 modules — every layer reached every other
through a deferred import — which is why nothing could be imported, tested or reused
alone, and why ``result/signature.py`` could not stop knowing about the node classes.
The edges are now directional; this test is what keeps them that way.

When a new edge is needed in the wrong direction, add a hook to ``interfaces.py``
(a protocol, a factory, a registration) rather than an exception here.  If an
exception is genuinely unavoidable, add it to ``_ALLOWED_UPWARD`` with the reason
in the same commit — an empty list is the goal, and it was reached on 2026-10-14.
"""

from __future__ import annotations

import pathlib

from import_graph import runtime_importers

import pynbodyext

#: Layers of ``core.calculate``, lowest first.  A module may import from any layer
#: at or below its own.
_ORDER = (
    "interfaces",
    "display",
    "params",
    "diagnostics",
    "result",
    "store",
    "runtime",
    "nodes",
    "bins",
    # many objects x many calculators: runs nodes, reads and writes the store
    "sample",
    "study",  # ADR-021: detached definitions and observations
    # the package facade wires everything together, so it sits at the top
    "",
)

_PACKAGE = "pynbodyext.core.calculate"

#: ``(importer, imported)`` pairs that break the rule and are known.  Empty: every
#: edge points downwards today.
_ALLOWED_UPWARD: frozenset[tuple[str, str]] = frozenset()


def _layer(relative_path: str) -> str | None:
    """The layer a module of ``core.calculate`` belongs to, or None if outside.

    ``pynbodyext.core.calculate`` itself is the facade (``""``), a subpackage is its
    own layer (``nodes``, ``result``, …) and a top-level module is that module's
    name (``interfaces``, ``display``).
    """
    parts = relative_path.split(".")
    if parts[:3] != ["pynbodyext", "core", "calculate"]:
        return None
    if len(parts) == 3:
        return ""  # the facade wires everything together
    return parts[3]  # a subpackage (nodes/, result/, …) or a top-level module


def _layer_of_module(module: str) -> str | None:
    if not module.startswith(_PACKAGE):
        return None
    rest = module[len(_PACKAGE) :].lstrip(".")
    return rest.split(".")[0] if rest else ""


def test_every_import_in_the_calculator_core_points_downwards() -> None:
    package = pathlib.Path(pynbodyext.__file__).parent
    root = package / "core" / "calculate"
    importers = runtime_importers(root, package_root=package.parent)

    violations: list[str] = []
    for importer, modules in importers.items():
        source_layer = _layer(importer)
        if source_layer is None:
            continue
        for module in modules:
            target_layer = _layer_of_module(module)
            if target_layer is None:
                continue  # outside the core: covered by their own boundary tests
            if _ORDER.index(target_layer) > _ORDER.index(source_layer):
                if (source_layer, target_layer) in _ALLOWED_UPWARD:
                    continue
                violations.append(f"{importer}: {source_layer} -> {target_layer} ({module})")

    assert violations == []


def test_the_layers_are_the_ones_the_rule_names() -> None:
    """A renamed or new subpackage must be added to ``_ORDER`` deliberately."""
    package = pathlib.Path(pynbodyext.__file__).parent
    core = package / "core" / "calculate"
    present = {
        path.stem if path.is_file() else path.name
        for path in core.iterdir()
        if (path.is_file() and path.suffix == ".py" and path.name != "__init__.py")
        or (path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__")
    }
    assert present == {layer for layer in _ORDER if layer}


# The scanner's own self-test lives in ``test_package_layers_contract.py``, which
# uses the same helpers; one copy is enough to keep them honest.
