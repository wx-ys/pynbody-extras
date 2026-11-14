"""Capture save provenance without loading optional database dependencies."""

from __future__ import annotations

import os
import weakref
from typing import TYPE_CHECKING, Any

from pynbodyext.core.calculate.interfaces import SavedInput

from .sim_identity import _is_file_name, id_based_sim_identity, snapshot_identity

if TYPE_CHECKING:
    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.result.result import Result
    from pynbodyext.core.calculate.store.base import ResultStore

    from .options import RunOptions
    from .sim_identity import SimIdentityProvider


def capture_input(sim: Any, provider: SimIdentityProvider = snapshot_identity) -> SavedInput:
    """Freeze input identity, selection and unit convention before DAG transforms."""
    root = getattr(sim, "ancestor", sim)
    signature = tuple(provider(sim))
    root_signature = tuple(provider(root))
    partial = (
        getattr(root, "partial_load", False) or getattr(getattr(root, "_load_control", None), "_ids", None) is not None
    )
    # Partial loaders need an explicit logical identity and custom loader binding.
    if partial and not (
        len(root_signature) == 7
        and root_signature[1] == "pynbodyext.logical-snapshot/v1"
        and root_signature[6] != "default"
    ):
        signature = id_based_sim_identity(sim)
        root_signature = id_based_sim_identity(root)
    filename = getattr(root, "filename", None)
    location = os.path.abspath(os.fspath(filename)) if filename is not None and _is_file_name(filename) else None
    indices = None if root is sim else tuple(int(i) for i in sim.get_index_list(root))
    basis = tuple(str(unit) for unit in (getattr(root, "_autoconvert", None) or ())[:3])
    return SavedInput(signature, root_signature, location, len(root), indices, basis)


def execute(
    calculator: CalculatorBase,
    sim: Any,
    options: RunOptions,
    *,
    provider: SimIdentityProvider | None = None,
    store: ResultStore | None = None,
    reuse: bool = True,
) -> Result:
    """One executor, with frozen save facts even when persistence is deferred."""
    from .engine import EvalEngine

    catalogue = getattr(sim, "_halo_catalogue", None)
    catalogue_link = None if catalogue is None else (weakref.ref(catalogue), getattr(sim, "_halo_number", None))
    # Pure calculations also accept arbitrary inputs. Capture failure must not
    # turn a valid non-pynbody computation into a persistence requirement.
    try:
        captured = capture_input(sim, provider or id_based_sim_identity)
    except (AttributeError, TypeError):
        captured = None
    engine = EvalEngine(sim_identity=provider) if provider is not None else EvalEngine()
    result = engine.run(calculator, sim, options, store=store, reuse=reuse)
    result._save_catalogue = catalogue_link
    result._save_input = captured
    result._save_signature = result.provenance.calculator_signature_text if result.provenance is not None else None
    return result
