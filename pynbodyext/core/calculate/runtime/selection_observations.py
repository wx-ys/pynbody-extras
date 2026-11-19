"""Detached facts about consumed Filter views, their uses and measurement frames."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cache
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from weakref import ref

import numpy as np
from pynbody import units
from pynbody.family import Family
from pynbody.transformation import GenericTranslation, Rotation

from pynbodyext.core.calculate.params.fields import ParamView
from pynbodyext.core.calculate.result.enums import BuiltinKinds, NodeStatus
from pynbodyext.core.calculate.store.identity import sim_identity_key

from .sim_identity import id_based_sim_identity

if TYPE_CHECKING:
    from collections.abc import Iterator
    from weakref import ReferenceType

    from pynbodyext.core.calculate.interfaces import SavedInput
    from pynbodyext.core.calculate.nodes.base import CalculatorBase
    from pynbodyext.core.calculate.result.result import ResultNode
    from pynbodyext.core.calculate.result.signature import CalculatorSignature

    from .input import FilterResult

Path = tuple[str, ...]
Indices = tuple[int, ...] | None
_SCHEMA = "pynbodyext.selection-observations/v1"


@dataclass(frozen=True, slots=True)
class _ParameterMap(Mapping[str, Any]):
    entries: tuple[tuple[str, Any], ...]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, key: str) -> Any:
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)


def _tuples(value: Any) -> Any:
    if isinstance(value, dict) and "$map" in value:
        return _ParameterMap(tuple((key, _tuples(item)) for key, item in value["$map"]))
    if isinstance(value, dict) and "$float" in value:
        return float(value["$float"])
    if isinstance(value, (tuple, list)):
        return tuple(_tuples(item) for item in value)
    return value


def _lists(value: Any) -> Any:
    if isinstance(value, _ParameterMap):
        return {"$map": [[key, _lists(item)] for key, item in value.entries]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": str(value)}
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    return value


def _indices(value: Any) -> Indices:
    return None if value is None else tuple(value)


@dataclass(frozen=True, slots=True)
class FrameStep:
    """One actual operation, its signed displacement or matrix, and its particle target."""

    kind: str
    signature: str | None = field(compare=False)
    target_indices: Indices
    values: tuple[Any, ...] = ()
    units: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "signature": self.signature,
            "target_indices": None if self.target_indices is None else list(self.target_indices),
            "values": _lists(self.values),
            "units": self.units,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrameStep:
        return cls(
            data["kind"],
            data["signature"],
            _indices(data["target_indices"]),
            _tuples(data["values"]),
            data["units"],
            data["reason"],
        )


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Physical frame at the point of selection or measurement, including its root version."""

    root_signature: tuple[Any, ...] = field(compare=False)
    root_count: int
    family_layout: tuple[tuple[str, int], ...]
    unit_basis: tuple[str, ...]
    position_units: str | None
    velocity_units: str | None
    steps: tuple[FrameStep, ...]
    _root_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_root_key", sim_identity_key(self.root_signature))

    @property
    def complete(self) -> bool:
        return all(step.kind in {"rotation", "shift_pos", "shift_vel"} and step.reason is None for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_signature": _lists(self.root_signature),
            "root_count": self.root_count,
            "family_layout": _lists(self.family_layout),
            "unit_basis": list(self.unit_basis),
            "position_units": self.position_units,
            "velocity_units": self.velocity_units,
            "steps": [step.to_dict() for step in self.steps],
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionContext:
        return cls(
            _tuples(data["root_signature"]),
            data["root_count"],
            _tuples(data["family_layout"]),
            tuple(data["unit_basis"]),
            data["position_units"],
            data["velocity_units"],
            tuple(FrameStep.from_dict(step) for step in data["steps"]),
        )


@dataclass(frozen=True, slots=True)
class ResolvedParameter:
    """The value passed to a Filter hook, with its path through any Boolean operands."""

    path: Path
    value: Any
    units: str | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": list(self.path), "value": _lists(self.value), "units": self.units, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResolvedParameter:
        return cls(tuple(data["path"]), _tuples(data["value"]), data["units"], data.get("reason"))


@dataclass(frozen=True, slots=True)
class SelectionOrigin:
    origin_id: int
    parent_origin_id: int | None
    parent_indices: Indices
    indices: tuple[int, ...]
    filter_signature: str
    parameters: tuple[ResolvedParameter, ...]
    context: SelectionContext

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "parent_origin_id": self.parent_origin_id,
            "parent_indices": None if self.parent_indices is None else list(self.parent_indices),
            "indices": list(self.indices),
            "filter_signature": self.filter_signature,
            "parameters": [p.to_dict() for p in self.parameters],
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionOrigin:
        return cls(
            data["origin_id"],
            data["parent_origin_id"],
            _indices(data["parent_indices"]),
            tuple(data["indices"]),
            data["filter_signature"],
            tuple(ResolvedParameter.from_dict(p) for p in data["parameters"]),
            SelectionContext.from_dict(data["context"]),
        )


@dataclass(frozen=True, slots=True)
class SelectionUse:
    origin_id: int
    output_path: Path
    occurrence_path: Path
    role: str


@dataclass(frozen=True, slots=True)
class OutputObservation:
    output_path: Path
    context: SelectionContext


@dataclass(frozen=True, slots=True)
class SelectionObservations:
    """Origins are in consumption order. Each use belongs to one successful leaf output."""

    origins: tuple[SelectionOrigin, ...]
    uses: tuple[SelectionUse, ...]
    outputs: tuple[OutputObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "origins": [origin.to_dict() for origin in self.origins],
            "uses": [
                {
                    "origin_id": u.origin_id,
                    "role": u.role,
                    "output_path": list(u.output_path),
                    "occurrence_path": list(u.occurrence_path),
                }
                for u in self.uses
            ],
            "outputs": [{"output_path": list(o.output_path), "context": o.context.to_dict()} for o in self.outputs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SelectionObservations:
        if data.get("schema") != _SCHEMA:
            raise ValueError("unsupported selection observations schema")
        return cls(
            tuple(SelectionOrigin.from_dict(o) for o in data["origins"]),
            tuple(
                SelectionUse(u["origin_id"], tuple(u["output_path"]), tuple(u["occurrence_path"]), u["role"])
                for u in data["uses"]
            ),
            tuple(
                OutputObservation(tuple(o["output_path"]), SelectionContext.from_dict(o["context"]))
                for o in data["outputs"]
            ),
        )


def _field_units(sim: Any, name: str | None) -> str | None:
    """Read unit metadata without loading a field or causing an observer read."""
    if name is None:
        return None
    root = sim.ancestor
    array = root._arrays.get(name)
    if array is not None:
        return str(array.units)
    families = root._family_arrays.get(name, {})
    if families:
        names = {str(families[fam].units) for fam in sim.families() if fam in families}
        if len(names) == 1:
            return names.pop()
    basis = getattr(root, "_autoconvert", None)
    if basis and name in {"pos", "vel"}:
        return str(basis[0 if name == "pos" else 1])
    return None


def _parameter(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("resolved mapping parameter has non-string keys")
        return _ParameterMap(tuple((key, _parameter(value[key])) for key in sorted(value)))
    if isinstance(value, np.ndarray):
        return _parameter(value.tolist())
    if isinstance(value, np.generic):
        return _parameter(value.item())
    if isinstance(value, (Family, units.UnitBase)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return tuple(_parameter(v) for v in value)
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f"unsupported resolved parameter type {type(value).__qualname__}")


@dataclass(frozen=True, slots=True)
class _Edge:
    path: Path
    node_id: str


@dataclass(frozen=True, slots=True)
class _Use:
    path: Path
    origin_id: int
    role: str


@dataclass(slots=True)
class SelectionBinding:
    """A filter scope becomes an origin only when active_sim hands its view to a consumer."""

    recorder: SelectionRecorder
    owner_id: str
    filter_id: str
    path: Path
    role: str
    parent_origin_id: int | None
    origin_id: int | None = None

    def consume(self, result: FilterResult, sim: Any) -> None:
        if self.origin_id is None:
            self.origin_id = self.recorder.consume(self, result, sim)


class SelectionRecorder:
    """Run-local evidence; only the frozen observations leave the execution context."""

    def __init__(self, sim: Any, saved: SavedInput, stack: list[ResultNode]) -> None:
        self.root = sim.ancestor
        self.source = sim
        try:
            sim_identity_key(saved.root_signature)
        except (TypeError, ValueError):
            # A pure run can use opaque process-local cache keys. Do not retain such objects.
            saved = replace(saved, root_signature=id_based_sim_identity(self.root))
        self.saved = saved
        self.stack = stack
        self._direct: set[tuple[str, str, Path]] = set()
        self._view_indices: dict[int, tuple[ReferenceType, tuple[int, ...]]] = {}
        self._filter_entry: dict[str, tuple[FrameStep, ...]] = {}
        self._during_measurement: dict[str, list[FrameStep]] = defaultdict(list)
        self.layout = tuple((str(fam), len(self.root[fam])) for fam in self.root.families())
        self.frames: list[tuple[Any, FrameStep]] = []
        self.origins: list[SelectionOrigin] = []
        self._origin_keys: dict[tuple[Any, ...], int] = {}
        self.signatures: dict[str, CalculatorSignature] = {}
        self.parameters: dict[str, tuple[ResolvedParameter, ...]] = {}
        self.measurements: dict[str, SelectionContext] = {}
        self._filter_evidence: dict[str, tuple[tuple[ResolvedParameter, ...], SelectionContext]] = {}
        self.events: dict[str, list[_Edge | _Use]] = defaultdict(list)
        self.transform_nodes: set[str] = set()
        self.pipeline_nodes: set[str] = set()
        self._external_frames(sim)

    def _target_indices(self, sim: Any) -> Indices:
        if sim is self.root:
            return None
        if sim is self.source:
            return self.saved.indices
        stored = self._view_indices.get(id(sim))
        if stored is not None and stored[0]() is sim:
            return stored[1]
        indices = tuple(map(int, sim.get_index_list(self.root)))
        self._view_indices[id(sim)] = (ref(sim), indices)
        return indices

    def native_application(self, target: Any, handle: Any, *, own_run: bool) -> None:
        node = next(reversed(self.stack), None)
        if node is None or getattr(target, "ancestor", None) is not self.root:
            return
        node_id = node.node_id
        if own_run and node_id in self.transform_nodes:
            # The transform lifecycle will inspect the returned native receipt.
            return
        step = FrameStep(
            "external",
            None,
            self._target_indices(target),
            reason=f"external {type(handle).__name__} applied inside a science hook",
        )
        self.frames.append((handle, step))
        self._during_measurement[node_id].append(step)

    def _external_frames(self, sim: Any) -> None:
        lineage = [sim]
        while lineage[-1] is not self.root:
            base = getattr(lineage[-1], "_subsnap_base", self.root)
            lineage.append(base)
        for target in reversed(lineage):
            for handle in target._transformations:
                if any(known is handle for known, _ in self.frames):
                    continue
                actual = getattr(handle, "sim", target)
                if getattr(actual, "ancestor", None) is not self.root:
                    actual = target
                self.frames.append(
                    (
                        handle,
                        FrameStep(
                            "external",
                            None,
                            self._target_indices(actual),
                            reason=f"external transformation {type(handle).__name__} on the particle input",
                        ),
                    )
                )

    def context(self, sim: Any) -> SelectionContext:
        self._external_frames(sim)
        return SelectionContext(
            self.saved.root_signature,
            self.saved.root_count,
            self.layout,
            self.saved.unit_basis,
            _field_units(sim, "pos"),
            _field_units(sim, "vel"),
            tuple(step for handle, step in self.frames if not getattr(handle, "_reverted", False)),
        )

    def applied(self, node: CalculatorBase, node_id: str, handle: Any, target: Any) -> None:
        # A compound native handle exposes the SimSnap it actually mutated.
        actual = getattr(handle, "sim", getattr(target, "sim", target))
        if not hasattr(actual, "get_index_list") or actual.ancestor is not self.root:
            actual = self.source
            kind = None
        else:
            # Subclasses may perform additional mutations. Certify an explicit declaration only.
            kind = vars(type(node)).get("frame_operation")
        target_indices = self._target_indices(actual)
        signature = self.signatures[node_id].to_json(full=True)
        if kind is not None and any(prior is handle for prior, _ in self.frames):
            return
        if kind == "rotation" and type(handle) is Rotation:
            matrix = np.asarray(handle.matrix)
            if matrix.shape == (3, 3) and np.all(np.isfinite(matrix)):
                step = FrameStep(kind, signature, target_indices, _tuples(matrix.tolist()))
            else:
                step = FrameStep("unsupported", signature, target_indices, reason="invalid rotation matrix")
        elif (
            kind in {"shift_pos", "shift_vel"}
            and type(handle) is GenericTranslation
            and kind == f"shift_{handle.arname}"
        ):
            shift = np.asarray(handle.shift)
            if shift.shape in {(), (1,), (3,)} and np.all(np.isfinite(shift)):
                values = tuple(float(v) for v in np.broadcast_to(shift, (3,)))
                shift_units = getattr(handle.shift, "units", None)
                unit = str(shift_units) if shift_units is not None else _field_units(actual, handle.arname)
                step = FrameStep(kind, signature, target_indices, values, unit)
            else:
                step = FrameStep(
                    "unsupported", signature, target_indices, reason="per-particle or non-finite translation"
                )
        else:
            step = FrameStep(
                "unsupported",
                signature,
                target_indices,
                reason=f"{type(node).__name__} supplied no supported frame operation",
            )
        self.frames.append((handle, step))

    def reverted(self, handle: Any) -> None:
        self.frames[:] = [(h, s) for h, s in self.frames if h is not handle and not getattr(h, "_reverted", False)]

    def register(self, node: CalculatorBase, node_id: str, signature: CalculatorSignature, sim: Any) -> None:
        self.signatures[node_id] = signature
        if node.kind == BuiltinKinds.FILTER:
            self._filter_entry[node_id] = self.context(sim).steps
        if node.kind == BuiltinKinds.TRANSFORM:
            self.transform_nodes.add(node_id)
        if node._manages_output_scopes:
            self.pipeline_nodes.add(node_id)

    def link(self, owner_id: str, node_id: str, edge: Path | None) -> None:
        events = self.events[owner_id]
        if edge in {("left",), ("right",)}:
            # Boolean signatures canonicalize commutative operands. An occurrence must
            # still address the same component when the caller wrote them in reverse.
            payload = self.signatures[owner_id].payload
            if payload.get("node") == "filter_op" and edge is not None:
                if payload[edge[0]] != self.signatures[node_id].payload:
                    edge = ("right",) if edge == ("left",) else ("left",)
        events.append(_Edge(edge if edge is not None else ("dependencies", str(len(events))), node_id))

    def prepared(self, node_id: str, node: CalculatorBase, sim: Any, params: Any, values: dict[str, Any]) -> None:
        items: Mapping[str, Any]
        if isinstance(params, ParamView):
            items = params.data
        elif isinstance(params, Mapping):
            items = params
        elif isinstance(params, SimpleNamespace):
            items = vars(params)
        elif is_dataclass(params) and not isinstance(params, type):
            items = {f.name: getattr(params, f.name) for f in fields(params)}
        elif isinstance(params, tuple) and hasattr(params, "_asdict"):
            items = params._asdict()
        elif params is None and not values:
            items = {}
        else:
            unavailable = f"unavailable prepared parameters of type {type(params).__qualname__}"
            self.parameters[node_id] = tuple(
                ResolvedParameter((name,), None, None, unavailable) for name in values
            ) or (ResolvedParameter((), None, None, unavailable),)
            return
        frozen = []
        for name, val in items.items():
            spec = node.dynamic_param_spec(name)
            unit = getattr(val, "units", None)
            if unit is None:
                unit = spec.target_units if spec.target_units is not None else _field_units(sim, spec.field_name)
            try:
                saved = _parameter(val)
                reason = None
            except TypeError as error:
                saved, reason = None, str(error)
            frozen.append(ResolvedParameter((name,), saved, str(unit) if unit is not None else None, reason))
        self.parameters[node_id] = tuple(frozen)

    def filter_evidence(self, node_id: str) -> tuple[tuple[ResolvedParameter, ...], SelectionContext]:
        cached = self._filter_evidence.get(node_id)
        if cached is not None:
            return cached
        parameters = list(self.parameters.get(node_id, ()))
        frame = self.measurements[node_id]
        operands = []
        for event in self.events.get(node_id, ()):
            if isinstance(event, _Edge) and event.path in {("left",), ("right",), ("child",)}:
                child_params, child_frame = self.filter_evidence(event.node_id)
                parameters.extend(replace(p, path=event.path + p.path) for p in child_params)
                operands.append(child_frame)
        if operands:
            positions = {c.position_units for c in operands if c.position_units is not None}
            velocities = {c.velocity_units for c in operands if c.velocity_units is not None}
            if all(c.steps == operands[0].steps for c in operands) and len(positions) <= 1 and len(velocities) <= 1:
                frame = replace(
                    operands[0],
                    position_units=next(iter(positions), frame.position_units),
                    velocity_units=next(iter(velocities), frame.velocity_units),
                )
            else:
                mixed = FrameStep(
                    "unsupported",
                    self.signatures[node_id].to_json(full=True),
                    None,
                    reason="Boolean filter operands measured particles in different coordinate frames",
                )
                frame = replace(frame, steps=(*frame.steps, mixed))
        evidence = (tuple(sorted(parameters, key=lambda p: p.path)), frame)
        self._filter_evidence[node_id] = evidence
        return evidence

    def measured_units(self, node_id: str, sim: Any) -> None:
        # Calculation may have loaded coordinate fields that were absent on entry.
        frame = self.measurements[node_id]
        pos = frame.position_units or _field_units(sim, "pos")
        vel = frame.velocity_units or _field_units(sim, "vel")
        external = tuple(s for s in self._during_measurement.pop(node_id, ()) if s not in frame.steps)
        if external or pos != frame.position_units or vel != frame.velocity_units:
            self.measurements[node_id] = replace(
                frame, position_units=pos, velocity_units=vel, steps=frame.steps + external
            )

    def register_filter(self, node_id: str, parent: SelectionBinding | None, result: FilterResult) -> None:
        if node_id in self._filter_entry:
            result._observation = (ref(self), node_id, None if parent is None else parent.origin_id)

    def consume_direct(self, node_id: str, parent_id: int | None, result: FilterResult, sim: Any) -> None:
        if not self.stack:
            return
        owner = self.stack[-1].node_id
        edges = self.events[owner]
        for event in reversed(edges):
            if isinstance(event, _Edge) and event.node_id == node_id:
                key = (owner, node_id, event.path)
                if key not in self._direct:
                    self._direct.add(key)
                    self.consume(
                        SelectionBinding(self, owner, node_id, event.path, "output_filter", parent_id), result, sim
                    )
                return

    def consume(self, binding: SelectionBinding, result: FilterResult, sim: Any) -> int:
        parent_id = binding.parent_origin_id
        parent = self.origins[parent_id].indices if parent_id is not None else self._target_indices(result.source_sim)
        target = self._target_indices(sim)
        indices = tuple(range(len(self.root))) if target is None else target
        signature = self.signatures[binding.filter_id].to_json(full=True)
        # A filter can own a reversible transform that has already unwound at
        # consumption. Its mask belongs to the frame in which it was calculated.
        parameters, context = self.filter_evidence(binding.filter_id)
        if context.steps == self._filter_entry[binding.filter_id]:
            # A frame-insensitive cached mask can be consumed in a later frame.
            context = self.context(result.source_sim)
        # Two recipes can reach the same physical frame. Preserve both method histories for browsing.
        key = (parent_id, parent, indices, signature, parameters, context, tuple(s.signature for s in context.steps))
        idx = self._origin_keys.get(key)
        if idx is None:
            idx = len(self.origins)
            self._origin_keys[key] = idx
            self.origins.append(SelectionOrigin(idx, parent_id, parent, indices, signature, parameters, context))
        self.events[binding.owner_id].append(_Use(binding.path, idx, binding.role))
        return idx

    def finish(self, root_id: str, nodes: Mapping[str, ResultNode]) -> SelectionObservations:
        outputs: list[OutputObservation] = []
        uses: list[SelectionUse] = []

        @cache
        def dependencies(node_id: str, include_outputs: bool) -> tuple[_Use, ...]:
            local = []
            for event in self.events.get(node_id, ()):
                if isinstance(event, _Use):
                    local.append(event)
                elif include_outputs or event.path[:1] != ("outputs",):
                    for use in dependencies(event.node_id, True):
                        role = "transform_measure" if node_id in self.transform_nodes else use.role
                        local.append(_Use(event.path + use.path, use.origin_id, role))
            return tuple(local)

        def visit(node_id: str, output: Path, occurrence: Path, inherited: tuple[_Use, ...]) -> None:
            local = inherited + tuple(replace(use, path=occurrence + use.path) for use in dependencies(node_id, False))
            if node_id in self.pipeline_nodes:
                for event in self.events.get(node_id, ()):
                    if isinstance(event, _Edge) and event.path[:1] == ("outputs",):
                        visit(event.node_id, (*output, event.path[1]), occurrence + event.path, local)
            elif node_id in self.measurements and nodes[node_id].status == NodeStatus.OK:
                outputs.append(OutputObservation(output, self.measurements[node_id]))
                uses.extend(SelectionUse(use.origin_id, output, use.path, use.role) for use in local)

        visit(root_id, (), (), ())
        return SelectionObservations(tuple(self.origins), tuple(uses), tuple(outputs))
