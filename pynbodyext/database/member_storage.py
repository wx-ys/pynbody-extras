"""Translate detached runtime evidence into staged and confirmed relational metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert

from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.core.calculate.runtime.sim_identity import particle_index_digest
from pynbodyext.core.calculate.store.identity import sim_identity_key
from pynbodyext.orm import AnalysisRepository
from pynbodyext.orm.models import DataObject, Selection

from .member_models import OutputObservation, SelectionOrigin, SelectionUse, StagedMemberInput, StagedObservations
from .models import DatabaseNamespace

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from pynbodyext.core.calculate.interfaces import SavedInput
    from pynbodyext.core.calculate.runtime.selection_observations import SelectionContext, SelectionObservations

    from .database import Database


def metadata_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def producer_source(source: Any) -> tuple[tuple[str, ...], str | None] | None:
    """Capture the caller's scientific source, including a uniquely pinned origin."""
    from .members import Member
    from .objects import Input

    if not isinstance(source, Input):
        return None
    parent_id = source.origin.id if isinstance(source, Member) and len(source.origins) == 1 else None
    return source.object_locator, parent_id


def _validate_indices(indices: tuple[int, ...], context: SelectionContext) -> None:
    if any(type(i) is not int or i < 0 or i >= context.root_count for i in indices):
        raise ValueError("observed selection indices are outside the captured root")
    boundaries = np.cumsum([count for _, count in context.family_layout])
    if sum(count for _, count in context.family_layout) != context.root_count:
        raise ValueError("observed family layout conflicts with captured root count")
    if np.any(np.diff(np.searchsorted(boundaries, indices, side="right")) < 0):
        raise ValueError("selection indices must follow root family-block order; cross-block reordering is unsupported")


def _read_target(session: Session, selection_id: int, context: SelectionContext) -> tuple[int, ...]:
    selection = session.get(Selection, selection_id)
    if selection is None:
        raise ValueError("archived coordinate target Selection is missing")
    if (
        selection.root_input.identity_key != sim_identity_key(context.root_signature)
        or selection.expected_count != context.root_count
    ):
        raise ValueError("archived coordinate target conflicts with the frame's root revision or particle count")
    definition = json.loads(selection.definition_json)
    if selection.view_input.kind != "subset" or not isinstance(definition, dict) or definition.get("kind") != "indices":
        raise ValueError("archived coordinate target requires a root-relative index Selection")
    # Only exact index definitions reach the ORM validator. Metadata browsing never reconstructs a filter.
    AnalysisRepository._validate_selection(selection)
    target = tuple(definition["indices"])
    _validate_indices(target, context)
    if particle_index_digest(target) != selection.view_input.signature[-1]:
        raise ValueError("archived coordinate target conflicts with its particle digest")
    return target


def read_context(session: Session, text: str) -> SelectionContext:
    """Expand and verify the exact, ordered targets against the frame's own root revision."""
    from pynbodyext.core.calculate.runtime.selection_observations import FrameStep, SelectionContext

    raw = json.loads(text)
    steps = raw.pop("steps")
    raw["steps"] = ()
    frame = SelectionContext.from_dict(raw)
    if not steps:
        return frame
    targets: dict[int, tuple[int, ...]] = {}
    resolved = []
    for step in steps:
        if "target_selection_id" in step:
            selection_id = step.pop("target_selection_id")
            if type(selection_id) is not int:
                raise ValueError("invalid archived coordinate target Selection ID")
            if step.get("target_indices") is not None:
                raise ValueError("archived coordinate target has conflicting index definitions")
            if selection_id not in targets:
                targets[selection_id] = _read_target(session, selection_id, frame)
            step["target_indices"] = targets[selection_id]
        elif step.get("target_indices") is not None:
            _validate_indices(tuple(step["target_indices"]), frame)
        resolved.append(FrameStep.from_dict(step))
    return replace(frame, steps=tuple(resolved))


class _ObservationArchive:
    """Intern each exact sequence once per save and preserve actual parent provenance."""

    def __init__(
        self,
        db: Database,
        session: Session,
        desc: SavedInput,
        input_id: int,
        source: tuple[tuple[str, ...], str | None] | None,
    ) -> None:
        self.db = db
        self.session = session
        self.desc = desc
        self.input = session.scalars(select(DataObject).where(DataObject.id == input_id)).one()
        root = self.input if self.input.parent_id is None else self.input.parent
        if root is None:
            raise ValueError("member source has no registered root")
        self.root = root
        self.namespace = session.scalars(select(DatabaseNamespace.uuid).where(DatabaseNamespace.id == 1)).one()
        self.source = (
            source
            if source is not None and source[0][0] == self.namespace
            else (self.input_locator(self.input.identity_key), None)
        )
        self.views: dict[tuple[int, ...], tuple[DataObject, Selection]] = {}
        self.contexts: dict[int, str] = {}

    def input_locator(self, key: str) -> tuple[str, ...]:
        from .objects import input_locator

        return input_locator(self.namespace, key)

    def view(self, indices: tuple[int, ...]) -> tuple[DataObject, Selection]:
        if indices not in self.views:
            signature = self.desc.root_signature + ("subset", particle_index_digest(indices))
            view = replace(self.desc, signature=signature, indices=indices)
            self.views[indices] = self.db._record_exact_view(self.session, self.root, view, staged=True)
        return self.views[indices]

    def context(self, context: SelectionContext) -> str:
        cached = self.contexts.get(id(context))
        if cached is not None:
            return cached
        if tuple(context.root_signature) != self.desc.root_signature or context.root_count != self.desc.root_count:
            raise ValueError("observed coordinate context conflicts with captured root revision or count")
        # Serialize only the small frame facts. Repeated targets remain references even while staging.
        steps = tuple(
            replace(step, target_indices=None) if step.target_indices is not None else step for step in context.steps
        )
        raw = replace(context, steps=steps).to_dict()
        for step, actual in zip(raw["steps"], context.steps, strict=True):
            if actual.target_indices is not None:
                _validate_indices(actual.target_indices, context)
                _, selection = self.view(actual.target_indices)
                step.pop("target_indices")
                step["target_selection_id"] = selection.id
        text = metadata_json(raw)
        self.contexts[id(context)] = text
        return text

    def freeze(self, observations: SelectionObservations) -> dict[str, Any]:
        origins: dict[int, dict[str, Any]] = {}
        for observed in observations.origins:
            context = self.context(observed.context)
            _validate_indices(observed.indices, observed.context)
            member, _ = self.view(observed.indices)
            if observed.parent_origin_id is not None:
                parent = origins[observed.parent_origin_id]
                parent_id = parent["id"]
                if observed.parent_indices is None:
                    raise ValueError("observed parent origin requires an exact filter input")
                source_input, _ = self.view(observed.parent_indices)
                if source_input.id != parent["member_input_id"]:
                    raise ValueError("observed parent origin conflicts with the filter input")
                source_locator = self.input_locator(source_input.identity_key)
            elif observed.parent_indices == self.desc.indices:
                source_input = self.input
                source_locator, parent_id = self.source
            else:
                if observed.parent_indices is not None:
                    _validate_indices(observed.parent_indices, observed.context)
                source_input = self.root if observed.parent_indices is None else self.view(observed.parent_indices)[0]
                source_locator, parent_id = self.input_locator(source_input.identity_key), None
            signature = CalculatorSignature.from_json(observed.filter_signature)
            if not signature.stable:
                raise ValueError("archived member filters must have a stable signature")
            parameters = metadata_json([p.to_dict() for p in observed.parameters])
            key = metadata_json(
                [
                    source_locator,
                    source_input.identity_key,
                    parent_id,
                    member.identity_key,
                    signature.hash,
                    parameters,
                    context,
                ]
            )
            origins[observed.origin_id] = {
                "id": hashlib.sha256(key.encode()).hexdigest(),
                "source_locator_json": metadata_json(source_locator),
                "source_input_id": source_input.id,
                "parent_origin_id": parent_id,
                "member_input_id": member.id,
                "particle_count": len(observed.indices),
                "filter_hash": signature.hash,
                "filter_signature": signature.to_json(full=True),
                "parameters_json": parameters,
                "context_json": context,
            }
        uses = [
            {
                "origin_id": origins[u.origin_id]["id"],
                "output_path_json": metadata_json(u.output_path),
                "occurrence_path_json": metadata_json(u.occurrence_path),
                "role": u.role,
            }
            for u in observations.uses
        ]
        outputs = [
            {
                "output_path_json": metadata_json(output.output_path),
                "source_locator_json": metadata_json(self.source[0]),
                "context_json": self.context(output.context),
            }
            for output in observations.outputs
        ]
        return {
            "origins": list(origins.values()),
            "uses": uses,
            "outputs": outputs,
            "input_ids": [view.id for view, _ in self.views.values()],
        }


def prepare_observations(
    db: Database,
    session: Session,
    desc: SavedInput,
    input_id: int,
    result: Any,
    *,
    source: tuple[tuple[str, ...], str | None] | None = None,
) -> dict[str, Any] | None:
    observed = result.selection_observations
    if observed is None:
        return None
    return _ObservationArchive(db, session, desc, input_id, source or result._save_source).freeze(observed)


def stage_observations(
    db: Database,
    session: Session,
    desc: SavedInput,
    input_id: int,
    result: Any,
    operation_id: str,
    *,
    source: tuple[tuple[str, ...], str | None] | None = None,
) -> None:
    staged = prepare_observations(db, session, desc, input_id, result, source=source)
    if staged is not None:
        session.add(StagedObservations(operation_id=operation_id, observations_json=metadata_json(staged)))


def publish_origins(session: Session, staged: dict[str, Any]) -> None:
    if staged["input_ids"]:
        session.execute(delete(StagedMemberInput).where(StagedMemberInput.input_id.in_(staged["input_ids"])))
    # Parent origins occur before their descendants. The PK is a digest of the full observation.
    for origin in staged["origins"]:
        session.execute(insert(SelectionOrigin).values(**origin).on_conflict_do_nothing(index_elements=["id"]))


def publish_observations(
    db: Database, session: Session, operation_id: str, version_id: str, root_signature: tuple[Any, ...]
) -> None:
    staged = session.get(StagedObservations, operation_id)
    if staged is None:
        return
    data = json.loads(staged.observations_json)
    for input_id in data["input_ids"]:
        if db._description_in(session, input_id).root_signature != root_signature:
            raise ValueError("staged particle selection belongs to another registered root")
    publish_origins(session, data)
    for row in data["outputs"]:
        session.execute(insert(OutputObservation).values(version_id=version_id, **row).on_conflict_do_nothing())
    for row in data["uses"]:
        session.execute(insert(SelectionUse).values(version_id=version_id, **row).on_conflict_do_nothing())
    session.delete(staged)
