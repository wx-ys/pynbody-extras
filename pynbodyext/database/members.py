"""Navigate exact particle inputs and their independently confirmed origins."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, overload

from sqlalchemy import inspect, select

from pynbodyext.core.calculate.nodes.expr import ConstantProperty
from pynbodyext.core.calculate.nodes.filters import FilterBase
from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.filters import FamilyFilter
from pynbodyext.orm.models import DataObject

from .member_models import SelectionOrigin
from .member_storage import metadata_json, prepare_observations, publish_origins, read_context
from .objects import Input

if TYPE_CHECKING:
    from pynbodyext.core.calculate.runtime.selection_observations import SelectionContext

    from .database import Database


def _selector(selector: Any) -> FilterBase:
    result = FamilyFilter(selector) if isinstance(selector, str) else selector
    if not isinstance(result, FilterBase):
        raise TypeError("member selector must be a Filter instance or a family name")
    if not result.to_signature().stable:
        raise ValueError("member selector requires a stable filter signature")
    return result


class Origin:
    """One observed filter application, reached without opening particle data."""

    def __init__(self, db: Database, row: SelectionOrigin) -> None:
        self._database = db
        self._row = row

    @property
    def id(self) -> str:
        return self._row.id

    @property
    def source_locator(self) -> tuple[str, ...]:
        return tuple(json.loads(self._row.source_locator_json))

    @property
    def parent_origin_id(self) -> str | None:
        return self._row.parent_origin_id

    @property
    def filter_signature(self) -> CalculatorSignature:
        return CalculatorSignature.from_json(self._row.filter_signature)

    @property
    def parameters(self) -> tuple[Any, ...]:
        from pynbodyext.core.calculate.runtime.selection_observations import ResolvedParameter

        return tuple(ResolvedParameter.from_dict(p) for p in json.loads(self._row.parameters_json))

    @property
    def context(self) -> SelectionContext:
        self._database._check()
        with self._database._repo.session() as session:
            return read_context(session, self._row.context_json)

    @property
    def members(self) -> Members:
        return Members(member_from_origins(self._database, [self._row]))

    @property
    def used_by(self) -> Any:
        return member_from_origins(self._database, [self._row]).used_by

    def load(self, *, frame: str = "snapshot", location_id: int | None = None) -> Any:
        return member_from_origins(self._database, [self._row]).load(frame=frame, location_id=location_id)

    def __repr__(self) -> str:
        return f"<selection origin {self.id} {self.filter_signature.pretty()}>"


class Member(Input):
    """An exact ordered particle sequence with one or more matching origins."""

    def __init__(self, db: Database, rows: Sequence[SelectionOrigin], key: str) -> None:
        self._database = db
        self._input_id = rows[0].member_input_id
        self._key = key
        self.origins = tuple(Origin(db, row) for row in rows)

    @property
    def origin(self) -> Origin:
        if len(self.origins) != 1:
            raise ValueError(f"ambiguous member origin; choose one of {[o.id for o in self.origins]}")
        return self.origins[0]

    @property
    def count(self) -> int:
        return self.origins[0]._row.particle_count

    @property
    def used_by(self) -> Any:
        from .result_queries import used_by

        return used_by(self)

    def load(self, *, frame: str = "snapshot", location_id: int | None = None) -> Any:
        """Load original particle coordinates or replay this origin's observed frame."""
        if frame == "selection":
            from .selection_frames import load_in_frame

            return load_in_frame(self._database, self.input_id, self.origin.context, location_id=location_id)
        if frame not in ("snapshot", "original"):
            raise ValueError("member frame must be 'snapshot' or 'selection'")
        sim = super().load(location_id=location_id)
        root = sim.ancestor
        layout = tuple((str(family), len(root[family])) for family in root.families())
        if any(tuple(map(tuple, json.loads(o._row.context_json)["family_layout"])) != layout for o in self.origins):
            raise ValueError("loaded root family layout differs from observed membership; use its original revision")
        return sim

    def __repr__(self) -> str:
        return f"<member of {self.count} particles with {len(self.origins)} origin(s)>"


def member_from_origins(db: Database, rows: Iterable[SelectionOrigin], *, key: str | None = None) -> Member:
    """Construct a handle for rows already selected from one metadata snapshot."""
    origins = tuple(rows)
    if len({row.member_input_id for row in origins}) != 1:
        raise ValueError("member origins must identify one exact particle input")
    if key is None:
        with db._repo.session() as session:
            key = session.scalars(
                select(DataObject.identity_key).where(DataObject.id == origins[0].member_input_id)
            ).one()
    return Member(db, origins, key)


class MemberMatches(Sequence[Member]):
    """An immutable list with one entry per matching origin."""

    def __init__(self, members: Iterable[Member]) -> None:
        self._members = tuple(members)

    def __len__(self) -> int:
        return len(self._members)

    @overload
    def __getitem__(self, index: int) -> Member: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Member, ...]: ...
    def __getitem__(self, index: int | slice) -> Member | tuple[Member, ...]:
        return self._members[index]

    def one(self) -> Member:
        if len(self) != 1:
            raise ValueError(f"expected one member origin, found {len(self)}: {[m.origin.id for m in self]}")
        return self._members[0]


class _RegisterMember(ConstantProperty[int]):
    """Consume the selected view without issuing a scientific empty-input warning."""

    warns_on_empty_input = False


class Members:
    """Confirmed origins immediately below one scientific source or origin branch."""

    def __init__(self, source: Input) -> None:
        self._source = source
        self._database = source._database

    def _rows(self, filter_selector: Any = None, transform: Any = None, context: Any = None) -> list[Any]:
        from .recipe_queries import matches_calculator, matches_transforms

        db = self._database
        db._check()
        if not inspect(db._engine).has_table(SelectionOrigin.__tablename__):
            return []
        stmt = (
            select(SelectionOrigin, DataObject.identity_key)
            .join(DataObject, SelectionOrigin.member_input_id == DataObject.id)
            .where(SelectionOrigin.source_locator_json == metadata_json(self._source.object_locator))
            .order_by(SelectionOrigin.id)
        )
        if isinstance(self._source, Member):
            parent_match = SelectionOrigin.parent_origin_id.in_([o.id for o in self._source.origins])
            # Explicit work on an unpinned exact input records no scientific parent.
            stmt = stmt.where(
                parent_match | SelectionOrigin.parent_origin_id.is_(None)
                if len(self._source.origins) > 1
                else parent_match
            )
        if filter_selector is not None and not isinstance(filter_selector, type):
            filter_selector = _selector(filter_selector)
            stmt = stmt.where(SelectionOrigin.filter_hash == filter_selector.to_signature().hash)
        result = []
        with db._repo.session() as session:
            for row, key in session.execute(stmt):
                if isinstance(filter_selector, type):
                    signature = CalculatorSignature.from_json(row.filter_signature)
                    if not matches_calculator(signature.payload, filter_selector):
                        continue
                if transform is not None or context is not None:
                    actual = read_context(session, row.context_json)
                    if context is not None and (not actual.complete or not context.complete or actual != context):
                        continue
                    if transform is not None:
                        if any(step.signature is None for step in actual.steps):
                            continue
                        methods = [
                            CalculatorSignature.from_json(step.signature).payload
                            for step in actual.steps
                            if step.signature is not None
                        ]
                        if not matches_transforms(methods, transform):
                            continue
                result.append((row, key))
        return result

    def where(self, *, filter: Any = None, transform: Any = None, context: Any = None) -> MemberMatches:
        return MemberMatches(
            member_from_origins(self._database, [row], key=key) for row, key in self._rows(filter, transform, context)
        )

    def __getitem__(self, selector: Any) -> Member:
        selector = _selector(selector)
        rows = self._rows(selector)
        if not rows:
            raise KeyError(
                f"no registered member for {selector.to_signature().pretty()}; use members.register(selector)"
            )
        if len({row.member_input_id for row, _ in rows}) != 1:
            raise ValueError(f"ambiguous member inputs; choose among origins {[row.id for row, _ in rows]}")
        return member_from_origins(self._database, (row for row, _ in rows), key=rows[0][1])

    def register(self, selector: Any) -> Member:
        """Establish a member in snapshot coordinates, keeping no Run or result."""
        db = self._database
        db._check_write()
        selector = _selector(selector)
        source = self._source
        parent_id = source.origin.id if isinstance(source, Member) and len(source.origins) == 1 else None
        original = []
        for candidate in self.where(filter=selector):
            if candidate.origin.parent_origin_id == parent_id and not candidate.origin.context.steps:
                original.append(candidate)
        if len(original) == 1:
            return original[0]

        # The constant only consumes the view. It reads no field; EvalEngine resolves the filter and parameters once.
        exact = source._materialized()
        consumer = _RegisterMember(0).filter(selector)
        result = consumer.run(source, mode="manual", errors="raise")
        with db._repo.session() as session, session.begin():
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            staged = prepare_observations(db, session, result._save_input, exact.input_id, result)
            if staged is None:
                raise ValueError("filter did not record an exact particle selection")
            publish_origins(session, staged)
            direct = {
                u["origin_id"]
                for u in staged["uses"]
                if u["role"] == "output_filter"
                and u["output_path_json"] == "[]"
                and u["occurrence_path_json"] == metadata_json(("scope", "filter"))
            }
            if len(direct) != 1:
                raise ValueError("explicit registration requires one direct filter consumption")
            row = session.scalars(select(SelectionOrigin).where(SelectionOrigin.id == direct.pop())).one()
            if row.filter_hash != selector.to_signature().hash:
                raise ValueError("observed direct filter differs from the registered selector")
        return member_from_origins(db, [row])

    def __iter__(self) -> Iterator[Member]:
        return iter(self.where())

    def __repr__(self) -> str:
        return f"<members of {self._source.name!r}>"
