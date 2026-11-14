"""Detached database handles and consistent, metadata-only collection browsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from pynbodyext.core.calculate.result.signature import CalculatorSignature
from pynbodyext.orm.loading import parse_input_identity
from pynbodyext.orm.models import DataObject, Simulation as SimulationRow, Timestep

from .models import InputInfo, ResultAlias, ResultSummary


class Input:
    """A registered root or view. Reading results never loads particles."""

    def __init__(self, database: Any, input_id: int) -> None:
        self._database = database
        self._input_id = input_id
        self._key = database._repo.get_input(input_id).identity_key

    @property
    def input_id(self) -> int:
        return self._input_id

    @property
    def _row(self) -> Any:
        self._database._check()
        return self._database._repo.get_input(self.input_id)

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        row = self._row
        parsed = parse_input_identity(row.signature)
        if row.parent_id is not None:
            return row.identity_key
        return parsed.logical.timestep_id if parsed.logical else Path(row.signature[1]).name

    @property
    def results(self) -> Any:
        return Results(self)

    @property
    def selections(self) -> Any:
        return InputCollection(self._database, parent=self.input_id)

    @property
    def catalogues(self) -> Any:
        from .catalogues import Catalogues

        return Catalogues(self)

    def _default_catalogue(self) -> Any:
        db = self._database
        db._check()
        with db._repo.session() as session:
            info = session.get(InputInfo, self.input_id)
            default = None if info is None else info.default_catalogue_id
        catalogues = list(self.catalogues)
        if default is not None:
            matches = [c for c in catalogues if c.catalogue_id == default]
            if len(matches) == 1:
                return matches[0]
        if len(catalogues) == 1:
            return catalogues[0]
        if not catalogues:
            raise KeyError("snapshot has no registered catalogues")
        raise ValueError("ambiguous catalogue; select snap.catalogues[name] or set a default")

    @property
    def halos(self) -> Any:
        return self._default_catalogue().halos

    @property
    def subhalos(self) -> Any:
        return self._default_catalogue().subhalos

    def load(self, *, location_id: int | None = None) -> Any:
        """Reload the registered root or exact selection, preserving its units."""
        return self._database._load(self.input_id, location_id=location_id)[0]

    def _materialized(self) -> Input:
        return self

    def __repr__(self) -> str:
        return f"<{self._row.kind} {self.name!r} key={self.key!r}>"


class InputCollection:
    def __init__(
        self, database: Any, *, roots: bool = False, parent: int | None = None, simulation_id: int | None = None
    ) -> None:
        self._database = database
        self._roots = roots
        self._parent = parent
        self._simulation_id = simulation_id

    def _rows(self) -> list[Any]:
        db = self._database
        db._check()
        stmt = select(DataObject).order_by(DataObject.id)
        if self._roots:
            stmt = stmt.where(DataObject.parent_id.is_(None))
        if self._parent is not None:
            stmt = stmt.where(DataObject.parent_id == self._parent)
        if self._simulation_id is not None:
            stmt = stmt.join(Timestep).where(Timestep.simulation_id == self._simulation_id)
        with db._repo.session() as session:
            return list(session.scalars(stmt))

    def __iter__(self) -> Any:
        return iter([Input(self._database, row.id) for row in self._rows()])

    def __len__(self) -> int:
        return len(self._rows())

    def __getitem__(self, key: str) -> Input:
        items = list(self)
        exact = [item for item in items if item.key == key]
        found = exact or [item for item in items if item.name == key]
        if not found:
            raise KeyError(key)
        if len(found) != 1:
            raise ValueError(f"ambiguous snapshot name {key!r}; select one of {[item.key for item in found]}")
        return found[0]

    def __repr__(self) -> str:
        rows = self._rows()
        return f"<{len(rows)} registered inputs>\n" + "\n".join(f"{row.kind}: {row.identity_key}" for row in rows[:20])


class Results:
    def __init__(self, source: Input) -> None:
        self._source = source

    def __getitem__(self, calculator_or_name: Any) -> Any:
        source = self._source._materialized()
        db = source._database
        db._check()
        if isinstance(calculator_or_name, str):
            with db._repo.session() as session:
                alias = session.get(ResultAlias, (source.input_id, calculator_or_name))
                if alias is None:
                    raise KeyError(calculator_or_name)
                from pynbodyext.orm.models import Recipe

                recipe = session.get(Recipe, alias.calculator_hash)
                signature = CalculatorSignature.from_json(recipe.signature_text)
        else:
            signature = calculator_or_name.to_signature()
        result = db._store.load(sim_signature=source._row.signature, calculator_signature_text=signature.to_json())
        if result is None:
            raise KeyError(calculator_or_name)
        return db._attach(result, db._description(source.input_id), signature)

    def _current(self) -> Any:
        db = self._source._database
        db._check()
        versions, current = db._store._versions, db._store._table
        return versions.join(current, versions.c.version_id == current.c.version_id)

    def __len__(self) -> int:
        db = self._source._database
        with db._repo.session() as session:
            return int(
                session.scalar(
                    select(func.count())
                    .select_from(self._current())
                    .where(db._store._versions.c.sim_signature == self._source.key)
                )
                or 0
            )

    def __repr__(self) -> str:
        import json

        db = self._source._database
        versions = db._store._versions
        joined = self._current().outerjoin(ResultSummary, versions.c.version_id == ResultSummary.version_id)
        with db._repo.session() as session:
            rows = list(
                session.execute(
                    select(
                        versions.c.calculator_signature_hash,
                        versions.c.version_id,
                        versions.c.created_at,
                        ResultSummary.summary_json,
                    )
                    .select_from(joined)
                    .where(versions.c.sim_signature == self._source.key)
                    .order_by(versions.c.calculator_signature_hash)
                    .limit(20)
                )
            )
            hashes = [row.calculator_signature_hash for row in rows]
            aliases = list(
                session.scalars(
                    select(ResultAlias).where(
                        ResultAlias.input_id == self._source.input_id, ResultAlias.calculator_hash.in_(hashes)
                    )
                )
            )
        names = {a.calculator_hash: a.name for a in aliases}
        lines = []
        for row in rows:
            info = json.loads(row.summary_json) if row.summary_json is not None else {"type": "summary unavailable"}
            lines.append(
                f"{names.get(row.calculator_signature_hash, '')} | {str(info.get('calculator', ''))[:200]} | "
                f"{row.calculator_signature_hash[:12]} | {info.get('type')} shape={info.get('shape')} "
                f"{info.get('units', '')} | {row.version_id} | {row.created_at}"
            )
        return f"<{len(self)} results: name | calculator | hash | type/shape/units | version | saved>\n" + "\n".join(
            lines
        )


class Simulation:
    def __init__(self, database: Any, row: Any) -> None:
        self._database = database
        self.id, self.name = row.id, row.logical_id

    @property
    def snapshots(self) -> InputCollection:
        return InputCollection(self._database, roots=True, simulation_id=self.id)


class SimulationCollection:
    def __init__(self, database: Any) -> None:
        self._database = database

    def _rows(self) -> list[Any]:
        self._database._check()
        with self._database._repo.session() as session:
            return list(
                session.scalars(select(SimulationRow).where(SimulationRow.project_id == self._database.project))
            )

    def __len__(self) -> int:
        return len(self._rows())

    def __iter__(self) -> Any:
        return iter([Simulation(self._database, row) for row in self._rows()])

    def __getitem__(self, name: str) -> Simulation:
        for row in self._rows():
            if row.logical_id == name:
                return Simulation(self._database, row)
        raise KeyError(name)

    def __repr__(self) -> str:
        rows = self._rows()
        return f"<{len(rows)} simulations>\n" + "\n".join(row.logical_id for row in rows[:20])
