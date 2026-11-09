"""Sample tables: many objects × many calculators, keyed by signature.

A :class:`SampleTable` is the shape a statistical study takes — one row per
galaxy (or halo, or snapshot), one column per measured quantity::

    table = SampleTable(
        {"r_e": ParamContain().filter(Sphere("30 kpc") & FamilyFilter("star")), "m_star": ParamSum("mass")},
        store=SQLAlchemyResultStore("sqlite:///results.sqlite"),
    )
    table.compute(halos)  # each value computed once, then read back from the store
    big = table.where(lambda row: row["m_star"] > 1e10)
    big.column("r_e")  # a numpy array, ready for statistics
    table.to_dataframe()  # pandas; .to_parquet() / .to_hdf() from there
    table.save("sample.json")  # lossless: values with units, and every column's signature

Each column is identified by its calculator's **signature key** (the full hash), and
each row by the object's **simulation identity**
(:func:`~pynbodyext.core.calculate.runtime.sim_identity.snapshot_identity` by
default).  Those are exactly the two halves of a result-store key, so with a store
every cell is a store entry: rerunning a study, adding a column, or adding objects
only computes what is missing.  The table's :class:`SignatureCatalog` maps each
key back to its calculator and expression, so a saved table says precisely what
each column is.

Cells are computed column by column, each as its own run; dependencies two
columns share are not shared between their runs.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pynbodyext.core.calculate.display import InfoView
from pynbodyext.core.calculate.interfaces import is_calculator_node, is_stable_sim_identity
from pynbodyext.core.calculate.runtime.sim_identity import SimIdentityProvider, snapshot_identity
from pynbodyext.core.calculate.store.catalog import SignatureCatalog
from pynbodyext.core.calculate.store.codec import ValueCodec
from pynbodyext.core.calculate.store.identity import sim_identity_key

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

__all__ = ["SAMPLE_SCHEMA", "SampleRow", "SampleTable"]

#: Discriminator of the sample table's JSON file format.
SAMPLE_SCHEMA = "pynbodyext.sample.table/v1"

_ERROR_POLICIES = ("raise", "record")


@dataclass(slots=True)
class SampleRow:
    """One object of the sample: its name, simulation identity and values by column label."""

    name: str
    identity: tuple[Any, ...]
    values: dict[str, Any] = field(default_factory=dict)
    #: ``label -> "ErrorType: message"`` for cells that failed with ``errors="record"``.
    errors: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, label: str) -> Any:
        return self.values.get(label)

    def get(self, label: str, default: Any = None) -> Any:
        value = self.values.get(label)
        return default if value is None else value


def _default_name(identity: tuple[Any, ...], index: int) -> str:
    """A readable row name: ``snap_103``, ``snap_103:halo 7``, ``snap_103:subset 1a2b3c4d``."""
    if not is_stable_sim_identity(identity) or len(identity) < 2:
        return f"object {index}"
    if len(identity) >= 7 and identity[1] == "pynbodyext.logical-snapshot/v1":
        # Escape literal IDs and retain revision/loader distinctions in labels.
        base = json.dumps(identity[2:7], ensure_ascii=False, separators=(",", ":"))
        suffix = identity[7:]
    else:
        base = os.path.basename(str(identity[1])) or str(identity[1])
        suffix = identity[2:]
    if len(suffix) >= 2 and suffix[0] == "halo":
        return f"{base}:halo {suffix[1]}"
    if len(suffix) >= 2 and suffix[0] == "subset":
        return f"{base}:subset {str(suffix[1])[:8]}"
    return base


def _scalar(value: Any) -> Any:
    """A one-element array as a Python number; anything else unchanged."""
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(()).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _units(value: Any) -> str | None:
    units = getattr(value, "units", None)
    if units is None:
        return None
    text = str(units)
    return None if text in {"", "NoUnit()"} else text


class SampleTable(InfoView):
    """Objects (rows) × calculators (columns), every cell keyed for a result store.

    Parameters
    ----------
    columns : mapping of str to calculator, optional
        Column labels and the calculators that compute them.
    store : ResultStore, optional
        Where cells are looked up before computing and written after.  Without
        one, every :meth:`compute` computes.
    sim_identity : SimIdentityProvider, optional
        How an object becomes a row identity; defaults to
        :func:`~pynbodyext.core.calculate.runtime.sim_identity.snapshot_identity`.
    catalog : SignatureCatalog, optional
        Catalog to record the columns in (a new one by default).
    """

    def __init__(
        self,
        columns: Mapping[str, Any] | None = None,
        *,
        store: Any = None,
        sim_identity: SimIdentityProvider | None = None,
        catalog: SignatureCatalog | None = None,
    ) -> None:
        self.store = store
        self.sim_identity: SimIdentityProvider = sim_identity or snapshot_identity
        self.catalog = catalog if catalog is not None else SignatureCatalog()
        #: column label -> signature key
        self._columns: dict[str, str] = {}
        #: signature key -> live calculator (None for a column loaded without one)
        self._calculators: dict[str, Any] = {}
        self._rows: dict[str, SampleRow] = {}
        for label, calculator in (columns or {}).items():
            self.add_column(label, calculator)

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    def add_column(self, label: str, calculator: Any) -> str:
        """Add a column; return its signature key.

        Existing rows get no value for it until :meth:`compute` (or
        :meth:`fill_from_store`) runs.  With a persistent store, a calculator whose
        signature is not stable is computed but never stored, and a warning says so.
        """
        if not isinstance(label, str) or not label:
            raise ValueError("a column label must be a non-empty string")
        if label in self._columns:
            raise ValueError(f"column {label!r} already exists")
        if not is_calculator_node(calculator):
            raise TypeError(f"column {label!r}: expected a calculator, got {type(calculator)!r}")
        signature = calculator.to_signature()
        key = self.catalog.add(signature, label=label)
        if self._persistent() and not signature.stable:
            warnings.warn(
                f"column {label!r} has a session-specific signature; its values will not be stored",
                UserWarning,
                stacklevel=2,
            )
        self._columns[label] = key
        self._calculators[key] = calculator
        return key

    @property
    def columns(self) -> list[str]:
        """Column labels, in the order they were added."""
        return list(self._columns)

    def key(self, label: str) -> str:
        """The signature key of column *label*."""
        return self._columns[label]

    def calculator(self, label: str) -> Any:
        """The calculator of column *label* (rebuilt from the catalog for a loaded table)."""
        key = self._columns[label]
        calculator = self._calculators.get(key)
        if calculator is None:
            calculator = self.catalog.calculator(key)
            self._calculators[key] = calculator
        return calculator

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    @property
    def rows(self) -> list[SampleRow]:
        """Rows, in the order objects were first added."""
        return list(self._rows.values())

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Any:
        return iter(self._rows.values())

    def row(self, name: str) -> SampleRow:
        """The row called *name*."""
        for row in self._rows.values():
            if row.name == name:
                return row
        raise KeyError(f"no row named {name!r}")

    def _persistent(self) -> bool:
        return self.store is not None and bool(getattr(self.store, "persistent", True))

    def _store_for(self, identity: tuple[Any, ...]) -> Any:
        """The store to use for one row: none for a session identity in a persistent store."""
        if self.store is None:
            return None
        if self._persistent() and not is_stable_sim_identity(identity):
            return None
        return self.store

    def add(self, obj: Any, *, name: str | None = None, reuse: bool = True, errors: str = "raise") -> SampleRow:
        """Compute every column for one object and add (or update) its row."""
        if errors not in _ERROR_POLICIES:
            raise ValueError(f"errors must be one of {_ERROR_POLICIES}, got {errors!r}")
        identity = tuple(self.sim_identity(obj))
        identity_key = sim_identity_key(identity)
        row = self._rows.get(identity_key)
        if row is None:
            row = SampleRow(name or _default_name(identity, len(self._rows)), identity)
            self._rows[identity_key] = row
        elif name is not None:
            row.name = name
        store = self._store_for(identity)
        for label in self._columns:
            calculator = self.calculator(label)
            try:
                result = calculator.run(obj, store=store, sim_identity=self.sim_identity, reuse=reuse)
            except Exception as exc:
                if errors == "raise":
                    raise
                row.values[label] = None
                row.errors[label] = f"{type(exc).__name__}: {exc}"
                continue
            row.values[label] = result.value
            row.errors.pop(label, None)
        return row

    def compute(
        self, objects: Iterable[Any], *, names: Iterable[str] | None = None, reuse: bool = True, errors: str = "raise"
    ) -> SampleTable:
        """Add every object in *objects* (see :meth:`add`) and return the table.

        With a persistent store, objects whose identity is session-only (no file,
        or transformed outside the calculator) are computed but not stored; one
        warning counts them.
        """
        if errors not in _ERROR_POLICIES:
            raise ValueError(f"errors must be one of {_ERROR_POLICIES}, got {errors!r}")
        name_list = list(names) if names is not None else None
        session_only = 0
        for index, obj in enumerate(objects):
            name = name_list[index] if name_list is not None else None
            with warnings.catch_warnings():
                # One summary below instead of one engine warning per cell.
                warnings.filterwarnings("ignore", message=r"result of .* was not stored", category=UserWarning)
                row = self.add(obj, name=name, reuse=reuse, errors=errors)
            if self._persistent() and not is_stable_sim_identity(row.identity):
                session_only += 1
        if session_only:
            warnings.warn(
                f"{session_only} object(s) have a session-only identity; their values were computed but not stored",
                UserWarning,
                stacklevel=2,
            )
        return self

    def fill_from_store(self) -> SampleTable:
        """Fill empty cells from the store without computing; return the table."""
        if self.store is None:
            raise ValueError("this table has no store to read from")
        for row in self._rows.values():
            for label, key in self._columns.items():
                if row.values.get(label) is not None:
                    continue
                signature = self.catalog[key]
                hit = self.store.get(sim_signature=row.identity, calculator_signature=signature)
                if hit is not None:
                    row.values[label] = hit.value
                    row.errors.pop(label, None)
        return self

    @classmethod
    def from_store(
        cls,
        store: Any,
        columns: Mapping[str, Any],
        *,
        identities: Iterable[tuple[Any, ...]] | None = None,
        sim_identity: SimIdentityProvider | None = None,
    ) -> SampleTable:
        """Assemble a table from stored results only — nothing is computed.

        Rows are *identities* (default: every simulation identity the store holds a
        result for); a cell with no stored result is ``None``.
        """
        table = cls(columns, store=store, sim_identity=sim_identity)
        for index, given in enumerate(identities if identities is not None else store.sim_signatures()):
            identity = tuple(given)
            table._rows.setdefault(sim_identity_key(identity), SampleRow(_default_name(identity, index), identity))
        return table.fill_from_store()

    # ------------------------------------------------------------------
    # Reading values
    # ------------------------------------------------------------------

    def column(self, label: str) -> Any:
        """The values of column *label*, one per row.

        A float array (``nan`` for a missing value) when every value is a number
        or a one-element array; otherwise a list.
        """
        if label not in self._columns:
            raise KeyError(f"no column {label!r}")
        values = [_scalar(row.values.get(label)) for row in self._rows.values()]
        if all(v is None or (isinstance(v, (int, float, np.number)) and not isinstance(v, bool)) for v in values):
            return np.array([np.nan if v is None else float(v) for v in values])
        return values

    def units(self, label: str) -> str | None:
        """The units of column *label*, read from its first value that has any."""
        for row in self._rows.values():
            text = _units(row.values.get(label))
            if text is not None:
                return text
        return None

    def missing(self) -> list[tuple[str, str]]:
        """``(row name, column label)`` of every empty cell."""
        return [
            (row.name, label) for row in self._rows.values() for label in self._columns if row.values.get(label) is None
        ]

    def where(self, predicate: Callable[[SampleRow], bool]) -> SampleTable:
        """A new table with the rows for which ``predicate(row)`` is true.

        ``row["m_star"]`` reads a value (``None`` when missing); the new table
        shares this one's columns, catalog and store.
        """
        out = SampleTable(store=self.store, sim_identity=self.sim_identity, catalog=self.catalog)
        out._columns = dict(self._columns)
        out._calculators = self._calculators
        out._rows = {identity: row for identity, row in self._rows.items() if predicate(row)}
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def column_metadata(self) -> dict[str, dict[str, Any]]:
        """What each column is: ``{label: {"key", "expression", "units"}}``."""
        return {
            label: {"key": key, "expression": self.catalog[key].pretty(), "units": self.units(label)}
            for label, key in self._columns.items()
        }

    def to_records(self, *, identity: bool = False) -> list[dict[str, Any]]:
        """One plain dict per row: ``name``, optionally ``identity``, then the values."""
        records = []
        for row in self._rows.values():
            record: dict[str, Any] = {"name": row.name}
            if identity:
                record["identity"] = row.identity
            record.update({label: _scalar(row.values.get(label)) for label in self._columns})
            records.append(record)
        return records

    def to_dataframe(self) -> Any:
        """The table as a :class:`pandas.DataFrame` indexed by row name.

        One-element arrays become numbers (units are in the metadata).  The
        column metadata and the row identities are kept in ``df.attrs["pynbodyext"]``,
        so the frame still says what each column computed.  Save it with pandas
        (``df.to_parquet``, ``df.to_hdf``), which needs ``pyarrow`` or ``tables``.
        """
        import pandas as pd

        df = pd.DataFrame.from_records(self.to_records(), columns=["name", *self._columns]).set_index("name")
        df.attrs["pynbodyext"] = {
            "schema": SAMPLE_SCHEMA,
            "columns": self.column_metadata(),
            "identities": {row.name: [str(part) for part in row.identity] for row in self._rows.values()},
        }
        return df

    def to_dict(self, *, codec: ValueCodec | None = None) -> dict[str, Any]:
        """A lossless JSON-compatible description: catalog, columns, identities, values."""
        codec = codec or ValueCodec()
        return {
            "schema": SAMPLE_SCHEMA,
            "catalog": self.catalog.to_dict(),
            "columns": [{"label": label, "key": key} for label, key in self._columns.items()],
            "rows": [
                {
                    "name": row.name,
                    "identity": codec.encode(row.identity),
                    "values": {label: codec.encode(value) for label, value in row.values.items()},
                    "errors": dict(row.errors),
                }
                for row in self._rows.values()
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        store: Any = None,
        sim_identity: SimIdentityProvider | None = None,
        codec: ValueCodec | None = None,
    ) -> SampleTable:
        """Rebuild a table from :meth:`to_dict`.  Column calculators are rebuilt on first use."""
        if data.get("schema") != SAMPLE_SCHEMA:
            raise ValueError(f"unsupported sample table schema: {data.get('schema')!r}")
        codec = codec or ValueCodec()
        table = cls(store=store, sim_identity=sim_identity, catalog=SignatureCatalog.from_dict(data["catalog"]))
        for column in data["columns"]:
            key = table.catalog.key(column["key"])
            table._columns[column["label"]] = key
            table._calculators.setdefault(key, None)
        for item in data["rows"]:
            identity = tuple(codec.decode(item["identity"]))
            values = {label: codec.decode(value) for label, value in item["values"].items()}
            table._rows[sim_identity_key(identity)] = SampleRow(item["name"], identity, values, dict(item.get("errors", {})))
        return table

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the table (values with units, and every column's signature) as JSON."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], *, store: Any = None, sim_identity: SimIdentityProvider | None = None
    ) -> SampleTable:
        """Read a table written by :meth:`save`."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle), store=store, sim_identity=sim_identity)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_title(self) -> str:
        return "SampleTable"

    def _display_rows(self) -> list[tuple[str, Any]]:
        rows: list[tuple[str, Any]] = [
            ("rows", len(self._rows)),
            ("columns", len(self._columns)),
            ("missing cells", len(self.missing())),
            ("store", type(self.store).__name__ if self.store is not None else "-"),
        ]
        for label, key in self._columns.items():
            expression = self.catalog[key].pretty()
            if len(expression) > 120:
                expression = expression[:117] + "..."
            rows.append((label, f"{key[:12]}  {expression}"))
        return rows

    def __repr__(self) -> str:
        return f"SampleTable(rows={len(self._rows)}, columns={self.columns!r})"
