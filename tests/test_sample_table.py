"""Signature catalog and sample tables: keys that can be read back, cells computed once.

A :class:`SignatureCatalog` turns a key (a full signature hash) back into the
calculator it identifies.  A :class:`SampleTable` is objects × calculators, each
cell keyed exactly like a result-store entry, so a study only ever computes what
the store does not already hold.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pynbody
import pytest
from pynbody.array import SimArray

from pynbodyext.core.calculate import (
    InMemoryResultStore,
    LambdaProperty,
    Param,
    PropertyBase,
    SampleTable,
    SignatureCatalog,
)
from pynbodyext.filters import Sphere

CALLS: list[str] = []


@PropertyBase.dataclass
class Counted(PropertyBase[float]):
    """``sum(field) * scale``, recording every real computation."""

    signature_name = "tests.sample.Counted"

    field: Param[str] = "mass"
    scale: Param[float] = 1.0

    def calculate(self, sim, params=None) -> float:
        CALLS.append(self.field)
        return float(sim[self.field].sum()) * self.scale


@PropertyBase.dataclass
class TotalMassWithUnits(PropertyBase[SimArray]):
    signature_name = "tests.sample.TotalMassWithUnits"

    def calculate(self, sim, params=None) -> SimArray:
        return SimArray(float(sim["mass"].sum()), "Msol")


@PropertyBase.dataclass
class FailsOnB(PropertyBase[float]):
    signature_name = "tests.sample.FailsOnB"

    def calculate(self, sim, params=None) -> float:
        if str(sim.filename).endswith("b"):
            raise RuntimeError("no stars")
        return 1.0


class _PersistentMemoryStore(InMemoryResultStore):
    persistent = True


@pytest.fixture(autouse=True)
def _reset_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def obj(path, scale: float = 1.0, n: int = 5):
    sim = pynbody.new(n)
    sim["mass"] = np.arange(float(n)) * scale
    sim["temp"] = np.ones(n) * scale
    sim._filename = str(path)
    return sim


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_a_key_reads_back_as_its_calculator() -> None:
    catalog = SignatureCatalog()
    calc = Counted(scale=2.0).filter(Sphere("30 kpc"))
    key = catalog.add(calc, label="m2")

    assert key == calc.to_signature().hash
    assert catalog[key].pretty() == calc.to_signature().pretty()
    assert catalog[key[:12]].hash == key  # the short hash shown in cards
    assert catalog["m2"].hash == key
    assert catalog.calculator(key[:8]).to_signature().hash == key
    assert catalog.label(key) == "m2"
    assert calc in catalog and key[:10] in catalog and "nope" not in catalog


def test_a_short_or_ambiguous_prefix_is_refused() -> None:
    catalog = SignatureCatalog()
    keys = [catalog.add(Counted(scale=float(i))) for i in range(40)]
    with pytest.raises(KeyError):
        catalog[keys[0][:7]]  # shorter than eight characters
    # force an ambiguity on a common prefix
    catalog._entries["0" * 64] = catalog._entries[keys[0]].__class__("0" * 64, catalog[keys[0]])
    catalog._entries["0" * 63 + "1"] = catalog._entries[keys[1]].__class__("0" * 63 + "1", catalog[keys[1]])
    with pytest.raises(KeyError, match="ambiguous"):
        catalog["0" * 12]


def test_a_label_names_one_calculator() -> None:
    catalog = SignatureCatalog()
    key = catalog.add(Counted(), label="m")
    assert catalog.add(Counted(), label="m") == key  # the same calculator again is fine
    with pytest.raises(ValueError, match="already names"):
        catalog.add(Counted(scale=3.0), label="m")
    catalog.add(Counted(), label="mass")  # relabel
    assert catalog.label(key) == "mass"
    assert "m" not in catalog
    catalog.remove("mass")
    assert len(catalog) == 0


def test_find_searches_labels_and_expressions() -> None:
    catalog = SignatureCatalog()
    catalog.add(Counted(field="temp"), label="t_sum")
    catalog.add(Sphere("30 kpc"))
    assert [entry.label for entry in catalog.find("T_SUM")] == ["t_sum"]
    assert [entry.expression for entry in catalog.find("30 kpc")] == ['Sphere("30 kpc", (0, 0, 0))']


def test_a_catalog_round_trips_through_a_file(tmp_path) -> None:
    catalog = SignatureCatalog()
    key = catalog.add(Counted(scale=2.0), label="m2")
    catalog.add(LambdaProperty(lambda sim: 1.0), label="lam")  # kept, but cannot be rebuilt
    catalog.save(tmp_path / "catalog.json")

    loaded = SignatureCatalog.load(tmp_path / "catalog.json")
    assert list(loaded) == list(catalog)
    assert loaded.calculator("m2").to_signature().hash == key
    with pytest.raises(ValueError, match="not constructible"):
        loaded.calculator("lam")


def test_a_catalog_file_with_a_wrong_key_is_refused(tmp_path) -> None:
    catalog = SignatureCatalog()
    catalog.add(Counted(), label="m")
    data = catalog.to_dict()
    data["entries"][0]["key"] = "f" * 64
    with pytest.raises(ValueError, match="hashes to"):
        SignatureCatalog.from_dict(data)


def test_a_store_lists_its_calculators_and_objects(tmp_path) -> None:
    store = InMemoryResultStore()
    a, b = obj(tmp_path / "a"), obj(tmp_path / "b")
    for sim in (a, b):
        Counted().run(sim, store=store, sim_identity=lambda s: ("sim", str(s.filename)))
    Counted(field="temp").run(a, store=store, sim_identity=lambda s: ("sim", str(s.filename)))

    catalog = store.catalog()
    assert set(catalog) == {Counted().to_signature().hash, Counted(field="temp").to_signature().hash}
    assert store.sim_signatures() == [("sim", str(tmp_path / "a")), ("sim", str(tmp_path / "b"))]


# ---------------------------------------------------------------------------
# Computing a table
# ---------------------------------------------------------------------------


def test_each_cell_is_computed_once(tmp_path) -> None:
    store = InMemoryResultStore()
    objects = [obj(tmp_path / "a", 1.0), obj(tmp_path / "b", 2.0)]
    table = SampleTable({"m": Counted(), "t": Counted(field="temp")}, store=store).compute(objects)
    assert CALLS == ["mass", "temp", "mass", "temp"]

    # a new table over the same store, as in a later session: nothing is recomputed
    again = SampleTable({"m": Counted(), "t": Counted(field="temp")}, store=store).compute(objects)
    assert CALLS == ["mass", "temp", "mass", "temp"]
    assert again.to_records() == table.to_records()


def test_a_new_column_computes_only_itself(tmp_path) -> None:
    store = InMemoryResultStore()
    objects = [obj(tmp_path / "a"), obj(tmp_path / "b")]
    table = SampleTable({"m": Counted()}, store=store).compute(objects)
    CALLS.clear()

    table.add_column("m3", Counted(scale=3.0))
    assert table.missing() == [("a", "m3"), ("b", "m3")]
    table.compute(objects)
    assert CALLS == ["mass", "mass"]  # only the new column
    assert list(table.column("m3")) == [30.0, 30.0]


def test_equal_numbers_share_a_cell(tmp_path) -> None:
    store = InMemoryResultStore()
    objects = [obj(tmp_path / "a")]
    SampleTable({"m": Counted(scale=2)}, store=store).compute(objects)
    SampleTable({"m": Counted(scale=2.0)}, store=store).compute(objects)
    assert CALLS == ["mass"]


def test_reuse_false_recomputes(tmp_path) -> None:
    store = InMemoryResultStore()
    objects = [obj(tmp_path / "a")]
    table = SampleTable({"m": Counted()}, store=store)
    table.compute(objects)
    table.compute(objects, reuse=False)
    assert CALLS == ["mass", "mass"]


def test_a_table_without_a_store_always_computes(tmp_path) -> None:
    objects = [obj(tmp_path / "a")]
    table = SampleTable({"m": Counted()})
    table.compute(objects)
    table.compute(objects)
    assert CALLS == ["mass", "mass"]
    assert len(table) == 1  # the same object is one row


def test_rows_are_named_after_the_object(tmp_path) -> None:
    sim = obj(tmp_path / "snap_103", n=6)
    halo = sim[np.array([0, 2, 5])]
    halo._halo_number = 7
    subset = sim[np.array([1, 3])]
    table = SampleTable({"m": Counted()}).compute([sim, halo, subset])
    names = [row.name for row in table]
    assert names[:2] == ["snap_103", "snap_103:halo 7"]
    assert names[2].startswith("snap_103:subset ")
    assert SampleTable({"m": Counted()}).compute([sim], names=["main"]).rows[0].name == "main"


def test_errors_can_be_recorded_instead_of_raised(tmp_path) -> None:
    objects = [obj(tmp_path / "a"), obj(tmp_path / "b")]
    with pytest.raises(RuntimeError, match="no stars"):
        SampleTable({"f": FailsOnB()}).compute(objects)

    table = SampleTable({"f": FailsOnB(), "m": Counted()}).compute(objects, errors="record")
    b = table.row("b")
    assert b["f"] is None and b["m"] == 10.0
    assert b.errors == {"f": "RuntimeError: no stars"}
    assert table.missing() == [("b", "f")]


@pytest.mark.parametrize("names", [["one"], ["one", "two", "three"]])
def test_compute_checks_name_count_before_calculating_any_row(tmp_path, names) -> None:
    store = InMemoryResultStore()
    table = SampleTable({"mass": Counted()}, store=store)
    objects = (obj(tmp_path / basename) for basename in ("a", "b"))
    with pytest.raises(ValueError, match="names.*objects|objects.*names"):
        table.compute(objects, names=iter(names))

    assert len(table) == 0
    assert CALLS == []
    assert store.list_refs() == []


def test_row_lookup_reports_ambiguous_display_names(tmp_path) -> None:
    objects = [obj(tmp_path / folder / "snap") for folder in ("a", "b")]
    table = SampleTable().compute(objects)
    assert [row.name for row in table] == ["snap", "snap"]
    with pytest.raises(ValueError, match="ambiguous.*snap|snap.*ambiguous"):
        table.row("snap")
    assert table.rows[0].identity != table.rows[1].identity


# ---------------------------------------------------------------------------
# Persistent stores
# ---------------------------------------------------------------------------


def test_session_only_objects_are_computed_but_not_stored() -> None:
    store = _PersistentMemoryStore()
    fileless = pynbody.new(3)
    fileless["mass"] = np.ones(3)
    with pytest.warns(UserWarning, match="1 object"):
        table = SampleTable({"m": Counted(), "t": Counted(scale=2.0)}, store=store).compute([fileless])
    assert table.rows[0]["m"] == 3.0
    assert store.list_refs() == []


def test_an_unstable_column_is_flagged_when_added() -> None:
    with pytest.warns(UserWarning, match="session-specific"):
        SampleTable({"lam": LambdaProperty(lambda sim: 1.0)}, store=_PersistentMemoryStore())


def test_a_persistent_store_survives_a_reload(tmp_path) -> None:
    store = _PersistentMemoryStore()
    SampleTable({"m": Counted()}, store=store).compute([obj(tmp_path / "a")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = SampleTable({"m": Counted()}, store=store).compute([obj(tmp_path / "a")])
    assert CALLS == ["mass"]
    assert again.rows[0]["m"] == 10.0


# ---------------------------------------------------------------------------
# Reading, filtering, exporting
# ---------------------------------------------------------------------------


def _table(tmp_path) -> SampleTable:
    objects = [obj(tmp_path / name, scale) for name, scale in (("a", 1.0), ("b", 2.0), ("c", 4.0))]
    return SampleTable({"m": Counted(), "mu": TotalMassWithUnits()}).compute(objects)


def test_columns_filters_and_units(tmp_path) -> None:
    table = _table(tmp_path)
    mass = table.column("m")
    assert isinstance(mass, np.ndarray)
    assert np.array_equal(mass, [10.0, 20.0, 40.0])
    assert table.units("mu") == "Msol"
    assert table.units("m") is None
    heavy = table.where(lambda row: row["m"] > 15)
    assert [row.name for row in heavy] == ["b", "c"]
    assert heavy.columns == table.columns
    assert np.isnan(table.where(lambda row: False).column("m")).size == 0


def test_a_missing_value_is_nan_in_a_numeric_column(tmp_path) -> None:
    table = _table(tmp_path)
    table.rows[1].values["m"] = None
    column = table.column("m")
    assert column[0] == 10.0 and np.isnan(column[1])


def test_complex_column_preserves_imaginary_values_and_missing_rows(tmp_path) -> None:
    table = SampleTable({"wave": LambdaProperty(lambda sim: np.complex64(2 + 3j))})
    table.add(obj(tmp_path / "a"))
    table.add(obj(tmp_path / "b")).values["wave"] = 10**30
    table.add(obj(tmp_path / "c")).values.pop("wave")

    values = table.column("wave")
    assert isinstance(values, np.ndarray)
    assert np.issubdtype(values.dtype, np.complexfloating)
    assert values[0] == 2 + 3j
    assert values[1] == complex(10**30)
    assert np.isnan(values[2])


def test_to_dataframe_keeps_what_each_column_is(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    table = _table(tmp_path)
    df = table.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["a", "b", "c"]
    assert list(df["mu"]) == [10.0, 20.0, 40.0]
    meta = df.attrs["pynbodyext"]["columns"]
    assert meta["m"]["key"] == Counted().to_signature().hash
    assert meta["m"]["expression"] == 'Counted("mass", 1.0)' == Counted().to_signature().pretty()
    assert meta["mu"]["units"] == "Msol"
    json.dumps(df.attrs)  # the metadata itself is plain JSON


def test_save_and_load_are_lossless(tmp_path) -> None:
    table = _table(tmp_path)
    table.save(tmp_path / "sample.json")
    loaded = SampleTable.load(tmp_path / "sample.json")

    assert loaded.columns == table.columns
    assert [row.identity for row in loaded] == [row.identity for row in table]
    assert loaded.to_records() == table.to_records()
    mu = loaded.rows[0]["mu"]
    assert isinstance(mu, SimArray) and str(mu.units) == "Msol"
    # the columns can be computed again from the saved signatures
    CALLS.clear()
    loaded.compute([obj(tmp_path / "d", 8.0)])
    assert loaded.row("d")["m"] == 80.0 and CALLS == ["mass"]


def test_from_store_assembles_without_computing(tmp_path) -> None:
    store = InMemoryResultStore()
    objects = [obj(tmp_path / "a"), obj(tmp_path / "b", 2.0)]
    SampleTable({"m": Counted()}, store=store).compute(objects)
    CALLS.clear()

    table = SampleTable.from_store(store, {"m": Counted(), "t": Counted(field="temp")})
    assert CALLS == []
    assert [row.name for row in table] == ["a", "b"]
    assert list(table.column("m")) == [10.0, 20.0]
    assert table.missing() == [("a", "t"), ("b", "t")]


def test_column_labels_are_checked() -> None:
    table = SampleTable({"m": Counted()})
    with pytest.raises(ValueError, match="already exists"):
        table.add_column("m", Counted(scale=2.0))
    with pytest.raises(TypeError):
        table.add_column("x", lambda sim: 1.0)
    with pytest.raises(ValueError):
        table.add_column("", Counted())
    with pytest.raises(ValueError, match="errors"):
        table.compute([], errors="ignore")
    with pytest.raises(ValueError, match="errors"):
        table.add(pynbody.new(1), errors="ignore")


def test_repr_and_card_describe_the_columns(tmp_path) -> None:
    table = _table(tmp_path)
    assert repr(table) == "SampleTable(rows=3, columns=['m', 'mu'])"
    rows = dict(table._display_rows())
    assert rows["rows"] == 3 and rows["missing cells"] == 0
    assert rows["m"].startswith(Counted().to_signature().hash[:12])
    assert isinstance(table._repr_html_(), str)
    assert isinstance(table.catalog._repr_html_(), str)
