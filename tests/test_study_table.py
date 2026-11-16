"""Portable StudyTable acceptance scenarios."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from decimal import Decimal

import numpy as np
import pytest
from pynbody.array import SimArray

import pynbodyext as pe
from pynbodyext.calculate import PropertyBase
from pynbodyext.properties import ParamSum
from test_database import write_snapshot


Measurement = namedtuple("Measurement", ["value"])


@PropertyBase.dataclass
class ReturnValue(PropertyBase):
    value_kind: str

    def calculate(self, sim, params=None):
        return {
            "none": None,
            "zero": 0,
            "nan": float("nan"),
            "empty": np.array([]),
            "nested": {"values": [SimArray([2.0], "Msol")]},
            "opaque": Decimal("1.5"),
            "opaque_nested": {"value": [Decimal("1.5")]},
            "named_tuple": Measurement(1),
        }[self.value_kind]


@pytest.fixture
def table_env(tmp_path):
    sim = write_snapshot(tmp_path / "snap")
    with pe.Database(tmp_path / "results.sqlite") as db:
        obj = db.register(sim, simulation="box", snapshot="99")
        yield db, obj


def test_a12_a20_a24_a30_portable_values_without_backend_or_calculator(table_env, tmp_path):
    db, obj = table_env
    kinds = ("none", "zero", "nan", "empty", "nested")
    calculations = {kind: ReturnValue(kind) for kind in kinds}
    table = db.study(objects=[obj], calculations=calculations).run()
    assert obj.results.get(calculations["none"]).value is None
    assert table.cell(obj, "none").has_value
    nested = table.cell(obj, "nested").value
    nested["values"][0][0] = 100
    assert table.cell(obj, "nested").value["values"][0][0] == 2
    path = tmp_path / "published.json"
    table.save(path)
    db.close()
    db.path.unlink()
    script = """
import sys
import numpy as np
from pynbodyext import StudyTable
assert 'sqlalchemy' not in sys.modules
assert 'pynbodyext.database' not in sys.modules
table = StudyTable.load(sys.argv[1])
obj, = table.objects
assert table.cell(obj, 'none').status == 'succeeded'
assert table.cell(obj, 'none').value is None
assert table.cell(obj, 'zero').value == 0
assert np.isnan(table.cell(obj, 'nan').value)
assert table.cell(obj, 'empty').value.size == 0
assert table.cell(obj, 'nested').value['values'][0].in_units('Msol')[0] == 2
assert table.cell(obj, 'nested').ref.database_id == obj.database_id
assert 'test_study_table' not in sys.modules
assert 'sqlalchemy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(path)], check=True, env=os.environ.copy())


def test_a22_backup_namespace_validates_actual_pinned_version(table_env, tmp_path):
    db, obj = table_env
    dest = tmp_path / "relocated # backup.sqlite"
    shutil.copyfile(db.path, dest)
    table = db.study(objects=[obj], calculations={"mass": ParamSum("mass")}).run()
    ref = table.cell(obj, "mass").ref
    with pe.Database(dest) as backup:
        assert backup.database_id == ref.database_id
        assert backup.study(objects=table.objects, calculations=table.calculations).run().cell(obj, "mass").ref != ref
    with pe.Database(dest, read_only=True) as reopened:
        with pytest.raises(KeyError, match="no stored result"):
            reopened.fetch(ref)


@pytest.mark.parametrize("problem", ["version", "duplicate", "ref", "opaque", "omitted", "digest"])
def test_a25_malformed_tables_rejected(table_env, tmp_path, monkeypatch, problem):
    db, obj = table_env
    table = db.study(objects=[obj], calculations={"mass": ParamSum("mass")}).run()
    path = tmp_path / "table.json"
    table.save(path)
    data = json.loads(path.read_text())
    if problem == "version":
        data["schema"] = "pynbodyext.study.table/v999"
    elif problem == "ref":
        data["cells"][0]["ref"]["record_ref"]["version_id"] = None
    elif problem == "opaque":
        data["cells"][0]["value"] = {"__kind": "seq", "items": [{"__kind": "opaque", "pickle_b64": "bad"}]}
    elif problem == "omitted":
        data["cells"].clear()
    elif problem == "digest":
        data["definition_id"] = "0" * 64
    text = json.dumps(data)
    path.write_text(text if problem != "duplicate" else text.replace("{", '{"schema": null,', 1))

    def forbidden(*args, **kwargs):
        pytest.fail("unexpected pickle decode")

    monkeypatch.setattr("pynbodyext.core.calculate.store.codec.pickle.loads", forbidden)
    with pytest.raises(ValueError):
        pe.StudyTable.load(path)


def test_a26_atomic_export_with_explicit_overwrite(table_env, tmp_path):
    db, obj = table_env
    good = db.study(objects=[obj], calculations={"mass": ParamSum("mass")}).run()
    bad = db.study(objects=[obj], calculations={"opaque": ReturnValue("opaque")}).run()
    assert bad.cell(obj, "opaque").value == Decimal("1.5")
    path = tmp_path / "table.json"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        good.save(path)
    with pytest.raises((ValueError, TypeError), match="portable"):
        bad.save(path, overwrite=True)
    assert path.read_bytes() == b"original"
    good.save(path, overwrite=True)
    assert pe.StudyTable.load(path).cell(obj, "mass").value == 10


def test_a29_filter_and_projection_preserve_units_status_refs_and_order(table_env, tmp_path):
    db, obj = table_env
    other_sim = write_snapshot(tmp_path / "other" / "snap")
    other = db.register(other_sim, simulation="box", snapshot="100")
    calc = ParamSum("mass")
    ref = calc.run(obj).store_ref
    study = db.study(objects=[other, obj], calculations={"mass": calc})
    table = study.collect(missing="keep")
    column = table.column("mass", units="kg")
    assert column.mask.tolist() == [True, False]
    assert column[1] == pytest.approx(float(SimArray(10, "Msol").in_units("kg")))
    with pytest.raises(ValueError):
        table.column("mass", units="s")
    with pytest.raises(ValueError, match="missing"):
        table[other]["mass"]
    kept = table.where(lambda row: row.cell("mass").status == "succeeded")
    assert kept.definition_id == table.definition_id
    assert kept.objects == table.objects[1:]
    assert kept.cell(obj, "mass").ref.record_ref == ref
    path = tmp_path / "filtered.json"
    kept.save(path)
    restored = pe.StudyTable.load(path)
    assert restored.objects == kept.objects and restored.definition_id == kept.definition_id
    pytest.importorskip("pandas")
    frame = table.to_dataframe(units={"mass": "kg"})
    assert frame["mass.status"].tolist() == ["missing", "succeeded"]
    assert frame["mass.ref"].iloc[1].record_ref == ref


def test_a21_foreign_live_handles_cannot_select_values(table_env):
    db, obj = table_env
    calc = ParamSum("mass")
    table = db.study(objects=[obj], calculations={"mass": calc}).run()
    from test_database import write_snapshot

    sim = write_snapshot(db.path.parent / "same" / "snap")
    with pe.Database(db.path.with_name("foreign.sqlite")) as foreign:
        match = foreign.register(sim, simulation="box", snapshot="99")
        assert obj.key == match.key
        with pytest.raises(ValueError, match="database"):
            table.cell(match, "mass")
        with pytest.raises(ValueError, match="database"):
            foreign.fetch(table.cell(obj, "mass").ref)


@pytest.mark.parametrize("value_kind", ["opaque_nested", "named_tuple"])
def test_a26_nonportable_container_does_not_replace_existing_file(table_env, tmp_path, value_kind):
    db, obj = table_env
    table = db.study(objects=[obj], calculations={"nested": ReturnValue(value_kind)}).run()
    target = tmp_path / "table.json"
    target.write_text("old")
    with pytest.raises(ValueError, match="portable"):
        table.save(target, overwrite=True)
    assert target.read_text() == "old"
