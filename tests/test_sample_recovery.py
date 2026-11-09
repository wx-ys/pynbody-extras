"""Recovery keeps each sample cell value and diagnostic consistent."""

from typing import ClassVar

import pynbody
import pytest

from pynbodyext.core.calculate import InMemoryResultStore, Param, PropertyBase, SampleTable


@PropertyBase.dataclass
class Recoverable(PropertyBase[float]):
    signature_name = "tests.recovery.Recoverable"
    failing: ClassVar[bool] = True
    scale: Param[float] = 1.0

    def calculate(self, sim, params=None):
        if type(self).failing:
            raise RuntimeError("temporary failure")
        return 7.0 * self.scale


@pytest.fixture(autouse=True)
def reset_failure():
    Recoverable.failing = True
    yield
    Recoverable.failing = True


def identity(sim):
    return ("sim", "recovery-fixture")


def failed_table(store):
    sim = pynbody.new(1)
    calc = Recoverable()
    table = SampleTable({"value": calc, "missing": Recoverable(scale=2.0)}, store=store, sim_identity=identity)
    row = table.add(sim, errors="record")
    assert row.errors == {"value": "RuntimeError: temporary failure", "missing": "RuntimeError: temporary failure"}
    return sim, calc, table, row


def test_store_fill_clears_only_recovered_error():
    store = InMemoryResultStore()
    sim, calc, table, row = failed_table(store)
    Recoverable.failing = False
    calc.run(sim, store=store, sim_identity=identity)
    assert table.fill_from_store() is table
    assert row.values == {"value": 7.0, "missing": None}
    assert row.errors == {"missing": "RuntimeError: temporary failure"}
    assert table.missing() == [(row.name, "missing")]


def test_cache_miss_preserves_existing_diagnostics():
    sim, calc, table, row = failed_table(InMemoryResultStore())
    previous_values, previous_errors = dict(row.values), dict(row.errors)
    table.fill_from_store()
    assert row.values == previous_values
    assert row.errors == previous_errors


def test_nonempty_cell_is_not_overwritten_or_cleared():
    store = InMemoryResultStore()
    sim, calc, table, row = failed_table(store)
    row.values["value"] = 9.0
    Recoverable.failing = False
    calc.run(sim, store=store, sim_identity=identity)
    previous_values, previous_errors = dict(row.values), dict(row.errors)
    table.fill_from_store()
    assert row.values == previous_values
    assert row.errors == previous_errors


class UnavailableStore(InMemoryResultStore):
    def get(self, **kwargs):
        raise OSError("backend unavailable")


def test_failed_backend_read_preserves_existing_diagnostics():
    sim, calc, table, row = failed_table(InMemoryResultStore())
    Recoverable.failing = False
    calc.run(sim, store=table.store, sim_identity=identity)
    table.store = UnavailableStore()
    previous_values, previous_errors = dict(row.values), dict(row.errors)
    with pytest.raises(OSError, match="backend unavailable"):
        table.fill_from_store()
    assert row.values == previous_values
    assert row.errors == previous_errors
