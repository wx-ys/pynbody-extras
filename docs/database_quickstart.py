"""Run from the repository: PYTHONPATH=. .venv/bin/python docs/database_quickstart.py.

Creates its own four-particle Tipsy snapshot and SQLite database in a temporary
output directory. No downloaded simulation data is required. The output remains
on disk so you can reopen the database and inspect its contents.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pynbody
from pynbody.array import SimArray
from pynbody.halo.number_array import HaloNumberCatalogue

import pynbodyext as pe
from pynbodyext.properties import ParamSum


def write_snapshot(directory: Path):
    """Write a real file with explicit physical-unit metadata, then load it."""
    path = directory / "snapshot_099"
    (directory / "units.param").write_text("dKpcUnit = 1\ndMsolUnit = 1\nbComove = 0\n", encoding="utf-8")
    sim = pynbody.new(dm=4)
    sim["pos"] = SimArray(np.arange(12.0).reshape(4, 3), "kpc")
    sim["vel"] = SimArray(np.zeros((4, 3)), "km s^-1")
    sim["mass"] = SimArray([1.0, 2.0, 3.0, 4.0], "Msol")
    sim.properties["time"] = 0.0
    sim.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=str(path))
    loaded = pynbody.load(str(path))
    loaded.physical_units()
    return loaded


def main() -> None:
    directory = Path(tempfile.mkdtemp(prefix="pynbody-database-quickstart-"))
    sim = write_snapshot(directory)
    cc = ParamSum("mass")

    # Calculate first; saving later creates the schema and records the input.
    result = cc.run(sim)
    with pe.Database(directory / "analysis.sqlite") as db:
        first_ref = result.save(db, name="total_mass")
        snap = db.snapshots["snapshot_099"]
        snapshot_key = snap.key
        np.testing.assert_allclose(snap.results["total_mass"].value.in_units("Msol"), 10.0)
        assert snap.results[cc].store_ref == first_ref
        assert snap.results[cc].calculator.to_signature().hash == cc.to_signature().hash

        # auto reuses the saved result; refresh produces a new pinned version.
        with db.use(mode="auto"):
            cached = cc.run(sim)
            assert cached.from_store and cached.store_ref == first_ref
            refreshed = cc.run(sim, mode="refresh")
            assert not refreshed.from_store and refreshed.store_ref != first_ref
            assert len(db.versions(first_ref)) == 2
        np.testing.assert_allclose(db.fetch(first_ref).value.in_units("Msol"), 10.0)

        # An exact particle selection preserves order and repeated indices.
        subset_result = cc.run(sim[[2, 0, 2]], database=db)
        selected = next(iter(snap.selections))
        selection_key = selected.key
        np.testing.assert_allclose(selected.load()["mass"].in_units("Msol"), [3.0, 1.0, 3.0])
        np.testing.assert_allclose(subset_result.value.in_units("Msol"), 7.0)

        # The qualified number-array adapter records this live catalogue on save.
        # 7 is the original halo ID, not the seventh row.
        sim["grp"] = np.array([7, 7, 12, -1])
        catalogue = HaloNumberCatalogue(sim, array="grp", ignore=-1)
        halo_result = cc.run(catalogue[7])
        halo_ref = halo_result.save(db, name="halo_mass")
        halo = snap.halos[7]
        np.testing.assert_allclose(halo.load()["mass"].in_units("Msol"), [1.0, 2.0])
        assert halo.results["halo_mass"].store_ref == halo_ref
        print(snap.results)
        print(snap.catalogues)

    # Reopen without the live sim/catalogue. Cached runs must not invoke a loader.
    del sim, catalogue
    with pe.Database(directory / "analysis.sqlite") as db:
        calls = []

        def unavailable_loader(path):
            calls.append(path)
            raise AssertionError("cache hit attempted to load particles")

        db.bind_loader("default", unavailable_loader)
        snap = db.snapshots[snapshot_key]
        assert cc.run(snap).store_ref == refreshed.store_ref
        assert cc.run(snap.selections[selection_key]).store_ref == subset_result.store_ref
        assert cc.run(snap.halos[7]).store_ref == halo_ref
        assert calls == []
        assert snap.results[cc].from_store
        np.testing.assert_allclose(db.fetch(first_ref).value.in_units("Msol"), 10.0)

    print("Validated: save, browse, refresh, history, subset replay, halo replay and zero-loader cache hits.")
    print(f"Output directory: {directory}")
    print(f"Database: {directory / 'analysis.sqlite'}")


if __name__ == "__main__":
    main()
