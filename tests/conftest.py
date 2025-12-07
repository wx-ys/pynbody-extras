import gc
from pathlib import Path

import pytest
import pynbody
import pynbody.test_utils


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data():
    """
    Ensure test data is available for the whole test session.

    This runs once before any tests. If data preparation fails we abort the test run.
    """
    try:
        pynbody.test_utils.ensure_test_data_available("gadget", "arepo")
    except Exception as exc:  # defensive: surface any failure early
        pytest.exit(f"Failed to prepare pynbody test data: {exc}")
    # Optional sanity check: ensure path exists
    base = Path("testdata")
    if not base.exists():
        pytest.exit("Test data directory 'testdata' is missing after ensure_test_data_available()")
    yield
    # session teardown (nothing special)
    # no explicit cleanup here — snapshots are closed in individual fixtures


@pytest.fixture(scope="module")
def snap():
    """
    Load snapshot for tests. Module-scoped so tests in same module reuse the object.
    """
    f = pynbody.load("testdata/gadget3/data/snapshot_103/snap_103.hdf5")
    yield f
    # explicit cleanup to release memory between modules
    try:
        del f
    finally:
        gc.collect()


@pytest.fixture(scope="module")
def subfind():
    """
    Load subhalo/subfind data for tests that need it.
    """
    f = pynbody.load("testdata/gadget3/data/subhalos_103/subhalo_103")
    yield f
    try:
        del f
    finally:
        gc.collect()