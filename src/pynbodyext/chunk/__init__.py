
from importlib.util import find_spec

DASK_AVAILABLE: bool = False

if find_spec("dask") is not None:
    DASK_AVAILABLE = True


