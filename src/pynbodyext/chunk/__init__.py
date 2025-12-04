
from importlib.util import find_spec

DASK_AVAILABLE: bool = False
_dask_array_type: type | None = None


if find_spec("dask") is not None:
    DASK_AVAILABLE = True
    from dask.array import Array as DaskArray
    _dask_array_type = DaskArray


def is_dask_array(obj: object) -> bool:
    """Check if an object is a Dask array."""
    if not DASK_AVAILABLE or _dask_array_type is None:
        return False
    return isinstance(obj, _dask_array_type)



