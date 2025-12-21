
from pynbodyext.util.deps import DASK_AVAILABLE

_dask_array_type: type | None = None

if DASK_AVAILABLE:
    from dask.array import Array as DaskArray
    _dask_array_type = DaskArray

def is_dask_array(obj: object) -> bool:
    """Check if an object is a Dask array."""
    if not DASK_AVAILABLE or _dask_array_type is None:
        return False
    return isinstance(obj, _dask_array_type)



