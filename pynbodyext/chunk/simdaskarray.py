# SimDaskArray: Dask-backed array with pynbody SimArray unit-tracking

from functools import wraps
from typing import Any

import dask.array as da
import numpy as np
from dask.array.core import Array as DaskArray, finalize as dask_finalize
from pynbody import units as pyn_units

# Import SimArray and unit helpers from pynbody
from pynbody.array import SimArray

# Methods that do not alter units (mirror unyt_dask_array)
_USE_UNARY_DECORATOR = {
    "min", "max", "sum", "mean", "std", "cumsum",
    "squeeze", "rechunk", "clip", "view", "swapaxes",
    "round", "copy", "repeat", "astype", "reshape", "topk",
}


def _is_iterable(obj):
    return isinstance(obj, (tuple, list))

def _extract_sidecar(obj):
    if _is_iterable(obj):
        has_sim_like = any(isinstance(o, (SimDaskArray, SimArray)) for o in obj)
        if has_sim_like:
            # keep the same type as input
            seq_type = type(obj)
            return seq_type(_extract_sidecar(o) for o in obj)
        else:
            return obj
    return obj._sidecar if isinstance(obj, SimDaskArray) else obj

def _extract_dask(obj):
    if _is_iterable(obj):
        return [_extract_dask(o) for o in obj]
    return obj.to_dask() if isinstance(obj, SimDaskArray) else obj

def _extract_plain_numpy(obj):
    if _is_iterable(obj):
        return [_extract_plain_numpy(o) for o in obj]
    return obj.view(np.ndarray) if isinstance(obj, SimArray) else obj

def _wrap_unary_preserving_units(dask_func, current_sim_dask):
    """Wrap a Dask unary op so the result keeps the same sidecar/units."""
    @wraps(dask_func)
    def wrapper(*args, **kwargs):
        da_out = dask_func(*args, **kwargs)
        return _create_with_sidecar(da_out, current_sim_dask._sidecar)
    return wrapper

def _post_ufunc(dask_superfunc, sidecar_result):
    def wrapper(*args, **kwargs):
        dask_result = dask_superfunc(*args, **kwargs)
        # If we have a SimArray sidecar_result (units, sim, name), propagate units
        if isinstance(sidecar_result, SimArray) and sidecar_result.units is not None:
            return _create_with_sidecar(dask_result, sidecar_result)
        return dask_result
    return wrapper


def _units_via_registry(func, *sidecar_inputs, **kwargs):
    rule = SimArray._ufunc_registry.get(func, None)
    if rule is None:
        return None
    result = rule(*sidecar_inputs, **kwargs)
    if isinstance(result, tuple):
        out_units, modified_inputs = result
        return out_units, modified_inputs
    else:
        return result, None


def _prep_ufunc(ufunc, *inputs, extract_dask=False, **kwargs):
    """Prepare inputs/units for numpy/dask ufunc, using SimArray registry when available."""
    sidecar_inputs = _extract_sidecar(inputs)

    # Try SimArray registry first
    reg_units_and_inputs = _units_via_registry(ufunc, *sidecar_inputs, **kwargs)
    if reg_units_and_inputs is not None:
        out_units, modified_inputs = reg_units_and_inputs

        if modified_inputs is not None:
            sidecar_call_inputs = SimArray._simarray_to_plain_ndarray(modified_inputs)
        else:
            sidecar_call_inputs = _extract_plain_numpy(sidecar_inputs)

        try:
            _ = ufunc(*sidecar_call_inputs, **kwargs)
        except Exception:
            pass

        if extract_dask:
            dask_inputs = list(_extract_dask(inputs))
        else:
            dask_inputs = list(inputs)

        if modified_inputs is not None:
            for i, (orig_si, mod_si) in enumerate(zip(sidecar_inputs, modified_inputs, strict=False)):
                if not isinstance(orig_si, (SimArray, SimDaskArray)):
                    dask_inputs[i] = mod_si

        sidecar_result = SimArray(
            [1.0],
            units=out_units,
            sim=sidecar_inputs[0].sim if hasattr(sidecar_inputs[0], "sim") else None,
        )

        return tuple(dask_inputs), sidecar_result

    return None

def _finalize_to_sim(results, unit_like):
    # Finalize dask compute result to SimArray or SimArray scalar
    result = dask_finalize(results)
    units = unit_like.units if isinstance(unit_like, SimArray) else unit_like
    sim = unit_like.sim if isinstance(unit_like, SimArray) else None
    sim = sim.base if sim is not None and hasattr(sim, "base") else sim
    name = unit_like.name if isinstance(unit_like, SimArray) else None
    if np.isscalar(result):
        result = np.array(result)
    if isinstance(result, np.ndarray):
        out = SimArray(result, units=units, sim=sim)
        out._name = name
        return out
    else:
        # Scalar -> wrap into 0-d SimArray
        out = SimArray(np.asarray(result), units=units, sim=sim)
        out = out.squeeze()
        out._name = name
        return out

def _create_with_sidecar(dask_array, sidecar_simarray):
    # Wrap a dask Array with SimDaskArray and copy unit state from sidecar
    if isinstance(dask_array, SimDaskArray):
        dask_array = dask_array.to_dask()
    (cls, args) = dask_array.__reduce__()

    out = SimDaskArray(*args)
    out._set_sidecar(sidecar_simarray)
    return out



# Put near the other helpers
def _op_to_np(func_name):
    return {
        "__add__": np.add,
        "__sub__": np.subtract,
        "__mul__": np.multiply,
        "__rmul__": np.multiply,
        "__truediv__": np.true_divide,
        "__rtruediv__": np.true_divide,
        "__pow__": np.power,
        "__lt__": np.less,
        "__le__": np.less_equal,
        "__gt__": np.greater,
        "__ge__": np.greater_equal,
        "__eq__": np.equal,
        "__ne__": np.not_equal,
        "__abs__": np.abs,
    }.get(func_name)

def _wrap_binary_op_with_units(the_func):
    # Ensure dunder ops return SimDaskArray with correct units
    def wrapper(self, *args, **kwargs):
        funcname = the_func.__name__
        npufunc = _op_to_np(funcname)
        if npufunc is None:
            # Fallback: just call super and return plain dask result
            return the_func(self, *args, **kwargs)

        # Prepare unit result using the sidecars (and align units if compatible)
        ufunc_args = (self,) + args
        args_for_dask, sidecar_result = _prep_ufunc(
            npufunc, *ufunc_args, extract_dask=True, **kwargs
        )

        # Remove the first argument (self) for the actual operator call
        args_for_call = list(args_for_dask)
        args_for_call.pop(0)

        # Call the base operator
        dask_result = the_func(self, *args_for_call, **kwargs)

        # Wrap with units if we have sidecar_result units
        if isinstance(sidecar_result, SimArray):
            return _create_with_sidecar(dask_result, sidecar_result)
        else:
            return dask_result
    return wrapper

class SimDaskArray(DaskArray):
    """Dask Array subclass that tracks pynbody SimArray units and sim context."""


    _sidecar: SimArray
    def __new__(cls, dask_graph, name, chunks, dtype=None, meta=None, shape=None):
        obj = super().__new__(cls, dask_graph, name, chunks, dtype, meta, shape)
        # Default sidecar: dimensionless SimArray of length-1
        obj._sidecar = SimArray([1.0], units=pyn_units.no_unit, sim=None)
        return obj

    def _set_sidecar(self, sidecar_simarray: SimArray) -> None:
        self._sidecar = sidecar_simarray

    def to_dask(self):
        (_, args) = super().__reduce__()
        return DaskArray(*args)

    def __reduce__(self):
        (_, args) = super().__reduce__()
        state = {
            "_sidecar": self._sidecar,
            "units": self.units,
            #"name": self.name,
        }
        return SimDaskArray, args, state

    def __setstate__(self, state):
        self._sidecar = state["_sidecar"]

    __hash__ = DaskArray.__hash__  # make unhashable like Dask Array

    @property
    def pb_name(self):
        return self._sidecar.name

    @pb_name.setter
    def pb_name(self, value):
        self._sidecar._name = value

    @property
    def ancestor(self):
        return self

    @property
    def derived(self):
        return self._sidecar.derived

    @derived.setter
    def derived(self, value):
        self._sidecar.derived = value

    @property
    def sim(self):
        return self._sidecar.sim

    @sim.setter
    def sim(self, value):
        self._sidecar.sim = value

    @property
    def family(self):
        return self._sidecar.family

    @family.setter
    def family(self, value):
        self._sidecar.family = value

    @property
    def units(self):
        return self._sidecar.units

    @units.setter
    def units(self, value):
        self._sidecar.units = value

    @property
    def base(self):
        return None

    @property
    def flags(self):
        return self._sidecar.flags


    in_units = SimArray.in_units
    """ Return a new SimDaskArray in the specified units. """


    def convert_units(self, new_units):
        new = self.in_units(new_units)
        self.dask = new.dask
        self._name = new._name
        self._chunks = new._chunks
        self._meta = new._meta
        self.units = new.units


    set_units_like = SimArray.set_units_like
    set_default_units = SimArray.set_default_units



    def __repr__(self):
        base = super().__repr__().replace("dask.array", "SimDaskArray")
        units_str = f", units={self.units}>"
        return base.replace(">", units_str)

    def _repr_html_(self):
        base_table = super()._repr_html_()
        table = base_table.split("\n")
        new = []
        for row in table:
            if "</tbody>" in row:
                u = self.units
                new.append(f"    <tr><th> Units </th><td> {u} </td> <td> {u} </td></tr>")
            new.append(row)
        return "\n".join(new)

    # Hooks to propagate units through numpy/dask
    def _elemwise(self, ufunc, *args, **kwargs):
        args, sidecar_res = _prep_ufunc(ufunc, *args, **kwargs)
        return _post_ufunc(super()._elemwise, sidecar_res)(ufunc, *args, **kwargs)

    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        inputs, sidecar_res = _prep_ufunc(numpy_ufunc, *inputs, extract_dask=True, **kwargs)
        wrapped = _post_ufunc(super().__array_ufunc__, sidecar_res)
        return wrapped(numpy_ufunc, method, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        prep = _prep_ufunc(func, *args, extract_dask=True, **kwargs)
        if prep is None: # No unit handling available
            return super().__array_function__(func, types, args, kwargs)
        args, sidecar_res = prep
        types = [type(i) for i in args]
        wrapped = _post_ufunc(super().__array_function__, sidecar_res)
        return wrapped(func, types, args, kwargs)

    def __getitem__(self, index):
        return _wrap_unary_preserving_units(super().__getitem__, self)(index)

    def __getattribute__(self, name):
        result = super().__getattribute__(name)
        if name in _USE_UNARY_DECORATOR:
            return _wrap_unary_preserving_units(result, self)
        return result

    def __dask_postcompute__(self):
        # Called after compute(); finalize to SimArray with sidecar units
        return _finalize_to_sim, ((self._sidecar,))

    # Binary ops: rely on generic __array_ufunc__ path via numpy ufuncs
    # For reductions that change units specially, add overrides as needed
    def prod(self, *args, **kwargs):
        # Use SimArray rule: axis=None => same units; axis=i => power by size along axis
        dask_res = super().prod(*args, **kwargs)
        # Infer expected unit via SimArray sidecar using a tiny array of matching shape semantics
        # We apply np.multiply unit mapping like SimArray.prod does
        if "axis" in kwargs and kwargs["axis"] is not None:
            axis = kwargs["axis"]
            power = self.shape[axis]
            unit = self.units ** power
        else:
            unit = self.units
        return _create_with_sidecar(dask_res, SimArray([1.0], units=unit, sim=self.sim))

    def conversion_context(self):
        """Return a dictionary of contextual information that may be required for unit conversion.

        This is typically cosmological scalefactor and Hubble parameter."""
        if self._sidecar.sim is not None:
            return self._sidecar.sim.conversion_context()
        else:
            return {}

    @_wrap_binary_op_with_units
    def __abs__(self):
        return super().__abs__()

    @_wrap_binary_op_with_units
    def __pow__(self, other):
        return super().__pow__(other)

    @_wrap_binary_op_with_units
    def __mul__(self, other):
        return super().__mul__(other)

    @_wrap_binary_op_with_units
    def __rmul__(self, other):
        return super().__rmul__(other)

    @_wrap_binary_op_with_units
    def __truediv__(self, other):
        return super().__truediv__(other)

    @_wrap_binary_op_with_units
    def __rtruediv__(self, other):
        return super().__rtruediv__(other)

    @_wrap_binary_op_with_units
    def __add__(self, other):
        return super().__add__(other)

    @_wrap_binary_op_with_units
    def __sub__(self, other):
        return super().__sub__(other)

    @_wrap_binary_op_with_units
    def __lt__(self, other):
        return super().__lt__(other)

    @_wrap_binary_op_with_units
    def __le__(self, other):
        return super().__le__(other)

    @_wrap_binary_op_with_units
    def __gt__(self, other):
        return super().__gt__(other)

    @_wrap_binary_op_with_units
    def __ge__(self, other):
        return super().__ge__(other)

    @_wrap_binary_op_with_units
    def __eq__(self, other):
        return super().__eq__(other)

    @_wrap_binary_op_with_units
    def __ne__(self, other):
        return super().__ne__(other)


# Dirty-bit support for SimDaskArray, mirroring SimArray._dirty_fn but using pb_name

def _dirty_fn_dask(method):
    """Mark underlying SimDaskArray as dirty when mutating SimDaskArray in-place."""
    def wrapped(a, *args, **kwargs):
        # a is SimDaskArray
        sim = getattr(a, "sim", None)
        name = getattr(a, "pb_name", None)
        # Mark the corresponding pynbody array as "dirty" similar to SimArray._dirty_fn, but using pb_name
        if sim is not None and name is not None:
            sim._dirty(name)

        return method(a, *args, **kwargs)

    wrapped.__name__ = method.__name__
    return wrapped

_dirty_methods = [
    "__setitem__", "__setslice__",
    "__irshift__", "__imod__", "__iand__", "__ifloordiv__",
    "__ilshift__", "__imul__", "__ior__", "__ixor__",
    "__isub__", "__invert__", "__iadd__", "__itruediv__",
    "__idiv__", "__ipow__",
]

for name in _dirty_methods:
    if hasattr(SimDaskArray, name):
        setattr(SimDaskArray, name, _dirty_fn_dask(getattr(SimDaskArray, name)))

# Factories

def sim_from_dask(dask_array, units=None, sim=None, name=None, family=None):
    """Wrap a plain dask Array into SimDaskArray with specified units/sim/name."""
    (cls, args) = dask_array.__reduce__()
    out = SimDaskArray(*args)
    if isinstance(units, str):
        units = pyn_units.Unit(units)
    sidecar = SimArray([1.0], units=units or pyn_units.no_unit, sim=sim)
    sidecar._name = name
    sidecar.family = family
    out._set_sidecar(sidecar)
    return out

def sim_from_delayed(delayed_obj, shape, dtype, chunks, units=None, sim=None, name=None, family=None):
    """Construct SimDaskArray via dask.array.from_delayed."""
    base_da = da.from_delayed(delayed_obj, shape=shape, dtype=dtype)
    if chunks is not None:
        base_da = base_da.rechunk(chunks)
    return sim_from_dask(base_da, units=units, sim=sim, name=name, family=family)

# Reduction helper (optional), similar to unyt.reduce_with_units
_ALLOWED_REDUCTIONS = {"median", "diagonal", "nanmean", "nanstd", "nanmin", "nanmax", "nansum"}

def reduce_with_units(dask_func: Any, sim_dask_arr: SimDaskArray, *args: Any, **kwargs: Any) -> SimDaskArray:
    """Call a dask.array reduction function and preserve units when appropriate."""
    name = getattr(dask_func, "__name__", "")
    if name in _ALLOWED_REDUCTIONS:
        return _wrap_unary_preserving_units(dask_func, sim_dask_arr)(sim_dask_arr, *args, **kwargs)
    # Fallback: try matching numpy function name to infer units via sidecar
    np_func = getattr(np, name, None)
    if np_func:
        newargs, sidecar_res = _prep_ufunc(np_func, sim_dask_arr, *args, extract_dask=True, **kwargs)
        return _post_ufunc(dask_func, sidecar_res)(*newargs)
    else:
        raise ValueError("could not deduce np equivalent of dask reduction")
