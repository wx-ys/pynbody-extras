"""
Property calculators and expression trees.

This module defines :class:`~pynbodyext.properties.base.PropertyBase` –
a :class:`~pynbodyext.calculate.CalculatorBase` subclass for lazily
evaluated "properties" of a :class:`pynbody.snapshot.SimSnap` – and a
small expression system for combining them.

See also
--------
* :mod:`pynbodyext.calculate` for the general calculator API.

Basic usage
-----------

A property is a calculator; you call it with a snapshot:

.. code-block:: python

   from pynbodyext.properties.base import ParamSum

   # total stellar mass within some filter
   mstar = ParamSum("mass")

   # later, attach filters/transforms as needed
   from pynbodyext.filters import FamilyFilter, Sphere

   mstar_inner = (
       mstar
       .with_filter(Sphere("30 kpc") & FamilyFilter("stars"))
   )

   value = mstar_inner(sim)

Composing properties
--------------------

:class:`PropertyBase` supports arithmetic operators and builds
:class:`OpProperty` expression trees:

.. code-block:: python

   from pynbodyext.properties.base import PropertyBase, ParamSum

   mstar = ParamSum("mass").with_filter(FamilyFilter("stars"))
   mgas  = ParamSum("mass").with_filter(FamilyFilter("gas"))

   # total mass, using add
   mtot = mstar + mgas

   # specific quantity
   frac = mstar / (mstar + mgas)

   val = frac(sim)

Internally, no calculation happens at construction time; instead
``frac`` is an :class:`OpProperty` describing the expression.
When you call ``frac(sim)``, the tree is evaluated, and structurally
identical subtrees can be cached within that top-level run.


Domain-specific examples
------------------------

Half-mass radius (or similar "contain" radii) can be computed via
:class:`ParameterContain`:

.. code-block:: python

   from pynbodyext.properties.base import ParameterContain
   from pynbodyext.filters import FamilyFilter

   re = (
       ParameterContain("r", 0.5, "mass")
       .with_filter(FamilyFilter("stars"))
   )

   re_val = re(sim)   # SimArray with units of "r"

3D volume density and 2D surface density can be computed from simple
wrappers:

.. code-block:: python

   from pynbodyext.properties.base import VolumeDensity, SurfaceDensity
   from pynbodyext.filters.filt import Sphere

   # mean mass density inside 10 kpc sphere
   rho = VolumeDensity(Sphere("10 kpc"), parameter="mass")

   # surface density in annulus 5–10 kpc
   sigma = SurfaceDensity(rmax="10 kpc", rmin="5 kpc", parameter="mass")

   rho_val   = rho(sim)
   sigma_val = sigma(sim)

Extending
---------

To add a new property, subclass :class:`PropertyBase` and implement
:meth:`calculate`. For example:

.. code-block:: python

   from pynbodyext.properties.base import PropertyBase

   class MyProp(PropertyBase[SimArray]):
       def instance_signature(self) -> tuple:
           return (self.__class__.__name__,)

       def calculate(self, sim: SimSnap) -> SimArray:
           arr = sim["mass"] / sim["rho"]
           return arr

Because :class:`PropertyBase` inherits from
:class:`~pynbodyext.calculate.CalculatorBase`, your property can:

* be filtered and transformed;
* participate in profiling and caching;
* appear inside :class:`OpProperty` expression trees;
* be combined with other calculators via the ``&`` operator.

"""

import operator
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar, Union

import numpy as np
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import CalculatorBase
from pynbodyext.chunk import is_dask_array
from pynbodyext.util._type import get_signature_safe

__all__ = ["PropertyBase", "ConstantProperty", "LambdaProperty", "OpProperty",
           "ParamSum", "ParameterContain","VolumeDensity", "SurfaceDensity", "RadiusAtSurfaceDensity"]

from pynbodyext.filters.filt import Annulus, BandPass, ValueLike, ValueLikeFunc, VolumeFilter

TProp = TypeVar("TProp", bound=SimArray | float | np.ndarray | int | tuple | bool, covariant=True)

Addable = Union[ SimArray, float, np.ndarray, int, "PropertyBase[TProp]"]


# ---------------- Utility helpers ----------------

_NUMERIC_TYPES = (int, float, np.integer, np.floating)

def _is_numeric_scalar(x: Any) -> bool:
    if isinstance(x, _NUMERIC_TYPES):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False

def _extract_numeric(x: Any) -> float | np.ndarray | Any:
    # For folding: convert ndarray scalar to Python scalar if size==1
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x

# ---------------- Base class with evaluation caching ----------------

class PropertyBase(CalculatorBase[TProp], Generic[TProp]):
    """A lazily evaluated calculator: SimSnap -> TProp."""


    _description: str
    def calculate(self, sim: SimSnap) -> TProp:
        raise NotImplementedError

    # Automatically add a per-call evaluation cache wrapper to any subclass __call__
    def __call__(self, sim: SimSnap) -> TProp:
        return super().__call__(sim)

    def children(self) -> list[CalculatorBase[Any]]:
        return super().children()

    def __repr__(self) -> str:
        description = getattr(self, "_description", "")
        return f"<Prop {self.__class__.__name__} {description}>"

    # -------- composition helpers --------
    @classmethod
    def _as_property(cls, other: Addable) -> "PropertyBase":
        """
        Convert `other` to a PropertyBase. If it's already a PropertyBase, return it;
        otherwise wrap it into ConstantProperty.
        """
        if isinstance(other, PropertyBase):
            return other
        return ConstantProperty(other)


    # ---- Operator construction (returns OpProperty with simplification) ----
    def _make_op(self, op_name: str, other: Addable, reverse: bool = False) -> "PropertyBase":
        left = self
        right = self._as_property(other)

        if reverse:
            left, right = right, left

        # Associative flattening for add/mul
        if op_name in ("add", "mul"):
            operands: list[PropertyBase] = []
            def _gather(node: PropertyBase) -> None:
                if isinstance(node, OpProperty) and node.op_name == op_name:
                    for c in node.operands:
                        _gather(c)
                else:
                    operands.append(node)
            _gather(left)
            _gather(right)

            # Constant folding (only if all are ConstantProperty and all numeric scalars/arrays)
            all_const = all(isinstance(o, ConstantProperty) for o in operands)
            if all_const:
                raw_values = [o._value for o in operands]  # type: ignore[attr-defined]
                if all(_is_numeric_scalar(v) or isinstance(v, np.ndarray) for v in raw_values):
                    # Perform fold
                    func = operator.add if op_name == "add" else operator.mul
                    acc = _extract_numeric(raw_values[0])
                    for v in raw_values[1:]:
                        acc = func(acc, _extract_numeric(v))
                    return ConstantProperty(acc)  # folded
            return OpProperty(op_name, operands)

        # Non-associative binary or unary (we treat all unary via helper methods)
        return OpProperty(op_name, [left, right])

    def clip(self, vmin: Addable | None = None, vmax: Addable | None = None) -> "OpProperty":
        """ Return a new PropertyBase that clips the values to [vmin, vmax]."""

        return OpProperty("clip", [self, self._as_property(vmin),self._as_property(vmax)])

    # -------- arithmetic operators --------
    def __add__(self, other: Addable) -> "PropertyBase":
        return self._make_op("add", other)
    def __radd__(self, other: Addable) -> "PropertyBase":
        return self._make_op("add", other, reverse=True)
    def __sub__(self, other: Addable) -> "PropertyBase":
        return OpProperty("sub", [self, self._as_property(other)])
    def __rsub__(self, other: Addable) -> "PropertyBase":
        return OpProperty("sub", [self._as_property(other), self])
    def __mul__(self, other: Addable) -> "PropertyBase":
        return self._make_op("mul", other)
    def __rmul__(self, other: Addable) -> "PropertyBase":
        return self._make_op("mul", other, reverse=True)
    def __truediv__(self, other: Addable) -> "PropertyBase":
        return OpProperty("truediv", [self, self._as_property(other)])
    def __rtruediv__(self, other: Addable) -> "PropertyBase":
        return OpProperty("truediv", [self._as_property(other), self])
    def __pow__(self, other: Addable) -> "PropertyBase":
        return OpProperty("pow", [self, self._as_property(other)])
    def __rpow__(self, other: Addable) -> "PropertyBase":
        return OpProperty("pow", [self._as_property(other), self])

    # -------- unary operators --------
    def __neg__(self) -> "PropertyBase":
        return OpProperty("neg", [self])
    def __pos__(self) -> "PropertyBase":
        return OpProperty("pos", [self])
    def __abs__(self) -> "PropertyBase":
        return OpProperty("abs", [self])

    # Comparisons return PropertyBase (OpProperty) (structural CSE applies)
    #def __lt__(self, other: Addable) -> "PropertyBase":
    #    return OpProperty("lt", [self, self._as_property(other)])
    #def __le__(self, other: Addable) -> "PropertyBase":
    #    return OpProperty("le", [self, self._as_property(other)])
    #def __gt__(self, other: Addable) -> "PropertyBase":
    #   return OpProperty("gt", [self, self._as_property(other)])
    #def __ge__(self, other: Addable) -> "PropertyBase":
    #    return OpProperty("ge", [self, self._as_property(other)])
    #def __eq__(self, other: Addable) -> "PropertyBase":  # type: ignore[override]
    #    return OpProperty("eq", [self, self._as_property(other)])
    def __ne__(self, other: Addable) -> "PropertyBase":  # type: ignore[override]
        return OpProperty("ne", [self, self._as_property(other)])

    # Make instances hashable by identity even though __eq__ is overridden symbolically
    __hash__ = object.__hash__

    def __bool__(self) -> bool:
        raise TypeError("PropertyBase is symbolic; comparing returns an expression. "
                        "Evaluate with (expr)(sim) or use explicit numpy/pynbody ops.")


    def cached(self) -> "LambdaProperty":
        """
        Return a cached wrapper: evaluate at most once per SimSnap (by id(sim)).
        Use only when SimSnap is effectively read-only during evaluation.
        """
        return LambdaProperty(lambda sim: self(sim), cache=True)

class ConstantProperty(PropertyBase[TProp]):
    """
    Wrap a constant (float/int/np.ndarray/SimArray) as a PropertyBase.
    """
    def __init__(self, value: TProp):
        self._value = value

    def signature(self) -> tuple:
        v = self._value
        # For scalars, use value directly; for arrays/simarrays use object id to avoid heavy hashing
        if isinstance(v, (int, float, np.floating)):
            return ("Const", float(v))
        return ("ConstObj", id(v))

    def calculate(self, sim: SimSnap) -> TProp:
        return self._value
    def __repr__(self) -> str:
        return f"Const({repr(self._value)})"

class LambdaProperty(PropertyBase[Any]):
    """
    A composable property defined by a function:
    func: (sim: SimSnap) -> Any
    """
    def __init__(self, func: Callable[[SimSnap], Any], cache: bool = False):
        self._func = func
        self._cache_enabled = cache
        self._cache: dict[int, Any] = {}

    def clear_cache(self):
        self._cache.clear()

    def signature(self) -> tuple:
        return ("Lambda", get_signature_safe(self._func, fallback_to_id=True), self._cache_enabled)

    def calculate(self, sim: SimSnap) -> Any:
        if not self._cache_enabled:
            return self._func(sim)
        key = id(sim)
        if key in self._cache:
            return self._cache[key]
        val = self._func(sim)
        self._cache[key] = val
        return val
    def __repr__(self) -> str:
        return f"LambdaProperty(func={self._func}, cache={self._cache_enabled})"
# ---------------- OpProperty expression tree ----------------

class OpProperty(PropertyBase[Any]):
    """
    Structured operation node with operands list. Supports:
      add (n-ary), mul (n-ary), sub, truediv, pow,
      neg, pos, abs,
      lt, le, gt, ge, eq, ne.

    Constant folding + flattening performed at construction in PropertyBase helpers.
    """
    __slots__ = ("op_name", "operands")

    def __init__(self, op_name: str, operands: list[PropertyBase]):
        self.op_name = op_name
        self.operands = operands

    def signature(self) -> tuple:
        return ("Op",
                self.op_name,
                tuple(child.signature() for child in self.operands))

    def children(self) -> list[CalculatorBase[Any]]:
        # For structure tree: operands + parent's filter/transformation
        return [*self.operands, *super().children()]

    def calculate_children(self) -> list[CalculatorBase[Any]]:
        # In semantic flow, under "3) calculate", show expression operands
        return list(self.operands)

    def calculate(self, sim: SimSnap) -> Any:
        # Evaluate operands; thanks to evaluation wrapper + signatures, identical subtrees
        # within this top-level call are computed once.
        func_map: dict[str, Callable[[Any, Any], Any]] = {
            "add": operator.add,
            "mul": operator.mul,
            "sub": operator.sub,
            "truediv": operator.truediv,
            "pow": operator.pow,
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
            "eq": operator.eq,
            "ne": operator.ne,
        }
        unary_map: dict[str, Callable[[Any], Any]] = {
            "neg": operator.neg,
            "pos": lambda x: +x,
            "abs": operator.abs,
        }

        if self.op_name in unary_map:
            val = self.operands[0](sim)
            return unary_map[self.op_name](val)

        # support np.clip as clip operation
        if self.op_name == "clip":
            val = self.operands[0](sim)
            vmin = self.operands[1](sim)
            vmax = self.operands[2](sim)
            new_val = np.clip(val, vmin, vmax)
            if isinstance(val, SimArray):
                val[...] = new_val
            else:
                val = new_val
            return val

        if self.op_name in ("add", "mul"):
            opf = func_map[self.op_name]
            vals = [opnd(sim) for opnd in self.operands]
            # Sequential fold
            acc = vals[0]
            for v in vals[1:]:
                acc = opf(acc, v)
            return acc

        # Binary ops (including comparisons)
        if self.op_name in func_map:
            left = self.operands[0](sim)
            right = self.operands[1](sim)
            return func_map[self.op_name](left, right)

        raise ValueError(f"Unknown op_name {self.op_name}")

    def _expr_stats(self):
        """Return (num_basic, num_ops, depth) for this expression tree."""
        def helper(node, depth=1):
            if isinstance(node, OpProperty):
                ops = 1
                basics = 0
                max_depth = depth
                for child in node.operands:
                    b, o, d = helper(child, depth + 1)
                    basics += b
                    ops += o
                    max_depth = max(max_depth, d)
                return basics, ops, max_depth
            else:
                return 1, 0, depth
        return helper(self)

    def __repr__(self) -> str:
        num_basic, num_ops, depth = self._expr_stats()
        return (f"OpProperty(op='{self.op_name}', operands={len(self.operands)}, "
                f"basic={num_basic}, ops={num_ops}, depth={depth})")


# ---------------- Domain-specific leaves ----------------

class ParamSum(PropertyBase[SimArray]):
    """Calculate sum of a parameter for selected particles."""

    def __init__(self, parameter: str):
        """
        Parameters
        ----------
        parameter : str
            Parameter to sum up (e.g., 'mass', 'sfr')
        """
        self.parameter = parameter
        self._description = f"({parameter})"

    def instance_signature(self):
        return (self.__class__.__name__, self.parameter)

    def calculate(self, sim : SimSnap) -> SimArray:
        """
        Calculate the sum of the specified parameter for the given simulation snapshot.
        """
        return sim[self.parameter].sum()


class ParameterContain(PropertyBase[SimArray]):
    """
    Calculates the value of a key at which a certain fraction of a
    cumulative parameter is contained.

    For example, it can calculate the radius that contains half of the total mass
    (i.e., the half-mass radius).

    The calculation uses linear interpolation between points for improved accuracy.
    """
    def __init__(self, cal_key: str = "r", frac: float | Sequence[float] = 0.5, parameter: str = "mass"):
        """
        Initializes the calculator.

        Sort particles by cal_key, and then cumsum parameter,
        return cal_key where the cumsum of parameter equals frac * sum of parameter

        Parameters
        ----------
        cal_key : str, default 'r'
            Key to sort particles by ('r' for 3D radius, 'rxy' for projected radius)
        frac : float or Sequence of floats, default 0.5
            Fraction of the total parameter to include (must be between 0 and 1)
        parameter : str, default 'mass'
            Parameter to sum up (e.g., 'mass', 'sfr')

        """
        self.cal_key = cal_key
        self.parameter = parameter
        self.frac = frac
        # Normalize and validate frac now (type-safe for mypy)
        if isinstance(frac, (int, float, np.floating)):
            fval = float(frac)
            if not (0 < fval < 1):
                raise ValueError(f"Fraction must be between 0 and 1, got {frac}")
            self._frac_array: np.ndarray = np.array([fval], dtype=float)
            self._frac_is_scalar: bool = True
        elif isinstance(frac, Sequence):
            arr = np.asarray(frac, dtype=float)
            if arr.ndim != 1:
                raise ValueError("frac must be a 1D sequence of floats")
            if not np.all((arr > 0) & (arr < 1)):
                raise ValueError(f"Each fraction must be between 0 and 1, got {arr}")
            self._frac_array = arr
            self._frac_is_scalar = False
        else:
            raise TypeError("frac must be a float or a sequence of floats")

    def instance_signature(self):
        return (self.__class__.__name__, self.cal_key, self.parameter, get_signature_safe(self.frac, fallback_to_id=True))

    def calculate(self, sim: SimSnap) -> SimArray:
        """
        Parameters
        ----------
        sim : SimSnap
            Input snapshot.

        Returns
        -------
        SimArray
            Value(s) of `cal_key` at the fractional cumulative points.
            For multiple fractions, returns an array (length = len(frac)).

        Algorithm
        ---------
        1. Sort particles by `cal_key`.
        2. Compute cumulative sum of `parameter`.
        3. Normalize cumulative to (0, 1).
        4. Interpolate `cal_key` values at each requested fraction `frac`.

        Raises
        ------
        ValueError
            If any fraction is not in (0,1).
        """

        # Get the parameter and cal_key arrays
        parameter_array = sim[self.parameter]
        cal_key_array = sim[self.cal_key]

        if is_dask_array(cal_key_array) or is_dask_array(parameter_array):
            return self._calculate_dask(sim)

        # Sort the arrays
        indices = np.argsort(cal_key_array)
        cal_key_sorted = cal_key_array[indices]
        parameter_sorted = parameter_array[indices]

        # Compute the cumulative sum and normalize to (0,1)
        parameter_cumsum = parameter_sorted.cumsum()

        # normalize to (0,1)
        denom = float(parameter_cumsum[-1] - parameter_cumsum[0])
        if denom <= 0.0:
            raise ValueError(
                f"Non-positive total '{self.parameter}' encountered; cannot normalize cumulative."
                )
        parameter_cumsum = (parameter_cumsum - parameter_cumsum[0]) / denom

        # Interpolate using the normalized array version of frac
        results_arr = np.interp(self._frac_array, parameter_cumsum, cal_key_sorted)
        results = float(results_arr[0]) if self._frac_is_scalar else results_arr

        cal_key_crit = SimArray(results)
        cal_key_crit.units = cal_key_array.units
        cal_key_crit.sim = sim.ancestor

        return cal_key_crit

    def _calculate_dask(self, sim: SimSnap) -> SimArray:
        import dask.array as da
        key_da = sim[self.cal_key]
        param_da = sim[self.parameter]

        # 1) Materialize key to NumPy and compute global argsort
        #    This avoids dask.array global sort limitations while keeping param lazy.
        key_np = np.asarray(key_da.compute())
        idx_np = np.argsort(key_np)

        # Reorder key and param
        key_sorted_np = key_np[idx_np]                  # NumPy array
        param_sorted_da = da.take(param_da, idx_np)     # Dask array, reordered lazily

        # 2) Cumulative sum and normalization (match NumPy path)
        cum_da = da.cumsum(param_sorted_da)

        # Compute only first and last scalars for normalization
        first = float(cum_da[0].compute())
        last  = float(cum_da[-1].compute())
        denom = last - first
        if denom <= 0.0:
            raise ValueError(
                f"Non-positive total '{self.parameter}' encountered; cannot normalize cumulative."
            )

        cum_norm_da = (cum_da - first) / denom

        # 3) Vectorized interpolation via searchsorted against normalized cumulative
        frac_da = da.from_array(self._frac_array, chunks=(len(self._frac_array),))
        pos_da = da.searchsorted(cum_norm_da, frac_da, side="left")

        # Clamp positions to [1, N-1] to allow neighbor access
        n = key_sorted_np.shape[0]
        pos_da = da.clip(pos_da, 1, max(1, n - 1))

        i0 = pos_da - 1
        i1 = pos_da

        x0 = cum_norm_da[i0]
        x1 = cum_norm_da[i1]

        # Use NumPy key array for neighbors (fast, already sorted)
        y0 = key_sorted_np[np.asarray(i0.compute(), dtype=int)]
        y1 = key_sorted_np[np.asarray(i1.compute(), dtype=int)]

        # Linear interpolation weights with safe denominator
        w = (frac_da - x0) / da.maximum(x1 - x0, np.finfo(float).eps)

        # y0/y1 are NumPy; convert w to NumPy to finish interpolation cheaply
        w_np = np.asarray(w.compute(), dtype=float)
        res_np = y0 + w_np * (y1 - y0)

        # 4) Wrap result
        results = float(res_np[0]) if self._frac_is_scalar else res_np
        cal_key_crit = SimArray(results)
        cal_key_crit.units = key_da.units
        cal_key_crit.sim = sim.ancestor
        return cal_key_crit


class VolumeDensity(PropertyBase[SimArray]):

    def __init__(self, rmax: VolumeFilter | ValueLike | ValueLikeFunc, parameter: str = "mass", rmin: ValueLike | ValueLikeFunc = 0.):
        self.rmax = rmax
        self.rmin = rmin
        self.parameter = parameter


    def instance_signature(self):
        return (self.__class__.__name__,
                get_signature_safe(self.rmax, fallback_to_id=True),
                get_signature_safe(self.parameter, fallback_to_id=True),
                get_signature_safe(self.rmin, fallback_to_id=True))

    def calculate(self, sim: SimSnap) -> SimArray:
        if isinstance(self.rmax, VolumeFilter):
            selector = self.rmax
        else:
            selector = Annulus(self.rmin,self.rmax,)
        param_sum = sim[selector][self.parameter].sum()
        volume = selector.volume(sim)

        den = param_sum / volume
        if isinstance(volume, float):
            den.units = sim[self.parameter].units / sim["pos"].units**3
        den.sim = sim.ancestor
        den.in_units(sim[self.parameter].units / sim["pos"].units**3)
        return den

class SurfaceDensity(PropertyBase[SimArray]):

    def __init__(self, rmax: ValueLike | ValueLikeFunc, parameter: str = "mass", rmin: ValueLike | ValueLikeFunc = 0.):
        self.rmax = rmax
        self.rmin = rmin
        self.parameter = parameter

    def instance_signature(self):
        return (self.__class__.__name__,
                get_signature_safe(self.rmax, fallback_to_id=True),
                get_signature_safe(self.parameter, fallback_to_id=True),
                get_signature_safe(self.rmin, fallback_to_id=True))

    def calculate(self, sim: SimSnap) -> SimArray:
        rmax = self.rmax(sim) if callable(self.rmax) else self.rmax
        rmin = self.rmin(sim) if callable(self.rmin) else self.rmin
        rmin = self._in_sim_units(rmin, "pos",sim)
        rmax = self._in_sim_units(rmax, "pos",sim)
        selector = BandPass("rxy",rmin, rmax)
        param_sum = sim[selector][self.parameter].sum()

        den = param_sum / (np.pi * (rmax**2 - rmin**2))

        den.units = sim[self.parameter].units / sim["pos"].units**2
        den.sim = sim.ancestor
        return den

class RadiusAtSurfaceDensity(PropertyBase[SimArray]):
    """
    Calculate the radius at which the surface density equals a given value.

    In ``mode='shell'`` the surface density at radius r is measured in the
    shell [r - eps/2, r + eps/2].

    In ``mode='total'`` the surface density is Σ(<r) = M(<r) / (π r^2).

    A 1D bisection in r is used to solve Σ(r) = target.
    """

    def __init__(self, target: ValueLike | ValueLikeFunc, parameter: str = "mass", mode: str = "shell", r_key: str = "rxy", eps: float = 0.01):
        """
        Parameters
        ----------
        target : float, SimArray, or callable
            Target surface density (in Σ units) or a callable `target(sim)`.
        parameter : str, default 'mass'
            Parameter to compute surface density from (e.g., 'mass', 'sfr').
        mode : {'shell', 'total'}, default 'shell'
            'shell' -> Σ_shell(r) in shell [r - eps/2, r + eps/2]
            'total' -> Σ(<r) = M(<r) / (π r^2)
        r_key : str, default 'rxy'
            Radius key in the snapshot.
        eps : float, default 0.01
            Shell width in radius units (same units as `r_key`).
        """
        if mode not in ("shell", "total"):
            raise ValueError("mode must be 'shell' or 'total'")
        self.target = target
        self.parameter = parameter
        self.mode = mode
        self.r_key = r_key
        self.eps = eps

    def instance_signature(self) -> tuple:
        return (
            self.__class__.__name__,
            get_signature_safe(self.target, fallback_to_id=True),
            self.parameter,
            self.mode,
            self.r_key,
            self.eps,
        )

    def _target_value(self, sim: SimSnap) -> float:
        """Target surface density as scalar in sim surface-density units."""
        surf_units = sim[self.parameter].units / sim["pos"].units**2
        if callable(self.target):
            raw_target = self.target(sim)
        else:
            raw_target = self.target
        return  self._in_sim_units(raw_target, surf_units, sim, target_units=surf_units)

    @staticmethod
    def _sigma_at_radius(
        r_val: float,
        r_sorted: np.ndarray,
        m_cum: np.ndarray,
        eps: float,
        mode: str,
    ) -> float:
        """
        Surface density at radius r_val using sorted radii & cumulative mass.

        All inputs are in base units: r in pos-units, mass in parameter-units.
        """
        sigma = 0.0
        if r_val <= 0:
            return sigma

        if mode == "total":
            hi = np.searchsorted(r_sorted, r_val, side="right")
            if hi > 0:
                m_inside = m_cum[hi - 1]
                area = np.pi * (r_val**2)
                if area > 0:
                    sigma = float(m_inside / area)
            return sigma

        # mode == "shell": shell [r_val - eps/2, r_val + eps/2]
        rin = r_val - 0.5 * eps
        rout = r_val + 0.5 * eps
        if rout <= 0:
            return 0.0
        rin = max(rin, 0.0)

        lo = np.searchsorted(r_sorted, rin, side="left")
        hi = np.searchsorted(r_sorted, rout, side="right")

        if hi > 0 and hi > lo:
            m_shell = m_cum[hi - 1] - (m_cum[lo - 1] if lo > 0 else 0.0)
            area_shell = np.pi * (rout ** 2 - rin ** 2)
            if area_shell > 0:
                sigma = float(m_shell / area_shell)
        return sigma


    def calculate(self, sim: SimSnap) -> SimArray:
        r_arr = sim[self.r_key]
        m_arr = sim[self.parameter]

        if is_dask_array(r_arr) or is_dask_array(m_arr):
            raise NotImplementedError(
                "RadiusAtSurfaceDensity currently only supports non-dask arrays"
            )

        # Work in base units
        pos_units = sim["pos"].units
        mass_units = sim[self.parameter].units

        r_vals = np.asarray(r_arr.in_units(pos_units))
        m_vals = np.asarray(m_arr.in_units(mass_units))

        # Sort and build cumulative mass
        idx = np.argsort(r_vals)
        r_sorted = r_vals[idx]
        m_sorted = m_vals[idx]
        m_cum = np.cumsum(m_sorted)

        r_min = float(r_sorted[0])
        r_max = float(r_sorted[-1])
        if r_max <= r_min:
            raise ValueError("Degenerate radius distribution")

        t = self._target_value(sim)

        # Choose a bracket where Σ(r) crosses t.
        # For shell mode we avoid the very edges so shells are well-defined.
        half_eps = 0.5 * self.eps
        r_lo = max(r_min + half_eps, r_min * 1.0001)
        r_hi = max(r_lo + self.eps, r_max - half_eps)

        sigma_lo = self._sigma_at_radius(r_lo, r_sorted, m_cum, self.eps, self.mode)
        sigma_hi = self._sigma_at_radius(r_hi, r_sorted, m_cum, self.eps, self.mode)

        # Expect monotonically decreasing profile with radius
        s_max, s_min = sigma_lo, sigma_hi
        if t > s_max or t < s_min:
            raise ValueError(
                f"Target surface density {t} outside profile range "
                f"[{s_min}, {s_max}] in units "
                f"{sim[self.parameter].units / sim['pos'].units**2}"
            )

        # Ensure (sigma_lo - t) and (sigma_hi - t) have opposite signs
        if (sigma_lo - t) * (sigma_hi - t) > 0:
            raise ValueError("Could not bracket the root for Σ(r) = target")

        # Bisection until interval smaller than eps/4
        while (r_hi - r_lo) > (self.eps / 4.0):
            r_mid = 0.5 * (r_lo + r_hi)
            sigma_mid = self._sigma_at_radius(
                r_mid, r_sorted, m_cum, self.eps, self.mode
            )
            if (sigma_lo - t) * (sigma_mid - t) <= 0:
                r_hi, sigma_hi = r_mid, sigma_mid
            else:
                r_lo, sigma_lo = r_mid, sigma_mid

        r_star = 0.5 * (r_lo + r_hi)

        r_out = SimArray(r_star)
        r_out.units = pos_units
        r_out.sim = sim.ancestor
        return r_out
