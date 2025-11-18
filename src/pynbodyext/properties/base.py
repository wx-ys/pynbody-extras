

import functools
import inspect
import operator
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any, Generic, TypeVar, Union

import numpy as np
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

from pynbodyext.calculate import CalculatorBase

__all__ = ["PropertyBase", "ParamSum", "ParameterContain", "ConstantProperty", "LambdaProperty", "OpProperty"]


# Ephemeral per-top-level-call evaluation cache (sim-local, auto-cleared)
_EVAL_CTX: ContextVar[dict[tuple[int, tuple], Any] | None] = ContextVar("_EVAL_CTX", default=None)
_EVAL_SIM_TOKEN: ContextVar[int | None] = ContextVar("_EVAL_SIM_TOKEN", default=None)


TProp = TypeVar("TProp", bound=SimArray | float | np.ndarray | int, covariant=True)

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

def _canon(value: Any) -> Any:
    """
    Canonicalize a value to a hashable, lightweight representation for signature.
    - Numbers/bools/str: keep (cast numpy scalars to Python scalars).
    - PropertyBase: use its signature.
    - np.ndarray / SimArray: use object id to avoid heavy hashing.
    - Sequences: tuple of canonicalized items.
    - Dicts: sorted items, canonicalized.
    - Callables: ("CALLABLE", id(func)).
    - Others: ("ID", id(obj)).
    """
    result: Any
    if isinstance(value, np.generic):
        result = value.item()
    elif isinstance(value, (int, float, bool, str)):
        result = value
    elif isinstance(value, PropertyBase):
        result = ("PROP", value.signature())
    elif isinstance(value, (np.ndarray, SimArray)):
        result = ("ARR", id(value))
    elif isinstance(value, (list, tuple)):        # type: ignore
        result = tuple(_canon(v) for v in value)
    elif isinstance(value, dict):
        result = ("DICT", tuple(sorted((k, _canon(v)) for k, v in value.items())))
    elif callable(value):
        result = ("CALLABLE", id(value))
    else:
        result = ("ID", id(value))
    return result
# ---------------- Base class with evaluation caching ----------------

class PropertyBase(CalculatorBase[TProp], Generic[TProp]):
    """A lazily evaluated calculator: SimSnap -> TProp."""

    _init_sig: tuple

    def __call__(self, sim: SimSnap) -> TProp:
        raise NotImplementedError

    # Automatically add a per-call evaluation cache wrapper to any subclass __call__
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # let CalculatorBase wrap first (filter/transforms)

        # ---- wrap __init__ to capture a canonical _init_sig ----
        orig_init = cls.__dict__.get("__init__", None)
        if orig_init is None:
            # No __init__ defined in subclass: still create a wrapper that records empty init sig
            def _default_init(self: "PropertyBase[Any]", *args: Any, **kw: Any) -> None:
                self._init_sig = ()
                super(cls, self).__init__()  # call up the MRO if needed
            _default_init.__init_sig_wrapper__ = True  # type: ignore[attr-defined]
            cls.__init__ = _default_init  # type: ignore
        elif not getattr(orig_init, "__init_sig_wrapper__", False):
            sig = inspect.signature(orig_init)
            @functools.wraps(orig_init)
            def _wrapped_init(self: "PropertyBase[Any]", *args: Any, **kw: Any) -> None:
                bound = sig.bind_partial(self, *args, **kw)
                bound.apply_defaults()
                # preserve declared parameter order; skip 'self'
                items: list[tuple[str, Any]] = []
                for name, _param in sig.parameters.items():
                    if name == "self":
                        continue
                    if name in bound.arguments:
                        items.append((name, _canon(bound.arguments[name])))
                    else:
                        # not passed and no default applied; keep a placeholder
                        items.append((name, ("MISSING",)))
                self._init_sig = tuple(items)
                return orig_init(self, *args, **kw)
            _wrapped_init.__init_sig_wrapper__ = True  # type: ignore[attr-defined]
            cls.__init__ = _wrapped_init  # type: ignore


        # ---- wrap __call__ for per-eval caching ----
        user_call = cls.__dict__.get("__call__", None)
        if user_call is None:
            return
        # Avoid double-wrapping
        if getattr(user_call, "__evalctx_wrapper__", False):
            return
        @functools.wraps(user_call)
        def _wrapped_call(self: "PropertyBase[Any]", sim: SimSnap, *args: Any, **kw: Any) -> Any:
            token = id(sim)
            ctx = _EVAL_CTX.get()
            cur_token = _EVAL_SIM_TOKEN.get()
            owner = False
            if ctx is None or cur_token != token:
                # start a new ephemeral context for this top-level call
                ctx = {}
                _EVAL_CTX.set(ctx)
                _EVAL_SIM_TOKEN.set(token)
                owner = True

            key = (token, self.signature())
            if key in ctx:
                result = ctx[key]
            else:
                result = user_call(self, sim, *args, **kw)
                ctx[key] = result

            if owner:
                # auto-clear after this top-level call finishes
                _EVAL_CTX.set(None)
                _EVAL_SIM_TOKEN.set(None)
            return result

        _wrapped_call.__evalctx_wrapper__ = True  # type: ignore[attr-defined]
        cls.__call__ = _wrapped_call  # type: ignore[method-assign]


    # ----- signature for caching -----
    def signature(self) -> tuple:
        """
        Unified structural signature:
          (class_name,
           ("init", self._init_sig or ()),
           ("filter", id(self._filter)), ("trans", id(self._transformation)), ("rev", bool(self._revert_transformation)))
        Subclasses don't need to override this in most cases.
        """
        init_sig = getattr(self, "_init_sig", ())
        return (
            self.__class__.__name__,
            ("init", init_sig),
            ("filter", id(getattr(self, "_filter", None))),
            ("trans", id(getattr(self, "_transformation", None))),
            ("rev", bool(getattr(self, "_revert_transformation", True))),
        )


    # -------- composition helpers --------
    def _as_property(self, other: Addable) -> "PropertyBase":
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
    def __lt__(self, other: Addable) -> "PropertyBase":
        return OpProperty("lt", [self, self._as_property(other)])
    def __le__(self, other: Addable) -> "PropertyBase":
        return OpProperty("le", [self, self._as_property(other)])
    def __gt__(self, other: Addable) -> "PropertyBase":
        return OpProperty("gt", [self, self._as_property(other)])
    def __ge__(self, other: Addable) -> "PropertyBase":
        return OpProperty("ge", [self, self._as_property(other)])
    def __eq__(self, other: Addable) -> "PropertyBase":  # type: ignore[override]
        return OpProperty("eq", [self, self._as_property(other)])
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

    def __call__(self, sim: SimSnap) -> TProp:
        return self._value


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
        # Conservative: function object id + cache flag
        return ("Lambda", id(self._func), self._cache_enabled)

    def __call__(self, sim: SimSnap) -> Any:
        if not self._cache_enabled:
            return self._func(sim)
        key = id(sim)
        if key in self._cache:
            return self._cache[key]
        val = self._func(sim)
        self._cache[key] = val
        return val

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

    def __call__(self, sim: SimSnap) -> Any:
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

    def __call__(self, sim : SimSnap) -> SimArray:
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

    def __call__(self, sim: SimSnap) -> SimArray:
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
        cal_key_crit.units = cal_key_sorted.units
        cal_key_crit.sim = sim.ancestor

        return cal_key_crit
