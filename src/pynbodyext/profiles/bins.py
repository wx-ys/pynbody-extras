"""
Flexible 1D binning utilities for particle-based simulation data.

This module defines the :class:`~pynbodyext.profiles.bins.BinsSet` class, a pluggable helper
for constructing one-dimensional bins (edges, centers) from a simulation
(:class:`~pynbody.snapshot.SimSnap`) and assigning particles to those bins.

Extensibility
-------------
The design emphasizes extensibility via registries for three stages:

1. Data extraction (``bins_by``) — how to obtain the 1D quantity to be binned.
2. Edge construction (``bins_type``) — how to compute bin edges when the user supplies
   an integer number of bins.
3. Area/volume calculation (``bins_area``) — how to compute geometric measures (area or volume)
   per bin.

Registries
----------
You can register new behaviors using decorators:

- :meth:`~pynbodyext.profiles.bins.BinsSet.bins_by_register`
- :meth:`~pynbodyext.profiles.bins.BinsSet.bins_algorithm_register`
- :meth:`~pynbodyext.profiles.bins.BinsSet.bins_area_register`

Each decorator stores a callable under the provided (or inferred) name so that
users may pass a string instead of a function. This keeps user code concise:

>>> bins = BinsSet(bins_by="r", bins_area="annulus", bins_type="lin", nbins=20)
>>> bins(sim)  # materialize
>>> bins.rbins        # centers
>>> bins.bin_edges    # edges
>>> bins.npart_bins   # counts per bin
>>> bins.binsize      # area per bin

Error handling
--------------
The bin-building step validates inputs (e.g., explicit edges must be 1D with length ≥ 2).
Algorithms may raise :class:`ValueError` for invalid input domains (e.g., logarithmic bins
with non-positive minima).

See Also
--------
- :class:`~pynbodyext.profiles.profile.Profile`
- :class:`~pynbodyext.profiles.proarray.ProfileArray`
"""

from collections.abc import Callable
from typing import Any, Literal, TypeAlias, overload

import numpy as np
from pynbody.array import IndexedSimArray, SimArray
from pynbody.snapshot import SimSnap

SimOrNpArray: TypeAlias = SimArray | np.ndarray | IndexedSimArray
"""Type alias for arrays accepted or returned by binning routines."""

BinByFunc: TypeAlias = Callable[[SimSnap], SimOrNpArray]
"""Callable extracting a 1D array from a simulation."""

BinsAreaFunc: TypeAlias = Callable[["BinsSet", SimOrNpArray], SimOrNpArray]
"""Callable computing per-bin area or volume given bin edges."""

BinsAlgorithmFunc: TypeAlias = Callable[["BinsSet", SimOrNpArray], SimOrNpArray]
"""Callable constructing bin edges from raw data (returns length nbins+1)."""


RegistBinByString: TypeAlias = str | Literal["r","rxy"]
RegistBinAreaString: TypeAlias = str | Literal["length","annulus","cylindrical_shell","spherical_shell"]
RegistBinAlgorithmString: TypeAlias = str | Literal["lin","log","equaln"]


__all__ = ["BinsSet"]


class BinsSet:
    """
    Flexible, pluggable 1D binning helper for particle-based data.

    This class constructs bin edges and assigns particles to bins using:

    - a ``bins_by`` source (field name, registry key, or callable) to obtain the 1D
      quantity ``x`` to bin,
    - a ``bins_type`` algorithm (registry key or callable) to produce bin edges,
      or an explicit array of edges,
    - a ``bins_area`` calculator (registry key or callable) to compute bin area/volume
      per bin (optional but recommended).

    Attributes are populated after calling the instance with a :class:`~pynbody.snapshot.SimSnap`.

    Parameters
    ----------
    bins_by : str or :data:`~pynbodyext.profiles.bins.BinByFunc`
        How to obtain the 1D data array to bin. Resolution order:
        1. If callable: call directly.
        2. If string and in registry: use registered callable.
        3. Else: treat as simulation field key (e.g., ``sim["r"]``).
    bins_area : str or :data:`~pynbodyext.profiles.bins.BinsAreaFunc`
        Calculator for per-bin area or volume (returns array of length ``nbins``).
        If a string, must be a registry key.
    bins_type : str or :data:`~pynbodyext.profiles.bins.BinsAlgorithmFunc`
        Edge-building algorithm when ``nbins`` is an integer. Must produce an array
        of edges (length ``nbins + 1``). Ignored if explicit edges are supplied as ``nbins``.
    nbins : int or :class:`~numpy.ndarray` or :class:`~pynbody.array.SimArray`
        If int: the number of bins (used with ``bins_type``). If array-like: treated as explicit
        bin edges (length ≥ 2).
    bin_min : float, optional
        Minimum domain value for edge construction when using an algorithm.
        Defaults to the minimum of the data.
    bin_max : float, optional
        Maximum domain value for edge construction when using an algorithm.
        Defaults to the maximum of the data.
    **kwargs
        Extra parameters stored on the instance for use by custom algorithms and area/volume calculators.

    Attributes
    ----------
    x : :data:`~pynbodyext.profiles.bins.SimOrNpArray` or None
        The resolved data array being binned.
    bin_edges : :data:`~pynbodyext.profiles.bins.SimOrNpArray` or None
        Array of bin edges (length ``nbins + 1``).
    rbins : :data:`~pynbodyext.profiles.bins.SimOrNpArray` or None
        Bin centers (midpoints) (length ``nbins``).
    dr : :data:`~pynbodyext.profiles.bins.SimOrNpArray` or None
        Approximate half-widths computed via :func:`numpy.gradient`.
        For exact widths use :func:`numpy.diff` on :attr:`bin_edges`.
    binind : list[:class:`~numpy.ndarray`] or None
        List of index arrays per bin (particle indices).
    npart_bins : :class:`~numpy.ndarray` or None
        Number of particles in each bin (length ``nbins``).
    binsize : :data:`~pynbodyext.profiles.bins.SimOrNpArray` or None
        Area or volume per bin (depends on ``bins_area`` implementation).
    """
    _bins_by_registry: dict[str, BinByFunc] = {}
    _bins_area_registry: dict[str, BinsAreaFunc] = {}
    _bins_algorithm_registry: dict[str, BinsAlgorithmFunc] = {}

    def __init__(
        self,
        bins_by: RegistBinByString | BinByFunc,
        bins_area: RegistBinAreaString | BinsAreaFunc,
        bins_type: RegistBinAlgorithmString | BinsAlgorithmFunc,
        nbins: int | np.ndarray | SimArray,
        bin_min:float | None = None,
        bin_max:float | None = None,
        **kwargs: Any,
        ) -> None:
        self._bins_by = bins_by
        self._bins_area = bins_area
        self._bins_type = bins_type
        self._nbins = nbins
        self._bin_min = bin_min
        self._bin_max = bin_max
        self._kwargs = kwargs

        # Computed attributes (populated after __call__)
        self.x: SimArray | np.ndarray | None = None
        self.bin_edges: SimArray | np.ndarray | None = None
        self.rbins: SimArray | np.ndarray | None = None
        self.dr: SimArray | np.ndarray | None = None
        self.binind: list[np.ndarray] | None = None
        self.npart_bins: np.ndarray | None = None
        self.binsize: SimArray | np.ndarray | None = None  # area or volume

    def is_defined(self) -> bool:
        """
        Check if the :class:`~pynbodyext.profiles.bins.BinsSet` is materialized.

        Returns
        -------
        bool
            ``True`` if bin edges and derived attributes are defined, ``False`` otherwise.
        """
        return (self.bin_edges is not None
                and self.rbins is not None
                and self.dr is not None
                and self.binind is not None
                and self.npart_bins is not None
                and self.binsize is not None)

    def _resolve_x(self, sim: SimSnap) -> SimOrNpArray:
        """
        Resolve the data array to bin from ``self._bins_by``.

        Resolution order
        ----------------
        1. If ``_bins_by`` is callable: call it.
        2. If string and present in ``_bins_by_registry``: call registered function.
        3. Else: treat as a field key in the simulation.

        Parameters
        ----------
        sim : :class:`~pynbody.snapshot.SimSnap`
            Simulation snapshot.

        Returns
        -------
        :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            The resolved 1D data array.

        Raises
        ------
        ValueError
            If ``_bins_by`` is neither a callable nor a string.
        """
        if callable(self._bins_by):
            x = self._bins_by(sim)
        elif isinstance(self._bins_by, str):
            try:
                x = self._bins_by_registry[self._bins_by](sim)
            except KeyError:
                # fall back to simulation field lookup
                x = sim[self._bins_by]
        else:
            raise ValueError(
                f"Invalid bins_by: {self._bins_by}, required callable or registry keys: {list(self._bins_by_registry)}"
            )
        return x

    def _coerce_edges_units(self, edges: np.ndarray, x: SimOrNpArray) -> SimOrNpArray:
        """
        Attach units and simulation reference to edges if ``x`` is a :class:`~pynbody.array.SimArray`.

        Parameters
        ----------
        edges : :class:`~numpy.ndarray`
            Raw bin edges.
        x : :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            The source data array.

        Returns
        -------
        :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Either the original edges array or a :class:`~pynbody.array.SimArray` clone
            with units/sim copied.
        """
        if isinstance(x, SimArray):
            out = SimArray(edges)
            out.units = x.units
            out.sim = x.sim
            return out
        return edges

    def _build_edges(self, x: SimOrNpArray) -> SimOrNpArray:
        """
        Construct bin edges from explicit array or using the registered algorithm.

        Behavior
        --------
        - If ``_nbins`` is an array-like: interpret directly as edges.
        - Else: use ``_bins_type`` (callable or registry entry) to build edges spanning
          the domain defined by (``bin_min``, ``bin_max``) or (``min(x)``, ``max(x)``).

        Parameters
        ----------
        x : :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Data array used to determine domain and quantiles.

        Returns
        -------
        :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Array of bin edges (length ``nbins + 1``).

        Raises
        ------
        ValueError
            If explicit edges are invalid (not 1D or length < 2) or if ``_bins_type`` is unrecognized.
        """
        # If nbins is an array, interpret as explicit edges
        if not isinstance(self._nbins, (int, np.integer)):
            arr = np.asarray(self._nbins)
            if arr.ndim != 1 or arr.shape[0] < 2:
                raise ValueError("Explicit bin_edges must be a 1D array of length >= 2")
            return self._coerce_edges_units(arr, x)

        if callable(self._bins_type):
            edges = self._bins_type(self, x)
            return self._coerce_edges_units(edges, x)
        elif isinstance(self._bins_type, str):
            edges = self._bins_algorithm_registry[self._bins_type](self, x)
        else:
            raise ValueError(
                f"Invalid bins_type: {self._bins_type}, required callable or registry keys: {list(self._bins_algorithm_registry)}"
            )

        return self._coerce_edges_units(edges, x)

    def _calc_area_or_volume(self, bin_edges: SimOrNpArray) -> SimOrNpArray:
        """
        Compute per-bin area or volume using ``self._bins_area``.

        Parameters
        ----------
        bin_edges : :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Bin edge array (length ``nbins + 1``).

        Returns
        -------
        :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Area/volume per bin (length ``nbins``).

        Raises
        ------
        ValueError
            If ``_bins_area`` is not a callable or registered key.
        """
        if callable(self._bins_area):
            return self._bins_area(self, bin_edges)
        elif isinstance(self._bins_area, str):
            return self._bins_area_registry[self._bins_area](self, bin_edges)
        else:
            raise ValueError(f"Invalid bins_area: {self._bins_area}, required callable or registry keys: {list(self._bins_area_registry)}")

    @staticmethod
    def _calc_binmid(bin_edges: SimOrNpArray) -> SimOrNpArray:
        """
        Compute bin centers (midpoints) from edges.

        Parameters
        ----------
        bin_edges : :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Edge array (length ``nbins + 1``).

        Returns
        -------
        :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Midpoints (length ``nbins``).
        """
        return 0.5 * (bin_edges[:-1] + bin_edges[1:])


    def _assign_particles(self, x: SimOrNpArray, bin_edges: SimOrNpArray) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Assign particle indices to bins via :func:`numpy.digitize`.

        Out-of-range values are clamped to the first or last bin.

        Parameters
        ----------
        x : :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Data array being binned.
        bin_edges : :data:`~pynbodyext.profiles.bins.SimOrNpArray`
            Edge array (length ``nbins + 1``).

        Returns
        -------
        binind : list[:class:`~numpy.ndarray`]
            List of index arrays, one per bin.
        npart_bins : :class:`~numpy.ndarray`
            Number of particles per bin.
        """
        partbin = np.digitize(np.asarray(x), np.asarray(bin_edges), right=True) - 1
        nbins = len(bin_edges) - 1
        if nbins <= 0:
            return [], np.array([], dtype=int)


        # keep only values that fell into a valid bin
        valid_mask = (partbin >= 0) & (partbin < nbins)
        if not np.any(valid_mask):
            return [np.empty(0, dtype=int) for _ in range(nbins)], np.zeros(nbins, dtype=int)

        bin_id = partbin[valid_mask]
        idx = np.nonzero(valid_mask)[0]

        # counts per bin (fast, vectorized)
        counts = np.bincount(bin_id, minlength=nbins).astype(int)

        # group indices by bin with a single stable sort, then slice by cumulative counts
        order = np.argsort(bin_id, kind="stable")
        idx_sorted = idx[order]
        starts = np.concatenate(([0], np.cumsum(counts)))
        binind = [idx_sorted[starts[i]:starts[i+1]] for i in range(nbins)]

        return binind, counts

    def __call__(self, sim: SimSnap, inplace: bool = False) -> "BinsSet":
        """
        Materialize binning for a simulation.

        Default behavior (``inplace=False``)
        ------------------------------------
        Acts as a factory: returns a NEW :class:`~pynbodyext.profiles.bins.BinsSet` instance with all computed
        attributes populated, leaving the original instance (the “configuration object”) untouched.

        Backward-compatible mode (``inplace=True``)
        -------------------------------------------
        Retains previous semantics: mutates ``self`` in place and returns it.

        Steps
        -----
        1. Resolve data array ``x``.
        2. Build or accept bin edges.
        3. Compute centers and approximate half-widths.
        4. Assign particles to bins.
        5. Compute per-bin area/volume.

        Parameters
        ----------
        sim : :class:`~pynbody.snapshot.SimSnap`
            Simulation snapshot.
        inplace : bool, default ``False``
            If ``True``, mutate and return ``self``. If ``False``, return a new materialized instance.

        Returns
        -------
        :class:`~pynbodyext.profiles.bins.BinsSet`
            A materialized bin set (new object unless ``inplace=True``).
        """
        target: BinsSet
        if inplace:
            target = self
        else:
            # Create a fresh instance carrying over configuration
            target = BinsSet(
                bins_by=self._bins_by,
                bins_area=self._bins_area,
                bins_type=self._bins_type,
                nbins=self._nbins,
                bin_min=self._bin_min,
                bin_max=self._bin_max,
                **self._kwargs,
            )

        target.x = target._resolve_x(sim)
        target.bin_edges = target._build_edges(target.x)

        # Safety clamp: ensure valid edges
        if target.bin_edges.ndim != 1 or target.bin_edges.shape[0] < 2:
            target.bin_edges = np.asarray([0.0, 1.0])

        target.rbins = target._calc_binmid(target.bin_edges)
        target.dr = np.gradient(target.rbins)   # approx. half-widths
        target.binind, target.npart_bins = target._assign_particles(target.x, target.bin_edges)
        target.binsize = target._calc_area_or_volume(target.bin_edges)

        return target


    def spawn_with_same_edges(self, sim: SimSnap) -> "BinsSet":
        """
        Create a new :class:`~pynbodyext.profiles.bins.BinsSet` for ``sim``, reusing
        the already-computed edges/centers/widths, and recomputing bin assignments.
        """
        child = BinsSet(
            bins_by=self._bins_by,
            bins_area=self._bins_area,
            bins_type=self._bins_type,
            nbins=self.bin_edges,   # explicit edges
            bin_min=self._bin_min,
            bin_max=self._bin_max,
            **self._kwargs
        )
        child.x = child._resolve_x(sim)
        child.bin_edges = self.bin_edges
        child.rbins = self.rbins
        child.dr = self.dr
        child.binind, child.npart_bins = child._assign_particles(child.x, child.bin_edges)
        child.binsize = self.binsize
        return child

    @classmethod
    def available_options(cls) -> dict:
        return {
            "bins_by": list(cls._bins_by_registry.keys()),
            "bins_area": list(cls._bins_area_registry.keys()),
            "bins_type": list(cls._bins_algorithm_registry.keys()),
        }

    def __repr__(self)->str:
        return (
            f"BinsSet(bins_by={self._bins_by}, bins_area={self._bins_area}, "
            f"bins_type={self._bins_type}, nbins={self._nbins}, "
            f"bin_min={self._bin_min}, bin_max={self._bin_max})"
        )


    # ------------------- registry helpers ---------------------------------#

    @classmethod
    @overload
    def bins_by_register(
        cls,
        fn: BinByFunc,
        name: str | None = None,
    ) -> BinByFunc: ...
    @classmethod
    @overload
    def bins_by_register(
        cls,
        fn: None = None,
        name: str | None = None,
    ) -> Callable[[BinByFunc], BinByFunc]: ...

    @classmethod
    def bins_by_register(
        cls,
        fn: BinByFunc | None = None,
        name: str | None = None
        ) -> BinByFunc | Callable[[BinByFunc], BinByFunc]:
        """
        Register a function that extracts a 1D array from a simulation.

        The decorated function must accept ``sim: SimSnap`` and return a
        1D `SimOrNpArray`.

        Parameters
        ----------
        fn : Callable[[SimSnap], SimOrNpArray], optional
            Function to register. If omitted, used as a decorator factory.
        name : str, optional
            Registry key. Defaults to ``fn.__name__``.

        Returns
        -------
        BinByFunc or Callable
            The original function (when called directly) or a decorator.

        Examples
        --------
        >>> @BinsSet.bins_by_register(name="speed")
        ... def extract_speed(sim):
        ...     v = sim["vel"]
        ...     return np.sqrt((v**2).sum(axis=1))

        Raises
        ------
        ValueError
            If a duplicate key is registered (not enforced here but could be added).
        """
        def decorator(func: BinByFunc) -> BinByFunc:
            cls._bins_by_registry[name or func.__name__] = func
            return func
        if fn is None:
            return decorator
        return decorator(fn)



    @classmethod
    @overload
    def bins_area_register(
        cls,
        fn: BinsAreaFunc,
        name: str | None = None,
    ) -> BinsAreaFunc: ...
    @classmethod
    @overload
    def bins_area_register(
        cls,
        fn: None = None,
        name: str | None = None,
    ) -> Callable[[BinsAreaFunc], BinsAreaFunc]: ...

    @classmethod
    def bins_area_register(
        cls,
        fn: BinsAreaFunc | None = None,
        name: str | None = None
    ) -> BinsAreaFunc | Callable[[BinsAreaFunc], BinsAreaFunc]:
        """
        Register an area/volume calculator.

        The function must accept ``(self: BinsSet, bin_edges: SimOrNpArray)``
        and return an array of length nbins.

        Parameters
        ----------
        fn : Callable[[BinsSet, SimOrNpArray], SimOrNpArray], optional
            Function to register or omitted for decorator usage.
        name : str, optional
            Registry key; defaults to the function name.

        Returns
        -------
        BinsAreaFunc or Callable
            The registered function or a decorator.

        Examples
        --------
        >>> @BinsSet.bins_area_register
        ... def shell_volume(self, edges):
        ...     return 4/3 * np.pi * (edges[1:]**3 - edges[:-1]**3)
        """
        def decorator(func: BinsAreaFunc) -> BinsAreaFunc:
            cls._bins_area_registry[name or func.__name__] = func
            return func
        if fn is None:
            return decorator
        return decorator(fn)


    @classmethod
    @overload
    def bins_algorithm_register(
        cls,
        fn: BinsAlgorithmFunc,
        name: str | None = None,
    ) -> BinsAlgorithmFunc: ...
    @classmethod
    @overload
    def bins_algorithm_register(
        cls,
        fn: None = None,
        name: str | None = None,
    ) -> Callable[[BinsAlgorithmFunc], BinsAlgorithmFunc]: ...

    @classmethod
    def bins_algorithm_register(
        cls,
        fn: BinsAlgorithmFunc | None = None,
        name: str | None = None
    ) -> BinsAlgorithmFunc | Callable[[BinsAlgorithmFunc], BinsAlgorithmFunc]:
        """
        Register a bin edges construction algorithm.

        Signature: ``(self: BinsSet, x: SimOrNpArray) -> SimOrNpArray`` (edges).

        Parameters
        ----------
        fn : Callable[[BinsSet, SimOrNpArray], SimOrNpArray], optional
            Function to register or omitted for decorator usage.
        name : str, optional
            Registry key; defaults to function name.

        Returns
        -------
        BinsAlgorithmFunc or Callable
            The registered function or a decorator.

        Examples
        --------
        >>> @BinsSet.bins_algorithm_register(name="sqrt")
        ... def sqrt_edges(self, x):
        ...     nbins = int(self._nbins)
        ...     xmin, xmax = np.min(x), np.max(x)
        ...     return np.sqrt(np.linspace(xmin**2, xmax**2, nbins+1))
        """
        def decorator(func: BinsAlgorithmFunc) -> BinsAlgorithmFunc:
            cls._bins_algorithm_registry[name or func.__name__] = func
            return func
        if fn is None:
            return decorator
        return decorator(fn)


# ------------------- bins algorithms --------------------------------------#
@BinsSet.bins_algorithm_register(name="lin")
def linear_bins_algorithm(
    self: "BinsSet", x: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Build linearly spaced bin edges over the domain.

    Uses ``self._bin_min`` / ``self._bin_max`` when provided, else the min/max of ``x``.
    Returns an array of length ``nbins + 1``.
    """
    nbins = int(self._nbins)
    xmin = float(np.min(x)) if self._bin_min is None else self._bin_min
    xmax = float(np.max(x)) if self._bin_max is None else self._bin_max
    return np.linspace(xmin, xmax, nbins + 1)

@BinsSet.bins_algorithm_register(name="log")
def logarithmic_bins_algorithm(
    self: "BinsSet", x: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Build logarithmically spaced bin edges over the positive domain.

    Requires strictly positive lower bound. Returns an array of length ``nbins + 1``.
    """
    nbins = int(self._nbins)
    xmin = float(np.min(x)) if self._bin_min is None else self._bin_min
    xmax = float(np.max(x)) if self._bin_max is None else self._bin_max
    if xmin <= 0:
        raise ValueError("Logarithmic bins require xmin to be non-negative")
    return np.logspace(np.log10(xmin), np.log10(xmax), nbins + 1)

@BinsSet.bins_algorithm_register(name="equaln")
def equal_number_bins_algorithm(
    self: "BinsSet", x: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Build edges so that each bin contains approximately the same number of particles.

    Typically implemented via quantiles of ``x``. Returns an array of length ``nbins + 1``.
    """
    nbins = int(self._nbins)

    sorted_x = np.sort(x)
    if sorted_x.size == 0:
        raise ValueError("Cannot create bins: input array is empty")
    if self._bin_min is not None:
        sorted_x = sorted_x[sorted_x >= self._bin_min]
    if self._bin_max is not None:
        sorted_x = sorted_x[sorted_x <= self._bin_max]
    if sorted_x.size < 2:
        # Degenerate: fallback to two identical edges
        return np.array([sorted_x[0], sorted_x[0]], dtype=float)

    edges = [sorted_x[0]]
    for i in range(1, nbins):
        edges.append(sorted_x[int(i * len(sorted_x) / nbins)])
    edges.append(sorted_x[-1])
    return np.array(edges)

# ------------------- bins area/volume calculators -------------------------#

@BinsSet.bins_area_register(name="length")
def length_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Length of 1D bins: ``r_{i+1} - r_i`` for each bin.
    """
    return bin_edges[1:] - bin_edges[:-1]

@BinsSet.bins_area_register(name="annulus")
def annulus_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Area of 2D circular annuli: ``pi * (r_{i+1}^2 - r_i^2)`` for each bin.
    """
    return np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

@BinsSet.bins_area_register(name="spherical_shell")
def spherical_shell_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Volume of 3D spherical shells: ``4/3 * pi * (r_{i+1}^3 - r_i^3)`` for each bin.
    """
    return 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

@BinsSet.bins_area_register(name="cylindrical_shell")
def cylindrical_shell_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """
    Lateral surface area or volume proxy for cylindrical shells.

    Implementations typically read additional parameters (e.g., ``height``) from ``self._kwargs``.
    """
    z = self._kwargs.get("z", None)
    if z is None:
        raise ValueError("Parameter 'z' must be provided for cylindrical_shell area calculation")
    return np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)*z

