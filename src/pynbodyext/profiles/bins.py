

from collections.abc import Callable
from typing import Any, TypeAlias, overload

import numpy as np
from pynbody.array import SimArray
from pynbody.snapshot import SimSnap

SimOrNpArray: TypeAlias = SimArray | np.ndarray

BinByFunc: TypeAlias = Callable[[SimSnap], SimOrNpArray]
BinsAreaFunc: TypeAlias = Callable[["BinsSet", SimOrNpArray], SimOrNpArray]
BinsAlgorithmFunc: TypeAlias = Callable[["BinsSet", SimOrNpArray], SimOrNpArray]

class BinsSet:
    """Flexible, pluggable 1D binning helper for particle-based data.

    This class constructs bin edges and assigns particles to bins using:
    - a "bins_by" source (field name, registry key, or callable) to obtain the 1D quantity x to bin,
    - a "bins_type" algorithm (registry key or callable) to produce bin edges, or an explicit array of edges,
    - a "bins_area" calculator (registry key or callable) to compute bin area/volume per bin (optional but recommended).

    You can extend behavior by registering functions with the decorators:
    - BinsSet.bins_by_register
    - BinsSet.bins_algorithm_register
    - BinsSet.bins_area_register

    Typical usage:
        >>> bins = BinsSet(bins_by="r", bins_area="annulus", bins_type="lin", nbins=20)
        >>> bins(sim)  # materialize against a SimSnap
        >>> bins.rbins        # midpoints (bin centers)
        >>> bins.bin_edges    # edges (length nbins+1)
        >>> bins.binind       # per-bin indices into the original particle array
        >>> bins.npart_bins   # number of particles per bin
        >>> bins.binsize      # per-bin area/volume (units depend on bins_area)

    Notes:
    - If `nbins` is an int (or numpy integer), it specifies how many bins to create using `bins_type`.
        If `nbins` is an array (or `SimArray`), it is interpreted as explicit bin edges of length >= 2.
    - If `bins_by` is a string, the code first tries a registered "bins_by" function; if not found,
        it looks up the field in the simulation as `sim[bins_by]`.
    - If `bins_type` or `bins_area` is a string, it must be a key in the respective registries.
    - For non-uniform bins, `dr` is computed as `np.gradient(rbins)`, which approximates half-widths near edges.
        If you need exact widths use `np.diff(bin_edges)`.

    Attributes (set after calling instance with a SimSnap)
    -----------------------------------------------------
    x : SimOrNpArray | None
        The binned 1D data array.
    bin_edges : SimOrNpArray | None
        Bin edges (length nbins+1). Units/ties are coerced to match `x` if `x` is a SimArray.
    rbins : SimOrNpArray | None
        Bin centers (length nbins).
    dr : SimOrNpArray | None
        Approximate bin half-widths computed via `np.gradient(rbins)` (length nbins).
    binind : list[np.ndarray] | None
        Per-bin indices into the original particle array.
    npart_bins : np.ndarray | None
        Number of particles per bin (length nbins).
    binsize : SimOrNpArray | None
        Per-bin area/volume (length nbins); units depend on the area/volume calculator.
    """
    _bins_by_registry: dict[str, BinByFunc] = {}
    _bins_area_registry: dict[str, BinsAreaFunc] = {}
    _bins_algorithm_registry: dict[str, BinsAlgorithmFunc] = {}

    def __init__(
        self,
        bins_by: str | BinByFunc,
        bins_area: str | BinsAreaFunc,
        bins_type: str | BinsAlgorithmFunc,
        nbins: int | np.ndarray | SimArray,
        bin_min:float | None = None,
        bin_max:float | None = None,
        **kwargs: Any,
        ) -> None:
        """
        Parameters
        ----------
        bins_by : str | Callable[[SimSnap], SimOrNpArray]
            How to obtain the 1D array to bin. Can be:
            - registry key (see `BinsSet._bins_by_registry`),
            - simulation field name (e.g., "r"),
            - callable receiving the `SimSnap` and returning a 1D array-like.
        bins_area : str | Callable[[BinsSet, SimOrNpArray], SimOrNpArray]
            How to compute per-bin area/volume from bin edges (length nbins+1).
            Can be registry key or a callable `(self, bin_edges) -> array` of length nbins.
            Some implementations may read extra parameters from `self._kwargs`.
        bins_type : str | Callable[[BinsSet, SimOrNpArray], SimOrNpArray]
            How to compute bin edges when `nbins` is an integer.
            Can be registry key (e.g., "lin", "log", "equaln") or callable `(self, x) -> edges`.
        nbins : int | np.ndarray | SimArray
            If int, number of bins to construct. If 1D array, it is used as explicit bin edges.
        bin_min : float | None, optional
            Minimum value for the edge builder (when bins_type is used). Defaults to min(x) if None.
        bin_max : float | None, optional
            Maximum value for the edge builder (when bins_type is used). Defaults to max(x) if None.
        **kwargs : Any
            Extra parameters passed/stored on `self._kwargs`, available to custom algorithms and area calculators.
        """

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


    def _resolve_x(self, sim: SimSnap) -> SimOrNpArray:
        """Resolve the data to bin from `bins_by` against the simulation."""
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
        """If x is a SimArray with units/sim, coerce the edges to a matching SimArray."""
        if isinstance(x, SimArray):
            out = SimArray(edges)
            out.units = x.units
            out.sim = x.sim
            return out
        return edges

    def _build_edges(self, x: SimOrNpArray) -> SimOrNpArray:
        """Construct bin edges either from explicit array (`nbins`) or via `bins_type` algorithm."""
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
        """Compute per-bin area/volume from edges using `bins_area`."""
        if callable(self._bins_area):
            return self._bins_area(self, bin_edges)
        elif isinstance(self._bins_area, str):
            return self._bins_area_registry[self._bins_area](self, bin_edges)
        else:
            raise ValueError(f"Invalid bins_area: {self._bins_area}, required callable or registry keys: {list(self._bins_area_registry)}")

    @staticmethod
    def _calc_binmid(bin_edges: SimOrNpArray) -> SimOrNpArray:
        """Compute bin centers as midpoints of edges."""
        return 0.5 * (bin_edges[:-1] + bin_edges[1:])


    def _assign_particles(self, x: SimOrNpArray, bin_edges: SimOrNpArray) -> tuple[list[np.ndarray], np.ndarray]:
        """Assign each particle to a bin via np.digitize and clamp out-of-range to edge bins."""
        partbin = np.digitize(np.asarray(x), np.asarray(bin_edges), right=True) - 1
        nbins = len(bin_edges) - 1
        partbin[partbin < 0] = 0
        partbin[partbin >= nbins] = nbins - 1 if nbins > 0 else 0
        binind: list[np.ndarray] = []
        for i in range(nbins):
            binind.append(np.where(partbin == i)[0])
        npart_bins = np.array([len(ind) for ind in binind], dtype=int)
        return binind, npart_bins

    def __call__(self, sim: SimSnap) -> "BinsSet":
        """Materialize the binning against the provided simulation.

        Returns
        -------
        self : BinsSet
            The instance with computed fields populated.
        """
        self.x = self._resolve_x(sim)
        self.bin_edges = self._build_edges(self.x)

        # Safety clamp: ensure valid edges
        if self.bin_edges.ndim != 1 or self.bin_edges.shape[0] < 2:
            self.bin_edges = np.asarray([0.0, 1.0])
        self.rbins = self._calc_binmid(self.bin_edges)
        self.dr = np.gradient(self.rbins)   # approx. half-widths for non-uniform bins
        self.binind, self.npart_bins = self._assign_particles(self.x, self.bin_edges)
        self.binsize = self._calc_area_or_volume(self.bin_edges)
        return self


    def spawn_with_same_edges(self, sim: SimSnap) -> "BinsSet":
        """Create a new BinsSet on another simulation but reusing the same bin edges.

        Parameters
        ----------
        sim : SimSnap
            Target simulation to map the same edges onto.

        Returns
        -------
        BinsSet
            A child bins set with `x`, `binind`, `npart_bins` recomputed for `sim`,
            and `bin_edges`, `rbins`, `dr`, `binsize` copied.
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
        """Register a function that extracts a 1D array from a SimSnap.

        The function should have signature: `(sim: SimSnap) -> SimOrNpArray`.

        Parameters
        ----------
        fn : Callable[[SimSnap], SimOrNpArray]
            The function to register.
        name : str | None
            Optional registry key; defaults to `fn.__name__`.

        Returns
        -------
        BinByFunc
            The original function, for decorator usage.
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
        """Register a function that computes per-bin area/volume from edges.

        Signature: `(self: BinsSet, bin_edges: SimOrNpArray) -> SimOrNpArray` (length nbins).

        Parameters
        ----------
        fn : Callable[[BinsSet, SimOrNpArray], SimOrNpArray]
            The function to register.
        name : str | None
            Optional registry key; defaults to `fn.__name__`.

        Returns
        -------
        BinsAreaFunc
            The original function, for decorator usage.
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
        """Register a function that constructs bin edges from data.

        Signature: `(self: BinsSet, x: SimOrNpArray) -> SimOrNpArray` (length nbins+1).

        Parameters
        ----------
        fn : Callable[[BinsSet, SimOrNpArray], SimOrNpArray]
            The function to register.
        name : str | None
            Optional registry key; defaults to `fn.__name__`.

        Returns
        -------
        BinsAlgorithmFunc
            The original function, for decorator usage.
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
    """Equally spaced linear bin edges between [xmin, xmax]."""
    nbins = int(self._nbins)
    xmin = float(np.min(x)) if self._bin_min is None else self._bin_min
    xmax = float(np.max(x)) if self._bin_max is None else self._bin_max
    return np.linspace(xmin, xmax, nbins + 1)

@BinsSet.bins_algorithm_register(name="log")
def logarithmic_bins_algorithm(
    self: "BinsSet", x: SimOrNpArray
    ) -> SimOrNpArray:
    """Log-spaced bin edges between [xmin, xmax]; requires xmin > 0."""
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
    """Quantile-like edges so that each bin has ~equal number of points."""
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
@BinsSet.bins_area_register(name="annulus")
def annulus_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """Area of 2D circular annuli: π (r_{i+1}^2 - r_i^2)."""
    return np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)

@BinsSet.bins_area_register(name="spherical_shell")
def spherical_shell_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """Volume of spherical shells: (4/3) π (r_{i+1}^3 - r_i^3)."""
    return 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)

@BinsSet.bins_area_register(name="cylindrical_shell")
def cylindrical_shell_area(
    self: "BinsSet", bin_edges: SimOrNpArray
    ) -> SimOrNpArray:
    """Volume of cylindrical shells with height z: π (r_{i+1}^2 - r_i^2) z.

    Requires: `z` provided in `self._kwargs` (e.g., BinsSet(..., z=height)).
    """
    z = self._kwargs.get("z", None)
    if z is None:
        raise ValueError("Parameter 'z' must be provided for cylindrical_shell area calculation")
    return np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)*z

