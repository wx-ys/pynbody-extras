"""Accessors for retrieving the particles of individual bins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .selectors import normalize_flat_bin_selector, normalize_nd_bin_selector

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap

    from .result import BinNDResult


class BinParticlesAccessor:
    """Select the particles belonging to one or more bins.

    Returned by :attr:`BinNDResult.particles_at_bin`.  Accepts integer indices,
    slices, boolean bin masks, and N-D tuples (indexing ``"ij"``, last axis
    fastest), mirroring NumPy indexing.

    Examples
    --------
    >>> import pynbody
    >>> sim = pynbody.new(dm=6)
    >>> sim["x"] = [0, 1, 2, 3, 4, 5]
    >>> bins = Bin1D("x", vmin=0, vmax=6, nbins=3)(sim)
    >>> bins.particles_at_bin[0]["x"]  # particles in the first bin
    >>> bins.particles_at_bin[1:3]  # particles in the first two bins
    >>> bins.particles_at_bin[:, 0]  # 2-D: first bin along the second axis
    """

    def __init__(self, bins: BinNDResult) -> None:
        self._bins = bins

    def __repr__(self) -> str:
        return f"<BinParticlesAccessor for {self._bins}>"

    def __getitem__(self, selector: Any) -> SimSnap:
        """Return the ``SimSnap`` subset for the selected bin(s).

        Parameters
        ----------
        selector : int, slice, bool mask, sequence of int, or N-D tuple
            Bin selector as used for NumPy indexing.

        Returns
        -------
        SimSnap
            The particle subset (sim sub-snapshot) for the selected bin(s).

        Raises
        ------
        TypeError
            If ``selector`` is a string or otherwise unsupported.
        IndexError
            If a bin index is out of range.
        """
        if isinstance(selector, str):
            raise TypeError("particles_at_bin does not accept string queries; use bins[...] for per-bin arrays.")
        if isinstance(selector, tuple):
            flat = normalize_nd_bin_selector(selector, self._bins.shape_bins)
        else:
            flat = normalize_flat_bin_selector(selector, self._bins.nbins)

        if flat.size == 0:
            return self._bins.sim[np.asarray([], dtype=int)]

        bin_data = self._bins._bin_data
        bin_indptr = self._bins._bin_indptr
        assert bin_data is not None and bin_indptr is not None
        groups = [bin_data[bin_indptr[int(i)] : bin_indptr[int(i) + 1]] for i in flat]
        indices = np.concatenate(groups) if groups else np.asarray([], dtype=int)
        if indices.size:
            indices = np.unique(indices)
        return self._bins.sim[np.sort(indices)]
