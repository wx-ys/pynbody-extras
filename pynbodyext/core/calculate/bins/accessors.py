from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .selectors import normalize_flat_bin_selector, normalize_nd_bin_selector

if TYPE_CHECKING:
    from pynbody.snapshot import SimSnap

    from .result import BinNDResult


class BinParticlesAccessor:
    """Provides access to the particles in each bin via ``bins.particles_at_bin[...]``.

    You can select bins using integer indices, slices, or boolean arrays, just like with NumPy arrays.  For example:

    >>> bins.particles_at_bin[0]  # particles in the first bin
    >>> bins.particles_at_bin[1:5]  # particles in bins 1
    >>> bins.particles_at_bin[:, 0]  # particles in the first bin along the second axis (for 2D or higher)

    """

    def __init__(self, bins: BinNDResult) -> None:
        self._bins = bins

    def __repr__(self) -> str:
        return f"<BinParticlesAccessor for {self._bins}>"

    def __getitem__(self, selector: Any) -> SimSnap:
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
