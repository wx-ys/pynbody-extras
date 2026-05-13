from __future__ import annotations

from typing import Any

import numpy as np

from .selectors import normalize_flat_bin_selector, normalize_nd_bin_selector


class BinParticlesAccessor:
    def __init__(self, bins: Any) -> None:
        self._bins = bins

    def __getitem__(self, selector: Any) -> Any:
        if isinstance(selector, str):
            raise TypeError("particles_at_bin does not accept string queries; use bins[...] for per-bin arrays.")
        if isinstance(selector, tuple):
            flat = normalize_nd_bin_selector(selector, self._bins.shape_bins)
        else:
            flat = normalize_flat_bin_selector(selector, self._bins.nbins)

        if flat.size == 0:
            return self._bins.sim[np.asarray([], dtype=int)]

        groups = [
            self._bins.bin_data[self._bins.bin_indptr[int(i)] : self._bins.bin_indptr[int(i) + 1]]
            for i in flat
        ]
        indices = np.concatenate(groups) if groups else np.asarray([], dtype=int)
        if indices.size:
            indices = np.unique(indices)
        return self._bins.sim[np.sort(indices)]
