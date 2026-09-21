# pynbodyext: Extensions and Utilities for pynbody

`pynbodyext` extends [pynbody](https://github.com/pynbody/pynbody) with composable
calculators, binned profiles and ready-to-publish 2-D maps.

**Note: This project is under active development. Feedback and contributions are welcome!**


## Installation

```bash
git clone https://github.com/wx-ys/pynbody-extras.git
cd pynbody-extras
pip install -e .          # extras: [image] adaptive binning, [store] SQL result store
```

---

## Quick Start

The examples assume `sim` is a loaded `pynbody` snapshot.

### Calculators

Properties, filters and transforms are lazily composed objects: build the graph
once, evaluate it on as many snapshots as you like.

```python
import numpy as np

from pynbodyext.filters import FamilyFilter, Sphere
from pynbodyext.properties import ParamContain, ParamSum
from pynbodyext.transforms import ShiftPosTo, WrapBox

# half-mass radius of the stars inside 30 kpc, in a wrapped, centre-shifted frame
re = ParamContain(0.5).filter(Sphere("30 kpc") & FamilyFilter("star")).transform(
    WrapBox().then(ShiftPosTo("ssc"))
)

re(sim)                              # evaluate: the half-mass radius
stellar_mass = ParamSum("mass").filter(FamilyFilter("star"))
stellar_mass / (4 * np.pi * re**2)   # arithmetic builds a new calculator, evaluated on call
```

Calling a calculator returns its value; `run(...)` returns the whole execution —
tree, timings, warnings:

```python
res = re.run(sim, progress="node", perf_memory=True)   # value + tree, timings, warnings
print(res.value, res.report_summary())
```

`re.dependency_tree` prints the graph itself, without running it:

```
ParamContain(0.5, "r", "mass")<prop>
├─ TransformChain<trans>
│  ├─ WrapBox(None, "minirange")<trans>
│  └─ ShiftPosTo("ssc")<trans>
│     └─ CenPos("ssc")<prop>
└─ AndFilter<filt>
   ├─ Sphere("30 kpc", (0, 0, 0))<filt>
   └─ FamilyFilter("star")<filt>
```

One calculator can be reused inside another: `0.5 * re` above is a
calculator-valued input, resolved automatically at runtime, so you never order the
computation by hand.

### Profiles

`Bin1D` (and `BinND` for two axes) bins a snapshot's particles and answers per-bin
statistics by name — `"<field>.<stat>"`, or `count` for particles per bin.

```python
from pynbodyext.core.calculate import Bin1D

r = Bin1D("r", vmin="0 kpc", vmax="30 kpc", nbins=20, mode="equaln")
profile = r(sim)

profile["count"]        # particles per bin
profile["mass.sum"]     # total mass          profile["vz.mean"]   # mean vz
profile["vz.disp"]      # dispersion          profile["vz.p16"]    # 16th percentile
profile["mass.sum"].plot()
profile.star            # the same bins, star particles only
```

`mode` is `"linear"`, `"log"` or `"equaln"` (or pass explicit `edges`), and
re-binning a filtered snapshot profiles just that subset:
`r(sim[Sphere("30 kpc") & FamilyFilter("star")])`.

### Image post-processing and visualization

Two axes make a map; `bins2d["query"].image` is the `ImageData` for one of them —
the values plus their bin edges, units and labels. `image.display.*` shows it,
`image.process.*` works on it.

```python
from pynbodyext.core.calculate import Bin1D
from pynbodyext.plot import image

bins2d = (Bin1D("x", vmin="-50 kpc", vmax="50 kpc", nbins=128, alias="x")
          @ Bin1D("y", vmin="-50 kpc", vmax="50 kpc", nbins=128, alias="y"))(sim)

bins2d.imshow("mass.sum", cmap="inferno", colorbar=True)     # the one-liner

mass = bins2d["mass.sum"].image
mass.display.imshow(log=True, colorbar="bottom")             # decades -> log scale
smoothed = mass.process.smooth.gaussian(fwhm=1.0)            # NaN-aware, fwhm in axis units
observed = smoothed.process.psf.convolve(fwhm=3.0)           # what a telescope would see
detected = observed.process.noise.poisson(exposure=0.05, rng=1)       # seeded, maskable
restored = observed.process.psf.wiener(image.gaussian_psf(fwhm=3.0))  # and back again

velocity = bins2d["vz.mean"].image                           # a velocity map
binned = velocity.process.adaptive.bin(bins2d["mass.sum"], target_nbins=200)
binned.display.imshow(cmap="sauron", symmetric=True, colorbar="bottom")
velocity.display.contour(levels=[-100, 0, 100], colors="w")  # on the same axes

gas, dm = bins2d.gas["mass.sum"].image, bins2d.dm["mass.sum"].image   # two maps, one figure
gas.process.compose.imshow(
    dm, style=image.MapStyle(cmap="inferno", stretch="log"), other_style=image.MapStyle(cmap="cividis")
)
```

Every step returns a new `ImageData` with the operation recorded in `.ops` (so
`repr` shows the chain), and the free functions (`gaussian_smooth`,
`compose_maps`, …) stay available for plain arrays. `cmap="sauron"` is the
Cappellari & Emsellem SAURON/ATLAS³D velocity map (green on zero); the masks
behind the stitching are exposed too (`mask1, mask2 = gas.process.compose.masks()`).
Adaptive binning needs the optional `powerbin` (`pip install pynbodyext[image]`);
everything else only needs `numpy`, `scipy` and `matplotlib`.

---
