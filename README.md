# pynbodyext: Extensions and Utilities for pynbody

`pynbodyext` provides a set of extensions and utilities based on [pynbody](https://github.com/pynbody/pynbody) library.

**Note: This project is under active development. Feedback and contributions are welcome!**


## Installation

Clone the repository and install in editable (-e) mode:

```bash
git clone https://github.com/wx-ys/pynbody-extras.git
cd pynbody-extras
pip install -e .
```

---

## Quick Start



The examples below assume `sim` is an already loaded `pynbody` snapshot.

### Reusable calculators

Calculators are composable analysis objects. You can build them once and apply
them to many simulations.

```python
import numpy as np

from pynbodyext.filters import FamilyFilter
from pynbodyext.properties import ParamContain, ParamSum

# Half-mass radius of stars
re = ParamContain("r", 0.5, "mass").filter(FamilyFilter("star"))

# Total stellar mass
stellar_mass = ParamSum("mass").filter(FamilyFilter("star"))

# Derived quantity built from calculators
stellar_density = stellar_mass / (4 * np.pi * re**2)

# Direct call returns the public value
value = stellar_density(sim)
print(value)
```

A calculator behaves like a lazily defined analysis graph. Arithmetic between
calculators creates a new calculator rather than immediately evaluating anything.


### Filters and transforms

Filters select particles. Transforms temporarily modify the active frame before
evaluation.

```python
from pynbodyext.filters import FamilyFilter, Sphere
from pynbodyext.properties import ParamContain
from pynbodyext.transforms import ShiftPosTo, WrapBox

re = (ParamContain("r",0.5,"mass")
    .filter(
    Sphere("30 kpc") & FamilyFilter("star")
    # combine filters with logical operators (e.g., &, |, ~
    # the star particles within a sphere of radius 30 kpc
    ).transform(
        WrapBox(
        ).then(
        ShiftPosTo("ssc")
        )
        # apply a sequence of transforms to the simulation before computing the property:
        # means deal with the periodic boundary condition by wrapping particles into the box, 
        # and then shift the positions to the center
    )
)
# see the structure of the pipeline:
print(re.dependency_tree)
```
```
ParamContain("r", 0.5, "mass")<prop>
├─ TransformChain<trans>
│  ├─ WrapBox(None, "minirange")<trans>
│  └─ ShiftPosTo("ssc")<trans>
│     └─ CenPos("ssc")<prop>
└─ AndFilter<filt>
   ├─ Sphere("30 kpc", (0, 0, 0))<filt>
   └─ FamilyFilter("star")<filt>
```


### Run with diagnostics

Use ``run(...)`` when you want the full execution result instead of just the final
public value.

```python
# apply the pipeline to a simulation, with progress logging and memory performance tracking:
res = re.run(sim, progress="node",perf_memory=True)

print(res.value)
SimArray(3.41225841, 'kpc')

# You can also use the `pipeline_report` method to get a detailed report of the execution:
print(res.pipeline_report())
```
Example progress output:
```python
pynext.progress: run start ParamContain
pynext.progress: ├─ [n1] ParamContain <property> start
...
pynext.progress: │  │  │  ├─ [n5] CenPos <property> ok 411.45 ms
pynext.progress: │  │  ├─ [n4] ShiftPosTo <transform> ok 427.60 ms
pynext.progress: │  ├─ [n2] TransformChain <transform> ok 532.92 ms
...
pynext.progress: ├─ [n1] ParamContain <property> ok 816.12 ms
pynext.progress: run end ParamContain status=ok total=820.62 ms nodes=9 warnings=0 errors=0
```


If you only want the final public value, call the calculator directly:

```python
value = re(sim)
```

#### A Larger Example
The calculator system supports dynamic dependencies between nodes, so one
calculator can be reused inside another.

```python
from pynbodyext.filters import FamilyFilter, Sphere
from pynbodyext.properties import AngMomVec, KappaRot, ParamContain
from pynbodyext.transforms import AlignVec, ShiftPosTo, ShiftVelTo, WrapBox

# define half-mass radius for star particles within 30 kpc
re = ParamContain("r").filter(Sphere("30 kpc") & FamilyFilter("star"))

krot = KappaRot().filter(
    # we calculate the kappa_rot for star particles within 30 kpc
    Sphere("30 kpc") & FamilyFilter("star")
    ).transform(
        # the simulation is first wrapped into the box, 
        # then shifted to the center, 
        # then shift the velocities 
        # to mean velocity of star particles within 0.5*re,
        # and finally align the z-axis 
        # to the angular momentum vector of star particles within 2*re.
        WrapBox(
        ).then(
        ShiftPosTo("ssc")
        ).then(
        ShiftVelTo("com").filter(Sphere(0.5*re) & FamilyFilter("star"))
        ).then(
        AlignVec(
            AngMomVec().filter(Sphere(2 * re) & FamilyFilter("star"))
            )
        )
    )

```
Here ``0.5 * re`` and ``2 * re`` are calculator-valued inputs. They are resolved
automatically at runtime, so you do not need to manually order the computation.



---

### Using the Generalized Profile Builder

You can easily build radial profiles and extract sub-profiles using filters:

```python
from pynbodyext.filters import Sphere
from pynbodyext.profiles import RadialProfileBuilder

# Create a radial profile builder for 3D data, weighted by mass, with equal-number bins:
radial_pr = RadialProfileBuilder(ndim=3, weight="mass", bins_type="equaln")

# Generate the profile for your simulation:
pr = radial_pr(sim)

# Extract sub-profiles using filters:
subpr = pr.s  # or equivalently, pr[FamilyFilter("star")]
# 'subpr' has the same interface as 'pr'

# Access profile statistics:
subpr["z"]         # ProfileArray: mean z profile
subpr["z"]["abs"]  # ProfileArray: mean absolute z profile
subpr["z"]["p16"]  # ProfileArray: 16th percentile z profile

# Restrict particles to a sphere of radius 30 kpc:
subpr[Sphere("30 kpc")]  # Returns a new sub-profile
```

---

### Image post-processing and visualization

`pynbodyext.plot.image` turns the 2-D arrays produced by the calculator layer
(`BinND`, and later SPH renders or tessellations) into publication-ready figures.
Every operation is a free function over a plain 2-D array, and the same operations
are methods of `ImageData`, which carries the geometry, the units and the record
of what was done to it.

```python
from pynbodyext.plot import image

# 2-D binning: 200x200 pixels of projected z-velocity and mass
bins2d = (Bin1D("x", vmin=-50, vmax=50, nbins=200, alias="x")
        @ Bin1D("y", vmin=-50, vmax=50, nbins=200, alias="y"))(sim)

# `BinNDResult.imshow` itself goes through ImageData: extent, axis units and
# labels, and the choice between imshow and pcolormesh for uneven bins
bins2d.imshow("mass.sum", cmap="inferno", colorbar=True)

# Or hold the map and chain: geometry and metadata survive every step, and the
# calls are recorded in `.ops` (repr shows them)
density = image.ImageData.from_bins(bins2d, "mass.sum")
smoothed = density.smooth.gaussian(fwhm=1.0)        # fwhm in the units of the axes
observed = smoothed.psf.convolve(fwhm=3.0)          # what a telescope would see
observed.display.imshow(cmap="inferno", colorbar=True)

# ...and what the detector would do to it: seeded, maskable, recorded in .ops
detected = observed.noise.poisson(exposure=0.05, background=2.0, rng=1)
detected.display.imshow(cmap="inferno", colorbar=True)

# Deconvolution goes the other way (and amplifies noise: it is a visualisation tool)
restored = observed.psf.wiener(image.gaussian_psf(fwhm=3.0), balance=1e-6)

# A velocity map: adaptive bins of equal mass, green on zero velocity
velocity = image.ImageData.from_bins(bins2d, "vz.mean")
binned = velocity.adaptive.bin(bins2d["mass.sum"], target_nbins=200, min_signal=1e6)
binned.display.imshow(cmap="K_B_C_G_Y_R_W", symmetric=True, colorbar="bottom")
binned.image.smooth.box(size=3)       # the painted map is an ImageData too
# a colour bar can also be added afterwards, on the artist or on the map
binned.display.add_colorbar(loc="left", size="4%", tick_label_size=8)

# Contours are a display verb too: they follow the bins of the map they describe
velocity.display.draw(ax=ax, cmap="inferno")
velocity.display.contour(ax=ax, levels=[-200, -100, 0, 100, 200], colors="w", linewidths=0.6)

# Anything that takes a second map accepts a binned array directly: it carries its
# own orientation, so no `.T` (a raw ndarray is taken as image-oriented)
mass = image.as_image(bins2d["mass.sum"])   # (x, y) -> (row=y, column=x)

# Gas density and dark-matter density in one figure, crossfaded along a line
gas = image.ImageData.from_bins(bins2d, "mass.sum", units="1e10 Msol")
dm = image.ImageData.from_bins(bins2d, "dm_mass.sum", units="1e10 Msol")
gas.imshow_compose(
    dm,
    style1=image.MapStyle(cmap="inferno", stretch="log"),
    style2=image.MapStyle(cmap="cividis"),
    label1="gas", label2="dark matter",
)
```

Every capability is an accessor — `image.smooth.gaussian`, `image.psf.convolve`,
`image.noise.poisson`, `image.compose(other)`, `image.adaptive.bin`,
`image.display.imshow/contour/normalize/to_rgba` — and `ImageData` itself carries
only what it *is* (values, geometry, units, labels, `ops`, `with_data`,
`from_bins`). The underlying free functions (`gaussian_smooth(data, ...)`,
`compose_maps(...)`, …) remain available for plain arrays. A new family is a new module plus
`@ImageData.register_ops("tessellation")`, so extension never means editing base
classes. The velocity colour map (black → blue → cyan → **green on zero** → yellow
→ red → white, also known as `image.vel_cmap`) is registered with matplotlib, so
`cmap="K_B_C_G_Y_R_W"` works; the masks behind the stitching are exposed on their
own:

```python
mask1, mask2 = gas.compose.masks(line_angle=45, width=0.15)
blended = image.blend_images(rgb_gas, rgb_dm, mask1)
```

Adaptive binning needs the optional dependency `powerbin`
(`pip install pynbodyext[image]`); everything else only needs `numpy`, `scipy`
and `matplotlib`, which pynbody already brings.

---
