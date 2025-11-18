# pynbodyext: Extensions and Utilities for pynbody

`pynbodyext` provides a set of extensions and utilities based on [pynbody](https://github.com/pynbody/pynbody) library.

---

## Examples

### Defining and Using Parameterized Properties

You can define parameterized properties and use them directly on your simulation objects:

```python
import numpy as np
from pynbodyext.properties import ParameterContain, ParamSum
from pynbodyext.filters import FamilyFilter

# 're' and 'StarMass' are calculators.
# Use re(sim) to get the corresponding value for a simulation 'sim'.

re = ParameterContain().with_filter(FamilyFilter("star"))
StarMass = ParamSum("mass").with_filter(FamilyFilter("star"))

# Combine calculators, e.g., to compute density:
StarDensity = StarMass / (4 * np.pi * re * re)

# Retrieve the value defined by StarDensity for a specific simulation:
StarDensity(sim)
```

---

### Using the Generalized Profile Builder

You can easily build radial profiles and extract sub-profiles using filters:

```python
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