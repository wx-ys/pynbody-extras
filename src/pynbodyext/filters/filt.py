"""
This module provides convenient access to the powerful particle filtering classes
from `pynbody.filt`. These filters allow for the selection of particles based
on their spatial location, family type, or other properties.

Filters can be combined using logical operators (`&` for AND, `|` for OR, `~` for NOT)
to create complex selection criteria.

Examples
--------
Select all star particles within a 10 kpc sphere:

>>> sphere_filter = selector.Sphere('10 kpc')
>>> star_filter = selector.FamilyFilter('star')
>>> selection = sim[sphere_filter & star_filter]

"""
# Simsnap[Filter] -> Subsnap


from pynbody.filt import (
    Annulus,
    BandPass,
    Cuboid,
    Disc,
    FamilyFilter,
    Filter,  # The base class for all filters
    HighPass,
    LowPass,
    SolarNeighborhood,
    Sphere,
)

__all__ = ["Filter", "FamilyFilter", "Sphere", "Disc", "Annulus", "Cuboid", "BandPass", "HighPass", "LowPass", "SolarNeighborhood"]
