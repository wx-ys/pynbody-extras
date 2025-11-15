"""
Coordinate and Viewpoint Transformations
========================================

This module provides classes that apply transformations to a simulation snapshot,
such as centering on a halo, aligning with a disk's angular momentum, or wrapping
particles into a periodic box.

These classes typically return a `pynbody.transformation.Transformation` object,
which can be used within a `with` statement for temporary transformations.

"""
from .base import TransformBase
from .wrap import WrapBox

__all__ = ["WrapBox","TransformBase"]
