#pragma once

#include <pybind11/pybind11.h>

namespace image {

// Register the particle-to-cell scattering kernel on *m* (see scatter.cpp).
void register_scatter(pybind11::module_& m);

}  // namespace image
