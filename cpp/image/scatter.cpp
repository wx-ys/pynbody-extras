// Particle-to-cell scattering: the update loop of pynbody's `_render.render_image`,
// except that instead of accumulating a kernel-weighted scalar into the cell it
// records the pair — which cell, how far, which particle found it.  The reduction
// (a weighted quantile, say) then happens in Python.
//
// The Python fallback in `bins/sph_render.py` does exactly this in NumPy; this is
// the same arithmetic in a tight C++ loop, which is where the cost is.

#include "image/scatter.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace image {

namespace py = pybind11;

// forcecast + c_style guarantees contiguous float64: copies only when needed.
using PyArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

struct Ranges {
    long low[3];
    long high[3];
    std::size_t size;
};

// The cell range one particle reaches, clipped to the grid (and, in the first
// axis, to the slab being built).  Mirrors the Python `_scatter`.
Ranges ranges_of(const double* position, const double* support, std::size_t row, std::size_t dims,
                 const std::vector<double>& origins, const std::vector<double>& widths,
                 const std::vector<long>& counts, long low_row, long high_row) {
    Ranges r;
    r.size = 1;
    for (std::size_t axis = 0; axis < dims; ++axis) {
        const double coordinate = position[row * dims + axis];
        const double reach = support[row];
        double low = std::floor((coordinate - reach - origins[axis]) / widths[axis]);
        double high = std::ceil((coordinate + reach - origins[axis]) / widths[axis]);
        const long floor_bound = axis == 0 ? low_row : 0;
        const long ceil_bound = axis == 0 ? high_row : counts[axis];
        r.low[axis] = std::max((long)low, floor_bound);
        r.high[axis] = std::min((long)high, ceil_bound);
        if (r.high[axis] <= r.low[axis]) {
            r.size = 0;
            return r;
        }
        r.size *= (std::size_t)(r.high[axis] - r.low[axis]);
    }
    return r;
}

}  // namespace

// (cell, distance_squared, row) for every particle-cell pair inside one slab.
// `cell` is local to the slab (row 0 is `low_row`); the entries come out grouped
// by row, which lets a threaded fill write disjoint slices without a lock.
py::tuple scatter_pairs(PyArr position, PyArr smoothing, PyArr support,
                        const std::vector<double>& origins, const std::vector<double>& widths,
                        const std::vector<long>& counts, const std::vector<long>& strides, long low_row,
                        long high_row, int threads) {
    if (position.ndim() != 2)
        throw py::value_error("position must be (rows, dims)");
    const std::size_t rows = (std::size_t)position.shape(0);
    const std::size_t dims = (std::size_t)position.shape(1);
    if (dims < 2 || dims > 3) throw py::value_error("scatter_pairs handles 2 or 3 dimensions");
    if ((std::size_t)smoothing.shape(0) != rows || (std::size_t)support.shape(0) != rows)
        throw py::value_error("smoothing and support must have one entry per row");
    if (origins.size() != dims || widths.size() != dims || counts.size() != dims || strides.size() != dims)
        throw py::value_error("origins, widths, counts and strides must have one entry per dimension");

    const double* pos = position.data();
    const double* reach = support.data();

    std::vector<Ranges> ranges(rows);
    std::vector<std::size_t> offsets(rows + 1, 0);
    for (std::size_t row = 0; row < rows; ++row) {
        ranges[row] = ranges_of(pos, reach, row, dims, origins, widths, counts, low_row, high_row);
        offsets[row + 1] = offsets[row] + ranges[row].size;
    }
    const std::size_t total = offsets[rows];

    py::array_t<std::int64_t> cell(total);
    py::array_t<double> distance(total);
    py::array_t<std::int64_t> which_row(total);
    std::int64_t* cell_data = cell.mutable_data();
    double* distance_data = distance.mutable_data();
    std::int64_t* row_data = which_row.mutable_data();

#pragma omp parallel for num_threads(threads > 0 ? threads : omp_get_max_threads()) schedule(static)
    for (std::int64_t row = 0; row < (std::int64_t)rows; ++row) {
        const Ranges& r = ranges[row];
        if (r.size == 0) continue;
        std::size_t at = offsets[row];
        long index[3] = {0, 0, 0};
        for (std::size_t entry = 0; entry < r.size; ++entry) {
            std::int64_t flat = 0;
            double squared = 0.0;
            for (std::size_t axis = 0; axis < dims; ++axis) {
                const long j = r.low[axis] + index[axis];
                const double centre = origins[axis] + ((double)j + 0.5) * widths[axis];
                const double delta = centre - pos[row * (std::int64_t)dims + axis];
                squared += delta * delta;
                flat += j * strides[axis];
            }
            cell_data[at] = flat - low_row * strides[0];
            distance_data[at] = squared;
            row_data[at] = row;
            ++at;
            for (std::size_t axis = dims; axis-- > 0;) {  // last axis varies fastest
                if (++index[axis] < r.high[axis] - r.low[axis]) break;
                index[axis] = 0;
            }
        }
    }
    return py::make_tuple(cell, distance, which_row);
}

void register_scatter(py::module_& m) {
    m.def("scatter_pairs", &scatter_pairs, py::arg("position"), py::arg("smoothing"), py::arg("support"),
          py::arg("origins"), py::arg("widths"), py::arg("counts"), py::arg("strides"), py::arg("low_row"),
          py::arg("high_row"), py::arg("threads") = 0,
          "Every (cell, distance^2, row) particle-cell pair inside one slab of rows.");
}

}  // namespace image
