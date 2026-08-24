#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gravity/octree.hpp"
#include "gravity/direct.hpp"
#include "gravity/traversal.hpp"
#include "gravity/kernel.hpp"
#include "gravity/vec3.hpp"

namespace py = pybind11;
using namespace gravity;

// forcecast + c_style guarantees contiguous float64: copies only when needed.
using PyArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::vector<Vec3> extract_vec3(const PyArr& arr, const char* name) {
    if (arr.ndim() != 2 || arr.shape(1) != 3)
        throw py::value_error(std::string(name) + " must be (N,3) float64 array");
    const double* d = arr.data();
    size_t n = (size_t)arr.shape(0);
    std::vector<Vec3> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
        out.push_back(Vec3(d[3 * i], d[3 * i + 1], d[3 * i + 2]));
    return out;
}

static std::optional<KernelKind> parse_kernel_opt(std::optional<uint8_t> kernel) {
    if (!kernel.has_value())
        return std::nullopt;
    switch (*kernel) {
        case 0: return KernelKind::Plummer;
        case 1: return KernelKind::CubicSplineW2;
        default: throw py::value_error("kernel must be 0 (Plummer) or 1 (CubicSplineW2)");
    }
}

static std::optional<std::vector<double>> copy_opt_arr(const std::optional<PyArr>& arr,
                                                       const char* name, size_t n) {
    if (!arr.has_value())
        return std::nullopt;
    if ((size_t)arr->shape(0) != n)
        throw py::value_error(std::string(name) + " must be length N");
    const double* d = arr->data();
    return std::vector<double>(d, d + n);
}

// Builds an owning (N,3) float64 array from a vector of Vec3.
static py::array_t<double> make_n3(const std::vector<Vec3>& v) {
    py::array_t<double> a({(py::ssize_t)v.size(), (py::ssize_t)3});
    double* d = a.mutable_data();
    for (size_t i = 0; i < v.size(); ++i) {
        d[3 * i] = v[i].x;
        d[3 * i + 1] = v[i].y;
        d[3 * i + 2] = v[i].z;
    }
    return a;
}

struct PyOctree {
    Octree inner;
    bool payload_built = false;

    PyOctree(PyArr positions, std::optional<PyArr> masses, size_t leaf_capacity,
             uint8_t multipole_order, std::optional<PyArr> softenings,
             std::optional<uint8_t> kernel)
        : inner(Octree::from_owned(extract_vec3(positions, "positions"),
                                   copy_opt_arr(masses, "masses", (size_t)positions.shape(0)),
                                   copy_opt_arr(softenings, "softenings", (size_t)positions.shape(0)),
                                   leaf_capacity, multipole_order,
                                   parse_kernel_opt(kernel).value_or(KernelKind::Plummer))) {
        if (softenings.has_value() && !kernel.has_value())
            throw py::value_error(
                "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
        inner.build_mass_payload();
        payload_built = true;
    }

    void build_mass(std::optional<PyArr> masses) {
        if (masses.has_value())
            inner.set_masses(copy_opt_arr(masses, "masses", inner.positions.size()));
        inner.build_mass_payload();
        payload_built = true;
    }

    void set_softenings(std::optional<PyArr> softenings) {
        inner.set_softenings(copy_opt_arr(softenings, "softenings", inner.positions.size()));
    }

    void set_kernel(std::optional<uint8_t> kernel) {
        inner.set_kernel(parse_kernel_opt(kernel).value_or(KernelKind::Plummer));
    }

    void require_payload() const {
        if (!payload_built)
            throw py::value_error("mass payload not built; call build_mass() before compute_*");
    }

    py::array_t<double> compute_accelerations(double theta, size_t threads) {
        require_payload();
        size_t n = inner.positions.size();
        std::vector<Vec3> out(n);
        {
            py::gil_scoped_release release;
            if (threads > 0) omp_set_num_threads((int)threads);
            gravity::compute_accelerations(inner, theta, out);
        }
        return make_n3(out);
    }

    py::array_t<double> compute_potentials(double theta, size_t threads) {
        require_payload();
        size_t n = inner.positions.size();
        std::vector<double> out(n);
        {
            py::gil_scoped_release release;
            if (threads > 0) omp_set_num_threads((int)threads);
            gravity::compute_potentials(inner, theta, out);
        }
        return py::array_t<double>({(py::ssize_t)n}, out.data());
    }

    py::array_t<double> accelerations_at_points(PyArr points, double theta, size_t threads) {
        require_payload();
        auto pts = extract_vec3(points, "points");
        std::vector<Vec3> out(pts.size());
        {
            py::gil_scoped_release release;
            if (threads > 0) omp_set_num_threads((int)threads);
            gravity::accelerations_at_points(inner, pts, theta, out);
        }
        return make_n3(out);
    }

    py::array_t<double> potentials_at_points(PyArr points, double theta, size_t threads) {
        require_payload();
        auto pts = extract_vec3(points, "points");
        std::vector<double> out(pts.size());
        {
            py::gil_scoped_release release;
            if (threads > 0) omp_set_num_threads((int)threads);
            gravity::potentials_at_points(inner, pts, theta, out);
        }
        return py::array_t<double>({(py::ssize_t)pts.size()}, out.data());
    }
};

// ---- Direct-sum O(N^2) functions (mirror Rust gravity.rs signatures) ----

py::array_t<double> direct_accelerations_py(PyArr positions, std::optional<PyArr> masses,
                                            size_t threads, std::optional<PyArr> softenings,
                                            std::optional<uint8_t> kernel) {
    auto pos = extract_vec3(positions, "positions");
    size_t n = pos.size();
    auto mv = copy_opt_arr(masses, "masses", n);
    auto sv = copy_opt_arr(softenings, "softenings", n);
    auto kk = parse_kernel_opt(kernel);
    if (sv.has_value() && !kk.has_value())
        throw py::value_error(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
    std::vector<Vec3> acc;
    {
        py::gil_scoped_release release;
        if (threads > 0) omp_set_num_threads((int)threads);
        acc = kk.has_value()
                  ? direct_accelerations_kernel(pos, mv ? mv->data() : nullptr,
                                                sv ? sv->data() : nullptr, *kk)
                  : direct_accelerations(pos, mv ? mv->data() : nullptr);
    }
    return make_n3(acc);
}

py::array_t<double> direct_potentials_py(PyArr positions, std::optional<PyArr> masses,
                                         size_t threads, std::optional<PyArr> softenings,
                                         std::optional<uint8_t> kernel) {
    auto pos = extract_vec3(positions, "positions");
    size_t n = pos.size();
    auto mv = copy_opt_arr(masses, "masses", n);
    auto sv = copy_opt_arr(softenings, "softenings", n);
    auto kk = parse_kernel_opt(kernel);
    if (sv.has_value() && !kk.has_value())
        throw py::value_error(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
    std::vector<double> pot;
    {
        py::gil_scoped_release release;
        if (threads > 0) omp_set_num_threads((int)threads);
        pot = kk.has_value()
                  ? direct_potentials_kernel(pos, mv ? mv->data() : nullptr,
                                             sv ? sv->data() : nullptr, *kk)
                  : direct_potentials(pos, mv ? mv->data() : nullptr);
    }
    return py::array_t<double>({(py::ssize_t)pot.size()}, pot.data());
}

py::array_t<double> direct_accelerations_at_points_py(PyArr positions, PyArr targets,
                                                      std::optional<PyArr> masses, size_t threads,
                                                      std::optional<PyArr> softenings,
                                                      std::optional<uint8_t> kernel) {
    auto pos = extract_vec3(positions, "positions");
    size_t n = pos.size();
    auto mv = copy_opt_arr(masses, "masses", n);
    auto sv = copy_opt_arr(softenings, "softenings", n);
    auto tgt = extract_vec3(targets, "targets");
    auto kk = parse_kernel_opt(kernel);
    if (sv.has_value() && !kk.has_value())
        throw py::value_error(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
    std::vector<Vec3> acc;
    {
        py::gil_scoped_release release;
        if (threads > 0) omp_set_num_threads((int)threads);
        acc = kk.has_value()
                  ? direct_accelerations_kernel_at_points(pos, mv ? mv->data() : nullptr,
                                                          sv ? sv->data() : nullptr, tgt, *kk)
                  : direct_accelerations_at_points(pos, mv ? mv->data() : nullptr, tgt);
    }
    return make_n3(acc);
}

py::array_t<double> direct_potentials_at_points_py(PyArr positions, PyArr targets,
                                                   std::optional<PyArr> masses, size_t threads,
                                                   std::optional<PyArr> softenings,
                                                   std::optional<uint8_t> kernel) {
    auto pos = extract_vec3(positions, "positions");
    size_t n = pos.size();
    auto mv = copy_opt_arr(masses, "masses", n);
    auto sv = copy_opt_arr(softenings, "softenings", n);
    auto tgt = extract_vec3(targets, "targets");
    auto kk = parse_kernel_opt(kernel);
    if (sv.has_value() && !kk.has_value())
        throw py::value_error(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
    std::vector<double> pot;
    {
        py::gil_scoped_release release;
        if (threads > 0) omp_set_num_threads((int)threads);
        pot = kk.has_value()
                  ? direct_potentials_kernel_at_points(pos, mv ? mv->data() : nullptr,
                                                       sv ? sv->data() : nullptr, tgt, *kk)
                  : direct_potentials_at_points(pos, mv ? mv->data() : nullptr, tgt);
    }
    return py::array_t<double>({(py::ssize_t)pot.size()}, pot.data());
}

PYBIND11_MODULE(_native, m) {
    m.doc() = "pynbodyext native (C++) gravity extension";
    m.attr("__version__") = "0.0.1";

    py::class_<PyOctree>(m, "Octree")
        .def(py::init<PyArr, std::optional<PyArr>, size_t, uint8_t, std::optional<PyArr>,
                      std::optional<uint8_t>>(),
             py::arg("positions"), py::arg("masses") = py::none(),
             py::arg("leaf_capacity") = 32, py::arg("multipole_order") = 0,
             py::arg("softenings") = py::none(), py::arg("kernel") = py::none())
        .def("build_mass", &PyOctree::build_mass, py::arg("masses") = py::none())
        .def("set_softenings", &PyOctree::set_softenings, py::arg("softenings") = py::none())
        .def("set_kernel", &PyOctree::set_kernel, py::arg("kernel") = py::none())
        .def("compute_accelerations", &PyOctree::compute_accelerations, py::arg("theta"),
             py::arg("threads") = 0)
        .def("compute_potentials", &PyOctree::compute_potentials, py::arg("theta"),
             py::arg("threads") = 0)
        .def("accelerations_at_points", &PyOctree::accelerations_at_points, py::arg("points"),
             py::arg("theta"), py::arg("threads") = 0)
        .def("potentials_at_points", &PyOctree::potentials_at_points, py::arg("points"),
             py::arg("theta"), py::arg("threads") = 0);

    m.def("direct_accelerations_py", &direct_accelerations_py, py::arg("positions"),
          py::arg("masses") = py::none(), py::arg("threads") = 0, py::arg("softenings") = py::none(),
          py::arg("kernel") = py::none());
    m.def("direct_potentials_py", &direct_potentials_py, py::arg("positions"),
          py::arg("masses") = py::none(), py::arg("threads") = 0, py::arg("softenings") = py::none(),
          py::arg("kernel") = py::none());
    m.def("direct_accelerations_at_points_py", &direct_accelerations_at_points_py,
          py::arg("positions"), py::arg("targets"), py::arg("masses") = py::none(),
          py::arg("threads") = 0, py::arg("softenings") = py::none(), py::arg("kernel") = py::none());
    m.def("direct_potentials_at_points_py", &direct_potentials_at_points_py, py::arg("positions"),
          py::arg("targets"), py::arg("masses") = py::none(), py::arg("threads") = 0,
          py::arg("softenings") = py::none(), py::arg("kernel") = py::none());
}
