#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_native, m) {
    m.doc() = "pynbodyext native (C++) extension";
    m.attr("__version__") = "0.0.1";
}
