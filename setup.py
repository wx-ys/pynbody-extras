from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(".")


# Optional C++ extension providing high-performance implementations
# for tree and gravity operations. When a C++ compiler is not
# available, installation will fall back to the pure-Python package
# without this extension (optional=True, matching the old Rust behavior).
ext_modules = [
    Pybind11Extension(
        "pynbodyext._native",
        sources=[
            str(ROOT / "cpp" / "bindings" / "module.cpp"),
            *[str(p) for p in (ROOT / "cpp" / "gravity").glob("*.cpp")],
            *[str(p) for p in (ROOT / "cpp" / "gravity" / "multipole").glob("*.cpp")],
        ],
        include_dirs=[str(ROOT / "cpp")],
        cxx_std=17,
        optional=True,
        extra_compile_args=["-O2", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
