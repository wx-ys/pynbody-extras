from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(".")

# Set PYNBODYEXT_OPENMP=0 to force a single-threaded build.
OPENMP_ENV = "PYNBODYEXT_OPENMP"
# Set LIBOMP_PREFIX to a libomp install (with include/omp.h and lib/libomp.*)
# to use it instead of searching Homebrew.
LIBOMP_PREFIX_ENV = "LIBOMP_PREFIX"

_OPENMP_PROBE = """
#include <omp.h>
int main() {
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return (n > 0 && omp_get_max_threads() > 0) ? 0 : 1;
}
"""


# Without OpenMP the `#pragma omp` lines are ignored by design; do not warn about them.
_SERIAL_ARGS = ["-Wno-unknown-pragmas"]


def _libomp_prefixes() -> list[Path]:
    """Candidate libomp install prefixes on macOS (Homebrew's libomp is keg-only)."""
    candidates: list[Path] = []
    if os.environ.get(LIBOMP_PREFIX_ENV):
        candidates.append(Path(os.environ[LIBOMP_PREFIX_ENV]))
    brew = shutil.which("brew")
    if brew:
        try:
            out = subprocess.run([brew, "--prefix", "libomp"], capture_output=True, text=True, timeout=30, check=True)
            candidates.append(Path(out.stdout.strip()))
        except (OSError, subprocess.SubprocessError):
            pass
    candidates += [Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")]
    seen: list[Path] = []
    for prefix in candidates:
        if prefix not in seen and (prefix / "include" / "omp.h").is_file():
            seen.append(prefix)
    return seen


def _openmp_candidates() -> list[tuple[str, list[str], list[str]]]:
    """(label, compile args, link args) to try, most specific first."""
    candidates: list[tuple[str, list[str], list[str]]] = []
    if sys.platform == "darwin":
        # Apple clang rejects a bare -fopenmp; it needs -Xpreprocessor and libomp.
        for prefix in _libomp_prefixes():
            candidates.append(
                (
                    f"libomp at {prefix}",
                    ["-Xpreprocessor", "-fopenmp", f"-I{prefix / 'include'}"],
                    [f"-L{prefix / 'lib'}", "-lomp", f"-Wl,-rpath,{prefix / 'lib'}"],
                )
            )
    candidates.append(("-fopenmp", ["-fopenmp"], ["-fopenmp"]))
    return candidates


def _compiles_and_links(compiler, compile_args: list[str], link_args: list[str]) -> bool:
    """True if the build's own compiler can compile and link a small OpenMP program."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "openmp_probe.cpp"
        source.write_text(_OPENMP_PROBE)
        try:
            objects = compiler.compile([str(source)], output_dir=tmp, extra_postargs=compile_args)
            compiler.link_executable(objects, "openmp_probe", output_dir=tmp, extra_postargs=link_args)
        except Exception:
            return False
    return True


def _openmp_flags(compiler) -> tuple[list[str], list[str]]:
    """Compile and link args for OpenMP, or the single-threaded args if it is unavailable."""
    if os.environ.get(OPENMP_ENV, "").strip().lower() in {"0", "false", "no", "off"}:
        print(f"pynbodyext._native: {OPENMP_ENV}=0, building without OpenMP (single-threaded).", file=sys.stderr)
        return _SERIAL_ARGS, []
    for label, compile_args, link_args in _openmp_candidates():
        if _compiles_and_links(compiler, compile_args, link_args):
            print(f"pynbodyext._native: building with OpenMP ({label}).", file=sys.stderr)
            return compile_args, link_args
    hint = " On macOS, install it with `brew install libomp`." if sys.platform == "darwin" else ""
    print(
        "pynbodyext._native: WARNING: OpenMP not found (any compiler error just above comes from "
        f"this probe); building single-threaded, so gravity and SPH scatter use one core.{hint}",
        file=sys.stderr,
    )
    return _SERIAL_ARGS, []


class BuildExt(build_ext):
    """pybind11's build_ext, plus OpenMP detection for the compiler in use."""

    def build_extensions(self) -> None:
        compile_args, link_args = _openmp_flags(self.compiler)
        for ext in self.extensions:
            ext.extra_compile_args = [*ext.extra_compile_args, *compile_args]
            ext.extra_link_args = [*ext.extra_link_args, *link_args]
        super().build_extensions()


# Optional C++ extension providing high-performance implementations
# for tree and gravity operations. When a C++ compiler is not
# available, installation will fall back to the pure-Python package
# without this extension (optional=True, matching the old Rust behavior).
# OpenMP is added by BuildExt when the compiler supports it (Linux: -fopenmp;
# macOS: Homebrew libomp); otherwise the extension is built single-threaded.
ext_modules = [
    Pybind11Extension(
        "pynbodyext._native",
        sources=[
            str(ROOT / "cpp" / "bindings" / "module.cpp"),
            *[str(p) for p in (ROOT / "cpp" / "gravity").glob("*.cpp")],
            *[str(p) for p in (ROOT / "cpp" / "gravity" / "multipole").glob("*.cpp")],
            *[str(p) for p in (ROOT / "cpp" / "image").glob("*.cpp")],
        ],
        include_dirs=[str(ROOT / "cpp")],
        cxx_std=17,
        optional=True,
        extra_compile_args=["-O2"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
