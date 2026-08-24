# Rust → C++ Migration Design

Date: 2026-08-12
Status: Approved for planning

## Goal

Replace the Rust high-performance gravity implementation and its pyo3/numpy Python
bindings with a C++ implementation bound via pybind11, and remove **all** Rust
toolchain artifacts from the repository (source, `Cargo.toml`/`Cargo.lock`,
`target/`, setuptools-rust/maturin build wiring).

The Python-facing API stays functionally identical so existing consumers keep
working, with one deliberate rename of the internal extension module.

## Current state

- **Rust core** — `crates/gravity/` (~3,700 LOC): `octree`, `traversal`,
  `direct`, `kernel`, `multipole/{moment,derivatives,eval,local}`, plus unused
  `solver/`, `boundary`, `types`. All uncommitted WIP.
- **Rust bindings** — `crates/pynbodyext-rust/` (~730 LOC): pyo3 + numpy module
  exposing the `Octree` pyclass and four `direct_*_py` functions.
- **Empty stubs** — `crates/sph/` (14 LOC), `crates/src/main.rs` (3 LOC).
- **Python** — `pynbodyext.gravity` (`base.py`, `pyn_gravity.py`, `__init__.py`)
  imports `pynbodyext._rust`; `pynbodyext/util/deps.py` gates gravity availability
  on `module_available("pynbodyext._rust")`.
- **Build** — `setup.py` + `setuptools_rust.RustExtension(optional=True)`;
  `pyproject.toml` `[build-system]` includes `setuptools-rust`; `[tool.maturin]`
  section present. `uv.lock` used by uv.
- **Threading** — rayon: `threads=0` → global pool, `threads>0` → dedicated
  N-thread pool; 512-particle threshold in direct, 1024 in tree traversal.
- **Tests (Rust)** — `gravity_tests.rs` (tree-vs-direct equivalence, multipole
  convergence), `single_node.rs` (far-field multipole convergence), 
  `translate_multipole.rs` (M2M self-consistency). No Python gravity tests exist.

## Decisions (confirmed with user)

| Decision | Choice |
|---|---|
| Target language | **C++** (C++17) |
| Python binding | **pybind11** |
| Parallelism | **OpenMP** |
| Build | **setuptools + Pybind11Extension** (no CMake) |
| Directory | `cpp/` |
| Core/bindings split | `cpp/gravity/` (pure C++, no Python includes) + `cpp/bindings/module.cpp` (only file including pybind11) |
| Multipole dispatch | `template<int Order>` monomorphization (orders 0/2/3/4/5), explicit instantiations |
| Vector type | custom `Vec3 { double x,y,z; }` |
| Dropped modules | `multipole/local.rs` (unused FMM stub), `solver/`, `boundary`, `types` (all unused by hot path) |
| Module rename | `pynbodyext._rust` → `pynbodyext._native`; flag `GRAVITY_RUST_AVAILABLE` → `GRAVITY_NATIVE_AVAILABLE` |
| Scope | Full port; remove all Rust after port validated |

## Target layout

```
cpp/
  gravity/                    # core gravity library (pure C++, no pybind11)
    vec3.hpp                  # Vec3 { double x,y,z; } + arithmetic operators
    kernel.{hpp,cpp}          # KernelKind, kernel_potential_per_unit_mass, kernel_accel_factor, w2, w2_prime
    direct.{hpp,cpp}          # template DirectAccumulator, direct_self_impl, direct_at_points_impl, public direct_* fns
    octree.{hpp,cpp}          # Node, NodeBh, Octree (positions/masses/softenings/nodes, link caches, payloads)
    traversal.{hpp,cpp}       # linked-list treewalk (first_subnode/next_branch), leaf sums, template<int Order>
    multipole/
      moment.{hpp,cpp}        # MultipoleMoment (56 doubles), Moment0/2/3/4/5, MultipoleMoments variant, from_points, translate_multipole
      derivatives.{hpp,cpp}   # PotentialDerivatives1..5, dt_k recurrence
      eval.{hpp,cpp}          # template<int Order> evaluator struct + runtime-dispatch helpers for tests
    tests/
      test_core.cpp           # translate_multipole + single-node far-field tests (assert-based)
  bindings/
    module.cpp                # pybind11: pynbodyext._native (Octree class + 4 direct functions)
```

## Component mapping (Rust → C++)

| Rust | C++ | Notes |
|---|---|---|
| `kernel.rs` | `cpp/gravity/kernel` | enum + free functions; formulas verbatim |
| `direct.rs` trait `DirectAccumulator` | `template <typename A> A::init()` via struct/class templates | `AccelAccumulator`, `PotAccumulator`; generic `direct_self_impl<A>` / `direct_at_points_impl<A>`; 512 threshold → `#pragma omp parallel for if(...)` |
| `octree.rs` | `cpp/gravity/octree` | `Vec`→`std::vector`, `[f64;3]`→`Vec3`; `Option<Vec<T>>`→`std::optional<std::vector<T>>`; timing helpers (`GRAVITY_TIMING`) preserved |
| `traversal.rs` | `cpp/gravity/traversal` | exact `while (idx != NONE)` walk over `first_subnode`/`next_branch`; `template<int Order>` with instantiations 0/2/3/4/5; 1024 threshold → OpenMP |
| `multipole/moment.rs` | `multipole/moment` | `MultipoleMoments` enum → `std::variant<Moment0,Moment2,Moment3,Moment4,Moment5>`; `FACT` table `[1,1,2,6,24,120]` |
| `multipole/derivatives.rs` | `multipole/derivatives` | `dt_{k+1}=-(2k-1)dt_k/r` recurrence verbatim |
| `multipole/eval.rs` | `multipole/eval` | `template<int Order> struct MultipoleOrderEvaluator`; keep runtime-dispatch `gravity_potential_multipole`/`gravity_accel_multipole` for C++ tests |
| `multipole/local.rs` | — | **dropped** |
| `solver/mod.rs`, `solver/bh.rs` | — | **dropped** (unused) |
| `boundary.rs`, `types.rs` | — | **dropped** (unused) |
| `pynbodyext-rust/src/*.rs` | `cpp/bindings/module.cpp` | pybind11 `py::class_<Octree>` + `py::array_t<double>` + `py::gil_scoped_release` |

## Numerics preservation (bit-for-bit where feasible)

- `R2_TINY = f64::MIN_POSITIVE`, `MIN_SOFTENING = 0.0`.
- `FACT = {1,1,2,6,24,120}`.
- `multipole_min_separation_factor`: Plummer `2.8`, CubicSplineW2 `1.0`.
- Parallelism thresholds: 512 (direct), 1024 (tree).
- All kernel polynomials (`w2`, `w2_prime`), derivative recurrences, moment
  formulas and coefficients (`-1.0` in potential/accel assembly) unchanged.
- Tree-vs-direct equivalence tolerances (< 1e-10) carried into Python tests.

## Python API (unchanged surface, renamed module)

- Module: `pynbodyext._native` (was `_rust`).
- Rename surfaces (all committed files):
  - `pynbodyext/gravity/base.py` — import (line 51), module docstring (line 4),
    ImportError message (line 60: "Rust extension" → "C++ extension").
  - `pynbodyext/util/deps.py` — `GRAVITY_RUST_AVAILABLE` → `GRAVITY_NATIVE_AVAILABLE`
    (definition + `__all__`).
  - `pynbodyext/gravity/__init__.py` — flag import/usage + `__all__` entry.
- `benchmarks/bench_gravity.py` imports `pynbodyext.gravity.Gravity` (public API)
  only — no change needed. `asv.conf.json` / `scripts/asv_run.sh` unaffected.
- `Octree` pybind11 class, same constructor:
  `(positions, masses=None, leaf_capacity=32, multipole_order=0, softenings=None, kernel=None)`
  and methods: `build_mass`, `set_softenings`, `set_kernel`,
  `compute_accelerations(theta, threads=0)`, `compute_potentials(theta, threads=0)`,
  `accelerations_at_points(points, theta, threads=0)`, `potentials_at_points(points, theta, threads=0)`.
- Functions: `direct_accelerations_py`, `direct_potentials_py`,
  `direct_accelerations_at_points_py`, `direct_potentials_at_points_py` — same
  signatures `(positions, targets?, masses=None, threads=0, softenings=None, kernel=None)`.
- Error semantics preserved (`py::value_error`): length mismatches,
  "softenings require an explicit kernel", "mass payload not built", bad kernel value.
- NumPy interop: fast path for contiguous C-order arrays; copy path for
  non-contiguous (mirrors `extract_vec3_from_pyarray2`).

## Build & packaging

- `setup.py`: replace `RustExtension` with `pybind11.setup_helpers.Pybind11Extension`:
  - name `pynbodyext._native`, sources = `cpp/bindings/module.cpp` +
    `cpp/gravity/**/*.cpp`, `cxx_std=17`, `include_pybind11()`.
  - OpenMP: `-fopenmp` (GCC/Clang) in `extra_compile_args`/`extra_link_args`,
    `/openmp` (MSVC).
  - keep `optional=True` (toolchain-less installs fall back to pure Python,
    matching today's behavior).
- `pyproject.toml`:
  - `[build-system] requires` → `["setuptools>=61", "pybind11>=2.12", "wheel"]`.
  - remove `[tool.maturin]` section.
  - add `pybind11` to `dev` extras.
- `uv.lock` regenerated via `uv lock` / `uv sync`.
- `pynbodyext/util/deps.py`: `GRAVITY_NATIVE_AVAILABLE = module_available("pynbodyext._native")`.
- `pynbodyext/gravity/__init__.py`: use renamed flag; update warning text
  ("C++ extension not available").

## Tests & validation

- **`tests/test_gravity.py`** (new, replaces Rust `gravity_tests.rs`):
  - `tree_accelerations`/`tree_potentials` with `theta=0.0` vs `direct_*` → per-component < 1e-10.
  - at-points queries vs `direct_*_at_points`.
  - multipole-order convergence (error non-increasing with order; order-5
    significantly better than order-0), for both accel and potential.
  - softened kernels (Plummer, Spline) vs NumPy reference.
- **`cpp/gravity/tests/test_core.cpp`** (replaces `single_node.rs` +
  `translate_multipole.rs`): assert-based; built via `make cpp-test` target or
  a small build script; validates M2M translation and far-field order-0–5
  multipole convergence at the C++ level.
- Benchmarks: port criterion benches to Python/ASV where meaningful, or drop.

## Removal checklist

- Delete `crates/` (gravity, pynbodyext-rust, sph, src, target).
- Delete `Cargo.toml`, `Cargo.lock`.
- Purge `setuptools-rust`/`maturin` from setup.py, pyproject.toml.
- `.pre-commit-config.yaml` — remove the six `cargo fmt/clippy/test` hooks for
  `crates/gravity` and `crates/pynbodyext-rust` (lines 20–61).
- Update `.gitignore` (drop cargo/`target` entries if present), `asv.conf.json`,
  `scripts/`, `docs/`, `benchmarks/` references to `_rust`/cargo/maturin.
- Rename `_rust` → `_native` and `GRAVITY_RUST_AVAILABLE` →
  `GRAVITY_NATIVE_AVAILABLE` across Python sources.

## Out of scope

- `multipole/local.rs` FMM machinery (unused stub) — not ported.
- New physics features (periodic boundaries, PM/FMM solvers).
- Any change to the pure-Python `pynbodyext` package outside the gravity module.

## Risks

- **Numerics drift**: mitigated by porting the equivalence tests (tree vs direct,
  < 1e-10) to Python and keeping all constants/formulas verbatim.
- **OpenMP portability**: `threads>0` dedicated-pool semantics approximate rayon
  exactly per-call (num_threads); `threads=0` → default. Behavior preserved.
- **pybind11 numpy fast path**: non-contiguous arrays copied, same as today.
