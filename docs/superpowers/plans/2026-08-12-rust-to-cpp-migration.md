# Rust → C++ Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Rust gravity core (`crates/gravity`) and its pyo3 bindings (`crates/pynbodyext-rust`) to C++17 bound with pybind11, expose it as `pynbodyext._native`, and delete all Rust toolchain artifacts.

**Architecture:** A pure-C++ core library under `cpp/gravity/` (no Python includes) plus a single pybind11 binding file `cpp/bindings/module.cpp`. The core is translated 1:1 from the Rust modules (`kernel`, `direct`, `octree`, `traversal`, `multipole/{moment,derivatives,eval}`), using C++ templates for the Rust `MultipoleOrder` trait monomorphization, `std::optional`/`std::variant` for `Option`/enum-with-data, `std::vector` for `Vec`, a custom `Vec3` for `[f64; 3]`, and OpenMP for rayon.

**Tech Stack:** C++17, pybind11 (header-only), OpenMP, setuptools (`Pybind11Extension`), NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-12-rust-to-cpp-migration-design.md`

## Global Constraints

- **Language:** C++17. Compile all `.cpp` with `-std=c++17 -O2 -fopenmp` (GCC/Clang) or `/std:c++17 /O2 /openmp` (MSVC).
- **Directory root:** `cpp/`. Core in `cpp/gravity/`, bindings in `cpp/bindings/`.
- **Module name:** `pynbodyext._native` (renamed from `_rust`). Flag `GRAVITY_RUST_AVAILABLE` → `GRAVITY_NATIVE_AVAILABLE`.
- **Dropped Rust modules** (do NOT port): `multipole/local.rs` (FMM stub), `solver/`, `boundary.rs`, `types.rs`.
- **Numerics must match** the Rust source (which remains in `crates/` until Task 14). Constants: `R2_TINY = std::numeric_limits<double>::min()` (== Rust `f64::MIN_POSITIVE`), `MIN_SOFTENING = 0.0`, `FACT = {1.0,1.0,2.0,6.0,24.0,120.0}`, `NO_INDEX = std::numeric_limits<size_t>::max()` (== `usize::MAX`), parallel thresholds **512** (direct) and **1024** (tree), `multipole_min_separation_factor`: Plummer `2.8`, CubicSplineW2 `1.0`.
- **Rust → C++ mapping rules:**
  - `Vec<[f64; 3]>` → `std::vector<Vec3>`, `[f64; 3]` → `Vec3`.
  - `Option<T>` → `std::optional<T>`; `Option<&[T]>` in args → `const T*` (nullptr = None); `usize::MAX` sentinel → `NO_INDEX`.
  - `[f64; 8]` / `[usize; 8]` → `std::array<double,8>` / `std::array<size_t,8>`.
  - trait/generic monomorphization → `template<typename>` / `template<int>` with explicit instantiations.
  - rayon `par_iter`/`par_iter_mut` → `#pragma omp parallel for` with `if(n >= threshold)`.
  - `x.mul_add(y, z)` → `x*y + z` (tests use 1e-10 tolerances, not bit-exact).
  - `x.powi(k)` → `std::pow(x, k)` (or repeated multiplication).
  - `unsafe { slice.get_unchecked(i) }` → `vec[i]` (bounds are guaranteed by construction in the ported code).
- **Public Python API unchanged:** `pynbodyext.gravity` exports `Gravity`, `KernelKind`, `calculate_potential`, `calculate_acceleration`, `GRAVITY_NATIVE_AVAILABLE`. `Gravity` methods and the `Octree` binding constructor/method signatures are preserved exactly (see Task 12).
- **Do not touch** `pynbodyext/core/`, `pynbodyext/profiles/`, `pynbodyext/transforms/`, `pynbodyext/filters/`, `pynbodyext/properties/`, `benchmarks/` except where the plan explicitly says so.

## File Structure

| File | Responsibility |
|---|---|
| `cpp/gravity/vec3.hpp` | `Vec3` struct + operators + `dot3`/`norm2` |
| `cpp/gravity/common.hpp` | `R2_TINY`, `MIN_SOFTENING`, `NO_INDEX`, `inv_r_from_r2`, `inv_r_and_inv_r3_from_r2`, `timing_enabled`, `log_timing` |
| `cpp/gravity/kernel.{hpp,cpp}` | `KernelKind`, `kernel_potential_per_unit_mass`, `kernel_accel_factor`, `w2`, `w2_prime` |
| `cpp/gravity/multipole/moment.{hpp,cpp}` | `MultipoleMoment` (56 doubles), `Moment0/2/3/4`, `MultipoleMoments` variant, `from_points`, `add_assign`, `translate_multipole` |
| `cpp/gravity/multipole/derivatives.{hpp,cpp}` | `PotentialDerivatives1/2/3/4` + full `PotentialDerivatives` |
| `cpp/gravity/multipole/eval.{hpp,cpp}` | `template<int Order> MultipoleEval` specializations; runtime-dispatch `gravity_potential_multipole`/`gravity_accel_multipole` |
| `cpp/gravity/direct.{hpp,cpp}` | `AccelAccumulator`, `PotAccumulator`, `direct_self_impl<A>`, `direct_at_points_impl<A>`, public `direct_*` functions |
| `cpp/gravity/octree.{hpp,cpp}` | `NodeBh`, `Node`, `Octree` (build, treewalk links, BH/hmax/multipole payloads) |
| `cpp/gravity/traversal.{hpp,cpp}` | `TraversalCtx`, `leaf_potential_sum`, `leaf_acceleration_sum`, `template<int Order>` treewalks, dispatch, 4 public compute methods |
| `cpp/gravity/tests/test_core.cpp` | assert-based C++ test binary (all core tests) |
| `cpp/bindings/module.cpp` | pybind11 module `_native`: `Octree` class + 4 direct functions |
| `Makefile` | add `cpp-test` target |
| `setup.py` | `Pybind11Extension("pynbodyext._native", ...)` + OpenMP flags |
| `pyproject.toml` | build-system requires `pybind11`, drop setuptools-rust/maturin |
| `pynbodyext/gravity/base.py` | import `pynbodyext._native`, docstring + error text |
| `pynbodyext/util/deps.py` | `GRAVITY_NATIVE_AVAILABLE` |
| `pynbodyext/gravity/__init__.py` | renamed flag + warning text |
| `tests/test_gravity.py` | end-to-end Python tests through the extension |

---

## Task 1: Vec3, common helpers, and C++ test scaffold

**Files:**
- Create: `cpp/gravity/vec3.hpp`
- Create: `cpp/gravity/common.hpp`
- Create: `cpp/gravity/tests/test_core.cpp`
- Modify: `Makefile` (add `cpp-test` target)

**Interfaces:**
- Produces:
  - `namespace gravity { struct Vec3 { double x,y,z; ... }; }` with `operator[]`, `operator+`, `operator-`, `operator*`, `dot3`, `norm2`.
  - `constexpr double R2_TINY; constexpr double MIN_SOFTENING; constexpr size_t NO_INDEX;`
  - `double inv_r_from_r2(double r2);` and `void inv_r_and_inv_r3_from_r2(double r2, double& inv_r, double& inv_r3);`
  - `bool timing_enabled(); void log_timing(const char* label, double ms);` (reads `GRAVITY_TIMING` env var; only used when `-D GRAVITY_TIMING` — port `octree.rs:12-27` semantics).
  - Test harness: `CHECK(cond)`, `CHECK_NEAR(a,b,tol)` macros incrementing global `failures`; `main()` returning nonzero on failure.

- [ ] **Step 1: Write `cpp/gravity/vec3.hpp`**

```cpp
#pragma once
namespace gravity {
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    const double& operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
};
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
inline Vec3 operator-(const Vec3& a) { return Vec3(-a.x, -a.y, -a.z); }
inline Vec3 operator*(const Vec3& a, double s) { return Vec3(a.x*s, a.y*s, a.z*s); }
inline double dot3(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline double norm2(const Vec3& a) { return dot3(a, a); }
} // namespace gravity
```

- [ ] **Step 2: Write `cpp/gravity/common.hpp`**

```cpp
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace gravity {
constexpr double R2_TINY = std::numeric_limits<double>::min();
constexpr double MIN_SOFTENING = 0.0;
constexpr size_t NO_INDEX = std::numeric_limits<size_t>::max();

inline double inv_r_from_r2(double r2) { return 1.0 / std::sqrt(r2 + R2_TINY); }

inline void inv_r_and_inv_r3_from_r2(double r2, double& inv_r, double& inv_r3) {
    inv_r = inv_r_from_r2(r2);
    inv_r3 = inv_r * inv_r * inv_r;
}

// Port of octree.rs timing_enabled()/log_timing(): prints "[gravity-timing] ..."
// to stderr only when GRAVITY_TIMING env var is truthy. Use a function-local
// static bool initialized once from std::getenv.
inline bool timing_enabled();
inline void log_timing(const char* label, double dt_ms);
} // namespace gravity
```

- [ ] **Step 3: Write `cpp/gravity/tests/test_core.cpp` scaffold**

```cpp
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <optional>
#include "gravity/vec3.hpp"
#include "gravity/common.hpp"

static int failures = 0;
#define CHECK(cond) \
    do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)
#define CHECK_NEAR(a, b, tol) \
    do { double _a = (a), _b = (b); if (std::fabs(_a - _b) > (tol)) { \
        std::fprintf(stderr, "FAIL %s:%d: |%g - %g| > %g\n", __FILE__, __LINE__, _a, _b, (double)(tol)); ++failures; } } while (0)

static void test_vec3() {
    gravity::Vec3 a(1.0, 2.0, 3.0), b(4.0, 5.0, 6.0);
    gravity::Vec3 c = a + b;
    CHECK(c.x == 5.0 && c.y == 7.0 && c.z == 9.0);
    CHECK(a[0] == 1.0 && a[1] == 2.0 && a[2] == 3.0);
    CHECK_NEAR(gravity::norm2(a), 14.0, 1e-12);
    CHECK_NEAR(gravity::dot3(a, b), 32.0, 1e-12);
}

static void test_common() {
    CHECK_NEAR(gravity::inv_r_from_r2(4.0), 0.5, 1e-12);
    double ir, ir3;
    gravity::inv_r_and_inv_r3_from_r2(4.0, ir, ir3);
    CHECK_NEAR(ir, 0.5, 1e-12);
    CHECK_NEAR(ir3, 0.125, 1e-12);
    CHECK(gravity::R2_TINY > 0.0);
    CHECK(gravity::NO_INDEX == std::numeric_limits<size_t>::max());
}

int main() {
    test_vec3();
    test_common();
    if (failures) { std::printf("%d FAILURE(S)\n", failures); return 1; }
    std::printf("ALL PASS\n");
    return 0;
}
```

- [ ] **Step 4: Add `cpp-test` target to `Makefile`**

Append to the Makefile (after the `test:` target):

```make
.PHONY: cpp-test
cpp-test:
	g++ -std=c++17 -O2 -fopenmp -Icpp -o /tmp/pynbodyext_cpp_test \
		cpp/gravity/tests/test_core.cpp \
		cpp/gravity/*.cpp cpp/gravity/multipole/*.cpp \
		-lm && /tmp/pynbodyext_cpp_test
```

(As later tasks add `.cpp` files, this glob picks them up. Verify the glob compiles empty-file lists gracefully — if a task's glob matches nothing, add an explicit `-x c++` guard is not needed; the `.cpp` files exist from Task 2 onward.)

- [ ] **Step 5: Run the test**

Run: `make cpp-test`
Expected: prints `ALL PASS` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add cpp/gravity/vec3.hpp cpp/gravity/common.hpp cpp/gravity/tests/test_core.cpp Makefile
git commit -m "feat: add Vec3, common numerics helpers, and C++ test scaffold"
```

---

## Task 2: kernel

**Files:**
- Create: `cpp/gravity/kernel.hpp`, `cpp/gravity/kernel.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp` (add `test_kernel()` + call in `main`)

**Interfaces:**
- Produces:
  - `enum class KernelKind { Plummer = 0, CubicSplineW2 = 1 };`
  - `double multipole_min_separation_factor(KernelKind k);` (Plummer → 2.8, CubicSplineW2 → 1.0)
  - `bool multipole_soft_ok(KernelKind k, double r, double h);` (port `kernel.rs:32-38`)
  - `double kernel_potential_per_unit_mass(KernelKind k, double r, double h);` (port `kernel.rs:41-56`)
  - `double kernel_accel_factor(KernelKind k, double r, double h);` (port `kernel.rs:62-82`)

- [ ] **Step 1: Write `cpp/gravity/kernel.hpp`**

```cpp
#pragma once
namespace gravity {
enum class KernelKind { Plummer = 0, CubicSplineW2 = 1 };
double multipole_min_separation_factor(KernelKind k);
bool multipole_soft_ok(KernelKind k, double r, double h);
double kernel_potential_per_unit_mass(KernelKind k, double r, double h);
double kernel_accel_factor(KernelKind k, double r, double h);
} // namespace gravity
```

- [ ] **Step 2: Write `cpp/gravity/kernel.cpp`** — port `crates/gravity/src/kernel.rs` verbatim. The full formulas (load-bearing, do not alter):

```cpp
#include "gravity/kernel.hpp"
#include <cmath>
namespace gravity {

double multipole_min_separation_factor(KernelKind k) {
    switch (k) {
        case KernelKind::Plummer: return 2.8;
        case KernelKind::CubicSplineW2: return 1.0;
    }
    return 1.0;
}

bool multipole_soft_ok(KernelKind k, double r, double h) {
    if (h <= 0.0) return true;
    return r > multipole_min_separation_factor(k) * h;
}

double kernel_potential_per_unit_mass(KernelKind k, double r, double h) {
    if (r == 0.0) return 0.0;
    if (k == KernelKind::Plummer) return -1.0 / std::sqrt(r*r + h*h);
    // CubicSplineW2
    if (h <= 0.0) return -1.0 / r;
    double h_inv = 1.0 / h;
    double u = r * h_inv;
    double W2;
    if (u < 0.5) {
        double u2 = u*u, u4 = u2*u2, u5 = u4*u;
        W2 = (16.0/3.0)*u2 - (48.0/5.0)*u4 + (32.0/5.0)*u5 - 14.0/5.0;
    } else if (u < 1.0) {
        double inv_u = 1.0/u, u2 = u*u, u3 = u2*u, u4 = u2*u2, u5 = u4*u;
        W2 = (1.0/15.0)*inv_u + (32.0/3.0)*u2 - 16.0*u3 + (48.0/5.0)*u4 - (32.0/15.0)*u5 - 16.0/5.0;
    } else {
        W2 = -1.0/u;
    }
    return W2 * h_inv;
}

double kernel_accel_factor(KernelKind k, double r, double h) {
    if (r == 0.0) return 0.0;
    if (k == KernelKind::Plummer) {
        double s2 = r*r + h*h;
        return 1.0 / (std::sqrt(s2) * s2);
    }
    // CubicSplineW2
    if (h <= 0.0) return 1.0 / (r*r*r);
    double h_inv = 1.0 / h;
    double u = r * h_inv;
    double W2p;
    if (u < 0.5) {
        double u2 = u*u, u3 = u2*u, u4 = u2*u2;
        W2p = (32.0/3.0)*u - (192.0/5.0)*u3 + 32.0*u4;
    } else if (u < 1.0) {
        double u2 = u*u, u3 = u2*u, u4 = u2*u2;
        W2p = -(1.0/15.0)*(1.0/u2) + (64.0/3.0)*u - 48.0*u2 + (192.0/5.0)*u3 - (32.0/3.0)*u4;
    } else {
        W2p = 1.0/(u*u);
    }
    return W2p * (h_inv*h_inv) / r;
}

} // namespace gravity
```

- [ ] **Step 3: Add `test_kernel()` to test_core.cpp and register it in `main`**

```cpp
static void test_kernel() {
    using namespace gravity;
    CHECK_NEAR(kernel_potential_per_unit_mass(KernelKind::Plummer, 1.0, 0.0), -1.0, 1e-12);
    CHECK_NEAR(kernel_potential_per_unit_mass(KernelKind::Plummer, 0.0, 0.1), 0.0, 1e-15);
    CHECK_NEAR(kernel_potential_per_unit_mass(KernelKind::Plummer, 3.0, 4.0), -1.0/5.0, 1e-12);
    CHECK_NEAR(kernel_accel_factor(KernelKind::Plummer, 2.0, 0.0), 1.0/8.0, 1e-12);
    CHECK_NEAR(kernel_potential_per_unit_mass(KernelKind::CubicSplineW2, 2.0, 1.0), -1.0/2.0, 1e-12); // u>=1 -> -1/u
    CHECK(multipole_min_separation_factor(KernelKind::Plummer) == 2.8);
    CHECK(multipole_min_separation_factor(KernelKind::CubicSplineW2) == 1.0);
    CHECK(multipole_soft_ok(KernelKind::Plummer, 3.0, 1.0));   // 3 > 2.8*1
    CHECK(!multipole_soft_ok(KernelKind::Plummer, 2.0, 1.0));  // 2 < 2.8
}
```
Add `test_kernel();` to `main()` before the failure check.

- [ ] **Step 4: Run the test**

Run: `make cpp-test`
Expected: prints `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add cpp/gravity/kernel.hpp cpp/gravity/kernel.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port kernel (KernelKind, softened kernels)"
```

---

## Task 3: multipole/moment

**Files:**
- Create: `cpp/gravity/multipole/moment.hpp`, `cpp/gravity/multipole/moment.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Produces (in `namespace gravity`):
  - `struct MultipoleMoment { double m000, m100, m010, m001, m200, m020, m002, m110, m101, m011, m300, m030, m003, m210, m201, m120, m102, m021, m012, m111, m400, m040, m004, m310, m301, m130, m103, m031, m013, m220, m202, m022, m211, m121, m112, m500, m050, m005, m410, m401, m140, m104, m041, m014, m320, m302, m230, m203, m032, m023, m221, m212, m122, m311, m131, m113; }` (56 doubles, exact order from `moment.rs:18-81`) + `static MultipoleMoment zero();` + `static MultipoleMoment from_points(const std::vector<Vec3>& positions, const double* masses, const std::vector<size_t>& indices, const Vec3& center, unsigned char order);` + `void add_assign(const MultipoleMoment& o);`
  - `struct Moment0 { double m000; };` `struct Moment2 { m000,m100,m010,m001,m200,m020,m002,m110,m101,m011 };` `struct Moment3 { ...20 fields... };` `struct Moment4 { ...35 fields... };` (field lists identical to `moment.rs:281-309`); each with a constructor `explicit MomentX(const MultipoleMoment& v)` that copies only those fields. `using Moment5 = MultipoleMoment;`
  - `using MultipoleMoments = std::variant<std::vector<Moment0>, std::vector<Moment2>, std::vector<Moment3>, std::vector<Moment4>, std::vector<Moment5>>;` plus free function `MultipoleMoments multipole_moments_from_full(std::vector<MultipoleMoment> full, unsigned char order);` (port `moment.rs:332-343`; orders 0|1 → O0, 2 → O2, 3 → O3, 4 → O4, else O5).
  - `MultipoleMoment translate_multipole(const MultipoleMoment& m_child, const Vec3& shift, unsigned char order);` (port `moment.rs:479-538`; `shift = C_parent - C_child`).

- [ ] **Step 1: Write `cpp/gravity/multipole/moment.hpp`**

Declare the structs/types listed above. `MultipoleMoments` needs `#include <variant>`.

- [ ] **Step 2: Write `cpp/gravity/multipole/moment.cpp`** — port `crates/gravity/src/multipole/moment.rs` verbatim:
  - `MultipoleMoment::zero()` → `return MultipoleMoment{};` (zero-init).
  - `from_points` → port `from_points` + `from_points_const<O>` as a single function with runtime `order` branching inside the particle loop. The 56 accumulation formulas in `moment.rs:122-196` must be transcribed exactly (e.g. `m.m200 += 0.5 * mass * x * x;`, `m.m300 += (1.0/6.0) * mass * x*x*x;`, `m.m400 += (1.0/24.0)*...`). Replace `x.powi(k)` with repeated multiplication or `std::pow`.
  - `add_assign` → port `moment.rs:202-259` (all 56 `+=`).
  - `translate_multipole` → port `moment.rs:479-538` exactly, including the `FACT` table and the `get_moment`/`set_moment` index mapping. In C++ implement `get_moment`/`set_moment` as functions taking `(l, m, n)` with a `switch` on `(l*100 + m*10 + n)` or nested if-chains mapping to the 56 fields (port `moment.rs:352-474`).

- [ ] **Step 3: Add `test_moment()` to test_core.cpp**

```cpp
static void test_moment() {
    using namespace gravity;
    // Two unit-mass particles at (1,0,0) and (-1,0,0), center origin.
    std::vector<Vec3> pos = {Vec3(1,0,0), Vec3(-1,0,0)};
    std::vector<size_t> idx = {0, 1};
    MultipoleMoment m = MultipoleMoment::from_points(pos, nullptr, idx, Vec3(0,0,0), 5);
    CHECK_NEAR(m.m000, 2.0, 1e-12);
    CHECK_NEAR(m.m100, 0.0, 1e-12);   // dipole vanishes by symmetry
    CHECK_NEAR(m.m200, 1.0, 1e-12);   // 0.5*(1+1)
    CHECK_NEAR(m.m020, 0.0, 1e-12);
    CHECK_NEAR(m.m002, 0.0, 1e-12);
    // translate: shift by (5,0,0). Mass-conservation: m000 unchanged.
    MultipoleMoment t = translate_multipole(m, Vec3(5,0,0), 5);
    CHECK_NEAR(t.m000, 2.0, 1e-12);
    // from_points about shifted center should match translated moment's m100.
    MultipoleMoment m2 = MultipoleMoment::from_points(pos, nullptr, idx, Vec3(5,0,0), 5);
    CHECK_NEAR(t.m100, m2.m100, 1e-10);
    CHECK_NEAR(t.m200, m2.m200, 1e-10);
}
```
Add `test_moment();` to `main()`. Include `#include "gravity/multipole/moment.hpp"` at the top of test_core.cpp.

- [ ] **Step 4: Run the test**

Run: `make cpp-test`
Expected: prints `ALL PASS`. (The translate-vs-direct 56-coefficient check is deferred to Task 9.)

- [ ] **Step 5: Commit**

```bash
git add cpp/gravity/multipole/moment.hpp cpp/gravity/multipole/moment.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port multipole moments (P2M, M2M translate, compact storage)"
```

---

## Task 4: multipole/derivatives

**Files:**
- Create: `cpp/gravity/multipole/derivatives.hpp`, `cpp/gravity/multipole/derivatives.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Produces: `struct PotentialDerivatives1 { d000, d100, d010, d001 };` `PotentialDerivatives2 { +d200,d020,d002,d110,d101,d011 };` `PotentialDerivatives3 { +d300,d030,d003,d210,d201,d120,d102,d021,d012,d111 };` `PotentialDerivatives4 { +d400,d040,d004,d310,d301,d130,d103,d031,d013,d220,d202,d022,d211,d121,d112 };` `PotentialDerivatives` (full 56 through order 5, field order per `derivatives.rs:186-207`). Each has a static factory `static PotentialDerivativesX new_derivatives(double dx, double dy, double dz, double eps2);` (full version takes `int order` and computes only up to that order). All structs are zero-initialized.

- [ ] **Step 1: Write `cpp/gravity/multipole/derivatives.hpp`**

Declare the 5 structs with the exact field lists above and their factory functions.

- [ ] **Step 2: Write `cpp/gravity/multipole/derivatives.cpp`** — port `crates/gravity/src/multipole/derivatives.rs` verbatim:
  - `r2 = dx*dx + dy*dy + dz*dz + eps2 + R2_TINY; r = sqrt(r2); r_inv = 1.0/r;`
  - Recurrence `dt_1 = r_inv; dt_2 = -dt_1*r_inv; dt_3 = -3*dt_2*r_inv; dt_4 = -5*dt_3*r_inv; dt_5 = -7*dt_4*r_inv; dt_6 = -9*dt_5*r_inv;`
  - `PotentialDerivatives1::new_derivatives` → port `derivatives.rs:19-35` (uses `dt_1`, `dt_2`).
  - `PotentialDerivatives2::new_derivatives` → port `derivatives.rs:48-77` (adds `dt_3`, direction-cosine squares; note the in-place `dt_2 *= r_inv` before the quadrupole terms — preserve that ordering).
  - `PotentialDerivatives3::new_derivatives` → port `derivatives.rs:94-138` verbatim.
  - `PotentialDerivatives4::new_derivatives` → compute a full `PotentialDerivatives` with `order=4` and copy the 35 fields (port `derivatives.rs:162-180`).
  - `PotentialDerivatives::new_derivatives(dx,dy,dz,eps2,order)` → port `derivatives.rs:210-313` verbatim including the early-return `max` guards (`max = min(order,5)`; after computing order-0 set `d000` then return if `max==0`, etc.).

- [ ] **Step 3: Add `test_derivatives()` to test_core.cpp**

```cpp
static void test_derivatives() {
    using namespace gravity;
    // Monopole: d000 = 1/r at displacement (3,0,0), eps2=0.
    auto d1 = PotentialDerivatives1::new_derivatives(3.0, 0.0, 0.0, 0.0);
    CHECK_NEAR(d1.d000, 1.0/3.0, 1e-12);
    CHECK_NEAR(d1.d100, -1.0/9.0, 1e-12);  // d(1/r)/dx = -x/r^3
    CHECK_NEAR(d1.d010, 0.0, 1e-12);
    // Full order-5 with eps2: r2 = 3^2+4^2 = 25 -> r=5, d000 = 1/sqrt(25)=0.2
    auto d5 = PotentialDerivatives::new_derivatives(3.0, 4.0, 0.0, 0.0, 5);
    CHECK_NEAR(d5.d000, 0.2, 1e-12);
    CHECK_NEAR(d5.d100, -3.0/125.0, 1e-12); // -x/r^3 = -3/125
}
```
Add `test_derivatives();` to `main()` and `#include "gravity/multipole/derivatives.hpp"`.

- [ ] **Step 4: Run the test**

Run: `make cpp-test`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add cpp/gravity/multipole/derivatives.hpp cpp/gravity/multipole/derivatives.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port multipole potential derivatives (orders 1-5)"
```

---

## Task 5: multipole/eval

**Files:**
- Create: `cpp/gravity/multipole/eval.hpp`, `cpp/gravity/multipole/eval.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Produces:
  - `template <int Order> struct MultipoleEval;` with full specializations for `0, 2, 3, 4, 5`. Each specializes (matching `eval.rs` trait):
    - `using Moment = ...;` (`Moment0/2/3/4/Moment5`)
    - `using Derivatives = ...;`
    - `static Moment from_points(const std::vector<Vec3>& positions, const double* masses, const std::vector<size_t>& indices, const Vec3& center);`
    - `static Moment translate(const Moment& m, const Vec3& shift);`
    - `static void add_assign(Moment& acc, const Moment& other);`
    - `static double potential(const Moment& m, const Derivatives& d);`
    - `static Vec3 acceleration(const Moment& m, const Derivatives& d);`
    - `static Derivatives derivatives(double dx, double dy, double dz, double eps2);`
  - Runtime-dispatch helpers for tests: `double gravity_potential_multipole(const MultipoleMoment& m, const PotentialDerivatives& d, unsigned char order);` and `Vec3 gravity_accel_multipole(const MultipoleMoment& m, const PotentialDerivatives& d, unsigned char order);` (port `eval.rs:493-581`).

- [ ] **Step 1: Write `cpp/gravity/multipole/eval.hpp`**

Declare `template<int> struct MultipoleEval;` (primary, undefined) and the runtime-dispatch function declarations.

- [ ] **Step 2: Write `cpp/gravity/multipole/eval.cpp`** — port `crates/gravity/src/multipole/eval.rs` verbatim, splitting into explicit `template<> struct MultipoleEval<0>`, `<2>`, `<3>`, `<4>`, `<5>`:
  - **Order 0** → port `eval.rs:69-110` (`Moment=Moment0`, `Derivatives=PotentialDerivatives1`; translate = copy; potential = `-m.m000*d.d000`; acceleration = `Vec3(-m.m000*d.d100, -m.m000*d.d010, -m.m000*d.d001)`).
  - **Order 2** → port `eval.rs:114-168` (translate via `moment_to_full_o2` → `translate_multipole` → compact back; potential/acceleration formulas verbatim).
  - **Order 3** → port `eval.rs:172-238`.
  - **Order 4** → port `eval.rs:242-327` (acceleration includes the 10-term order-4 sums).
  - **Order 5** → port `eval.rs:331-368`; potential/acceleration delegate to `gravity_potential_multipole_o5`/`gravity_accel_multipole_o5`.
  - Keep `gravity_potential_multipole_o5` and `gravity_accel_multipole_o5` as file-local functions (`eval.rs:424-486`); port `gravity_potential_multipole`/`gravity_accel_multipole` runtime dispatchers (`eval.rs:493-581`) as public functions.
  - Add explicit template instantiations:
    ```cpp
    template struct MultipoleEval<0>;
    template struct MultipoleEval<2>;
    template struct MultipoleEval<3>;
    template struct MultipoleEval<4>;
    template struct MultipoleEval<5>;
    ```

- [ ] **Step 3: Add `test_eval()` to test_core.cpp**

```cpp
static void test_eval() {
    using namespace gravity;
    // Single unit mass at origin; monopole moment about origin; target at (2,0,0).
    // The treewalk evaluates derivatives at displacement (COM - target), i.e.
    // source-minus-target, so here the derivative displacement is (-2,0,0).
    std::vector<Vec3> pos = {Vec3(0,0,0)};
    std::vector<size_t> idx = {0};
    MultipoleMoment full = MultipoleMoment::from_points(pos, nullptr, idx, Vec3(0,0,0), 5);
    PotentialDerivatives d = PotentialDerivatives::new_derivatives(-2.0, 0.0, 0.0, 0.0, 5);
    CHECK_NEAR(gravity_potential_multipole(full, d, 0), -0.5, 1e-12);   // -1/r
    CHECK_NEAR(gravity_potential_multipole(full, d, 5), -0.5, 1e-12);
    Vec3 a = gravity_accel_multipole(full, d, 5);
    CHECK_NEAR(a.x, -0.25, 1e-12);   // -m000*d100 = -1*(+0.25); d100 = -(-2)/8
    CHECK_NEAR(a.y, 0.0, 1e-12);
    CHECK_NEAR(a.z, 0.0, 1e-12);
    // Moment2 translate invariance: a pure monopole stays m000 under shift.
    Moment2 m2(full);
    Moment2 t2 = MultipoleEval<2>::translate(m2, Vec3(3,0,0));
    CHECK_NEAR(t2.m000, 1.0, 1e-12);
}
```
Add `test_eval();` to `main()` and `#include "gravity/multipole/eval.hpp"` and `#include "gravity/multipole/moment.hpp"`.

- [ ] **Step 4: Run the test**

Run: `make cpp-test`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add cpp/gravity/multipole/eval.hpp cpp/gravity/multipole/eval.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port multipole evaluators (template<Order> + runtime dispatch)"
```

---

## Task 6: direct

**Files:**
- Create: `cpp/gravity/direct.hpp`, `cpp/gravity/direct.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Produces (in `namespace gravity`):
  - Accumulators (mirror `direct.rs:34-103`): `struct AccelAccumulator { double ax=0, ay=0, az=0; using output_type = Vec3; void add_newton(double dx,double dy,double dz,double m,double invr); void add_softened(double dx,double dy,double dz,double m,double r,double h,KernelKind k); Vec3 finish() const; };` and `struct PotAccumulator { double phi=0; using output_type = double; void add_newton(...); void add_softened(...); double finish() const; };`
  - `template <typename A> std::vector<typename A::output_type> direct_self_impl(const std::vector<Vec3>& positions, const double* masses, const double* softenings, std::optional<KernelKind> kernel);`
  - `template <typename A> std::vector<typename A::output_type> direct_at_points_impl(const std::vector<Vec3>& positions, const double* masses, const double* softenings, const std::vector<Vec3>& targets, std::optional<KernelKind> kernel);`
  - Public functions — all 8, exact signatures (port `direct.rs:326-393`, names identical to Rust):
    - `std::vector<Vec3> direct_accelerations(const std::vector<Vec3>& positions, const double* masses);`
    - `std::vector<Vec3> direct_accelerations_at_points(const std::vector<Vec3>& positions, const double* masses, const std::vector<Vec3>& targets);`
    - `std::vector<double> direct_potentials(const std::vector<Vec3>& positions, const double* masses);`
    - `std::vector<double> direct_potentials_at_points(const std::vector<Vec3>& positions, const double* masses, const std::vector<Vec3>& targets);`
    - `std::vector<double> direct_potentials_kernel(const std::vector<Vec3>& positions, const double* masses, const double* softenings, KernelKind kernel);`
    - `std::vector<Vec3> direct_accelerations_kernel(const std::vector<Vec3>& positions, const double* masses, const double* softenings, KernelKind kernel);`
    - `std::vector<double> direct_potentials_kernel_at_points(const std::vector<Vec3>& positions, const double* masses, const double* softenings, const std::vector<Vec3>& targets, KernelKind kernel);`
    - `std::vector<Vec3> direct_accelerations_kernel_at_points(const std::vector<Vec3>& positions, const double* masses, const double* softenings, const std::vector<Vec3>& targets, KernelKind kernel);`

- [ ] **Step 1: Write `cpp/gravity/direct.hpp`**

Declare the accumulators, the two templates, and the 8 public functions. Include `<vector>`, `<optional>`, `"gravity/vec3.hpp"`, `"gravity/kernel.hpp"`.

- [ ] **Step 2: Write `cpp/gravity/direct.cpp`** — port `crates/gravity/src/direct.rs` verbatim:
  - `add_newton`: accel → `g = m * invr3` with `invr3 = invr*invr*invr`, accumulate `dx*g, dy*g, dz*g` (port `direct.rs:51-59`); potential → `phi += -m * invr` (port `direct.rs:90-93`).
  - `add_softened`: accel → `g = m * kernel_accel_factor(k,r,h)`, accumulate (port `direct.rs:62-67`); potential → `phi += m * kernel_potential_per_unit_mass(k,r,h)` (port `direct.rs:95-97`).
  - `direct_self_impl` (port `direct.rs:113-216`): if `n==0` return `{}`. Unit mass when `masses==nullptr` (fill a local `std::vector<double> ones(n,1.0)`). `has_softening = (softenings != nullptr) && kernel.has_value()`. If `n < 512`: symmetric pairwise loop over `i`, inner `j in (i+1)..n`, updating both `accums[i]` and `accums[j]` (port `direct.rs:136-174`, including the `h = max(hi,hj)` softened branch). Else: `#pragma omp parallel for if(n >= 512) schedule(static)` over `i`, each iteration builds a local accumulator (port `direct.rs:176-214`); collect results in order.
  - `direct_at_points_impl` (port `direct.rs:219-319`): if `n_tgt==0 || n_src==0` return `{}`; unit masses same as above; `#pragma omp parallel for if(n_tgt >= 512)` over targets.
  - Public functions: thin wrappers calling the impl with the right accumulator and `kernel`/`softenings` (port `direct.rs:326-393`).
  - Explicit template instantiations at the bottom:
    ```cpp
    template std::vector<Vec3> direct_self_impl<AccelAccumulator>(const std::vector<Vec3>&, const double*, const double*, std::optional<KernelKind>);
    template std::vector<double> direct_self_impl<PotAccumulator>(const std::vector<Vec3>&, const double*, const double*, std::optional<KernelKind>);
    template std::vector<Vec3> direct_at_points_impl<AccelAccumulator>(const std::vector<Vec3>&, const double*, const double*, const std::vector<Vec3>&, std::optional<KernelKind>);
    template std::vector<double> direct_at_points_impl<PotAccumulator>(const std::vector<Vec3>&, const double*, const double*, const std::vector<Vec3>&, std::optional<KernelKind>);
    ```

- [ ] **Step 3: Add `test_direct()` to test_core.cpp**

```cpp
static void test_direct() {
    using namespace gravity;
    // Two unit masses at (1,0,0) and (-1,0,0). Potential at each: -1/2 - 1/2 = -1? No:
    // particle 0 sees particle 1 at distance 2 -> phi = -1*1/2. Self excluded.
    std::vector<Vec3> pos = {Vec3(1,0,0), Vec3(-1,0,0)};
    double ones[2] = {1.0, 1.0};
    auto acc = direct_accelerations(pos, ones);
    // acceleration on particle 0 from particle 1: m*x/r^3 = 1*(-2)/8 = -0.25
    CHECK_NEAR(acc[0].x, -0.25, 1e-12);
    CHECK_NEAR(acc[0].y, 0.0, 1e-12);
    CHECK_NEAR(acc[1].x, 0.25, 1e-12);
    auto pot = direct_potentials(pos, ones);
    CHECK_NEAR(pot[0], -0.5, 1e-12);
    CHECK_NEAR(pot[1], -0.5, 1e-12);
    // At-points: query at origin.
    std::vector<Vec3> q = {Vec3(0,0,0)};
    auto acc_q = direct_accelerations_at_points(pos, ones, q);
    CHECK_NEAR(acc_q[0].x, 0.0, 1e-12); // symmetric
    auto pot_q = direct_potentials_at_points(pos, ones, q);
    CHECK_NEAR(pot_q[0], -2.0, 1e-12);  // -1/1 - 1/1
}
```
Add `test_direct();` to `main()` and `#include "gravity/direct.hpp"`.

- [ ] **Step 4: Run the test**

Run: `make cpp-test`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add cpp/gravity/direct.hpp cpp/gravity/direct.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port direct O(N^2) summation with OpenMP"
```

---

## Task 7: octree

**Files:**
- Create: `cpp/gravity/octree.hpp`, `cpp/gravity/octree.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Consumes: `Vec3`, `common.hpp` helpers, `kernel.hpp`, `multipole/moment.hpp`, `multipole/eval.hpp`.
- Produces:
  - `struct NodeBh { Vec3 com; double mass; };`
  - `struct Node { Vec3 center; double half_size; double size2; std::optional<std::array<size_t,8>> children; std::vector<size_t> indices; };`
  - `class Octree` (public fields, mirror `octree.rs:86-109`):
    - `std::vector<Vec3> positions; std::optional<std::vector<double>> masses; std::optional<std::vector<double>> softenings; std::vector<Node> nodes; std::vector<size_t> first_subnode; std::vector<size_t> next_branch; std::optional<std::vector<NodeBh>> bh; std::optional<MultipoleMoments> multipoles; std::optional<std::vector<double>> hmax; unsigned char multipole_order; size_t leaf_capacity; KernelKind kernel;`
    - `static Octree from_owned(std::vector<Vec3> positions, std::optional<std::vector<double>> masses, std::optional<std::vector<double>> softenings, size_t leaf_capacity, unsigned char multipole_order, KernelKind kernel);`
    - `static Octree build(const std::vector<Vec3>& positions, const double* masses, size_t leaf_capacity, unsigned char multipole_order);` (convenience: Plummer kernel, no softening, then `build_mass_payload()` — port `octree.rs:359-375`)
    - `void build_mass_payload();` `void set_softenings(std::optional<std::vector<double>>);` `void set_kernel(KernelKind);` `void set_masses(std::optional<std::vector<double>>);` `const std::vector<NodeBh>& bh_payload() const;` (panics/throws if absent — port `octree.rs:382-386`; throw `std::runtime_error("BH payload not initialized; call build_mass_payload() before gravity queries")`)

- [ ] **Step 1: Write `cpp/gravity/octree.hpp`**

Declare `NodeBh`, `Node`, `Octree` per above. Include `<array>`, `<optional>`, `<vector>`, `"gravity/vec3.hpp"`, `"gravity/common.hpp"`, `"gravity/kernel.hpp"`, `"gravity/multipole/moment.hpp"`.

- [ ] **Step 2: Write `cpp/gravity/octree.cpp`** — port `crates/gravity/src/octree.rs` verbatim:
  - `bbox_of_points` (port `octree.rs:116-142`): min/max per axis, center = midpoints, `half = max(extent)/2`; if `half==0` → `1e-6`.
  - `from_owned` (port `octree.rs:149-228`): `leaf_capacity = max(1, leaf_capacity)`; indices `0..n`; make root node; `build_recursive(0)`; `build_treewalk_links()`. Guard: if `softenings` provided with wrong length, throw `std::invalid_argument("softenings length must match positions length")` (port the `assert_eq!`).
  - `make_node` / `subdivide_node` / `build_recursive` (port `octree.rs:234-305`): octant bit `oct = (x>=cx ? 1:0) | (y>=cy ? 2:0) | (z>=cz ? 4:0)`; child center offset `half/2`; empty octant → `NO_INDEX`.
  - `build_treewalk_links` (port `octree.rs:311-351`): `first_subnode`/`next_branch` filled with `NO_INDEX`; recursive helper linking children then recursing; `next_branch[0] = NO_INDEX`.
  - `build_bh_payload` (port `octree.rs:403-469`): bottom-up (`for (idx = nodes.size()-1; idx>=0; --idx)` using `ssize_t`); leaves sum masses/COM, internal nodes combine children; `if (mass > 0) com /= mass`.
  - `build_hmax_payload` (port `octree.rs:471-495`): bottom-up max of particle softenings (leaf) or child hmax (internal); returns `std::nullopt` when no softenings.
  - `build_mass_payload` (port `octree.rs:497-541`): builds `bh`, then `hmax`, then if `multipole_order > 0` builds `build_multipole_payload`.
  - `build_multipole_payload` (port `octree.rs:543-596`): bottom-up; leaves use `MultipoleMoment::from_points`; internal nodes translate children moments to COM via `translate_multipole(shift = center - child_bh.com)` and accumulate; then `multipole_moments_from_full(acc, order)`.
  - Include `#include "gravity/traversal.hpp"` is NOT needed here — traversal is separate (Task 8). The 4 compute methods live in traversal.

- [ ] **Step 3: Add `test_octree()` to test_core.cpp**

```cpp
static void test_octree() {
    using namespace gravity;
    // 20 random-ish particles, leaf_capacity=4, order=2.
    std::vector<Vec3> pos;
    for (int i = 0; i < 20; ++i) pos.push_back(Vec3((i*7)%11 / 10.0, (i*13)%17 / 10.0, (i*5)%9 / 10.0));
    std::vector<double> masses(20, 1.0);
    Octree tree = Octree::build(pos, masses.data(), 4, 2);
    // Every node is either a leaf (children absent) or has valid children.
    for (const auto& node : tree.nodes) {
        if (node.children.has_value()) {
            for (size_t c : *node.children) {
                CHECK(c == NO_INDEX || c < tree.nodes.size());
            }
        } else {
            CHECK(!node.indices.empty());
        }
    }
    CHECK(tree.nodes.size() >= 1);
    CHECK(tree.bh.has_value());
    CHECK(!tree.hmax.has_value());      // build_hmax_payload returns None when no softenings
    CHECK(tree.multipoles.has_value()); // order 2
    CHECK_NEAR((*tree.bh)[0].mass, 20.0, 1e-12); // root mass = sum
}
```
Add `test_octree();` to `main()` and `#include "gravity/octree.hpp"`.

- [ ] **Step 4: Run the test**

Run: `make cpp-test`
Expected: `ALL PASS`.

- [ ] **Step 5: Commit**

```bash
git add cpp/gravity/octree.hpp cpp/gravity/octree.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port octree construction and node payloads"
```

---

## Task 8: traversal

**Files:**
- Create: `cpp/gravity/traversal.hpp`, `cpp/gravity/traversal.cpp`
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Consumes: `Octree` (public fields), `kernel.hpp`, `multipole/eval.hpp`, `multipole/moment.hpp`, `common.hpp`.
- Produces (methods on `Octree`, declared in `octree.hpp` — the implementation lives in `traversal.cpp`):
  - `void Octree::compute_accelerations(double theta, std::vector<Vec3>& out) const;`
  - `void Octree::compute_potentials(double theta, std::vector<double>& out) const;`
  - `void Octree::accelerations_at_points(const std::vector<Vec3>& points, double theta, std::vector<Vec3>& out) const;`
  - `void Octree::potentials_at_points(const std::vector<Vec3>& points, double theta, std::vector<double>& out) const;`
  - (These are declared in `octree.hpp` so the class is complete; defined in `traversal.cpp`.)
- Internal (in `namespace gravity { namespace detail {`):
  - `struct LeafArgs { const std::vector<Vec3>* positions; const std::vector<size_t>* indices; const double* masses; const double* softenings; };`
  - `struct TargetArgs { const Vec3* target; std::optional<size_t> skip_self; std::optional<double> target_h_opt; KernelKind kernel; };`
  - `struct TraversalCtx { const std::vector<NodeBh>* bh; const double* masses; const double* softenings; const double* hmax; double theta2; double multipole_eps2; KernelKind kernel; };`
  - `void leaf_potential_sum(const LeafArgs&, const TargetArgs&, double& out);` and `void leaf_acceleration_sum(const LeafArgs&, const TargetArgs&, Vec3& out);`
  - `template <int Order> void potential_traversal_with_multipoles(const Octree& self, const Vec3& target, std::optional<size_t> skip_self, std::optional<double> target_h_opt, size_t node_idx, double& out, const TraversalCtx& ctx, const std::vector<typename MultipoleEval<Order>::Moment>& multipoles);` and the `acceleration_...` analog.

- [ ] **Step 1: Add the 4 method declarations to `cpp/gravity/octree.hpp`**

Inside `class Octree`, after the payload builders:

```cpp
    void compute_accelerations(double theta, std::vector<Vec3>& out) const;
    void compute_potentials(double theta, std::vector<double>& out) const;
    void accelerations_at_points(const std::vector<Vec3>& points, double theta, std::vector<Vec3>& out) const;
    void potentials_at_points(const std::vector<Vec3>& points, double theta, std::vector<double>& out) const;
```

- [ ] **Step 2: Write `cpp/gravity/traversal.hpp`**

Declare `LeafArgs`, `TargetArgs`, `TraversalCtx`, the leaf-sum functions, and the two `template<int Order>` traversal functions (all in `namespace gravity { namespace detail { ... } }`). Declare `void compute_accelerations(const Octree&, double, std::vector<Vec3>&);` etc. as free functions in `namespace gravity` that call the `Octree` methods (so the binding layer can call them), OR rely solely on the methods — pick free-function wrappers so traversal.hpp is self-contained:

```cpp
#pragma once
#include "gravity/octree.hpp"
namespace gravity {
void compute_accelerations(const Octree& tree, double theta, std::vector<Vec3>& out);
void compute_potentials(const Octree& tree, double theta, std::vector<double>& out);
void accelerations_at_points(const Octree& tree, const std::vector<Vec3>& points, double theta, std::vector<Vec3>& out);
void potentials_at_points(const Octree& tree, const std::vector<Vec3>& points, double theta, std::vector<double>& out);
} // namespace gravity
```

- [ ] **Step 3: Write `cpp/gravity/traversal.cpp`** — port `crates/gravity/src/traversal.rs` verbatim:
  - `node_soft_ok` (port `traversal.rs:56-77`): return true when no hmax; else `h = max(hmax[idx], MIN_SOFTENING)`, combine with `target_h_opt`, `if (h <= 0) return true`, `dist2 > (c*h)^2`.
  - `leaf_potential_sum` (port `traversal.rs:84-246`) — preserve all five branches: (a) constant target-h with masses + spline, (b) constant target-h with masses + Plummer, (c) no softening at all (`!use_softening`) with/without masses, (d) per-particle softenings present, (e) target-h only, no per-particle softenings. `skip` = `skip_self.value_or(NO_INDEX)`.
  - `leaf_acceleration_sum` (port `traversal.rs:249-377`) — same branch structure; Newtonian uses `inv_r_and_inv_r3_from_r2` then `out += Vec3(m*ddx*invr3, ...)`; softened uses `kernel_accel_factor`.
  - `potential_traversal_cached_no_multipoles` / `acceleration_traversal_cached_no_multipoles` (port `traversal.rs:387-519`) — **the treewalk loop is load-bearing; port exactly**:

```cpp
    // potential, no multipoles
    while (idx != NO_INDEX) {
        const NodeBh& node_bh = (*ctx.bh)[idx];
        if (node_bh.mass == 0.0) { idx = self.next_branch[idx]; continue; }
        const Node& node = self.nodes[idx];
        if (!node.children.has_value()) {
            LeafArgs leaf{&self.positions, &node.indices, ctx.masses, ctx.softenings};
            TargetArgs targ{&target, skip_self, target_h_opt, ctx.kernel};
            leaf_potential_sum(leaf, targ, out);
            idx = self.next_branch[idx];
            continue;
        }
        double dx = node_bh.com.x - target.x, dy = node_bh.com.y - target.y, dz = node_bh.com.z - target.z;
        double dist2 = dx*dx + dy*dy + dz*dz + ctx.multipole_eps2;
        bool soft_ok = softening_enabled ? node_soft_ok(idx, dist2, target_h_opt, ctx) : true;
        if (soft_ok && node.size2 < ctx.theta2 * dist2) {
            out += -node_bh.mass * inv_r_from_r2(dist2);
            idx = self.next_branch[idx];
        } else {
            idx = self.first_subnode[idx];
        }
    }
```

  The acceleration version adds `out += Vec3(mass*dx*invr3, mass*dy*invr3, mass*dz*invr3)` on accept (port `traversal.rs:507-513`).
  - `potential_traversal_with_multipoles<Order>` / `acceleration_traversal_with_multipoles<Order>` (port `traversal.rs:523-657`): identical loop; on accept call `auto d = MultipoleEval<Order>::derivatives(dx,dy,dz,ctx.multipole_eps2); out += MultipoleEval<Order>::potential(multipoles[idx], d);` (accel: `Vec3 a = MultipoleEval<Order>::acceleration(...); out += a;`).
  - Dispatch entry points (port `traversal.rs:661-723`): a `std::visit` over `*self.multipoles` choosing the right `template<int Order>` traversal; when `!self.multipoles.has_value()`, use the no-multipole path.
  - Public methods (port `traversal.rs:728-857`): build a `TraversalCtx` with `bh = &*self.bh` (throws via `bh_payload()` if absent), `theta2 = theta*theta`, `multipole_eps2 = R2_TINY`, `kernel = self.kernel`. For self-gravity, `target_h_opt = softenings ? softenings[i] : nullopt`, `skip_self = i`. Parallelize: `if (n < 1024)` serial loop else `#pragma omp parallel for if(n >= 1024) schedule(static)`; each iteration accumulates into a local `tmp` then writes `out[i]` (port the tmp pattern from `traversal.rs:783-791` to avoid false sharing).
  - Free-function wrappers call the methods.

- [ ] **Step 4: Add `test_traversal()` to test_core.cpp** (tree-vs-direct equivalence, port of `gravity_tests.rs:accelerations_match_direct_small_n`)

```cpp
static void test_traversal() {
    using namespace gravity;
    // n=256, theta=0.0 forces full leaf traversal; must match direct to 1e-10.
    // Matches gravity_tests.rs config: positions in [-0.5,0.5], masses in [0.5,1.5], leaf_capacity=32.
    const size_t n = 256;
    std::vector<Vec3> pos(n);
    std::vector<double> mass(n);
    unsigned s = 12345u;
    auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (double)(s >> 8) / 16777216.0; };
    for (size_t i = 0; i < n; ++i) pos[i] = Vec3(rnd()-0.5, rnd()-0.5, rnd()-0.5);
    for (size_t i = 0; i < n; ++i) mass[i] = 0.5 + rnd();
    Octree tree = Octree::build(pos, mass.data(), 32, 2);
    std::vector<Vec3> acc_t(n), acc_d;
    tree.compute_accelerations(0.0, acc_t);
    acc_d = direct_accelerations(pos, mass.data());
    for (size_t i = 0; i < n; ++i) {
        CHECK_NEAR(acc_t[i].x, acc_d[i].x, 1e-10);
        CHECK_NEAR(acc_t[i].y, acc_d[i].y, 1e-10);
        CHECK_NEAR(acc_t[i].z, acc_d[i].z, 1e-10);
    }
    std::vector<double> pot_t(n), pot_d;
    tree.compute_potentials(0.0, pot_t);
    pot_d = direct_potentials(pos, mass.data());
    for (size_t i = 0; i < n; ++i) CHECK_NEAR(pot_t[i], pot_d[i], 1e-10);
}
```
Add `test_traversal();` to `main()` and `#include "gravity/traversal.hpp"`.

- [ ] **Step 5: Run the test**

Run: `make cpp-test`
Expected: `ALL PASS`.

- [ ] **Step 6: Commit**

```bash
git add cpp/gravity/octree.hpp cpp/gravity/traversal.hpp cpp/gravity/traversal.cpp cpp/gravity/tests/test_core.cpp
git commit -m "feat: port Barnes-Hut tree traversal with OpenMP"
```

---

## Task 9: C++ test invariants (translate vs direct, single-node far-field)

**Files:**
- Modify: `cpp/gravity/tests/test_core.cpp`

**Interfaces:**
- Consumes: `moment.hpp` (`MultipoleMoment`, `translate_multipole`), `derivatives.hpp`, `eval.hpp` (`gravity_potential_multipole`, `gravity_accel_multipole`).

- [ ] **Step 1: Add `test_translate_vs_direct()`** — port of `translate_multipole.rs::compare_translate_vs_direct` (same config: n=200, positions in [0,1]³, masses [0,1], centers `[0.3,0.4,0.5]`/`[0.8,-0.2,0.1]`, all 56 coefficients < 1e-10)

```cpp
static void test_translate_vs_direct() {
    using namespace gravity;
    unsigned s = 999u;
    auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (double)(s >> 8) / 16777216.0; };
    std::vector<Vec3> pos(200);
    for (auto& p : pos) p = Vec3(rnd(), rnd(), rnd());
    std::vector<double> mass(200);
    for (auto& m : mass) m = rnd();
    std::vector<size_t> idx(200);
    for (size_t i = 0; i < 200; ++i) idx[i] = i;
    Vec3 B(0.3, 0.4, 0.5), A(0.8, -0.2, 0.1);
    MultipoleMoment aboutB = MultipoleMoment::from_points(pos, mass.data(), idx, B, 5);
    MultipoleMoment translated = translate_multipole(aboutB, A - B, 5);
    MultipoleMoment aboutA = MultipoleMoment::from_points(pos, mass.data(), idx, A, 5);
    // MultipoleMoment holds exactly 56 doubles in field order, so compare the
    // layout as arrays (add a static_assert(sizeof(MultipoleMoment) == 56*sizeof(double))).
    const double* t = &translated.m000;
    const double* a = &aboutA.m000;
    for (int i = 0; i < 56; ++i) CHECK_NEAR(t[i], a[i], 1e-10);
}
```

- [ ] **Step 2: Add `test_single_node_far_field()`** — port of `single_node.rs::single_node_multipole_vs_direct` (true mass-weighted COM, direct particle-sum reference, derivative displacement `com - target`, p90 relative error < 1e-2 per order)

```cpp
static void test_single_node_far_field() {
    using namespace gravity;
    // 4000 particles in [-0.1,0.1]^3, masses in [0.1, 1.0]; 400 targets at r in [20,30].
    unsigned s = 7u;
    auto rnd = [&s]() { s = s*1664525u + 1013904223u; return (double)(s >> 8) / 16777216.0; };
    const size_t n = 4000;
    std::vector<Vec3> pos(n);
    std::vector<double> mass(n);
    for (size_t i = 0; i < n; ++i) pos[i] = Vec3(-0.1 + 0.2*rnd(), -0.1 + 0.2*rnd(), -0.1 + 0.2*rnd());
    for (size_t i = 0; i < n; ++i) mass[i] = 0.1 + 0.9*rnd();
    // True mass-weighted COM.
    Vec3 com(0,0,0); double mtot = 0.0;
    for (size_t i = 0; i < n; ++i) { com = com + pos[i]*mass[i]; mtot += mass[i]; }
    com.x /= mtot; com.y /= mtot; com.z /= mtot;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    MultipoleMoment m = MultipoleMoment::from_points(pos, mass.data(), idx, com, 5);

    auto direct_potential = [&](const Vec3& target) {
        double phi = 0.0;
        for (size_t i = 0; i < n; ++i) {
            Vec3 dvec = pos[i] - target;
            double r2 = norm2(dvec);
            if (r2 == 0.0) continue;
            phi += -mass[i] / std::sqrt(r2);
        }
        return phi;
    };

    std::vector<double> err_per_order[6];
    for (int q = 0; q < 400; ++q) {
        // random direction on the unit sphere
        Vec3 dir;
        do { dir = Vec3(2.0*rnd()-1.0, 2.0*rnd()-1.0, 2.0*rnd()-1.0); } while (norm2(dir) < 1e-6 || norm2(dir) > 1.0);
        double norm = std::sqrt(norm2(dir));
        double r = 20.0 + 10.0*rnd();
        Vec3 target = com + Vec3(dir.x/norm*r, dir.y/norm*r, dir.z/norm*r);

        double phi_direct = direct_potential(target);
        Vec3 dvec = com - target;   // source-minus-target, matching the treewalk
        PotentialDerivatives d = PotentialDerivatives::new_derivatives(dvec.x, dvec.y, dvec.z, 0.0, 5);
        for (int order = 0; order <= 5; ++order) {
            double phi_mp = gravity_potential_multipole(m, d, (unsigned char)order);
            double err = phi_direct != 0.0 ? std::fabs(phi_mp - phi_direct) / std::fabs(phi_direct)
                                           : std::fabs(phi_mp - phi_direct);
            err_per_order[order].push_back(err);
        }
    }
    // p90 of relative error per order must be < 1e-2 (matches the Rust assertion).
    for (int order = 0; order <= 5; ++order) {
        auto& errs = err_per_order[order];
        std::sort(errs.begin(), errs.end());
        size_t k = std::min((size_t)(errs.size() * 0.9), errs.size() - 1);
        CHECK(errs[k] < 1e-2);
    }
}
```
(Add `#include <algorithm>` at the top of test_core.cpp for `std::sort`.)

- [ ] **Step 3: Register both in `main()` and run**

Add `test_translate_vs_direct();` and `test_single_node_far_field();` to `main()`. Run `make cpp-test`. Expected: `ALL PASS`.

- [ ] **Step 4: Commit**

```bash
git add cpp/gravity/tests/test_core.cpp
git commit -m "test: add M2M translate and single-node far-field invariants"
```

---

## Task 10: Build wiring + minimal module

**Files:**
- Create: `cpp/bindings/module.cpp` (minimal for now)
- Modify: `setup.py`, `pyproject.toml`

**Interfaces:**
- Produces: a buildable `pynbodyext._native` extension (empty module with `__version__`) so the build pipeline is proven before the real API lands in Task 12.

- [ ] **Step 1: Rewrite `setup.py`** replacing `setuptools_rust` with `Pybind11Extension`

```python
from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).parent


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
```

Note: For MSVC the OpenMP flag is `/openmp` on the compile line only. `sources` must not include `cpp/gravity/tests/test_core.cpp`.

- [ ] **Step 2: Edit `pyproject.toml`**

- `[build-system] requires` → `["setuptools>=61", "pybind11>=2.12", "wheel"]`.
- Remove the entire `[tool.maturin]` section (lines 63-67).
- In `[project.optional-dependencies] dev` (line 76-82): remove `"maturin>=1.10"` and add `"pybind11>=2.12"`.

- [ ] **Step 3: Write minimal `cpp/bindings/module.cpp`**

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_native, m) {
    m.doc() = "pynbodyext native (C++) extension";
    m.attr("__version__") = "0.0.1";
}
```

- [ ] **Step 4: Install pybind11 and build the extension**

Run:
```bash
.venv/bin/pip install "pybind11>=2.12"
.venv/bin/pip install -e . --no-build-isolation
.venv/bin/python -c "import pynbodyext._native; print(pynbodyext._native.__version__)"
```
Expected: prints `0.0.1`. If `--no-build-isolation` is unavailable, fall back to a normal `pip install -e .` (build isolation will fetch pybind11 from PyPI).

- [ ] **Step 5: Regenerate `uv.lock`** (pyproject changed)

Run: `uv lock` (or `uv sync`). Expected: lock file updated with pybind11, without maturin/setuptools-rust. Commit the lockfile change.

- [ ] **Step 6: Commit**

```bash
git add setup.py pyproject.toml cpp/bindings/module.cpp uv.lock
git commit -m "build: replace setuptools-rust with pybind11 extension (pynbodyext._native)"
```

---

## Task 11: Rename Python module `_rust` → `_native`

**Files:**
- Modify: `pynbodyext/gravity/base.py`, `pynbodyext/util/deps.py`, `pynbodyext/gravity/__init__.py`

**Interfaces:**
- Consumes: extension built as `pynbodyext._native` (Task 10).
- Produces: `GRAVITY_NATIVE_AVAILABLE` flag; `from pynbodyext._native import (...)`.

- [ ] **Step 1: Edit `pynbodyext/util/deps.py`**

Replace (lines 4 and 14):
```python
__all__ = ["GRAVITY_NATIVE_AVAILABLE", "DASK_AVAILABLE", "module_available", "PYNBODY_VERSION"]
...
GRAVITY_NATIVE_AVAILABLE: bool = module_available("pynbodyext._native")
```

- [ ] **Step 2: Edit `pynbodyext/gravity/base.py`**

- Module docstring line 3-4: "over a C++ backend (`pynbodyext._native`)".
- Import (line 51): `from pynbodyext._native import (Octree as _Octree, direct_accelerations_at_points_py as _direct_accelerations_at_points_py, direct_accelerations_py as _direct_accelerations_py, direct_potentials_at_points_py as _direct_potentials_at_points_py, direct_potentials_py as _direct_potentials_py)`.
- ImportError message (line 60): "pynbodyext.gravity requires the C++ extension module `pynbodyext._native` to be built. Install pynbodyext with C++ enabled or build the native bindings with pip install -e ."

- [ ] **Step 3: Edit `pynbodyext/gravity/__init__.py`**

- Line 15: `from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE`
- Line 17: `__all__ = ["GRAVITY_NATIVE_AVAILABLE"]`
- Lines 24-30 warning text: "pynbodyext.gravity: C++ extension not available; gravity calculations will be unavailable."

- [ ] **Step 4: Grep for stragglers and verify**

Run:
```bash
grep -rn "_rust\|GRAVITY_RUST_AVAILABLE" pynbodyext/ --include="*.py"
```
Expected: no matches. Then smoke test:
```bash
.venv/bin/python -c "from pynbodyext.util.deps import GRAVITY_NATIVE_AVAILABLE; print(GRAVITY_NATIVE_AVAILABLE)"
```
Expected: prints `True`.

- [ ] **Step 5: Commit**

```bash
git add pynbodyext/gravity/base.py pynbodyext/util/deps.py pynbodyext/gravity/__init__.py
git commit -m "refactor: rename gravity extension module _rust -> _native"
```

---

## Task 12: Full pybind11 binding API

**Files:**
- Modify: `cpp/bindings/module.cpp`

**Interfaces:**
- Consumes: `octree.hpp` (`Octree`), `direct.hpp`, `kernel.hpp`, `common.hpp`.
- Produces: pybind11 `class_<Octree>` named `Octree` and 4 module functions with signatures identical to the Rust bindings (`crates/pynbodyext-rust/src/gravity.rs`):
  - `Octree(positions, masses=None, leaf_capacity=32, multipole_order=0, softenings=None, kernel=None)`
  - `Octree.build_mass(masses=None)`; `Octree.set_softenings(softenings=None)`; `Octree.set_kernel(kernel=None)`
  - `Octree.compute_accelerations(theta, threads=0) -> (N,3) float64`
  - `Octree.compute_potentials(theta, threads=0) -> (N,) float64`
  - `Octree.accelerations_at_points(points, theta, threads=0) -> (M,3) float64`
  - `Octree.potentials_at_points(points, theta, threads=0) -> (M,) float64`
  - `direct_accelerations_py(positions, masses=None, threads=0, softenings=None, kernel=None) -> (N,3)`
  - `direct_potentials_py(positions, masses=None, threads=0, softenings=None, kernel=None) -> (N,)`
  - `direct_accelerations_at_points_py(positions, targets, masses=None, threads=0, softenings=None, kernel=None) -> (M,3)`
  - `direct_potentials_at_points_py(positions, targets, masses=None, threads=0, softenings=None, kernel=None) -> (M,)`

- [ ] **Step 1: Write helpers** — copy the NumPy extraction and kernel-parsing logic from `crates/pynbodyext-rust/src/gravity.rs:33-101`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <optional>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include "gravity/octree.hpp"
#include "gravity/direct.hpp"
#include "gravity/kernel.hpp"
namespace py = pybind11;
using namespace gravity;

// forcecast + c_style guarantees contiguous float64: copies only when needed.
using PyArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::vector<Vec3> extract_vec3(const PyArr& arr, const char* name) {
    if (arr.ndim() != 2 || arr.shape(1) != 3)
        throw py::value_error(std::string(name) + " must be (N,3) float64 array");
    const double* d = arr.data();
    size_t n = (size_t)arr.shape(0);
    std::vector<Vec3> out; out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(Vec3(d[3*i], d[3*i+1], d[3*i+2]));
    return out;
}

static std::optional<KernelKind> parse_kernel_opt(std::optional<uint8_t> kernel) {
    if (!kernel.has_value()) return std::nullopt;
    switch (*kernel) {
        case 0: return KernelKind::Plummer;
        case 1: return KernelKind::CubicSplineW2;
        default: throw py::value_error("kernel must be 0 (Plummer) or 1 (CubicSplineW2)");
    }
}

static std::optional<std::vector<double>> copy_opt_arr(const std::optional<PyArr>& arr, const char* name, size_t n) {
    if (!arr.has_value()) return std::nullopt;
    if ((size_t)arr->shape(0) != n)
        throw py::value_error(std::string(name) + " must be length N");
    const double* d = arr->data();
    return std::vector<double>(d, d + n);
}

// Runs `f` with the GIL released and, if threads>0, an OpenMP team of exactly
// `threads` threads (mirrors rayon's dedicated pool).
template <typename F>
void with_gil_released_threads(py::gil_scoped_release& release, size_t threads, F&& f) {
    if (threads > 0) omp_set_num_threads((int)threads);
    f();
}
```

Note: `py::gil_scoped_release` must be constructed before the compute and destructed after. For a simple design, construct it inside each method: `py::gil_scoped_release release; ...compute...; release.release();` — see Step 3.

- [ ] **Step 2: Implement the `Octree` class**

```cpp
static py::array_t<double> make_n3(const std::vector<Vec3>& v) {
    py::array_t<double> a({(py::ssize_t)v.size(), (py::ssize_t)3});
    double* d = a.mutable_data();
    for (size_t i = 0; i < v.size(); ++i) { d[3*i] = v[i].x; d[3*i+1] = v[i].y; d[3*i+2] = v[i].z; }
    return a;
}

struct PyOctree {
    Octree inner;
    bool payload_built = false;

    PyOctree(PyArr positions, std::optional<PyArr> masses, size_t leaf_capacity,
             uint8_t multipole_order, std::optional<PyArr> softenings, std::optional<uint8_t> kernel)
        : inner(Octree::from_owned(extract_vec3(positions, "positions"),
                                   copy_opt_arr(masses, "masses", (size_t)positions.shape(0)),
                                   copy_opt_arr(softenings, "softenings", (size_t)positions.shape(0)),
                                   leaf_capacity, multipole_order,
                                   parse_kernel_opt(kernel).value_or(KernelKind::Plummer))) {
        if (kernel.has_value() && softenings.has_value())
            ; // ok
        else if (softenings.has_value() && !kernel.has_value())
            throw py::value_error("softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
        inner.build_mass_payload();
        payload_built = true;
    }
};
```

Port the Rust constructor's exact behavior (`gravity.rs:121-226`): validate lengths, the "softenings require an explicit kernel" check (note the Rust checks `kernel.is_none() && softenings_vec.is_some()` before parsing), clamp `multipole_order` internally (`from_owned`), and call `build_mass_payload()`.

Register it:

```cpp
py::class_<PyOctree>(m, "Octree")
    .def(py::init<PyArr, std::optional<PyArr>, size_t, uint8_t, std::optional<PyArr>, std::optional<uint8_t>>(),
         py::arg("positions"), py::arg("masses") = py::none(),
         py::arg("leaf_capacity") = 32, py::arg("multipole_order") = 0,
         py::arg("softenings") = py::none(), py::arg("kernel") = py::none())
    .def("build_mass", &PyOctree::build_mass, py::arg("masses") = py::none())
    .def("set_softenings", &PyOctree::set_softenings, py::arg("softenings") = py::none())
    .def("set_kernel", &PyOctree::set_kernel, py::arg("kernel") = py::none())
    .def("compute_accelerations", &PyOctree::compute_accelerations, py::arg("theta"), py::arg("threads") = 0)
    .def("compute_potentials", &PyOctree::compute_potentials, py::arg("theta"), py::arg("threads") = 0)
    .def("accelerations_at_points", &PyOctree::accelerations_at_points, py::arg("points"), py::arg("theta"), py::arg("threads") = 0)
    .def("potentials_at_points", &PyOctree::potentials_at_points, py::arg("points"), py::arg("theta"), py::arg("threads") = 0);
```

Methods:
```cpp
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
    { py::gil_scoped_release release; if (threads > 0) omp_set_num_threads((int)threads); gravity::compute_accelerations(inner, theta, out); }
    return make_n3(out);
}
py::array_t<double> compute_potentials(double theta, size_t threads) {
    require_payload();
    size_t n = inner.positions.size();
    std::vector<double> out(n);
    { py::gil_scoped_release release; if (threads > 0) omp_set_num_threads((int)threads); gravity::compute_potentials(inner, theta, out); }
    return py::array_t<double>({(py::ssize_t)n}, out.data());
}
py::array_t<double> accelerations_at_points(PyArr points, double theta, size_t threads) {
    require_payload();
    auto pts = extract_vec3(points, "points");
    std::vector<Vec3> out(pts.size());
    { py::gil_scoped_release release; if (threads > 0) omp_set_num_threads((int)threads); gravity::accelerations_at_points(inner, pts, theta, out); }
    return make_n3(out);
}
py::array_t<double> potentials_at_points(PyArr points, double theta, size_t threads) {
    require_payload();
    auto pts = extract_vec3(points, "points");
    std::vector<double> out(pts.size());
    { py::gil_scoped_release release; if (threads > 0) omp_set_num_threads((int)threads); gravity::potentials_at_points(inner, pts, theta, out); }
    return py::array_t<double>({(py::ssize_t)pts.size()}, out.data());
}
```

- [ ] **Step 3: Implement the 4 direct functions**

Port `gravity.rs:443-704`. Each: extract positions (and targets), copy masses/softenings, validate the "softenings require an explicit kernel" rule, parse kernel, then compute under GIL release with the threads handling, returning the array. Use `direct_accelerations`/`direct_accelerations_kernel`/etc. For example:

```cpp
py::array_t<double> direct_accelerations_py(PyArr positions, std::optional<PyArr> masses, size_t threads,
                                            std::optional<PyArr> softenings, std::optional<uint8_t> kernel) {
    auto pos = extract_vec3(positions, "positions");
    size_t n = pos.size();
    auto mv = copy_opt_arr(masses, "masses", n);
    auto sv = copy_opt_arr(softenings, "softenings", n);
    auto kk = parse_kernel_opt(kernel);
    if (kk.has_value() && sv.has_value()) ; else if (!kk.has_value() && sv.has_value())
        throw py::value_error("softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)");
    std::vector<Vec3> acc;
    { py::gil_scoped_release release; if (threads > 0) omp_set_num_threads((int)threads);
      acc = kk.has_value() ? direct_accelerations_kernel(pos, mv ? mv->data() : nullptr, sv ? sv->data() : nullptr, *kk)
                           : direct_accelerations(pos, mv ? mv->data() : nullptr); }
    return make_n3(acc);
}
```
Mirror this for the other 3 functions (potentials, at-points variants). The `masses=None` case passes `nullptr` (unit mass).

- [ ] **Step 4: Register the module**

```cpp
PYBIND11_MODULE(_native, m) {
    m.doc() = "pynbodyext native (C++) gravity extension";
    // ... py::class_ registrations and m.def(...) for the 4 direct functions ...
}
```

- [ ] **Step 5: Rebuild and smoke test**

Run:
```bash
.venv/bin/pip install -e . --no-build-isolation
.venv/bin/python -c "
import numpy as np
from pynbodyext._native import Octree, direct_potentials_py
pos = np.random.rand(100,3); mass = np.ones(100)
o = Octree(pos, mass, 8, 2)
a = o.compute_accelerations(0.0)
print(a.shape, float(np.max(np.abs(a))))
print(direct_potentials_py(pos, mass).shape)
"
```
Expected: `(100, 3)` printed and a finite max value; `(100,)` printed. Verify a tree-vs-direct rough agreement here (`o.compute_accelerations(0.0)` vs `direct_accelerations_py(pos, mass)`) — full test is Task 13.

- [ ] **Step 6: Commit**

```bash
git add cpp/bindings/module.cpp
git commit -m "feat: pybind11 binding for Octree and direct-sum gravity"
```

---

## Task 13: Python end-to-end tests

**Files:**
- Create: `tests/test_gravity.py`

**Interfaces:**
- Consumes: `pynbodyext.gravity.Gravity`, `KernelKind`; `pynbodyext._native` (indirect).

- [ ] **Step 1: Write `tests/test_gravity.py`**

```python
from __future__ import annotations

import numpy as np
import pytest

from pynbodyext.gravity import Gravity, KernelKind


@pytest.fixture
def particles():
    rng = np.random.default_rng(42)
    n = 256
    pos = rng.uniform(-1.0, 1.0, (n, 3))
    mass = 1.0 + rng.uniform(0.0, 1.0, n)
    return pos, mass


def test_tree_accelerations_match_direct(particles):
    pos, mass = particles
    g = Gravity(pos, mass, leaf_capacity=8, multipole_order=2)
    acc_tree = g.tree_accelerations(theta=0.0, threads=1)
    acc_direct = g.direct_accelerations(threads=1)
    np.testing.assert_allclose(acc_tree, acc_direct, atol=1e-10)


def test_tree_potentials_match_direct(particles):
    pos, mass = particles
    g = Gravity(pos, mass, leaf_capacity=8, multipole_order=2)
    pot_tree = g.tree_potentials(theta=0.0, threads=1)
    pot_direct = g.direct_potentials(threads=1)
    np.testing.assert_allclose(pot_tree, pot_direct, atol=1e-10)


def test_at_points_match_direct(particles):
    pos, mass = particles
    g = Gravity(pos, mass, leaf_capacity=8, multipole_order=2)
    rng = np.random.default_rng(7)
    targets = rng.uniform(-1.0, 1.0, (128, 3))
    np.testing.assert_allclose(
        g.tree_accelerations(positions=targets, theta=0.0, threads=1),
        g.direct_accelerations(positions=targets, threads=1),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        g.tree_potentials(positions=targets, theta=0.0, threads=1),
        g.direct_potentials(positions=targets, threads=1),
        atol=1e-10,
    )


def test_multipole_order_convergence_accel(particles):
    pos, mass = particles
    rng = np.random.default_rng(1)
    n = 800
    big_pos = rng.uniform(-1.0, 1.0, (n, 3))
    big_mass = 1.0 + rng.uniform(0.0, 1.0, n)
    g = Gravity(big_pos, big_mass, leaf_capacity=64)
    direct = g.direct_accelerations(threads=1)
    errs = []
    for order in (0, 3, 4, 5):
        acc = g.tree_accelerations(theta=0.7, leaf_capacity=64, multipole_order=order, threads=1)
        errs.append(np.sqrt(np.mean(np.sum((acc - direct) ** 2, axis=1))))
    assert errs == sorted(errs)  # non-increasing with order
    assert errs[-1] <= 0.8 * errs[0]  # order-5 at least 20% better than order-0


def test_multipole_order_convergence_potential(particles):
    pos, mass = particles
    rng = np.random.default_rng(1)
    n = 800
    big_pos = rng.uniform(-1.0, 1.0, (n, 3))
    big_mass = 1.0 + rng.uniform(0.0, 1.0, n)
    g = Gravity(big_pos, big_mass, leaf_capacity=64)
    direct = g.direct_potentials(threads=1)
    errs = []
    for order in (0, 2, 3, 4, 5):
        pot = g.tree_potentials(theta=0.7, leaf_capacity=64, multipole_order=order, threads=1)
        errs.append(np.sqrt(np.mean((pot - direct) ** 2)))
    assert errs == sorted(errs)
    assert errs[-1] <= 0.8 * errs[0]


def test_plummer_softening_matches_numpy(particles):
    pos, mass = particles
    eps = 0.05
    g = Gravity(pos, mass, softening=eps, kernel=KernelKind.Plummer, leaf_capacity=8, multipole_order=2)
    pot = g.direct_potentials(threads=1)
    # NumPy reference: phi_i = -sum_j m_j / sqrt(r_ij^2 + eps^2), excluding self.
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    inv = 1.0 / np.sqrt(r2 + eps**2)
    np.fill_diagonal(inv, 0.0)
    ref = -np.sum(mass[None, :] * inv, axis=1)
    np.testing.assert_allclose(pot, ref, atol=1e-12)


def test_spline_softening_matches_numpy(particles):
    # Port of the W2 kernel: for h small enough that most pairs are Newtonian,
    # still verify a softened pair exactly. Use two particles.
    pos = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    h = 0.2
    g = Gravity(pos, mass, softening=h, kernel=KernelKind.Spline)
    pot = g.direct_potentials(threads=1)
    # W2(u), u=r/h=1.5 -> -1/u = -1/1.5
    ref = np.array([-1.0 / 1.5, -1.0 / 1.5])
    np.testing.assert_allclose(pot, ref, atol=1e-12)
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_gravity.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gravity.py
git commit -m "test: add Python end-to-end gravity tests for C++ backend"
```

---

## Task 14: Remove all Rust artifacts

**Files:**
- Delete: `crates/` (whole tree), `Cargo.toml`, `Cargo.lock`
- Modify: `.pre-commit-config.yaml`, `.gitignore`

**Interfaces:**
- Consumes: the C++ port is fully validated by Tasks 1-13 (C++ tests `ALL PASS`, Python tests PASS, `pynbodyext._native` imported by `pynbodyext.gravity`).

- [ ] **Step 1: Verify the port is green before deletion**

Run: `make cpp-test` and `.venv/bin/python -m pytest tests/test_gravity.py -q`
Expected: `ALL PASS` and all tests pass.

- [ ] **Step 2: Delete Rust source and manifests**

```bash
git rm -r crates Cargo.toml Cargo.lock
```

- [ ] **Step 3: Remove the 6 cargo hooks from `.pre-commit-config.yaml`**

Delete the blocks `cargo-fmt-gravity`, `cargo-clippy-gravity`, `cargo-test-gravity`, `cargo-fmt-pynbodyext-rust`, `cargo-clippy-pynbodyext-rust`, `cargo-test-pynbodyext-rust` (lines 20-61), leaving only the `ruff-check` and `mypy` hooks.

- [ ] **Step 4: Clean `.gitignore`**

Remove any `target/` and `Cargo.lock` entries if present (check current `.gitignore`; the rust target dirs were untracked). If none, no change needed.

- [ ] **Step 5: Grep for residual Rust references**

```bash
grep -rn "cargo\|maturin\|setuptools.rust\|_rust\|GRAVITY_RUST_AVAILABLE\|crates/" --include="*.py" --include="*.toml" --include="*.yaml" --include="*.md" --include="*.json" . 2>/dev/null | grep -v "\.venv\|docs/superpowers\|\.git/"
```
Expected: no matches outside `docs/superpowers/` and the `.venv`.

- [ ] **Step 6: Full verification**

Run:
```bash
make cpp-test
.venv/bin/python -m pytest tests/ -q
```
Expected: `ALL PASS` and the full Python suite passes.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: remove Rust implementation and toolchain artifacts"
```

---

## Self-Review Notes

- **Spec coverage:** Layout (`cpp/gravity` + `cpp/bindings`) → Tasks 1-8, 10, 12. Custom `Vec3` → Task 1. `template<int Order>` monomorphization → Task 5/8. Drop FMM/solver/boundary/types → no task ports them. Numerics preservation → Global Constraints + every port task. OpenMP → Tasks 6/8/12. Build (setuptools+pybind11) → Task 10. Module rename → Task 11. Python tests → Task 13. C++ tests (translate + single-node) → Task 9. Removal (crates, Cargo, pre-commit cargo hooks, gitignore, maturin/setuptools-rust) → Tasks 10/14. Benchmark strategy ("port where meaningful or drop") → criterion benches dropped; ASV Python benchmark already targets the public API and needs no change.
- **Ordering dependency:** Task 8 adds method declarations to `octree.hpp` and defines them in `traversal.cpp` — the Makefile glob compiles all `cpp/gravity/*.cpp`, so linking resolves them. Task 10's `setup.py` globs the same way.
- **Type consistency:** `Vec3`, `R2_TINY`, `NO_INDEX`, `KernelKind`, `MultipoleMoments`, `MultipoleEval<Order>`, `Octree`, and the `direct_*` names are defined once (Tasks 1-7) and used consistently in Tasks 8-12.
