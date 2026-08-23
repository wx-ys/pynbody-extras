#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <optional>
#include "gravity/vec3.hpp"
#include "gravity/common.hpp"
#include "gravity/kernel.hpp"
#include "gravity/multipole/moment.hpp"

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
    CHECK(!gravity::timing_enabled()); // GRAVITY_TIMING unset in the test environment
}

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

int main() {
    test_vec3();
    test_common();
    test_kernel();
    test_moment();
    if (failures) { std::printf("%d FAILURE(S)\n", failures); return 1; }
    std::printf("ALL PASS\n");
    return 0;
}
