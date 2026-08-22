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
