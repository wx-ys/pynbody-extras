#include "gravity/multipole/derivatives.hpp"
#include "gravity/common.hpp"
#include <cmath>

namespace gravity {

// ===========================================================================
// PotentialDerivatives1 — order-1 (monopole)
// ===========================================================================

PotentialDerivatives1 PotentialDerivatives1::new_derivatives(double dx, double dy, double dz, double eps2) {
    double r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
    double r = std::sqrt(r2);
    double r_inv = 1.0 / r;
    double dt_1 = r_inv;
    double dt_2 = -dt_1 * r_inv;
    double rx_r = dx * r_inv;
    double ry_r = dy * r_inv;
    double rz_r = dz * r_inv;
    return PotentialDerivatives1{
        dt_1,
        dt_2 * rx_r,
        dt_2 * ry_r,
        dt_2 * rz_r,
    };
}

// ===========================================================================
// PotentialDerivatives2 — order-2
// ===========================================================================

PotentialDerivatives2 PotentialDerivatives2::new_derivatives(double dx, double dy, double dz, double eps2) {
    double r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
    double r = std::sqrt(r2);
    double r_inv = 1.0 / r;
    double dt_1 = r_inv;
    double dt_2 = -dt_1 * r_inv;
    double dt_3 = -3.0 * dt_2 * r_inv;
    double rx_r = dx * r_inv;
    double ry_r = dy * r_inv;
    double rz_r = dz * r_inv;
    double rx_r2 = rx_r * rx_r;
    double ry_r2 = ry_r * ry_r;
    double rz_r2 = rz_r * rz_r;
    double d100 = dt_2 * rx_r;
    double d010 = dt_2 * ry_r;
    double d001 = dt_2 * rz_r;
    dt_2 *= r_inv;
    return PotentialDerivatives2{
        dt_1,
        d100,
        d010,
        d001,
        dt_3 * rx_r2 + dt_2,
        dt_3 * ry_r2 + dt_2,
        dt_3 * rz_r2 + dt_2,
        dt_3 * rx_r * ry_r,
        dt_3 * rx_r * rz_r,
        dt_3 * ry_r * rz_r,
    };
}

// ===========================================================================
// PotentialDerivatives3 — order-3
// ===========================================================================

PotentialDerivatives3 PotentialDerivatives3::new_derivatives(double dx, double dy, double dz, double eps2) {
    double r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
    double r = std::sqrt(r2);
    double r_inv = 1.0 / r;
    double dt_1 = r_inv;
    double dt_2 = -dt_1 * r_inv;
    double dt_3 = -3.0 * dt_2 * r_inv;
    double dt_4 = -5.0 * dt_3 * r_inv;
    double rx_r = dx * r_inv;
    double ry_r = dy * r_inv;
    double rz_r = dz * r_inv;
    double rx_r2 = rx_r * rx_r;
    double ry_r2 = ry_r * ry_r;
    double rz_r2 = rz_r * rz_r;
    double rx_r3 = rx_r2 * rx_r;
    double ry_r3 = ry_r2 * ry_r;
    double rz_r3 = rz_r2 * rz_r;
    double d100 = dt_2 * rx_r;
    double d010 = dt_2 * ry_r;
    double d001 = dt_2 * rz_r;
    dt_2 *= r_inv;
    double d200 = dt_3 * rx_r2 + dt_2;
    double d020 = dt_3 * ry_r2 + dt_2;
    double d002 = dt_3 * rz_r2 + dt_2;
    double d110 = dt_3 * rx_r * ry_r;
    double d101 = dt_3 * rx_r * rz_r;
    double d011 = dt_3 * ry_r * rz_r;
    dt_3 *= r_inv;
    return PotentialDerivatives3{
        dt_1,
        d100,
        d010,
        d001,
        d200,
        d020,
        d002,
        d110,
        d101,
        d011,
        dt_4 * rx_r3 + 3.0 * dt_3 * rx_r,
        dt_4 * ry_r3 + 3.0 * dt_3 * ry_r,
        dt_4 * rz_r3 + 3.0 * dt_3 * rz_r,
        dt_4 * rx_r2 * ry_r + dt_3 * ry_r,
        dt_4 * rx_r2 * rz_r + dt_3 * rz_r,
        dt_4 * ry_r2 * rx_r + dt_3 * rx_r,
        dt_4 * rz_r2 * rx_r + dt_3 * rx_r,
        dt_4 * ry_r2 * rz_r + dt_3 * rz_r,
        dt_4 * rz_r2 * ry_r + dt_3 * ry_r,
        dt_4 * rx_r * ry_r * rz_r,
    };
}

// ===========================================================================
// PotentialDerivatives4 — order-4 (delegates to full, truncated)
// ===========================================================================

PotentialDerivatives4 PotentialDerivatives4::new_derivatives(double dx, double dy, double dz, double eps2) {
    PotentialDerivatives d = PotentialDerivatives::new_derivatives(dx, dy, dz, eps2, 4);
    return PotentialDerivatives4{
        d.d000,
        d.d100,
        d.d010,
        d.d001,
        d.d200,
        d.d020,
        d.d002,
        d.d110,
        d.d101,
        d.d011,
        d.d300,
        d.d030,
        d.d003,
        d.d210,
        d.d201,
        d.d120,
        d.d102,
        d.d021,
        d.d012,
        d.d111,
        d.d400,
        d.d040,
        d.d004,
        d.d310,
        d.d301,
        d.d130,
        d.d103,
        d.d031,
        d.d013,
        d.d220,
        d.d202,
        d.d022,
        d.d211,
        d.d121,
        d.d112,
    };
}

// ===========================================================================
// PotentialDerivatives — full order-5
// ===========================================================================

PotentialDerivatives PotentialDerivatives::new_derivatives(double dx, double dy, double dz, double eps2, int order) {
    int max = order < 5 ? order : 5;
    double r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
    double r = std::sqrt(r2);
    double r_inv = 1.0 / r;

    double dt_1 = r_inv;
    double dt_2 = -dt_1 * r_inv;
    double dt_3 = -3.0 * dt_2 * r_inv;
    double dt_4 = -5.0 * dt_3 * r_inv;
    double dt_5 = -7.0 * dt_4 * r_inv;
    double dt_6 = -9.0 * dt_5 * r_inv;

    double rx_r = dx * r_inv;
    double ry_r = dy * r_inv;
    double rz_r = dz * r_inv;
    double rx_r2 = rx_r * rx_r;
    double ry_r2 = ry_r * ry_r;
    double rz_r2 = rz_r * rz_r;
    double rx_r3 = rx_r2 * rx_r;
    double ry_r3 = ry_r2 * ry_r;
    double rz_r3 = rz_r2 * rz_r;
    double rx_r4 = rx_r3 * rx_r;
    double ry_r4 = ry_r3 * ry_r;
    double rz_r4 = rz_r3 * rz_r;
    double rx_r5 = rx_r4 * rx_r;
    double ry_r5 = ry_r4 * ry_r;
    double rz_r5 = rz_r4 * rz_r;

    PotentialDerivatives d{};
    d.d000 = dt_1;
    if (max == 0) {
        return d;
    }

    d.d100 = dt_2 * rx_r;
    d.d010 = dt_2 * ry_r;
    d.d001 = dt_2 * rz_r;
    if (max == 1) {
        return d;
    }

    dt_2 *= r_inv;
    d.d200 = dt_3 * rx_r2 + dt_2;
    d.d020 = dt_3 * ry_r2 + dt_2;
    d.d002 = dt_3 * rz_r2 + dt_2;
    d.d110 = dt_3 * rx_r * ry_r;
    d.d101 = dt_3 * rx_r * rz_r;
    d.d011 = dt_3 * ry_r * rz_r;
    if (max == 2) {
        return d;
    }

    dt_3 *= r_inv;
    d.d300 = dt_4 * rx_r3 + 3.0 * dt_3 * rx_r;
    d.d030 = dt_4 * ry_r3 + 3.0 * dt_3 * ry_r;
    d.d003 = dt_4 * rz_r3 + 3.0 * dt_3 * rz_r;
    d.d210 = dt_4 * rx_r2 * ry_r + dt_3 * ry_r;
    d.d201 = dt_4 * rx_r2 * rz_r + dt_3 * rz_r;
    d.d120 = dt_4 * ry_r2 * rx_r + dt_3 * rx_r;
    d.d102 = dt_4 * rz_r2 * rx_r + dt_3 * rx_r;
    d.d021 = dt_4 * ry_r2 * rz_r + dt_3 * rz_r;
    d.d012 = dt_4 * rz_r2 * ry_r + dt_3 * ry_r;
    d.d111 = dt_4 * rx_r * ry_r * rz_r;
    if (max == 3) {
        return d;
    }

    dt_3 *= r_inv;
    dt_4 *= r_inv;
    d.d400 = dt_5 * rx_r4 + 6.0 * dt_4 * rx_r2 + 3.0 * dt_3;
    d.d040 = dt_5 * ry_r4 + 6.0 * dt_4 * ry_r2 + 3.0 * dt_3;
    d.d004 = dt_5 * rz_r4 + 6.0 * dt_4 * rz_r2 + 3.0 * dt_3;
    d.d310 = dt_5 * rx_r3 * ry_r + 3.0 * dt_4 * rx_r * ry_r;
    d.d301 = dt_5 * rx_r3 * rz_r + 3.0 * dt_4 * rx_r * rz_r;
    d.d130 = dt_5 * ry_r3 * rx_r + 3.0 * dt_4 * ry_r * rx_r;
    d.d103 = dt_5 * rz_r3 * rx_r + 3.0 * dt_4 * rx_r * rz_r;
    d.d031 = dt_5 * ry_r3 * rz_r + 3.0 * dt_4 * rz_r * ry_r;
    d.d013 = dt_5 * rz_r3 * ry_r + 3.0 * dt_4 * rz_r * ry_r;
    d.d220 = dt_5 * rx_r2 * ry_r2 + dt_4 * (rx_r2 + ry_r2) + dt_3;
    d.d202 = dt_5 * rx_r2 * rz_r2 + dt_4 * (rx_r2 + rz_r2) + dt_3;
    d.d022 = dt_5 * ry_r2 * rz_r2 + dt_4 * (ry_r2 + rz_r2) + dt_3;
    d.d211 = dt_5 * rx_r2 * ry_r * rz_r + dt_4 * ry_r * rz_r;
    d.d121 = dt_5 * ry_r2 * rx_r * rz_r + dt_4 * rx_r * rz_r;
    d.d112 = dt_5 * rz_r2 * rx_r * ry_r + dt_4 * rx_r * ry_r;
    if (max == 4) {
        return d;
    }

    dt_4 *= r_inv;
    dt_5 *= r_inv;
    d.d500 = dt_6 * rx_r5 + 10.0 * dt_5 * rx_r3 + 15.0 * dt_4 * rx_r;
    d.d050 = dt_6 * ry_r5 + 10.0 * dt_5 * ry_r3 + 15.0 * dt_4 * ry_r;
    d.d005 = dt_6 * rz_r5 + 10.0 * dt_5 * rz_r3 + 15.0 * dt_4 * rz_r;
    d.d410 = dt_6 * rx_r4 * ry_r + 6.0 * dt_5 * rx_r2 * ry_r + 3.0 * dt_4 * ry_r;
    d.d401 = dt_6 * rx_r4 * rz_r + 6.0 * dt_5 * rx_r2 * rz_r + 3.0 * dt_4 * rz_r;
    d.d140 = dt_6 * ry_r4 * rx_r + 6.0 * dt_5 * ry_r2 * rx_r + 3.0 * dt_4 * rx_r;
    d.d041 = dt_6 * ry_r4 * rz_r + 6.0 * dt_5 * ry_r2 * rz_r + 3.0 * dt_4 * rz_r;
    d.d104 = dt_6 * rz_r4 * rx_r + 6.0 * dt_5 * rz_r2 * rx_r + 3.0 * dt_4 * rx_r;
    d.d014 = dt_6 * rz_r4 * ry_r + 6.0 * dt_5 * rz_r2 * ry_r + 3.0 * dt_4 * ry_r;
    d.d320 = dt_6 * rx_r3 * ry_r2 + dt_5 * rx_r3 + 3.0 * dt_5 * rx_r * ry_r2 + 3.0 * dt_4 * rx_r;
    d.d302 = dt_6 * rx_r3 * rz_r2 + dt_5 * rx_r3 + 3.0 * dt_5 * rx_r * rz_r2 + 3.0 * dt_4 * rx_r;
    d.d230 = dt_6 * ry_r3 * rx_r2 + dt_5 * ry_r3 + 3.0 * dt_5 * ry_r * rx_r2 + 3.0 * dt_4 * ry_r;
    d.d032 = dt_6 * ry_r3 * rz_r2 + dt_5 * ry_r3 + 3.0 * dt_5 * ry_r * rz_r2 + 3.0 * dt_4 * ry_r;
    d.d203 = dt_6 * rz_r3 * rx_r2 + dt_5 * rz_r3 + 3.0 * dt_5 * rz_r * rx_r2 + 3.0 * dt_4 * rz_r;
    d.d023 = dt_6 * rz_r3 * ry_r2 + dt_5 * rz_r3 + 3.0 * dt_5 * rz_r * ry_r2 + 3.0 * dt_4 * rz_r;
    d.d311 = dt_6 * rx_r3 * ry_r * rz_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
    d.d131 = dt_6 * ry_r3 * rx_r * rz_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
    d.d113 = dt_6 * rz_r3 * rx_r * ry_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
    d.d122 = dt_6 * rx_r * ry_r2 * rz_r2 + dt_5 * rx_r * ry_r2 + dt_5 * rx_r * rz_r2 + dt_4 * rx_r;
    d.d212 = dt_6 * ry_r * rx_r2 * rz_r2 + dt_5 * ry_r * rx_r2 + dt_5 * ry_r * rz_r2 + dt_4 * ry_r;
    d.d221 = dt_6 * rz_r * rx_r2 * ry_r2 + dt_5 * rz_r * rx_r2 + dt_5 * rz_r * ry_r2 + dt_4 * rz_r;

    return d;
}

} // namespace gravity
