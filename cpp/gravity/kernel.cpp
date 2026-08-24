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
