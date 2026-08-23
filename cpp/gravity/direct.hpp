#pragma once
#include <vector>
#include <optional>
#include "gravity/vec3.hpp"
#include "gravity/kernel.hpp"

namespace gravity {

// ===========================================================================
// Accumulators — allow a single generic loop for both accel & potential
// ===========================================================================

/// Accumulator for gravitational acceleration [ax, ay, az].
struct AccelAccumulator {
    double ax = 0.0, ay = 0.0, az = 0.0;
    using output_type = Vec3;

    // g = m / r^3 -> g = m * invr^3
    void add_newton(double dx, double dy, double dz, double m, double invr) {
        double invr2 = invr * invr;
        double invr3 = invr2 * invr;
        double g = m * invr3;
        ax += dx * g;
        ay += dy * g;
        az += dz * g;
    }

    void add_softened(double dx, double dy, double dz, double m, double r, double h, KernelKind k) {
        double g = m * kernel_accel_factor(k, r, h);
        ax += dx * g;
        ay += dy * g;
        az += dz * g;
    }

    Vec3 finish() const { return Vec3(ax, ay, az); }
};

/// Accumulator for gravitational potential phi.
struct PotAccumulator {
    double phi = 0.0;
    using output_type = double;

    void add_newton(double /*dx*/, double /*dy*/, double /*dz*/, double m, double invr) {
        phi += -m * invr;
    }

    void add_softened(double /*dx*/, double /*dy*/, double /*dz*/, double m, double r, double h, KernelKind k) {
        phi += m * kernel_potential_per_unit_mass(k, r, h);
    }

    double finish() const { return phi; }
};

// ===========================================================================
// Generic direct-sum implementations (defined in direct.cpp)
// ===========================================================================

template <typename A>
std::vector<typename A::output_type> direct_self_impl(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, std::optional<KernelKind> kernel);

template <typename A>
std::vector<typename A::output_type> direct_at_points_impl(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, const std::vector<Vec3>& targets,
    std::optional<KernelKind> kernel);

// ===========================================================================
// Public API — thin wrappers over the generic implementations
// ===========================================================================

// Direct-sum O(N^2) gravitational accelerations (Newtonian, no softening).
std::vector<Vec3> direct_accelerations(const std::vector<Vec3>& positions, const double* masses);

// Direct-sum O(N^2) gravitational accelerations at arbitrary target points.
std::vector<Vec3> direct_accelerations_at_points(
    const std::vector<Vec3>& positions, const double* masses, const std::vector<Vec3>& targets);

// Direct-sum O(N^2) gravitational potentials (Newtonian, no softening).
std::vector<double> direct_potentials(const std::vector<Vec3>& positions, const double* masses);

// Direct-sum O(N^2) gravitational potentials at arbitrary target points.
std::vector<double> direct_potentials_at_points(
    const std::vector<Vec3>& positions, const double* masses, const std::vector<Vec3>& targets);

// Direct-sum O(N^2) gravitational potentials with softening kernel.
std::vector<double> direct_potentials_kernel(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, KernelKind kernel);

// Direct-sum O(N^2) gravitational accelerations with softening kernel.
std::vector<Vec3> direct_accelerations_kernel(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, KernelKind kernel);

// Direct-sum softened gravitational potentials at arbitrary target points.
std::vector<double> direct_potentials_kernel_at_points(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, const std::vector<Vec3>& targets, KernelKind kernel);

// Direct-sum softened gravitational accelerations at arbitrary target points.
std::vector<Vec3> direct_accelerations_kernel_at_points(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, const std::vector<Vec3>& targets, KernelKind kernel);

} // namespace gravity
