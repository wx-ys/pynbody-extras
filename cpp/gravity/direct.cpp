#include "gravity/direct.hpp"
#include "gravity/common.hpp"
#include <cmath>
#include <cstddef>
#include <algorithm>

namespace gravity {

// ===========================================================================
// Generic direct-sum implementations
// ===========================================================================

// Self-gravity: compute quantity on each source particle.
//
// For small N (< 512) uses a symmetric pairwise loop (each pair computed once,
// both particles updated). For larger N uses a parallelised per-particle loop.
template <typename A>
std::vector<typename A::output_type> direct_self_impl(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, std::optional<KernelKind> kernel) {
    const std::size_t n = positions.size();
    if (n == 0) {
        return {};
    }

    std::vector<double> ones;
    const double* masses_slice = masses;
    if (masses_slice == nullptr) {
        ones.assign(n, 1.0);
        masses_slice = ones.data();
    }

    const bool has_softening = (softenings != nullptr) && kernel.has_value();
    const KernelKind kernel_kind = kernel.value_or(KernelKind::Plummer);
    const bool kernel_is_spline = kernel_kind == KernelKind::CubicSplineW2;

    if (n < 512) {
        // Symmetric pairwise loop — each (i,j) pair computed once.
        std::vector<A> accums(n);

        for (std::size_t i = 0; i < n; ++i) {
            const Vec3& pi = positions[i];
            const double mi = masses_slice[i];
            const double hi = softenings ? softenings[i] : 0.0;

            for (std::size_t j = i + 1; j < n; ++j) {
                const Vec3& pj = positions[j];
                const double mj = masses_slice[j];

                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double r2 = dx * dx + dy * dy + dz * dz;

                if (has_softening) {
                    const double hj = softenings ? softenings[j] : 0.0;
                    const double h = std::max(hi, hj);
                    if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                        const double invr = 1.0 / std::sqrt(r2 + R2_TINY);
                        accums[i].add_newton(dx, dy, dz, mj, invr);
                        accums[j].add_newton(-dx, -dy, -dz, mi, invr);
                    } else {
                        const double r = std::sqrt(r2 + R2_TINY);
                        accums[i].add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                        accums[j].add_softened(-dx, -dy, -dz, mi, r, h, kernel_kind);
                    }
                } else {
                    const double invr = 1.0 / std::sqrt(r2 + R2_TINY);
                    accums[i].add_newton(dx, dy, dz, mj, invr);
                    accums[j].add_newton(-dx, -dy, -dz, mi, invr);
                }
            }
        }

        std::vector<typename A::output_type> out;
        out.reserve(n);
        for (const A& a : accums) {
            out.push_back(a.finish());
        }
        return out;
    } else {
        // Parallel per-particle — each particle sums over all others.
        std::vector<typename A::output_type> out(n);
#pragma omp parallel for if(n >= 512) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            A acc;
            const Vec3& pi = positions[static_cast<std::size_t>(i)];
            const double hi = softenings ? softenings[static_cast<std::size_t>(i)] : 0.0;

            for (std::size_t j = 0; j < n; ++j) {
                if (j == static_cast<std::size_t>(i)) {
                    continue;
                }
                const Vec3& pj = positions[j];
                const double mj = masses_slice[j];

                const double dx = pj.x - pi.x;
                const double dy = pj.y - pi.y;
                const double dz = pj.z - pi.z;
                const double r2 = dx * dx + dy * dy + dz * dz;

                if (has_softening) {
                    const double hj = softenings ? softenings[j] : 0.0;
                    const double h = std::max(hi, hj);
                    if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                        const double invr = 1.0 / std::sqrt(r2 + R2_TINY);
                        acc.add_newton(dx, dy, dz, mj, invr);
                    } else {
                        const double r = std::sqrt(r2 + R2_TINY);
                        acc.add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                    }
                } else {
                    const double invr = 1.0 / std::sqrt(r2 + R2_TINY);
                    acc.add_newton(dx, dy, dz, mj, invr);
                }
            }

            out[static_cast<std::size_t>(i)] = acc.finish();
        }
        return out;
    }
}

// Evaluate gravity from source particles at arbitrary target positions.
template <typename A>
std::vector<typename A::output_type> direct_at_points_impl(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, const std::vector<Vec3>& targets,
    std::optional<KernelKind> kernel) {
    const std::size_t n_src = positions.size();
    const std::size_t n_tgt = targets.size();
    if (n_tgt == 0 || n_src == 0) {
        return {};
    }

    std::vector<double> ones;
    const double* masses_slice = masses;
    if (masses_slice == nullptr) {
        ones.assign(n_src, 1.0);
        masses_slice = ones.data();
    }

    const bool has_softening = (softenings != nullptr) && kernel.has_value();
    const KernelKind kernel_kind = kernel.value_or(KernelKind::Plummer);
    const bool kernel_is_spline = kernel_kind == KernelKind::CubicSplineW2;

    std::vector<typename A::output_type> out(n_tgt);
#pragma omp parallel for if(n_tgt >= 512)
    for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n_tgt); ++t) {
        A acc;
        const Vec3& tgt = targets[static_cast<std::size_t>(t)];
        const double tx = tgt.x;
        const double ty = tgt.y;
        const double tz = tgt.z;

        for (std::size_t j = 0; j < n_src; ++j) {
            const Vec3& pj = positions[j];
            const double mj = masses_slice[j];

            const double dx = pj.x - tx;
            const double dy = pj.y - ty;
            const double dz = pj.z - tz;
            const double r2 = dx * dx + dy * dy + dz * dz;

            if (has_softening) {
                const double hj = softenings ? softenings[j] : 0.0;
                const double h = std::max(hj, 0.0);
                if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                    const double invr = 1.0 / std::sqrt(r2 + R2_TINY);
                    acc.add_newton(dx, dy, dz, mj, invr);
                } else {
                    const double r = std::sqrt(r2 + R2_TINY);
                    acc.add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                }
            } else {
                const double invr = 1.0 / std::sqrt(r2 + R2_TINY);
                acc.add_newton(dx, dy, dz, mj, invr);
            }
        }

        out[static_cast<std::size_t>(t)] = acc.finish();
    }
    return out;
}

// ===========================================================================
// Public API — thin wrappers over the generic implementations
// ===========================================================================

std::vector<Vec3> direct_accelerations(const std::vector<Vec3>& positions, const double* masses) {
    return direct_self_impl<AccelAccumulator>(positions, masses, nullptr, std::nullopt);
}

std::vector<Vec3> direct_accelerations_at_points(
    const std::vector<Vec3>& positions, const double* masses, const std::vector<Vec3>& targets) {
    return direct_at_points_impl<AccelAccumulator>(positions, masses, nullptr, targets, std::nullopt);
}

std::vector<double> direct_potentials(const std::vector<Vec3>& positions, const double* masses) {
    return direct_self_impl<PotAccumulator>(positions, masses, nullptr, std::nullopt);
}

std::vector<double> direct_potentials_at_points(
    const std::vector<Vec3>& positions, const double* masses, const std::vector<Vec3>& targets) {
    return direct_at_points_impl<PotAccumulator>(positions, masses, nullptr, targets, std::nullopt);
}

std::vector<double> direct_potentials_kernel(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, KernelKind kernel) {
    return direct_self_impl<PotAccumulator>(positions, masses, softenings, std::optional<KernelKind>(kernel));
}

std::vector<Vec3> direct_accelerations_kernel(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, KernelKind kernel) {
    return direct_self_impl<AccelAccumulator>(positions, masses, softenings, std::optional<KernelKind>(kernel));
}

std::vector<double> direct_potentials_kernel_at_points(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, const std::vector<Vec3>& targets, KernelKind kernel) {
    return direct_at_points_impl<PotAccumulator>(positions, masses, softenings, targets, std::optional<KernelKind>(kernel));
}

std::vector<Vec3> direct_accelerations_kernel_at_points(
    const std::vector<Vec3>& positions, const double* masses,
    const double* softenings, const std::vector<Vec3>& targets, KernelKind kernel) {
    return direct_at_points_impl<AccelAccumulator>(positions, masses, softenings, targets, std::optional<KernelKind>(kernel));
}

// ===========================================================================
// Explicit template instantiations
// ===========================================================================

template std::vector<Vec3> direct_self_impl<AccelAccumulator>(const std::vector<Vec3>&, const double*, const double*, std::optional<KernelKind>);
template std::vector<double> direct_self_impl<PotAccumulator>(const std::vector<Vec3>&, const double*, const double*, std::optional<KernelKind>);
template std::vector<Vec3> direct_at_points_impl<AccelAccumulator>(const std::vector<Vec3>&, const double*, const double*, const std::vector<Vec3>&, std::optional<KernelKind>);
template std::vector<double> direct_at_points_impl<PotAccumulator>(const std::vector<Vec3>&, const double*, const double*, const std::vector<Vec3>&, std::optional<KernelKind>);

} // namespace gravity
