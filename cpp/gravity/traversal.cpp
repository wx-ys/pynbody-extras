#include "gravity/traversal.hpp"

#include <algorithm>
#include <cmath>
#include <variant>

#include "gravity/common.hpp"

namespace gravity {
namespace detail {

// ---------------------------------------------------------------------------
// Softening helper for node-level opening criterion (port of traversal.rs:56-77)
// ---------------------------------------------------------------------------

static bool node_soft_ok(size_t idx, double dist2, std::optional<double> target_h_opt,
                         const TraversalCtx& ctx) {
    const double* hmax = ctx.hmax;
    if (hmax == nullptr) {
        return true;
    }

    double h = std::max(hmax[idx], MIN_SOFTENING);
    if (target_h_opt.has_value()) {
        h = std::max(h, std::max(*target_h_opt, MIN_SOFTENING));
    }
    if (h <= 0.0) {
        return true;
    }
    double c = multipole_min_separation_factor(ctx.kernel);
    double ch = c * h;
    return dist2 > ch * ch;
}

// ---------------------------------------------------------------------------
// Leaf-level direct summation (port of traversal.rs:84-246)
// ---------------------------------------------------------------------------

void leaf_potential_sum(const LeafArgs& leaf, const TargetArgs& targ, double& out) {
    const std::vector<Vec3>& positions = *leaf.positions;
    const std::vector<size_t>& indices = *leaf.indices;
    const double* masses = leaf.masses;
    const double* softenings = leaf.softenings;

    const Vec3& target = *targ.target;
    std::optional<size_t> skip_self = targ.skip_self;
    std::optional<double> target_h_opt = targ.target_h_opt;
    KernelKind kernel = targ.kernel;

    double tx = target.x;
    double ty = target.y;
    double tz = target.z;

    size_t skip = skip_self.value_or(NO_INDEX);

    double target_h = std::max(target_h_opt.value_or(MIN_SOFTENING), MIN_SOFTENING);
    bool use_softening = (softenings != nullptr) || (target_h > 0.0);

    // Fast path: masses present + constant target softening (no per-particle softenings).
    if (use_softening) {
        if (masses != nullptr && softenings == nullptr) {
            double h = target_h;
            if (h <= 0.0) {
                // fall through to no-softening logic below
            } else if (kernel == KernelKind::CubicSplineW2) {
                double hh = h * h;
                for (size_t pi : indices) {
                    if (pi == skip) {
                        continue;
                    }
                    const Vec3& p = positions[pi];
                    double ddx = p.x - tx;
                    double ddy = p.y - ty;
                    double ddz = p.z - tz;
                    double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                    double m = masses[pi];

                    if (r2 >= hh) {
                        out += -m * inv_r_from_r2(r2);
                    } else {
                        double r = std::sqrt(r2 + R2_TINY);
                        out += m * kernel_potential_per_unit_mass(kernel, r, h);
                    }
                }
                return;
            } else {
                for (size_t pi : indices) {
                    if (pi == skip) {
                        continue;
                    }
                    const Vec3& p = positions[pi];
                    double ddx = p.x - tx;
                    double ddy = p.y - ty;
                    double ddz = p.z - tz;
                    double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                    double r = std::sqrt(r2 + R2_TINY);
                    double m = masses[pi];
                    out += m * kernel_potential_per_unit_mass(kernel, r, h);
                }
                return;
            }
        }
    }

    if (!use_softening) {
        if (masses != nullptr) {
            for (size_t pi : indices) {
                if (pi == skip) {
                    continue;
                }
                const Vec3& p = positions[pi];
                double ddx = p.x - tx;
                double ddy = p.y - ty;
                double ddz = p.z - tz;
                double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                double inv_r = inv_r_from_r2(r2);
                double m = masses[pi];
                out += -m * inv_r;
            }
        } else {
            for (size_t pi : indices) {
                if (pi == skip) {
                    continue;
                }
                const Vec3& p = positions[pi];
                double ddx = p.x - tx;
                double ddy = p.y - ty;
                double ddz = p.z - tz;
                double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                double inv_r = inv_r_from_r2(r2);
                out += -inv_r;
            }
        }
        return;
    }

    bool kernel_is_spline = (kernel == KernelKind::CubicSplineW2);

    if (softenings != nullptr) {
        const double* hs = softenings;
        for (size_t pi : indices) {
            if (pi == skip) {
                continue;
            }
            const Vec3& p = positions[pi];
            double ddx = p.x - tx;
            double ddy = p.y - ty;
            double ddz = p.z - tz;
            double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            double m = (masses != nullptr) ? masses[pi] : 1.0;

            double hi = std::max(hs[pi], MIN_SOFTENING);
            double h = std::max(hi, target_h);

            if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                out += -m * inv_r_from_r2(r2);
            } else {
                double r = std::sqrt(r2 + R2_TINY);
                out += m * kernel_potential_per_unit_mass(kernel, r, h);
            }
        }
    } else {
        double h = target_h;
        for (size_t pi : indices) {
            if (pi == skip) {
                continue;
            }
            const Vec3& p = positions[pi];
            double ddx = p.x - tx;
            double ddy = p.y - ty;
            double ddz = p.z - tz;
            double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            double m = (masses != nullptr) ? masses[pi] : 1.0;

            if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                out += -m * inv_r_from_r2(r2);
            } else {
                double r = std::sqrt(r2 + R2_TINY);
                out += m * kernel_potential_per_unit_mass(kernel, r, h);
            }
        }
    }
}

void leaf_acceleration_sum(const LeafArgs& leaf, const TargetArgs& targ, Vec3& out) {
    const std::vector<Vec3>& positions = *leaf.positions;
    const std::vector<size_t>& indices = *leaf.indices;
    const double* masses = leaf.masses;
    const double* softenings = leaf.softenings;

    const Vec3& target = *targ.target;
    std::optional<size_t> skip_self = targ.skip_self;
    std::optional<double> target_h_opt = targ.target_h_opt;
    KernelKind kernel = targ.kernel;

    double tx = target.x;
    double ty = target.y;
    double tz = target.z;

    size_t skip = skip_self.value_or(NO_INDEX);

    double target_h = std::max(target_h_opt.value_or(MIN_SOFTENING), MIN_SOFTENING);
    bool use_softening = (softenings != nullptr) || (target_h > 0.0);

    if (!use_softening) {
        if (masses != nullptr) {
            for (size_t pi : indices) {
                if (pi == skip) {
                    continue;
                }
                const Vec3& p = positions[pi];
                double ddx = p.x - tx;
                double ddy = p.y - ty;
                double ddz = p.z - tz;
                double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                double inv_r, inv_r3;
                inv_r_and_inv_r3_from_r2(r2, inv_r, inv_r3);
                double m = masses[pi];
                out += Vec3(m * ddx * inv_r3, m * ddy * inv_r3, m * ddz * inv_r3);
            }
        } else {
            for (size_t pi : indices) {
                if (pi == skip) {
                    continue;
                }
                const Vec3& p = positions[pi];
                double ddx = p.x - tx;
                double ddy = p.y - ty;
                double ddz = p.z - tz;
                double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                double inv_r, inv_r3;
                inv_r_and_inv_r3_from_r2(r2, inv_r, inv_r3);
                out += Vec3(ddx * inv_r3, ddy * inv_r3, ddz * inv_r3);
            }
        }
        return;
    }

    bool kernel_is_spline = (kernel == KernelKind::CubicSplineW2);

    if (softenings != nullptr) {
        const double* hs = softenings;
        for (size_t pi : indices) {
            if (pi == skip) {
                continue;
            }
            const Vec3& p = positions[pi];
            double ddx = p.x - tx;
            double ddy = p.y - ty;
            double ddz = p.z - tz;
            double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            double m = (masses != nullptr) ? masses[pi] : 1.0;

            double hi = std::max(hs[pi], MIN_SOFTENING);
            double h = std::max(hi, target_h);

            if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                double inv_r, inv_r3;
                inv_r_and_inv_r3_from_r2(r2, inv_r, inv_r3);
                out += Vec3(m * ddx * inv_r3, m * ddy * inv_r3, m * ddz * inv_r3);
            } else {
                double r = std::sqrt(r2 + R2_TINY);
                double g = kernel_accel_factor(kernel, r, h);
                out += Vec3(m * ddx * g, m * ddy * g, m * ddz * g);
            }
        }
    } else {
        double h = target_h;
        for (size_t pi : indices) {
            if (pi == skip) {
                continue;
            }
            const Vec3& p = positions[pi];
            double ddx = p.x - tx;
            double ddy = p.y - ty;
            double ddz = p.z - tz;
            double r2 = ddx * ddx + ddy * ddy + ddz * ddz;
            double m = (masses != nullptr) ? masses[pi] : 1.0;

            if (h <= 0.0 || (kernel_is_spline && r2 >= h * h)) {
                double inv_r, inv_r3;
                inv_r_and_inv_r3_from_r2(r2, inv_r, inv_r3);
                out += Vec3(m * ddx * inv_r3, m * ddy * inv_r3, m * ddz * inv_r3);
            } else {
                double r = std::sqrt(r2 + R2_TINY);
                double g = kernel_accel_factor(kernel, r, h);
                out += Vec3(m * ddx * g, m * ddy * g, m * ddz * g);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Core treewalk (no multipoles) — port of traversal.rs:387-519
// ---------------------------------------------------------------------------

static void potential_traversal_no_multipoles(const Octree& self, const Vec3& target,
                                              std::optional<size_t> skip_self,
                                              std::optional<double> target_h_opt,
                                              size_t node_idx, double& out,
                                              const TraversalCtx& ctx) {
    double tx = target.x;
    double ty = target.y;
    double tz = target.z;
    bool softening_enabled = (ctx.hmax != nullptr) || target_h_opt.has_value();
    size_t idx = node_idx;
    while (idx != NO_INDEX) {
        const NodeBh& node_bh = (*ctx.bh)[idx];
        if (node_bh.mass == 0.0) {
            idx = self.next_branch[idx];
            continue;
        }
        const Node& node = self.nodes[idx];
        if (!node.children.has_value()) {
            LeafArgs leaf{&self.positions, &node.indices, ctx.masses, ctx.softenings};
            TargetArgs targ{&target, skip_self, target_h_opt, ctx.kernel};
            leaf_potential_sum(leaf, targ, out);
            idx = self.next_branch[idx];
            continue;
        }
        double dx = node_bh.com.x - tx, dy = node_bh.com.y - ty, dz = node_bh.com.z - tz;
        double dist2 = dx * dx + dy * dy + dz * dz + ctx.multipole_eps2;
        bool soft_ok = softening_enabled ? node_soft_ok(idx, dist2, target_h_opt, ctx) : true;
        if (soft_ok && node.size2 < ctx.theta2 * dist2) {
            out += -node_bh.mass * inv_r_from_r2(dist2);
            idx = self.next_branch[idx];
        } else {
            idx = self.first_subnode[idx];
        }
    }
}

static void acceleration_traversal_no_multipoles(const Octree& self, const Vec3& target,
                                                 std::optional<size_t> skip_self,
                                                 std::optional<double> target_h_opt,
                                                 size_t node_idx, Vec3& out,
                                                 const TraversalCtx& ctx) {
    double tx = target.x;
    double ty = target.y;
    double tz = target.z;
    bool softening_enabled = (ctx.hmax != nullptr) || target_h_opt.has_value();
    size_t idx = node_idx;
    while (idx != NO_INDEX) {
        const NodeBh& node_bh = (*ctx.bh)[idx];
        if (node_bh.mass == 0.0) {
            idx = self.next_branch[idx];
            continue;
        }
        const Node& node = self.nodes[idx];
        if (!node.children.has_value()) {
            LeafArgs leaf{&self.positions, &node.indices, ctx.masses, ctx.softenings};
            TargetArgs targ{&target, skip_self, target_h_opt, ctx.kernel};
            leaf_acceleration_sum(leaf, targ, out);
            idx = self.next_branch[idx];
            continue;
        }
        double dx = node_bh.com.x - tx, dy = node_bh.com.y - ty, dz = node_bh.com.z - tz;
        double dist2 = dx * dx + dy * dy + dz * dz + ctx.multipole_eps2;
        bool soft_ok = softening_enabled ? node_soft_ok(idx, dist2, target_h_opt, ctx) : true;
        if (soft_ok && node.size2 < ctx.theta2 * dist2) {
            double inv_r = inv_r_from_r2(dist2);
            double inv_r2 = inv_r * inv_r;
            double inv_r3 = inv_r2 * inv_r;
            out += Vec3(node_bh.mass * dx * inv_r3, node_bh.mass * dy * inv_r3,
                        node_bh.mass * dz * inv_r3);
            idx = self.next_branch[idx];
        } else {
            idx = self.first_subnode[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Core treewalk (with multipoles) — port of traversal.rs:523-657
// ---------------------------------------------------------------------------

template <int Order>
void potential_traversal_with_multipoles(
    const Octree& self, const Vec3& target, std::optional<size_t> skip_self,
    std::optional<double> target_h_opt, size_t node_idx, double& out, const TraversalCtx& ctx,
    const std::vector<typename MultipoleEval<Order>::Moment>& multipoles) {
    double tx = target.x;
    double ty = target.y;
    double tz = target.z;
    bool softening_enabled = (ctx.hmax != nullptr) || target_h_opt.has_value();
    size_t idx = node_idx;
    while (idx != NO_INDEX) {
        const NodeBh& node_bh = (*ctx.bh)[idx];
        if (node_bh.mass == 0.0) {
            idx = self.next_branch[idx];
            continue;
        }
        const Node& node = self.nodes[idx];
        if (!node.children.has_value()) {
            LeafArgs leaf{&self.positions, &node.indices, ctx.masses, ctx.softenings};
            TargetArgs targ{&target, skip_self, target_h_opt, ctx.kernel};
            leaf_potential_sum(leaf, targ, out);
            idx = self.next_branch[idx];
            continue;
        }
        double dx = node_bh.com.x - tx, dy = node_bh.com.y - ty, dz = node_bh.com.z - tz;
        double dist2 = dx * dx + dy * dy + dz * dz + ctx.multipole_eps2;
        bool soft_ok = softening_enabled ? node_soft_ok(idx, dist2, target_h_opt, ctx) : true;
        if (soft_ok && node.size2 < ctx.theta2 * dist2) {
            auto d = MultipoleEval<Order>::derivatives(dx, dy, dz, ctx.multipole_eps2);
            out += MultipoleEval<Order>::potential(multipoles[idx], d);
            idx = self.next_branch[idx];
        } else {
            idx = self.first_subnode[idx];
        }
    }
}

template <int Order>
void acceleration_traversal_with_multipoles(
    const Octree& self, const Vec3& target, std::optional<size_t> skip_self,
    std::optional<double> target_h_opt, size_t node_idx, Vec3& out, const TraversalCtx& ctx,
    const std::vector<typename MultipoleEval<Order>::Moment>& multipoles) {
    double tx = target.x;
    double ty = target.y;
    double tz = target.z;
    bool softening_enabled = (ctx.hmax != nullptr) || target_h_opt.has_value();
    size_t idx = node_idx;
    while (idx != NO_INDEX) {
        const NodeBh& node_bh = (*ctx.bh)[idx];
        if (node_bh.mass == 0.0) {
            idx = self.next_branch[idx];
            continue;
        }
        const Node& node = self.nodes[idx];
        if (!node.children.has_value()) {
            LeafArgs leaf{&self.positions, &node.indices, ctx.masses, ctx.softenings};
            TargetArgs targ{&target, skip_self, target_h_opt, ctx.kernel};
            leaf_acceleration_sum(leaf, targ, out);
            idx = self.next_branch[idx];
            continue;
        }
        double dx = node_bh.com.x - tx, dy = node_bh.com.y - ty, dz = node_bh.com.z - tz;
        double dist2 = dx * dx + dy * dy + dz * dz + ctx.multipole_eps2;
        bool soft_ok = softening_enabled ? node_soft_ok(idx, dist2, target_h_opt, ctx) : true;
        if (soft_ok && node.size2 < ctx.theta2 * dist2) {
            auto d = MultipoleEval<Order>::derivatives(dx, dy, dz, ctx.multipole_eps2);
            out += MultipoleEval<Order>::acceleration(multipoles[idx], d);
            idx = self.next_branch[idx];
        } else {
            idx = self.first_subnode[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch entry points (one dispatch per traversal, not per node)
// port of traversal.rs:661-723
// ---------------------------------------------------------------------------

namespace {

struct PotentialMultipoleVisitor {
    const Octree& self;
    const Vec3& target;
    std::optional<size_t> skip_self;
    std::optional<double> target_h_opt;
    size_t node_idx;
    double& out;
    const TraversalCtx& ctx;

    void operator()(const std::vector<Moment0>& m) const {
        potential_traversal_with_multipoles<0>(self, target, skip_self, target_h_opt, node_idx,
                                               out, ctx, m);
    }
    void operator()(const std::vector<Moment2>& m) const {
        potential_traversal_with_multipoles<2>(self, target, skip_self, target_h_opt, node_idx,
                                               out, ctx, m);
    }
    void operator()(const std::vector<Moment3>& m) const {
        potential_traversal_with_multipoles<3>(self, target, skip_self, target_h_opt, node_idx,
                                               out, ctx, m);
    }
    void operator()(const std::vector<Moment4>& m) const {
        potential_traversal_with_multipoles<4>(self, target, skip_self, target_h_opt, node_idx,
                                               out, ctx, m);
    }
    void operator()(const std::vector<Moment5>& m) const {
        potential_traversal_with_multipoles<5>(self, target, skip_self, target_h_opt, node_idx,
                                               out, ctx, m);
    }
};

struct AccelerationMultipoleVisitor {
    const Octree& self;
    const Vec3& target;
    std::optional<size_t> skip_self;
    std::optional<double> target_h_opt;
    size_t node_idx;
    Vec3& out;
    const TraversalCtx& ctx;

    void operator()(const std::vector<Moment0>& m) const {
        acceleration_traversal_with_multipoles<0>(self, target, skip_self, target_h_opt, node_idx,
                                                  out, ctx, m);
    }
    void operator()(const std::vector<Moment2>& m) const {
        acceleration_traversal_with_multipoles<2>(self, target, skip_self, target_h_opt, node_idx,
                                                  out, ctx, m);
    }
    void operator()(const std::vector<Moment3>& m) const {
        acceleration_traversal_with_multipoles<3>(self, target, skip_self, target_h_opt, node_idx,
                                                  out, ctx, m);
    }
    void operator()(const std::vector<Moment4>& m) const {
        acceleration_traversal_with_multipoles<4>(self, target, skip_self, target_h_opt, node_idx,
                                                  out, ctx, m);
    }
    void operator()(const std::vector<Moment5>& m) const {
        acceleration_traversal_with_multipoles<5>(self, target, skip_self, target_h_opt, node_idx,
                                                  out, ctx, m);
    }
};

} // namespace

static void potential_traversal_cached(const Octree& self, const Vec3& target,
                                       std::optional<size_t> skip_self,
                                       std::optional<double> target_h_opt, size_t node_idx,
                                       double& out, const TraversalCtx& ctx) {
    if (!self.multipoles.has_value()) {
        potential_traversal_no_multipoles(self, target, skip_self, target_h_opt, node_idx, out,
                                          ctx);
    } else {
        std::visit(PotentialMultipoleVisitor{self, target, skip_self, target_h_opt, node_idx, out,
                                             ctx},
                   *self.multipoles);
    }
}

static void acceleration_traversal_cached(const Octree& self, const Vec3& target,
                                          std::optional<size_t> skip_self,
                                          std::optional<double> target_h_opt, size_t node_idx,
                                          Vec3& out, const TraversalCtx& ctx) {
    if (!self.multipoles.has_value()) {
        acceleration_traversal_no_multipoles(self, target, skip_self, target_h_opt, node_idx, out,
                                             ctx);
    } else {
        std::visit(
            AccelerationMultipoleVisitor{self, target, skip_self, target_h_opt, node_idx, out, ctx},
            *self.multipoles);
    }
}

} // namespace detail
} // namespace gravity

namespace gravity {

// ---------------------------------------------------------------------------
// Public Octree methods (port of traversal.rs:728-857)
// ---------------------------------------------------------------------------

void Octree::compute_accelerations(double theta, std::vector<Vec3>& out) const {
    size_t n = positions.size();

    detail::TraversalCtx ctx{
        &bh_payload(),
        masses.has_value() ? masses->data() : nullptr,
        softenings.has_value() ? softenings->data() : nullptr,
        hmax.has_value() ? hmax->data() : nullptr,
        theta * theta,
        R2_TINY,
        kernel,
    };

    if (n < 1024) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = Vec3(0.0, 0.0, 0.0);
            const Vec3& target = positions[i];
            std::optional<double> target_h_opt =
                softenings.has_value() ? std::optional<double>((*softenings)[i]) : std::nullopt;
            detail::acceleration_traversal_cached(*this, target, i, target_h_opt, 0, out[i], ctx);
        }
    } else {
#pragma omp parallel for if (n >= 1024) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            Vec3 tmp(0.0, 0.0, 0.0);
            const Vec3& target = positions[static_cast<size_t>(i)];
            std::optional<double> target_h_opt =
                softenings.has_value() ? std::optional<double>((*softenings)[static_cast<size_t>(i)])
                                       : std::nullopt;
            detail::acceleration_traversal_cached(*this, target, static_cast<size_t>(i),
                                                  target_h_opt, 0, tmp, ctx);
            out[static_cast<size_t>(i)] = tmp;
        }
    }
}

void Octree::compute_potentials(double theta, std::vector<double>& out) const {
    size_t n = positions.size();

    detail::TraversalCtx ctx{
        &bh_payload(),
        masses.has_value() ? masses->data() : nullptr,
        softenings.has_value() ? softenings->data() : nullptr,
        hmax.has_value() ? hmax->data() : nullptr,
        theta * theta,
        R2_TINY,
        kernel,
    };

    if (n < 1024) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = 0.0;
            const Vec3& target = positions[i];
            std::optional<double> target_h_opt =
                softenings.has_value() ? std::optional<double>((*softenings)[i]) : std::nullopt;
            detail::potential_traversal_cached(*this, target, i, target_h_opt, 0, out[i], ctx);
        }
    } else {
#pragma omp parallel for if (n >= 1024) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            double tmp = 0.0;
            const Vec3& target = positions[static_cast<size_t>(i)];
            std::optional<double> target_h_opt =
                softenings.has_value() ? std::optional<double>((*softenings)[static_cast<size_t>(i)])
                                       : std::nullopt;
            detail::potential_traversal_cached(*this, target, static_cast<size_t>(i), target_h_opt,
                                               0, tmp, ctx);
            out[static_cast<size_t>(i)] = tmp;
        }
    }
}

void Octree::accelerations_at_points(const std::vector<Vec3>& points, double theta,
                                     std::vector<Vec3>& out) const {
    size_t n = points.size();

    detail::TraversalCtx ctx{
        &bh_payload(),
        masses.has_value() ? masses->data() : nullptr,
        softenings.has_value() ? softenings->data() : nullptr,
        hmax.has_value() ? hmax->data() : nullptr,
        theta * theta,
        R2_TINY,
        kernel,
    };

    if (n < 1024) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = Vec3(0.0, 0.0, 0.0);
            detail::acceleration_traversal_cached(*this, points[i], std::nullopt, std::nullopt, 0,
                                                  out[i], ctx);
        }
    } else {
#pragma omp parallel for if (n >= 1024) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            Vec3 tmp(0.0, 0.0, 0.0);
            detail::acceleration_traversal_cached(*this, points[static_cast<size_t>(i)],
                                                  std::nullopt, std::nullopt, 0, tmp, ctx);
            out[static_cast<size_t>(i)] = tmp;
        }
    }
}

void Octree::potentials_at_points(const std::vector<Vec3>& points, double theta,
                                  std::vector<double>& out) const {
    size_t n = points.size();

    detail::TraversalCtx ctx{
        &bh_payload(),
        masses.has_value() ? masses->data() : nullptr,
        softenings.has_value() ? softenings->data() : nullptr,
        hmax.has_value() ? hmax->data() : nullptr,
        theta * theta,
        R2_TINY,
        kernel,
    };

    if (n < 1024) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = 0.0;
            detail::potential_traversal_cached(*this, points[i], std::nullopt, std::nullopt, 0,
                                               out[i], ctx);
        }
    } else {
#pragma omp parallel for if (n >= 1024) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            double tmp = 0.0;
            detail::potential_traversal_cached(*this, points[static_cast<size_t>(i)], std::nullopt,
                                               std::nullopt, 0, tmp, ctx);
            out[static_cast<size_t>(i)] = tmp;
        }
    }
}

// ---------------------------------------------------------------------------
// Free-function wrappers
// ---------------------------------------------------------------------------

void compute_accelerations(const Octree& tree, double theta, std::vector<Vec3>& out) {
    tree.compute_accelerations(theta, out);
}

void compute_potentials(const Octree& tree, double theta, std::vector<double>& out) {
    tree.compute_potentials(theta, out);
}

void accelerations_at_points(const Octree& tree, const std::vector<Vec3>& points, double theta,
                             std::vector<Vec3>& out) {
    tree.accelerations_at_points(points, theta, out);
}

void potentials_at_points(const Octree& tree, const std::vector<Vec3>& points, double theta,
                          std::vector<double>& out) {
    tree.potentials_at_points(points, theta, out);
}

} // namespace gravity
