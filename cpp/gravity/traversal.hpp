#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "gravity/vec3.hpp"
#include "gravity/octree.hpp"
#include "gravity/kernel.hpp"
#include "gravity/multipole/eval.hpp"

namespace gravity {

// Free-function wrappers over the Octree traversal methods. These are what the
// binding layer (and callers that prefer free functions) invoke; each simply
// forwards to the matching Octree method.
void compute_accelerations(const Octree& tree, double theta, std::vector<Vec3>& out);
void compute_potentials(const Octree& tree, double theta, std::vector<double>& out);
void accelerations_at_points(const Octree& tree, const std::vector<Vec3>& points, double theta,
                             std::vector<Vec3>& out);
void potentials_at_points(const Octree& tree, const std::vector<Vec3>& points, double theta,
                          std::vector<double>& out);

namespace detail {

// Hot-path argument bundles passed through the treewalk.
struct LeafArgs {
    const std::vector<Vec3>* positions;
    const std::vector<size_t>* indices;
    const double* masses;      // nullptr when absent
    const double* softenings;  // nullptr when absent
};

struct TargetArgs {
    const Vec3* target;
    std::optional<size_t> skip_self;
    std::optional<double> target_h_opt;
    KernelKind kernel;
};

struct TraversalCtx {
    const std::vector<NodeBh>* bh;
    const double* masses;      // nullptr when absent
    const double* softenings;  // nullptr when absent
    const double* hmax;        // nullptr when absent
    double theta2;
    double multipole_eps2;
    KernelKind kernel;
};

// Leaf-level direct summation (called when the treewalk reaches a leaf).
void leaf_potential_sum(const LeafArgs& leaf, const TargetArgs& targ, double& out);
void leaf_acceleration_sum(const LeafArgs& leaf, const TargetArgs& targ, Vec3& out);

// Generic treewalk over a precomputed multipole payload of fixed order.
template <int Order>
void potential_traversal_with_multipoles(
    const Octree& self, const Vec3& target, std::optional<size_t> skip_self,
    std::optional<double> target_h_opt, size_t node_idx, double& out, const TraversalCtx& ctx,
    const std::vector<typename MultipoleEval<Order>::Moment>& multipoles);

template <int Order>
void acceleration_traversal_with_multipoles(
    const Octree& self, const Vec3& target, std::optional<size_t> skip_self,
    std::optional<double> target_h_opt, size_t node_idx, Vec3& out, const TraversalCtx& ctx,
    const std::vector<typename MultipoleEval<Order>::Moment>& multipoles);

} // namespace detail
} // namespace gravity
