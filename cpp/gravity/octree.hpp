#pragma once
#include <array>
#include <cstddef>
#include <optional>
#include <vector>
#include "gravity/vec3.hpp"
#include "gravity/common.hpp"
#include "gravity/kernel.hpp"
#include "gravity/multipole/moment.hpp"

namespace gravity {

// BH payload for a single node (total mass + center-of-mass).
struct NodeBh {
    Vec3 com;
    double mass;
};

// A single octree node.
struct Node {
    Vec3 center;
    double half_size;
    // Cached (2*half_size)^2 for fast opening-criterion checks.
    double size2;
    // Octant children indices into the nodes vec, or nullopt for a leaf.
    std::optional<std::array<size_t, 8>> children;
    // Particle indices in this node's subtree.
    std::vector<size_t> indices;
};

// A spatial octree over 3-D points, with optional masses and softenings.
//
// This is the central data structure for tree-based gravity solvers.
// It owns the particle data and all precomputed node-level payloads
// (BH center-of-mass, per-node max softening, multipole moments).
class Octree {
public:
    std::vector<Vec3> positions;
    std::optional<std::vector<double>> masses;
    // Optional per-particle softening length (same length as positions).
    std::optional<std::vector<double>> softenings;
    std::vector<Node> nodes;

    // ---- treewalk link caches ----
    // For each node, first existing child index (or NO_INDEX).
    std::vector<size_t> first_subnode;
    // For each node, next sibling to visit after finishing this subtree.
    std::vector<size_t> next_branch;

    // ---- node-level payloads ----
    std::optional<std::vector<NodeBh>> bh;
    std::optional<MultipoleMoments> multipoles;
    // Per-node maximum softening length.
    std::optional<std::vector<double>> hmax;

    // ---- configuration ----
    unsigned char multipole_order;
    size_t leaf_capacity;
    KernelKind kernel;

    static Octree from_owned(std::vector<Vec3> positions,
                             std::optional<std::vector<double>> masses,
                             std::optional<std::vector<double>> softenings,
                             size_t leaf_capacity,
                             unsigned char multipole_order,
                             KernelKind kernel);

    // Convenience: build an Octree from borrowed data with default settings
    // (no softening, Plummer kernel). For full control use from_owned.
    static Octree build(const std::vector<Vec3>& positions,
                        const double* masses,
                        size_t leaf_capacity,
                        unsigned char multipole_order);

    void build_mass_payload();
    void set_softenings(std::optional<std::vector<double>> softenings);
    void set_kernel(KernelKind kernel);
    void set_masses(std::optional<std::vector<double>> masses);
    const std::vector<NodeBh>& bh_payload() const;

    // ---- Barnes-Hut traversal (defined in traversal.cpp) ----
    void compute_accelerations(double theta, std::vector<Vec3>& out) const;
    void compute_potentials(double theta, std::vector<double>& out) const;
    void accelerations_at_points(const std::vector<Vec3>& points, double theta, std::vector<Vec3>& out) const;
    void potentials_at_points(const std::vector<Vec3>& points, double theta, std::vector<double>& out) const;

private:
    static std::pair<Vec3, double> bbox_of_points(const std::vector<Vec3>& pts);
    Node make_node(Vec3 center, double half_size, std::vector<size_t> indices);
    void subdivide_node(size_t node_idx);
    void build_recursive(size_t node_idx);
    void build_treewalk_links();
    std::vector<NodeBh> build_bh_payload() const;
    std::optional<std::vector<double>> build_hmax_payload() const;
    MultipoleMoments build_multipole_payload() const;
};

} // namespace gravity
