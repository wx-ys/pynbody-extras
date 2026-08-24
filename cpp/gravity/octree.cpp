#include "gravity/octree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gravity {

namespace {

void build_treewalk_links_rec(const std::vector<Node>& nodes,
                              std::vector<size_t>& first,
                              std::vector<size_t>& next,
                              size_t node_idx) {
    const Node& node = nodes[node_idx];
    if (!node.children.has_value()) {
        return;
    }
    const auto& children = *node.children;

    bool has_last = false;
    size_t last = NO_INDEX;
    for (size_t c : children) {
        if (c == NO_INDEX) {
            continue;
        }
        if (first[node_idx] == NO_INDEX) {
            first[node_idx] = c;
        }
        if (has_last) {
            next[last] = c;
        }
        last = c;
        has_last = true;
    }
    if (has_last) {
        next[last] = next[node_idx];
    }
    for (size_t c : children) {
        if (c == NO_INDEX) {
            continue;
        }
        if (nodes[c].children.has_value()) {
            build_treewalk_links_rec(nodes, first, next, c);
        }
    }
}

} // namespace

std::pair<Vec3, double> Octree::bbox_of_points(const std::vector<Vec3>& pts) {
    double minp[3] = {std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity()};
    double maxp[3] = {-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity()};
    for (const Vec3& p : pts) {
        for (int i = 0; i < 3; ++i) {
            if (p[i] < minp[i]) {
                minp[i] = p[i];
            }
            if (p[i] > maxp[i]) {
                maxp[i] = p[i];
            }
        }
    }
    Vec3 center((minp[0] + maxp[0]) / 2.0,
                (minp[1] + maxp[1]) / 2.0,
                (minp[2] + maxp[2]) / 2.0);
    double half = 0.0;
    for (int i = 0; i < 3; ++i) {
        half = std::max(half, (maxp[i] - minp[i]) / 2.0);
    }
    if (half == 0.0) {
        half = 1e-6;
    }
    return {center, half};
}

Octree Octree::from_owned(std::vector<Vec3> positions,
                          std::optional<std::vector<double>> masses,
                          std::optional<std::vector<double>> softenings,
                          size_t leaf_capacity,
                          unsigned char multipole_order,
                          KernelKind kernel) {
    auto bbox = bbox_of_points(positions);
    Vec3 center = bbox.first;
    double half = bbox.second;

    size_t n = positions.size();
    if (softenings.has_value() && softenings->size() != n) {
        throw std::invalid_argument("softenings length must match positions length");
    }

    Octree tree;
    tree.positions = std::move(positions);
    tree.masses = std::move(masses);
    tree.softenings = std::move(softenings);
    tree.multipole_order = multipole_order;
    tree.leaf_capacity = (leaf_capacity < 1) ? 1 : leaf_capacity;
    tree.kernel = kernel;

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    Node root = tree.make_node(center, half, std::move(indices));
    tree.nodes.push_back(std::move(root));

    tree.build_recursive(0);
    tree.build_treewalk_links();

    return tree;
}

Node Octree::make_node(Vec3 center, double half_size, std::vector<size_t> indices) {
    double s = half_size * 2.0;
    Node node;
    node.center = center;
    node.half_size = half_size;
    node.size2 = s * s;
    node.children = std::nullopt;
    node.indices = std::move(indices);
    return node;
}

void Octree::subdivide_node(size_t node_idx) {
    Node& n = nodes[node_idx];
    Vec3 center = n.center;
    double half = n.half_size;
    std::vector<size_t> parent_indices = std::move(n.indices);

    std::array<size_t, 8> child_indices;
    child_indices.fill(NO_INDEX);
    std::array<std::vector<size_t>, 8> buckets;

    for (size_t pi : parent_indices) {
        const Vec3& p = positions[pi];
        size_t oct = 0;
        if (p.x >= center.x) {
            oct |= 1;
        }
        if (p.y >= center.y) {
            oct |= 2;
        }
        if (p.z >= center.z) {
            oct |= 4;
        }
        buckets[oct].push_back(pi);
    }

    for (size_t oct = 0; oct < 8; ++oct) {
        if (buckets[oct].empty()) {
            continue;
        }
        Vec3 child_center = center;
        double offset = half / 2.0;
        child_center.x += (oct & 1) ? offset : -offset;
        child_center.y += (oct & 2) ? offset : -offset;
        child_center.z += (oct & 4) ? offset : -offset;
        Node child = make_node(child_center, offset, std::move(buckets[oct]));
        size_t idx = nodes.size();
        nodes.push_back(std::move(child));
        child_indices[oct] = idx;
    }

    nodes[node_idx].children = child_indices;
}

void Octree::build_recursive(size_t node_idx) {
    bool should_subdivide = nodes[node_idx].indices.size() > leaf_capacity;
    if (!should_subdivide) {
        return;
    }
    subdivide_node(node_idx);
    if (nodes[node_idx].children.has_value()) {
        // Copy by value: the recursive build_recursive(c) below push_backs into
        // `nodes`, which may reallocate it and invalidate a reference into it.
        const auto children = *nodes[node_idx].children;
        for (size_t c : children) {
            if (c == NO_INDEX) {
                continue;
            }
            build_recursive(c);
        }
    }
}

void Octree::build_treewalk_links() {
    size_t n = nodes.size();
    first_subnode.assign(n, NO_INDEX);
    next_branch.assign(n, NO_INDEX);

    // Root's next branch is the end-of-traversal marker.
    next_branch[0] = NO_INDEX;
    build_treewalk_links_rec(nodes, first_subnode, next_branch, 0);
}

Octree Octree::build(const std::vector<Vec3>& positions,
                     const double* masses,
                     size_t leaf_capacity,
                     unsigned char multipole_order) {
    std::optional<std::vector<double>> masses_opt;
    if (masses != nullptr) {
        masses_opt = std::vector<double>(positions.size());
        std::copy(masses, masses + positions.size(), masses_opt->begin());
    }
    Octree tree = Octree::from_owned(positions, std::move(masses_opt), std::nullopt,
                                     leaf_capacity, multipole_order, KernelKind::Plummer);
    tree.build_mass_payload();
    return tree;
}

const std::vector<NodeBh>& Octree::bh_payload() const {
    if (!bh.has_value()) {
        throw std::runtime_error(
            "BH payload not initialized; call build_mass_payload() before gravity queries");
    }
    return *bh;
}

void Octree::set_softenings(std::optional<std::vector<double>> softenings) {
    if (softenings.has_value() && softenings->size() != positions.size()) {
        throw std::invalid_argument("softenings length must match positions length");
    }
    this->softenings = std::move(softenings);
}

void Octree::set_kernel(KernelKind kernel) {
    this->kernel = kernel;
}

void Octree::set_masses(std::optional<std::vector<double>> masses) {
    this->masses = std::move(masses);
}

std::vector<NodeBh> Octree::build_bh_payload() const {
    const double* masses_ptr = masses.has_value() ? masses->data() : nullptr;
    std::vector<NodeBh> bh(nodes.size());
    for (size_t i = 0; i < bh.size(); ++i) {
        bh[i].com = Vec3(0.0, 0.0, 0.0);
        bh[i].mass = 0.0;
    }

    for (std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(nodes.size()) - 1; idx >= 0; --idx) {
        double mass = 0.0;
        Vec3 com(0.0, 0.0, 0.0);
        const Node& node = nodes[static_cast<size_t>(idx)];

        if (!node.children.has_value()) {
            if (!node.indices.empty()) {
                if (masses_ptr != nullptr) {
                    for (size_t pi : node.indices) {
                        const Vec3& p = positions[pi];
                        double m = masses_ptr[pi];
                        mass += m;
                        com.x += p.x * m;
                        com.y += p.y * m;
                        com.z += p.z * m;
                    }
                } else {
                    for (size_t pi : node.indices) {
                        const Vec3& p = positions[pi];
                        mass += 1.0;
                        com.x += p.x;
                        com.y += p.y;
                        com.z += p.z;
                    }
                }
                if (mass > 0.0) {
                    com.x /= mass;
                    com.y /= mass;
                    com.z /= mass;
                }
            }
        } else {
            for (size_t c : *node.children) {
                if (c == NO_INDEX) {
                    continue;
                }
                const NodeBh& child_bh = bh[c];
                if (child_bh.mass == 0.0) {
                    continue;
                }
                mass += child_bh.mass;
                com.x += child_bh.com.x * child_bh.mass;
                com.y += child_bh.com.y * child_bh.mass;
                com.z += child_bh.com.z * child_bh.mass;
            }
            if (mass > 0.0) {
                com.x /= mass;
                com.y /= mass;
                com.z /= mass;
            }
        }

        bh[static_cast<size_t>(idx)].mass = mass;
        bh[static_cast<size_t>(idx)].com = com;
    }

    return bh;
}

std::optional<std::vector<double>> Octree::build_hmax_payload() const {
    if (!softenings.has_value()) {
        return std::nullopt;
    }
    const std::vector<double>& hs = *softenings;
    std::vector<double> hmax(nodes.size(), 0.0);

    for (std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(nodes.size()) - 1; idx >= 0; --idx) {
        const Node& node = nodes[static_cast<size_t>(idx)];
        if (!node.children.has_value()) {
            double m = 0.0;
            for (size_t pi : node.indices) {
                m = std::max(m, std::max(hs[pi], MIN_SOFTENING));
            }
            hmax[static_cast<size_t>(idx)] = m;
        } else {
            double m = 0.0;
            for (size_t c : *node.children) {
                if (c == NO_INDEX) {
                    continue;
                }
                m = std::max(m, hmax[c]);
            }
            hmax[static_cast<size_t>(idx)] = m;
        }
    }

    return hmax;
}

void Octree::build_mass_payload() {
    std::vector<NodeBh> payload = build_bh_payload();
    bh = std::move(payload);

    hmax = build_hmax_payload();

    if (multipole_order > 0) {
        MultipoleMoments payload = build_multipole_payload();
        multipoles = std::move(payload);
    }
}

MultipoleMoments Octree::build_multipole_payload() const {
    const double* masses_ptr = masses.has_value() ? masses->data() : nullptr;
    if (!bh.has_value()) {
        throw std::runtime_error(
            "BH payload not initialized; call build_mass_payload() before building multipoles");
    }
    const std::vector<NodeBh>& bh_ref = *bh;
    unsigned char order = (multipole_order < 5) ? multipole_order : 5;
    std::vector<MultipoleMoment> moments(nodes.size(), MultipoleMoment::zero());

    for (std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(nodes.size()) - 1; idx >= 0; --idx) {
        const Node& node = nodes[static_cast<size_t>(idx)];
        const NodeBh& node_bh = bh_ref[static_cast<size_t>(idx)];
        if (node_bh.mass == 0.0) {
            continue;
        }

        if (!node.children.has_value()) {
            if (node.indices.empty()) {
                continue;
            }
            Vec3 center = node_bh.com;
            MultipoleMoment m = MultipoleMoment::from_points(positions, masses_ptr,
                                                              node.indices, center, order);
            moments[static_cast<size_t>(idx)] = m;
        } else {
            Vec3 center = node_bh.com;
            MultipoleMoment acc = MultipoleMoment::zero();
            for (size_t c : *node.children) {
                if (c == NO_INDEX) {
                    continue;
                }
                const NodeBh& child_bh = bh_ref[c];
                if (child_bh.mass == 0.0) {
                    continue;
                }
                Vec3 shift = center - child_bh.com;
                MultipoleMoment translated = translate_multipole(moments[c], shift, order);
                acc.add_assign(translated);
            }
            moments[static_cast<size_t>(idx)] = acc;
        }
    }

    return multipole_moments_from_full(std::move(moments), order);
}

} // namespace gravity
