use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::kernel::KernelKind;
use crate::multipole::{translate_multipole, MultipoleMoment, MultipoleMoments};

// ---------------------------------------------------------------------------
// Timing helpers (only active when GRAVITY_TIMING env var is set)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GRAVITY_TIMING")
            .map(|v| {
                let v = v.trim();
                !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(false)
    })
}

#[inline]
pub(crate) fn log_timing(label: &str, dt: Duration) {
    eprintln!("[gravity-timing] {label}: {:.3} ms", dt.as_secs_f64() * 1e3);
}

// ---------------------------------------------------------------------------
// Numeric helpers (also used by traversal)
// ---------------------------------------------------------------------------

/// Tiny additive term to avoid division by zero in 1/sqrt(r2).
pub(crate) const R2_TINY: f64 = f64::MIN_POSITIVE;
pub(crate) const MIN_SOFTENING: f64 = 0.0;

#[inline]
pub(crate) fn inv_r_from_r2(r2: f64) -> f64 {
    let s2 = r2 + R2_TINY;
    1.0 / s2.sqrt()
}

#[inline]
pub(crate) fn inv_r_and_inv_r3_from_r2(r2: f64) -> (f64, f64) {
    let s2 = r2 + R2_TINY;
    let inv_r = 1.0 / s2.sqrt();
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    (inv_r, inv_r3)
}

// ---------------------------------------------------------------------------
// Tree data structures
// ---------------------------------------------------------------------------

/// BH payload for a single node (total mass + center-of-mass).
#[derive(Clone, Copy)]
pub struct NodeBh {
    pub com: [f64; 3],
    pub mass: f64,
}

/// A single octree node.
#[derive(Clone)]
pub struct Node {
    pub center: [f64; 3],
    pub half_size: f64,
    /// Cached (2*half_size)^2 for fast opening-criterion checks.
    pub size2: f64,
    /// Octant children indices into the nodes vec, or `None` for a leaf.
    pub children: Option<[usize; 8]>,
    /// Particle indices in this node's subtree.
    pub indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------

/// A spatial octree over 3-D points, with optional masses and softenings.
///
/// This is the central data structure for tree-based gravity solvers.
/// It owns the particle data and all precomputed node-level payloads
/// (BH center-of-mass, per-node max softening, multipole moments).
#[derive(Clone)]
pub struct Octree {
    pub positions: Vec<[f64; 3]>,
    pub masses: Option<Vec<f64>>,
    /// Optional per-particle softening length (same length as positions).
    pub softenings: Option<Vec<f64>>,
    pub nodes: Vec<Node>,

    // ---- treewalk link caches ----
    /// For each node, first existing child index (or `usize::MAX`).
    pub first_subnode: Vec<usize>,
    /// For each node, next sibling to visit after finishing this subtree.
    pub next_branch: Vec<usize>,

    // ---- node-level payloads ----
    pub bh: Option<Vec<NodeBh>>,
    pub multipoles: Option<MultipoleMoments>,
    /// Per-node maximum softening length.
    pub hmax: Option<Vec<f64>>,

    // ---- configuration ----
    pub multipole_order: u8,
    pub leaf_capacity: usize,
    pub kernel: KernelKind,
}

impl Octree {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    fn bbox_of_points(pts: &[[f64; 3]]) -> ([f64; 3], f64) {
        let mut minp = [f64::INFINITY; 3];
        let mut maxp = [f64::NEG_INFINITY; 3];
        for p in pts {
            for i in 0..3 {
                if p[i] < minp[i] {
                    minp[i] = p[i];
                }
                if p[i] > maxp[i] {
                    maxp[i] = p[i];
                }
            }
        }
        let center = [
            (minp[0] + maxp[0]) / 2.0,
            (minp[1] + maxp[1]) / 2.0,
            (minp[2] + maxp[2]) / 2.0,
        ];
        let mut half: f64 = 0.0;
        for i in 0..3 {
            half = half.max((maxp[i] - minp[i]) / 2.0);
        }
        if half == 0.0 {
            half = 1e-6;
        }
        (center, half)
    }

    /// Build an Octree from owned data vectors.
    ///
    /// If `masses` is `None`, unit mass is assumed by all downstream
    /// computations.  `softenings` must have the same length as `positions`
    /// when provided.  `multipole_order` is clamped to at most 5.
    pub fn from_owned(
        positions: Vec<[f64; 3]>,
        masses: Option<Vec<f64>>,
        softenings: Option<Vec<f64>>,
        leaf_capacity: usize,
        multipole_order: u8,
        kernel: KernelKind,
    ) -> Self {
        let t_all = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let (center, half) = Octree::bbox_of_points(&positions);
        if let Some(t0) = t0 {
            log_timing("octree.bbox", t0.elapsed());
        }
        let n = positions.len();
        if let Some(ref hs) = softenings {
            assert_eq!(hs.len(), n, "softenings length must match positions length");
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let mut tree = Octree {
            positions,
            masses,
            softenings,
            nodes: Vec::new(),
            first_subnode: Vec::new(),
            next_branch: Vec::new(),
            bh: None,
            multipoles: None,
            hmax: None,
            multipole_order,
            leaf_capacity: leaf_capacity.max(1),
            kernel,
        };
        if let Some(t0) = t0 {
            log_timing("octree.init", t0.elapsed());
        }

        let indices: Vec<usize> = (0..n).collect();
        let root = tree.make_node(center, half, indices);
        tree.nodes.push(root);

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        tree.build_recursive(0);
        if let Some(t0) = t0 {
            log_timing("octree.build_recursive", t0.elapsed());
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        tree.build_treewalk_links();
        if let Some(t0) = t0 {
            log_timing("octree.build_treewalk_links", t0.elapsed());
        }

        if let Some(t_all) = t_all {
            log_timing("octree.from_owned.total", t_all.elapsed());
        }
        tree
    }

    // ------------------------------------------------------------------
    // Tree building helpers
    // ------------------------------------------------------------------

    fn make_node(&mut self, center: [f64; 3], half_size: f64, indices: Vec<usize>) -> Node {
        let s = half_size * 2.0;
        Node {
            center,
            half_size,
            size2: s * s,
            children: None,
            indices,
        }
    }

    fn subdivide_node(&mut self, node_idx: usize) {
        let (center, half, parent_indices) = {
            let n = &mut self.nodes[node_idx];
            let center = n.center;
            let half = n.half_size;
            let parent_indices = std::mem::take(&mut n.indices);
            (center, half, parent_indices)
        };
        let mut child_indices: [usize; 8] = [usize::MAX; 8];
        let mut buckets: [Vec<usize>; 8] = Default::default();
        {
            for &pi in &parent_indices {
                let p = self.positions[pi];
                let mut oct = 0usize;
                if p[0] >= center[0] {
                    oct |= 1;
                }
                if p[1] >= center[1] {
                    oct |= 2;
                }
                if p[2] >= center[2] {
                    oct |= 4;
                }
                buckets[oct].push(pi);
            }
        }
        for oct in 0..8 {
            if buckets[oct].is_empty() {
                continue;
            }
            let mut child_center = center;
            let offset = half / 2.0;
            child_center[0] += if (oct & 1) != 0 { offset } else { -offset };
            child_center[1] += if (oct & 2) != 0 { offset } else { -offset };
            child_center[2] += if (oct & 4) != 0 { offset } else { -offset };
            let child = self.make_node(child_center, offset, std::mem::take(&mut buckets[oct]));
            let idx = self.nodes.len();
            self.nodes.push(child);
            child_indices[oct] = idx;
        }
        self.nodes[node_idx].children = Some(child_indices);
    }

    fn build_recursive(&mut self, node_idx: usize) {
        let should_subdivide = {
            let n = &self.nodes[node_idx];
            n.indices.len() > self.leaf_capacity
        };
        if !should_subdivide {
            return;
        }
        self.subdivide_node(node_idx);
        if let Some(children) = self.nodes[node_idx].children {
            for &c in &children {
                if c == usize::MAX {
                    continue;
                }
                self.build_recursive(c);
            }
        }
    }

    // ------------------------------------------------------------------
    // Treewalk link cache
    // ------------------------------------------------------------------

    pub(crate) fn build_treewalk_links(&mut self) {
        let n = self.nodes.len();
        self.first_subnode = vec![usize::MAX; n];
        self.next_branch = vec![usize::MAX; n];

        fn rec(nodes: &[Node], first: &mut [usize], next: &mut [usize], node_idx: usize) {
            let Some(children) = nodes[node_idx].children else {
                return;
            };

            let mut last: Option<usize> = None;
            for c in children {
                if c == usize::MAX {
                    continue;
                }
                if first[node_idx] == usize::MAX {
                    first[node_idx] = c;
                }
                if let Some(prev) = last {
                    next[prev] = c;
                }
                last = Some(c);
            }
            if let Some(last_child) = last {
                next[last_child] = next[node_idx];
            }
            for c in children {
                if c == usize::MAX {
                    continue;
                }
                if nodes[c].children.is_some() {
                    rec(nodes, first, next, c);
                }
            }
        }

        // Root's next branch is the end-of-traversal marker.
        self.next_branch[0] = usize::MAX;
        let nodes_ref: &[Node] = &self.nodes;
        rec(nodes_ref, &mut self.first_subnode, &mut self.next_branch, 0);
    }

    // ------------------------------------------------------------------
    // Convenience builder (backward-compatible with old Tree3D::build)
    // ------------------------------------------------------------------

    /// Convenience: build an Octree from borrowed slices with default settings
    /// (no softening, Plummer kernel).  For full control use `from_owned`.
    pub fn build(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        leaf_capacity: usize,
        multipole_order: u8,
    ) -> Self {
        let mut tree = Octree::from_owned(
            positions.to_vec(),
            masses.map(|m| m.to_vec()),
            None,
            leaf_capacity,
            multipole_order,
            KernelKind::Plummer,
        );
        tree.build_mass_payload();
        tree
    }

    // ------------------------------------------------------------------
    // Payload building
    // ------------------------------------------------------------------

    #[inline]
    pub fn bh(&self) -> &Vec<NodeBh> {
        self.bh
            .as_ref()
            .expect("BH payload not initialized; call build_mass_payload() before gravity queries")
    }

    pub fn set_softenings(&mut self, softenings: Option<Vec<f64>>) {
        if let Some(ref hs) = softenings {
            assert_eq!(hs.len(), self.positions.len());
        }
        self.softenings = softenings;
    }

    pub fn set_kernel(&mut self, kernel: KernelKind) {
        self.kernel = kernel;
    }

    pub fn set_masses(&mut self, masses: Option<Vec<f64>>) {
        self.masses = masses;
    }

    fn build_bh_payload(&self) -> Vec<NodeBh> {
        let masses_opt = self.masses.as_deref();
        let mut bh = vec![
            NodeBh {
                mass: 0.0,
                com: [0.0; 3]
            };
            self.nodes.len()
        ];

        for idx in (0..self.nodes.len()).rev() {
            let mut mass = 0.0f64;
            let mut com = [0.0f64; 3];
            let node = &self.nodes[idx];

            if node.children.is_none() {
                if !node.indices.is_empty() {
                    if let Some(masses) = masses_opt {
                        for &pi in &node.indices {
                            let p = self.positions[pi];
                            let m = masses[pi];
                            mass += m;
                            com[0] += p[0] * m;
                            com[1] += p[1] * m;
                            com[2] += p[2] * m;
                        }
                    } else {
                        for &pi in &node.indices {
                            let p = self.positions[pi];
                            mass += 1.0;
                            com[0] += p[0];
                            com[1] += p[1];
                            com[2] += p[2];
                        }
                    }
                    if mass > 0.0 {
                        com[0] /= mass;
                        com[1] /= mass;
                        com[2] /= mass;
                    }
                }
            } else if let Some(children) = node.children {
                for &c in &children {
                    if c == usize::MAX {
                        continue;
                    }
                    let child_bh = &bh[c];
                    if child_bh.mass == 0.0 {
                        continue;
                    }
                    mass += child_bh.mass;
                    com[0] += child_bh.com[0] * child_bh.mass;
                    com[1] += child_bh.com[1] * child_bh.mass;
                    com[2] += child_bh.com[2] * child_bh.mass;
                }
                if mass > 0.0 {
                    com[0] /= mass;
                    com[1] /= mass;
                    com[2] /= mass;
                }
            }

            bh[idx] = NodeBh { mass, com };
        }

        bh
    }

    fn build_hmax_payload(&self) -> Option<Vec<f64>> {
        let hs = self.softenings.as_deref()?;
        let mut hmax = vec![0.0f64; self.nodes.len()];

        for idx in (0..self.nodes.len()).rev() {
            let node = &self.nodes[idx];
            if node.children.is_none() {
                let mut m = 0.0f64;
                for &pi in &node.indices {
                    m = m.max(hs[pi].max(MIN_SOFTENING));
                }
                hmax[idx] = m;
            } else if let Some(children) = node.children {
                let mut m = 0.0f64;
                for &c in &children {
                    if c == usize::MAX {
                        continue;
                    }
                    m = m.max(hmax[c]);
                }
                hmax[idx] = m;
            }
        }
        Some(hmax)
    }

    pub fn build_mass_payload(&mut self) {
        let t_all = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let payload = self.build_bh_payload();
        self.bh = Some(payload);
        if let Some(t0) = t0 {
            log_timing("octree.build_bh_payload", t0.elapsed());
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        self.hmax = self.build_hmax_payload();
        if let Some(t0) = t0 {
            log_timing("octree.build_hmax_payload", t0.elapsed());
        }

        if self.multipole_order > 0 {
            let t0 = if timing_enabled() {
                Some(Instant::now())
            } else {
                None
            };
            let payload = self.build_multipole_payload();
            self.multipoles = Some(payload);
            if let Some(t0) = t0 {
                log_timing("octree.build_multipole_payload", t0.elapsed());
            }
        }

        if let Some(t_all) = t_all {
            log_timing("octree.build_mass_payload.total", t_all.elapsed());
        }
    }

    fn build_multipole_payload(&self) -> MultipoleMoments {
        let masses_opt = self.masses.as_deref();
        let bh = self
            .bh
            .as_ref()
            .expect("BH payload not initialized; call build_mass_payload() before building multipoles");
        let order = self.multipole_order.min(5);
        let mut moments = vec![MultipoleMoment::zero(); self.nodes.len()];

        for idx in (0..self.nodes.len()).rev() {
            let node = &self.nodes[idx];
            let node_bh = &bh[idx];
            if node_bh.mass == 0.0 {
                continue;
            }

            if node.children.is_none() {
                if node.indices.is_empty() {
                    continue;
                }
                let center = node_bh.com;
                let m = MultipoleMoment::from_points(
                    &self.positions,
                    masses_opt,
                    &node.indices,
                    center,
                    order,
                );
                moments[idx] = m;
            } else if let Some(children) = node.children {
                let center = node_bh.com;
                let mut acc = MultipoleMoment::zero();
                for &c in &children {
                    if c == usize::MAX {
                        continue;
                    }
                    let child_bh = &bh[c];
                    if child_bh.mass == 0.0 {
                        continue;
                    }
                    let shift = [
                        center[0] - child_bh.com[0],
                        center[1] - child_bh.com[1],
                        center[2] - child_bh.com[2],
                    ];
                    let translated = translate_multipole(&moments[c], shift, order);
                    acc.add_assign(&translated);
                }
                moments[idx] = acc;
            }
        }

        MultipoleMoments::from_full(moments, order)
    }
}
