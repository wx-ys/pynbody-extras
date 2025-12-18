use rayon::prelude::*;

use crate::multipole::{
    gravity_accel_multipole,
    gravity_potential_multipole,
    translate_multipole,
    MultipoleMoment,
    PotentialDerivatives,
};

/// Simple 3D tree abstraction over point sets, with optional masses,
/// specialised for gravitational calculations.
pub trait Tree3D {
    /// Build a tree from positions and optional masses.
    /// If `masses` is `None`, unit mass is assumed.
    fn build(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        leaf_capacity: usize,
        multipole_order: u8,
    ) -> Self
    where
        Self: Sized;

    /// Compute gravitational accelerations on all particles in-place.
    fn compute_accelerations(&self, theta: f64, eps: f64, out: &mut [[f64; 3]]);

    /// Compute gravitational potentials on all particles in-place.
    fn compute_potentials(&self, theta: f64, eps: f64, out: &mut [f64]);

    /// Compute gravitational accelerations at arbitrary query points.
    fn accelerations_at_points(
        &self,
        points: &[[f64; 3]],
        theta: f64,
        eps: f64,
        out: &mut [[f64; 3]],
    );

    /// Compute gravitational potentials at arbitrary query points.
    fn potentials_at_points(
        &self,
        points: &[[f64; 3]],
        theta: f64,
        eps: f64,
        out: &mut [f64],
    );
}

#[derive(Clone)]
pub struct Node {
    pub center: [f64; 3],
    pub half_size: f64,
    pub children: Option<[usize; 8]>, // indices into nodes vec
    /// Indices of particles contained in this node's subtree.
    /// For leaf nodes this is the leaf's particle list; for
    /// internal nodes this is the union of all descendant leaves.
    pub indices: Vec<usize>,
}

#[derive(Clone, Copy)]
pub struct NodeBh {
    pub com: [f64; 3], // center of mass
    pub mass: f64,
}

#[derive(Clone)]
pub struct Octree {
    pub positions: Vec<[f64; 3]>,
    pub masses: Option<Vec<f64>>,      // optional per-particle masses
    pub nodes: Vec<Node>,
    // Optional BH payload (mass + COM per node).
    pub bh: Option<Vec<NodeBh>>,
    // Optional multipole payload per node (up to given order).
    pub multipoles: Option<Vec<MultipoleMoment>>,
    pub multipole_order: u8,
    pub leaf_capacity: usize,
}

impl Octree {
    fn bbox_of_points(pts: &[[f64; 3]]) -> ([f64; 3], f64) {
        let mut minp = [f64::INFINITY; 3];
        let mut maxp = [f64::NEG_INFINITY; 3];
        for p in pts {
            for i in 0..3 {
                if p[i] < minp[i] { minp[i] = p[i]; }
                if p[i] > maxp[i] { maxp[i] = p[i]; }
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

    /// Build an Octree from owned positions and optional masses.
    /// If `masses` is None, unit mass is assumed.
    pub fn from_owned(
        positions: Vec<[f64; 3]>,
        masses: Option<Vec<f64>>,
        leaf_capacity: usize,
        multipole_order: u8,
    ) -> Self {
        let (center, half) = Octree::bbox_of_points(&positions);
        let n = positions.len();
        let mut tree = Octree {
            positions,
            masses,
            nodes: Vec::new(),
            bh: None,
            multipoles: None,
            multipole_order,
            leaf_capacity: leaf_capacity.max(1),
        };
        let indices: Vec<usize> = (0..n).collect();
        let root = tree.make_node(center, half, indices);
        tree.nodes.push(root);
        tree.build_recursive(0);
        tree
    }

    pub fn set_masses(&mut self, masses: Option<Vec<f64>>) {
        self.masses = masses;
    }

    fn make_node(&mut self, center: [f64; 3], half_size: f64, indices: Vec<usize>) -> Node {
        Node {
            center,
            half_size,
            children: None,
            indices,
        }
    }

    fn subdivide_node(&mut self, node_idx: usize) {
        let (center, half, parent_indices) = {
            let n = &self.nodes[node_idx];
            (n.center, n.half_size, n.indices.clone())
        };
        let mut child_indices: [usize; 8] = [usize::MAX; 8];
        let mut buckets: [Vec<usize>; 8] = Default::default();
        {
            for &pi in &parent_indices {
                let p = self.positions[pi];
                let mut oct = 0usize;
                if p[0] >= center[0] { oct |= 1; }
                if p[1] >= center[1] { oct |= 2; }
                if p[2] >= center[2] { oct |= 4; }
                buckets[oct].push(pi);
            }
        }
        for oct in 0..8 {
            if buckets[oct].is_empty() { continue; }
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
        let parent = &mut self.nodes[node_idx];
        parent.children = Some(child_indices.map(|v| if v == usize::MAX { usize::MAX } else { v }));
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
                if c == usize::MAX { continue; }
                self.build_recursive(c);
            }
        }
    }

    fn build_bh_payload(&self) -> Vec<NodeBh> {
        let masses_opt = self.masses.as_deref();
        let mut bh = vec![NodeBh { mass: 0.0, com: [0.0; 3] }; self.nodes.len()];

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

    #[inline]
    fn bh(&self) -> &Vec<NodeBh> {
        self
            .bh
            .as_ref()
            .expect("BH payload not initialized; call build_mass() before gravity queries")
    }

    #[inline]
    pub fn build_mass_payload(&mut self) {
        let payload = self.build_bh_payload();
        self.bh = Some(payload);
        if self.multipole_order > 0 {
            let payload = self.build_multipole_payload();
            self.multipoles = Some(payload);
        }
    }

    fn build_multipole_payload(&self) -> Vec<MultipoleMoment> {
        let masses_opt = self.masses.as_deref();
        let bh = self
            .bh
            .as_ref()
            .expect("BH payload not initialized; call build_mass() before building multipoles");
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
                    if c == usize::MAX { continue; }
                    let child_bh = &bh[c];
                    if child_bh.mass == 0.0 { continue; }
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

        moments
    }

    fn node_potential_multipole(
        &self,
        target: &[f64; 3],
        node_idx: usize,
        eps2: f64,
        bh: &[NodeBh],
        multipoles: &[MultipoleMoment],
    ) -> f64 {
        let node_bh = &bh[node_idx];
        if node_bh.mass == 0.0 {
            return 0.0;
        }
        let dx = node_bh.com[0] - target[0];
        let dy = node_bh.com[1] - target[1];
        let dz = node_bh.com[2] - target[2];
        let r2 = dx * dx + dy * dy + dz * dz + eps2;
        if r2 == 0.0 {
            return 0.0;
        }
        let d = PotentialDerivatives::new(dx, dy, dz, eps2, self.multipole_order.min(5));
        let m = &multipoles[node_idx];
        gravity_potential_multipole(m, &d, self.multipole_order.min(5))
    }

    fn potential_traversal_cached(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        node_idx: usize,
        theta: f64,
        eps2: f64,
        out: &mut f64,
        bh: &[NodeBh],
        masses_opt: Option<&[f64]>,
    ) {
        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(node_idx);

        let multipoles_opt = self.multipoles.as_ref();

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            let node_bh = &bh[idx];

            if node_bh.mass == 0.0 {
                continue;
            }

            if node.children.is_none() {
                match masses_opt {
                    Some(masses) => {
                        for &pi in &node.indices {
                            if Some(pi) == skip_self {
                                continue;
                            }
                            let p = &self.positions[pi];
                            let ddx = p[0] - target[0];
                            let ddy = p[1] - target[1];
                            let ddz = p[2] - target[2];
                            let d2 = ddx * ddx + ddy * ddy + ddz * ddz + eps2;
                            if d2 == 0.0 {
                                continue;
                            }
                            let invr = 1.0 / d2.sqrt();
                            let m = masses[pi];
                            *out += -m * invr;
                        }
                    }
                    None => {
                        for &pi in &node.indices {
                            if Some(pi) == skip_self {
                                continue;
                            }
                            let p = &self.positions[pi];
                            let ddx = p[0] - target[0];
                            let ddy = p[1] - target[1];
                            let ddz = p[2] - target[2];
                            let d2 = ddx * ddx + ddy * ddy + ddz * ddz + eps2;
                            if d2 == 0.0 {
                                continue;
                            }
                            let invr = 1.0 / d2.sqrt();
                            *out += -1.0 * invr;
                        }
                    }
                }
                continue;
            }

            let dx = node_bh.com[0] - target[0];
            let dy = node_bh.com[1] - target[1];
            let dz = node_bh.com[2] - target[2];
            let dist2 = dx * dx + dy * dy + dz * dz + eps2;

            if dist2 == 0.0 {
                continue;
            }

            let s = node.half_size * 2.0;
            if (s * s) / dist2 < theta * theta {
                if let Some(multipoles) = multipoles_opt {
                    *out += self.node_potential_multipole(target, idx, eps2, bh, multipoles);
                } else {
                    let invr = 1.0 / dist2.sqrt();
                    let mass = node_bh.mass;
                    *out += -mass * invr;
                }
            } else if let Some(children) = node.children {
                for &c in &children {
                    if c == usize::MAX {
                        continue;
                    }
                    stack.push(c);
                }
            }
        }
    }

    fn acceleration_traversal_cached(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        node_idx: usize,
        theta: f64,
        eps2: f64,
        out: &mut [f64; 3],
        bh: &[NodeBh],
        masses_opt: Option<&[f64]>,
    ) {
        let mut stack: Vec<usize> = Vec::with_capacity(64);
        stack.push(node_idx);

        let multipoles_opt = self.multipoles.as_ref();

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            let node_bh = &bh[idx];

            if node_bh.mass == 0.0 {
                continue;
            }

            if node.children.is_none() {
                match masses_opt {
                    Some(masses) => {
                        for &pi in &node.indices {
                            if Some(pi) == skip_self {
                                continue;
                            }
                            let p = &self.positions[pi];
                            let ddx = p[0] - target[0];
                            let ddy = p[1] - target[1];
                            let ddz = p[2] - target[2];
                            let d2 = ddx * ddx + ddy * ddy + ddz * ddz + eps2;
                            if d2 == 0.0 {
                                continue;
                            }
                            let invr3 = 1.0 / (d2.sqrt() * d2);
                            let m = masses[pi];
                            out[0] += m * ddx * invr3;
                            out[1] += m * ddy * invr3;
                            out[2] += m * ddz * invr3;
                        }
                    }
                    None => {
                        for &pi in &node.indices {
                            if Some(pi) == skip_self {
                                continue;
                            }
                            let p = &self.positions[pi];
                            let ddx = p[0] - target[0];
                            let ddy = p[1] - target[1];
                            let ddz = p[2] - target[2];
                            let d2 = ddx * ddx + ddy * ddy + ddz * ddz + eps2;
                            if d2 == 0.0 {
                                continue;
                            }
                            let invr3 = 1.0 / (d2.sqrt() * d2);
                            let m = 1.0f64;
                            out[0] += m * ddx * invr3;
                            out[1] += m * ddy * invr3;
                            out[2] += m * ddz * invr3;
                        }
                    }
                }
                continue;
            }

            let dx = node_bh.com[0] - target[0];
            let dy = node_bh.com[1] - target[1];
            let dz = node_bh.com[2] - target[2];
            let dist2 = dx * dx + dy * dy + dz * dz + eps2;

            if dist2 == 0.0 {
                continue;
            }

            let s = node.half_size * 2.0;
            if (s * s) / dist2 < theta * theta {
                if let Some(multipoles) = multipoles_opt {
                    let d = PotentialDerivatives::new(dx, dy, dz, eps2, self.multipole_order.min(5));
                    let m = &multipoles[idx];
                    let acc = gravity_accel_multipole(m, &d, self.multipole_order.min(5));
                    out[0] += acc[0];
                    out[1] += acc[1];
                    out[2] += acc[2];
                } else {
                    let invr3 = 1.0 / (dist2.sqrt() * dist2);
                    let mass = node_bh.mass;
                    out[0] += mass * dx * invr3;
                    out[1] += mass * dy * invr3;
                    out[2] += mass * dz * invr3;
                }
            } else if let Some(children) = node.children {
                for &c in &children {
                    if c == usize::MAX {
                        continue;
                    }
                    stack.push(c);
                }
            }
        }
    }

    fn potential_at_point_node(
        &self,
        target: &[f64; 3],
        node_idx: usize,
        theta: f64,
        eps2: f64,
        out: &mut f64,
    ) {
        let bh = self.bh();
        let masses_opt = self.masses.as_deref();
        self.potential_traversal_cached(target, None, node_idx, theta, eps2, out, bh, masses_opt);
    }

    fn potential_on_particle_node(
        &self,
        target_idx: usize,
        node_idx: usize,
        theta: f64,
        eps2: f64,
        out: &mut f64,
    ) {
        let target = &self.positions[target_idx];
        let bh = self.bh();
        let masses_opt = self.masses.as_deref();
        self.potential_traversal_cached(target, Some(target_idx), node_idx, theta, eps2, out, bh, masses_opt);
    }

    fn acceleration_on_particle_node(
        &self,
        target_idx: usize,
        node_idx: usize,
        theta: f64,
        eps2: f64,
        out: &mut [f64; 3],
    ) {
        let target = &self.positions[target_idx];
        let bh = self.bh();
        let masses_opt = self.masses.as_deref();
        self.acceleration_traversal_cached(target, Some(target_idx), node_idx, theta, eps2, out, bh, masses_opt);
    }

    fn acceleration_at_point_node(
        &self,
        target: &[f64; 3],
        node_idx: usize,
        theta: f64,
        eps2: f64,
        out: &mut [f64; 3],
    ) {
        let bh = self.bh();
        let masses_opt = self.masses.as_deref();
        self.acceleration_traversal_cached(target, None, node_idx, theta, eps2, out, bh, masses_opt);
    }
}

impl Tree3D for Octree {
    fn build(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        leaf_capacity: usize,
        multipole_order: u8,
    ) -> Self {
        let positions_vec = positions.to_vec();
        let masses_vec = masses.map(|m| m.to_vec());
        let mut tree = Octree::from_owned(positions_vec, masses_vec, leaf_capacity, multipole_order);
        tree.build_mass_payload();
        tree
    }

    fn compute_accelerations(&self, theta: f64, eps: f64, out: &mut [[f64; 3]]) {
        let n = self.positions.len();
        let eps2 = eps * eps;

        if n < 1024 {
            for i in 0..n {
                out[i][0] = 0.0;
                out[i][1] = 0.0;
                out[i][2] = 0.0;
                self.acceleration_on_particle_node(i, 0, theta, eps2, &mut out[i]);
            }
        } else {
            out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
                let mut tmp = [0.0f64; 3];
                self.acceleration_on_particle_node(i, 0, theta, eps2, &mut tmp);
                out_i[0] = tmp[0];
                out_i[1] = tmp[1];
                out_i[2] = tmp[2];
            });
        }
    }

    fn compute_potentials(&self, theta: f64, eps: f64, out: &mut [f64]) {
        let n = self.positions.len();
        let eps2 = eps * eps;

        if n < 1024 {
            for i in 0..n {
                out[i] = 0.0;
                self.potential_on_particle_node(i, 0, theta, eps2, &mut out[i]);
            }
        } else {
            out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
                let mut tmp = 0.0f64;
                self.potential_on_particle_node(i, 0, theta, eps2, &mut tmp);
                *out_i = tmp;
            });
        }
    }

    fn accelerations_at_points(
        &self,
        points: &[[f64; 3]],
        theta: f64,
        eps: f64,
        out: &mut [[f64; 3]],
    ) {
        let n = points.len();
        let eps2 = eps * eps;

        if n < 1024 {
            for i in 0..n {
                out[i][0] = 0.0;
                out[i][1] = 0.0;
                out[i][2] = 0.0;
                self.acceleration_at_point_node(&points[i], 0, theta, eps2, &mut out[i]);
            }
        } else {
            out.par_iter_mut()
                .zip(points.par_iter())
                .for_each(|(out_i, p)| {
                    let mut tmp = [0.0f64; 3];
                    self.acceleration_at_point_node(p, 0, theta, eps2, &mut tmp);
                    out_i[0] = tmp[0];
                    out_i[1] = tmp[1];
                    out_i[2] = tmp[2];
                });
        }
    }

    fn potentials_at_points(
        &self,
        points: &[[f64; 3]],
        theta: f64,
        eps: f64,
        out: &mut [f64],
    ) {
        let n = points.len();
        let eps2 = eps * eps;

        if n < 1024 {
            for i in 0..n {
                out[i] = 0.0;
                self.potential_at_point_node(&points[i], 0, theta, eps2, &mut out[i]);
            }
        } else {
            out.par_iter_mut()
                .zip(points.par_iter())
                .for_each(|(out_i, p)| {
                    let mut tmp = 0.0f64;
                    self.potential_at_point_node(p, 0, theta, eps2, &mut tmp);
                    *out_i = tmp;
                });
        }
    }
}
