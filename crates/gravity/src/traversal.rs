use rayon::prelude::*;

use crate::kernel::{kernel_accel_factor, kernel_potential_per_unit_mass, KernelKind};
use crate::multipole::{
    Order0Eval, Order2Eval, Order3Eval, Order4Eval, Order5Eval, MultipoleOrder,
    MultipoleMoments,
};
use crate::octree::{
    inv_r_and_inv_r3_from_r2, inv_r_from_r2, Octree, R2_TINY, MIN_SOFTENING,
};
use crate::octree::NodeBh;

// ---------------------------------------------------------------------------
// Helper structs passed through the hot traversal path
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct LeafArgs<'a> {
    pub positions: &'a [[f64; 3]],
    pub indices: &'a [usize],
    pub masses_opt: Option<&'a [f64]>,
    pub softenings_opt: Option<&'a [f64]>,
}

#[derive(Clone, Copy)]
pub(crate) struct TargetArgs<'a> {
    pub target: &'a [f64; 3],
    pub skip_self: Option<usize>,
    pub target_h_opt: Option<f64>,
    pub kernel: KernelKind,
}

#[derive(Clone, Copy)]
pub(crate) struct QueryArgs<'a> {
    pub target: &'a [f64; 3],
    pub skip_self: Option<usize>,
    pub target_h_opt: Option<f64>,
    pub node_idx: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct TraversalCtx<'a> {
    pub bh: &'a [NodeBh],
    pub masses_opt: Option<&'a [f64]>,
    pub softenings_opt: Option<&'a [f64]>,
    pub hmax_opt: Option<&'a [f64]>,
    pub theta2: f64,
    pub multipole_eps2: f64,
    pub kernel: KernelKind,
}

// ---------------------------------------------------------------------------
// Softening helper for node-level opening criterion
// ---------------------------------------------------------------------------

#[inline]
fn node_soft_ok(
    idx: usize,
    dist2: f64,
    target_h_opt: Option<f64>,
    ctx: &TraversalCtx<'_>,
) -> bool {
    let Some(hmax) = ctx.hmax_opt else {
        return true;
    };

    let mut h = hmax[idx].max(MIN_SOFTENING);
    if let Some(ht) = target_h_opt {
        h = h.max(ht.max(MIN_SOFTENING));
    }
    if h <= 0.0 {
        return true;
    }
    let c = ctx.kernel.multipole_min_separation_factor();
    let ch = c * h;
    dist2 > ch * ch
}

// ---------------------------------------------------------------------------
// Leaf-level direct summation (called when the treewalk reaches a leaf)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn leaf_potential_sum(leaf: LeafArgs<'_>, targ: TargetArgs<'_>, out: &mut f64) {
    let positions = leaf.positions;
    let indices = leaf.indices;
    let masses_opt = leaf.masses_opt;
    let softenings_opt = leaf.softenings_opt;

    let target = targ.target;
    let skip_self = targ.skip_self;
    let target_h_opt = targ.target_h_opt;
    let kernel = targ.kernel;

    let tx = target[0];
    let ty = target[1];
    let tz = target[2];

    let skip = skip_self.unwrap_or(usize::MAX);

    let target_h = target_h_opt.unwrap_or(MIN_SOFTENING).max(MIN_SOFTENING);
    let use_softening = softenings_opt.is_some() || target_h > 0.0;

    // Fast path: masses present + constant target softening (no per-particle softenings).
    if use_softening {
        if let (Some(masses), None) = (masses_opt, softenings_opt) {
            let h = target_h;
            if h <= 0.0 {
                // fall through to no-softening logic below
            } else if kernel == KernelKind::CubicSplineW2 {
                let hh = h * h;
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    let p = unsafe { positions.get_unchecked(pi) };
                    let ddx = p[0] - tx;
                    let ddy = p[1] - ty;
                    let ddz = p[2] - tz;
                    let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                    let m = unsafe { *masses.get_unchecked(pi) };

                    if r2 >= hh {
                        let inv_r = inv_r_from_r2(r2);
                        *out += -m * inv_r;
                    } else {
                        let r = (r2 + R2_TINY).sqrt();
                        *out += m * kernel_potential_per_unit_mass(kernel, r, h);
                    }
                }
                return;
            } else {
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    let p = unsafe { positions.get_unchecked(pi) };
                    let ddx = p[0] - tx;
                    let ddy = p[1] - ty;
                    let ddz = p[2] - tz;
                    let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                    let r = (r2 + R2_TINY).sqrt();
                    let m = unsafe { *masses.get_unchecked(pi) };
                    *out += m * kernel_potential_per_unit_mass(kernel, r, h);
                }
                return;
            }
        }
    }

    if !use_softening {
        match masses_opt {
            Some(masses) => {
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    let p = unsafe { positions.get_unchecked(pi) };
                    let ddx = p[0] - tx;
                    let ddy = p[1] - ty;
                    let ddz = p[2] - tz;
                    let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                    let inv_r = inv_r_from_r2(r2);
                    let m = unsafe { *masses.get_unchecked(pi) };
                    *out += -m * inv_r;
                }
            }
            None => {
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    let p = unsafe { positions.get_unchecked(pi) };
                    let ddx = p[0] - tx;
                    let ddy = p[1] - ty;
                    let ddz = p[2] - tz;
                    let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                    let inv_r = inv_r_from_r2(r2);
                    *out += -inv_r;
                }
            }
        }
        return;
    }

    let kernel_is_spline = kernel == KernelKind::CubicSplineW2;

    match softenings_opt {
        Some(hs) => {
            for &pi in indices {
                if pi == skip {
                    continue;
                }
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        unsafe { *masses.get_unchecked(pi) }
                    }
                    None => 1.0,
                };

                let hi = unsafe { *hs.get_unchecked(pi) }.max(MIN_SOFTENING);
                let h = hi.max(target_h);

                if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                    let inv_r = inv_r_from_r2(r2);
                    *out += -m * inv_r;
                } else {
                    let r = (r2 + R2_TINY).sqrt();
                    *out += m * kernel_potential_per_unit_mass(kernel, r, h);
                }
            }
        }
        None => {
            let h = target_h;
            for &pi in indices {
                if pi == skip {
                    continue;
                }
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        unsafe { *masses.get_unchecked(pi) }
                    }
                    None => 1.0,
                };

                if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                    let inv_r = inv_r_from_r2(r2);
                    *out += -m * inv_r;
                } else {
                    let r = (r2 + R2_TINY).sqrt();
                    *out += m * kernel_potential_per_unit_mass(kernel, r, h);
                }
            }
        }
    }
}

#[inline]
pub(crate) fn leaf_acceleration_sum(leaf: LeafArgs<'_>, targ: TargetArgs<'_>, out: &mut [f64; 3]) {
    let positions = leaf.positions;
    let indices = leaf.indices;
    let masses_opt = leaf.masses_opt;
    let softenings_opt = leaf.softenings_opt;

    let target = targ.target;
    let skip_self = targ.skip_self;
    let target_h_opt = targ.target_h_opt;
    let kernel = targ.kernel;

    let tx = target[0];
    let ty = target[1];
    let tz = target[2];

    let skip = skip_self.unwrap_or(usize::MAX);

    let target_h = target_h_opt.unwrap_or(MIN_SOFTENING).max(MIN_SOFTENING);
    let use_softening = softenings_opt.is_some() || target_h > 0.0;

    if !use_softening {
        match masses_opt {
            Some(masses) => {
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    let p = unsafe { positions.get_unchecked(pi) };
                    let ddx = p[0] - tx;
                    let ddy = p[1] - ty;
                    let ddz = p[2] - tz;
                    let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                    let (_inv_r, inv_r3) = inv_r_and_inv_r3_from_r2(r2);
                    let m = unsafe { *masses.get_unchecked(pi) };
                    out[0] += m * ddx * inv_r3;
                    out[1] += m * ddy * inv_r3;
                    out[2] += m * ddz * inv_r3;
                }
            }
            None => {
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    let p = unsafe { positions.get_unchecked(pi) };
                    let ddx = p[0] - tx;
                    let ddy = p[1] - ty;
                    let ddz = p[2] - tz;
                    let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                    let (_inv_r, inv_r3) = inv_r_and_inv_r3_from_r2(r2);
                    out[0] += ddx * inv_r3;
                    out[1] += ddy * inv_r3;
                    out[2] += ddz * inv_r3;
                }
            }
        }
        return;
    }

    let kernel_is_spline = kernel == KernelKind::CubicSplineW2;

    match softenings_opt {
        Some(hs) => {
            for &pi in indices {
                if pi == skip {
                    continue;
                }
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        unsafe { *masses.get_unchecked(pi) }
                    }
                    None => 1.0,
                };

                let hi = unsafe { *hs.get_unchecked(pi) }.max(MIN_SOFTENING);
                let h = hi.max(target_h);
                if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                    let (_inv_r, inv_r3) = inv_r_and_inv_r3_from_r2(r2);
                    out[0] += m * ddx * inv_r3;
                    out[1] += m * ddy * inv_r3;
                    out[2] += m * ddz * inv_r3;
                } else {
                    let r = (r2 + R2_TINY).sqrt();
                    let g = kernel_accel_factor(kernel, r, h);
                    out[0] += m * ddx * g;
                    out[1] += m * ddy * g;
                    out[2] += m * ddz * g;
                }
            }
        }
        None => {
            let h = target_h;
            for &pi in indices {
                if pi == skip {
                    continue;
                }
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        unsafe { *masses.get_unchecked(pi) }
                    }
                    None => 1.0,
                };

                if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                    let (_inv_r, inv_r3) = inv_r_and_inv_r3_from_r2(r2);
                    out[0] += m * ddx * inv_r3;
                    out[1] += m * ddy * inv_r3;
                    out[2] += m * ddz * inv_r3;
                } else {
                    let r = (r2 + R2_TINY).sqrt();
                    let g = kernel_accel_factor(kernel, r, h);
                    out[0] += m * ddx * g;
                    out[1] += m * ddy * g;
                    out[2] += m * ddz * g;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Traversal methods on Octree
// ---------------------------------------------------------------------------

impl Octree {
    // ---- Core treewalk (no multipoles) ----

    fn potential_traversal_cached_no_multipoles(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        target_h_opt: Option<f64>,
        node_idx: usize,
        out: &mut f64,
        ctx: &TraversalCtx<'_>,
    ) {
        let tx = target[0];
        let ty = target[1];
        let tz = target[2];
        let softening_enabled = ctx.hmax_opt.is_some() || target_h_opt.is_some();
        let mut idx = node_idx;
        while idx != usize::MAX {
            let node_bh = unsafe { ctx.bh.get_unchecked(idx) };

            if node_bh.mass == 0.0 {
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
                continue;
            }

            let node = unsafe { self.nodes.get_unchecked(idx) };

            if node.children.is_none() {
                leaf_potential_sum(
                    LeafArgs {
                        positions: &self.positions,
                        indices: &node.indices,
                        masses_opt: ctx.masses_opt,
                        softenings_opt: ctx.softenings_opt,
                    },
                    TargetArgs {
                        target,
                        skip_self,
                        target_h_opt,
                        kernel: ctx.kernel,
                    },
                    out,
                );
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
                continue;
            }

            let dx = node_bh.com[0] - tx;
            let dy = node_bh.com[1] - ty;
            let dz = node_bh.com[2] - tz;
            let dist2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz)) + ctx.multipole_eps2;

            let soft_ok = if softening_enabled {
                node_soft_ok(idx, dist2, target_h_opt, ctx)
            } else {
                true
            };

            if soft_ok && node.size2 < ctx.theta2 * dist2 {
                let inv_r = inv_r_from_r2(dist2);
                *out += -node_bh.mass * inv_r;
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
            } else {
                idx = unsafe { *self.first_subnode.get_unchecked(idx) };
            }
        }
    }

    fn acceleration_traversal_cached_no_multipoles(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        target_h_opt: Option<f64>,
        node_idx: usize,
        out: &mut [f64; 3],
        ctx: &TraversalCtx<'_>,
    ) {
        let tx = target[0];
        let ty = target[1];
        let tz = target[2];
        let softening_enabled = ctx.hmax_opt.is_some() || target_h_opt.is_some();
        let mut idx = node_idx;
        while idx != usize::MAX {
            let node_bh = unsafe { ctx.bh.get_unchecked(idx) };

            if node_bh.mass == 0.0 {
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
                continue;
            }

            let node = unsafe { self.nodes.get_unchecked(idx) };

            if node.children.is_none() {
                leaf_acceleration_sum(
                    LeafArgs {
                        positions: &self.positions,
                        indices: &node.indices,
                        masses_opt: ctx.masses_opt,
                        softenings_opt: ctx.softenings_opt,
                    },
                    TargetArgs {
                        target,
                        skip_self,
                        target_h_opt,
                        kernel: ctx.kernel,
                    },
                    out,
                );
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
                continue;
            }

            let dx = node_bh.com[0] - tx;
            let dy = node_bh.com[1] - ty;
            let dz = node_bh.com[2] - tz;
            let dist2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz)) + ctx.multipole_eps2;

            let soft_ok = if softening_enabled {
                node_soft_ok(idx, dist2, target_h_opt, ctx)
            } else {
                true
            };

            if soft_ok && node.size2 < ctx.theta2 * dist2 {
                let inv_r = inv_r_from_r2(dist2);
                let inv_r2 = inv_r * inv_r;
                let inv_r3 = inv_r2 * inv_r;
                out[0] += node_bh.mass * dx * inv_r3;
                out[1] += node_bh.mass * dy * inv_r3;
                out[2] += node_bh.mass * dz * inv_r3;
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
            } else {
                idx = unsafe { *self.first_subnode.get_unchecked(idx) };
            }
        }
    }

    // ---- Core treewalk (with multipoles) — generic over MultipoleOrder ----

    fn potential_traversal_with_multipoles<E: MultipoleOrder>(
        &self,
        q: QueryArgs<'_>,
        out: &mut f64,
        ctx: &TraversalCtx<'_>,
        multipoles: &[E::Moment],
    ) {
        let target = q.target;
        let skip_self = q.skip_self;
        let target_h_opt = q.target_h_opt;

        let tx = target[0];
        let ty = target[1];
        let tz = target[2];
        let softening_enabled = ctx.hmax_opt.is_some() || target_h_opt.is_some();

        let mut idx = q.node_idx;
        while idx != usize::MAX {
            let node = &self.nodes[idx];
            let node_bh = &ctx.bh[idx];

            if node_bh.mass == 0.0 {
                idx = self.next_branch[idx];
                continue;
            }

            if node.children.is_none() {
                leaf_potential_sum(
                    LeafArgs {
                        positions: &self.positions,
                        indices: &node.indices,
                        masses_opt: ctx.masses_opt,
                        softenings_opt: ctx.softenings_opt,
                    },
                    TargetArgs {
                        target,
                        skip_self,
                        target_h_opt,
                        kernel: ctx.kernel,
                    },
                    out,
                );
                idx = self.next_branch[idx];
                continue;
            }

            let dx = node_bh.com[0] - tx;
            let dy = node_bh.com[1] - ty;
            let dz = node_bh.com[2] - tz;
            let dist2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz)) + ctx.multipole_eps2;

            let soft_ok = if softening_enabled {
                node_soft_ok(idx, dist2, target_h_opt, ctx)
            } else {
                true
            };

            if soft_ok && node.size2 < ctx.theta2 * dist2 {
                let d = E::derivatives(dx, dy, dz, ctx.multipole_eps2);
                *out += E::potential(&multipoles[idx], &d);
                idx = self.next_branch[idx];
            } else {
                idx = self.first_subnode[idx];
            }
        }
    }

    fn acceleration_traversal_with_multipoles<E: MultipoleOrder>(
        &self,
        q: QueryArgs<'_>,
        out: &mut [f64; 3],
        ctx: &TraversalCtx<'_>,
        multipoles: &[E::Moment],
    ) {
        let target = q.target;
        let skip_self = q.skip_self;
        let target_h_opt = q.target_h_opt;

        let tx = target[0];
        let ty = target[1];
        let tz = target[2];
        let softening_enabled = ctx.hmax_opt.is_some() || target_h_opt.is_some();
        let mut idx = q.node_idx;
        while idx != usize::MAX {
            let node = &self.nodes[idx];
            let node_bh = &ctx.bh[idx];

            if node_bh.mass == 0.0 {
                idx = self.next_branch[idx];
                continue;
            }

            if node.children.is_none() {
                leaf_acceleration_sum(
                    LeafArgs {
                        positions: &self.positions,
                        indices: &node.indices,
                        masses_opt: ctx.masses_opt,
                        softenings_opt: ctx.softenings_opt,
                    },
                    TargetArgs {
                        target,
                        skip_self,
                        target_h_opt,
                        kernel: ctx.kernel,
                    },
                    out,
                );
                idx = self.next_branch[idx];
                continue;
            }

            let dx = node_bh.com[0] - tx;
            let dy = node_bh.com[1] - ty;
            let dz = node_bh.com[2] - tz;
            let dist2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz)) + ctx.multipole_eps2;

            let soft_ok = if softening_enabled {
                node_soft_ok(idx, dist2, target_h_opt, ctx)
            } else {
                true
            };

            if soft_ok && node.size2 < ctx.theta2 * dist2 {
                let d = E::derivatives(dx, dy, dz, ctx.multipole_eps2);
                let acc = E::acceleration(&multipoles[idx], &d);
                out[0] += acc[0];
                out[1] += acc[1];
                out[2] += acc[2];
                idx = self.next_branch[idx];
            } else {
                idx = self.first_subnode[idx];
            }
        }
    }

    // ---- Dispatch entry points (one match per traversal, not per node) ----

    pub(crate) fn potential_traversal_cached(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        target_h_opt: Option<f64>,
        node_idx: usize,
        out: &mut f64,
        ctx: &TraversalCtx<'_>,
    ) {
        let q = QueryArgs { target, skip_self, target_h_opt, node_idx };
        match self.multipoles.as_ref() {
            None => self.potential_traversal_cached_no_multipoles(
                target, skip_self, target_h_opt, node_idx, out, ctx,
            ),
            Some(MultipoleMoments::O0(m)) => {
                self.potential_traversal_with_multipoles::<Order0Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O2(m)) => {
                self.potential_traversal_with_multipoles::<Order2Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O3(m)) => {
                self.potential_traversal_with_multipoles::<Order3Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O4(m)) => {
                self.potential_traversal_with_multipoles::<Order4Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O5(m)) => {
                self.potential_traversal_with_multipoles::<Order5Eval>(q, out, ctx, m)
            }
        }
    }

    pub(crate) fn acceleration_traversal_cached(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        target_h_opt: Option<f64>,
        node_idx: usize,
        out: &mut [f64; 3],
        ctx: &TraversalCtx<'_>,
    ) {
        let q = QueryArgs { target, skip_self, target_h_opt, node_idx };
        match self.multipoles.as_ref() {
            None => self.acceleration_traversal_cached_no_multipoles(
                target, skip_self, target_h_opt, node_idx, out, ctx,
            ),
            Some(MultipoleMoments::O0(m)) => {
                self.acceleration_traversal_with_multipoles::<Order0Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O2(m)) => {
                self.acceleration_traversal_with_multipoles::<Order2Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O3(m)) => {
                self.acceleration_traversal_with_multipoles::<Order3Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O4(m)) => {
                self.acceleration_traversal_with_multipoles::<Order4Eval>(q, out, ctx, m)
            }
            Some(MultipoleMoments::O5(m)) => {
                self.acceleration_traversal_with_multipoles::<Order5Eval>(q, out, ctx, m)
            }
        }
    }

    // ---- Public interface (used by BH solver) ----

    /// Compute gravitational accelerations on all particles.
    pub fn compute_accelerations(&self, theta: f64, out: &mut [[f64; 3]]) {
        let n = self.positions.len();

        let ctx = TraversalCtx {
            bh: self.bh(),
            masses_opt: self.masses.as_deref(),
            softenings_opt: self.softenings.as_deref(),
            hmax_opt: self.hmax.as_deref(),
            theta2: theta * theta,
            multipole_eps2: R2_TINY,
            kernel: self.kernel,
        };

        if n < 1024 {
            for (i, out_i) in out.iter_mut().enumerate() {
                out_i[0] = 0.0;
                out_i[1] = 0.0;
                out_i[2] = 0.0;
                let target = &self.positions[i];
                let target_h_opt = self.softenings.as_deref().map(|hs| hs[i]);
                self.acceleration_traversal_cached(target, Some(i), target_h_opt, 0, out_i, &ctx);
            }
        } else {
            out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
                out_i[0] = 0.0;
                out_i[1] = 0.0;
                out_i[2] = 0.0;
                let target = &self.positions[i];
                let target_h_opt = self.softenings.as_deref().map(|hs| hs[i]);
                self.acceleration_traversal_cached(target, Some(i), target_h_opt, 0, out_i, &ctx);
            });
        }
    }

    /// Compute gravitational potentials on all particles.
    pub fn compute_potentials(&self, theta: f64, out: &mut [f64]) {
        let n = self.positions.len();

        let ctx = TraversalCtx {
            bh: self.bh(),
            masses_opt: self.masses.as_deref(),
            softenings_opt: self.softenings.as_deref(),
            hmax_opt: self.hmax.as_deref(),
            theta2: theta * theta,
            multipole_eps2: R2_TINY,
            kernel: self.kernel,
        };

        if n < 1024 {
            for (i, out_i) in out.iter_mut().enumerate() {
                *out_i = 0.0;
                let target = &self.positions[i];
                let target_h_opt = self.softenings.as_deref().map(|hs| hs[i]);
                self.potential_traversal_cached(target, Some(i), target_h_opt, 0, out_i, &ctx);
            }
        } else {
            out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
                let mut tmp = 0.0f64;
                let target = &self.positions[i];
                let target_h_opt = self.softenings.as_deref().map(|hs| hs[i]);
                self.potential_traversal_cached(target, Some(i), target_h_opt, 0, &mut tmp, &ctx);
                *out_i = tmp;
            });
        }
    }

    /// Compute gravitational accelerations at arbitrary query points.
    pub fn accelerations_at_points(&self, points: &[[f64; 3]], theta: f64, out: &mut [[f64; 3]]) {
        let n = points.len();

        let ctx = TraversalCtx {
            bh: self.bh(),
            masses_opt: self.masses.as_deref(),
            softenings_opt: self.softenings.as_deref(),
            hmax_opt: self.hmax.as_deref(),
            theta2: theta * theta,
            multipole_eps2: R2_TINY,
            kernel: self.kernel,
        };

        if n < 1024 {
            out.iter_mut().zip(points.iter()).for_each(|(out_i, p)| {
                out_i[0] = 0.0;
                out_i[1] = 0.0;
                out_i[2] = 0.0;
                self.acceleration_traversal_cached(p, None, None, 0, out_i, &ctx);
            });
        } else {
            out.par_iter_mut()
                .zip(points.par_iter())
                .for_each(|(out_i, p)| {
                    let mut tmp = [0.0f64; 3];
                    self.acceleration_traversal_cached(p, None, None, 0, &mut tmp, &ctx);
                    out_i[0] = tmp[0];
                    out_i[1] = tmp[1];
                    out_i[2] = tmp[2];
                });
        }
    }

    /// Compute gravitational potentials at arbitrary query points.
    pub fn potentials_at_points(&self, points: &[[f64; 3]], theta: f64, out: &mut [f64]) {
        let n = points.len();

        let ctx = TraversalCtx {
            bh: self.bh(),
            masses_opt: self.masses.as_deref(),
            softenings_opt: self.softenings.as_deref(),
            hmax_opt: self.hmax.as_deref(),
            theta2: theta * theta,
            multipole_eps2: R2_TINY,
            kernel: self.kernel,
        };

        if n < 1024 {
            out.iter_mut().zip(points.iter()).for_each(|(out_i, p)| {
                *out_i = 0.0;
                self.potential_traversal_cached(p, None, None, 0, out_i, &ctx);
            });
        } else {
            out.par_iter_mut()
                .zip(points.par_iter())
                .for_each(|(out_i, p)| {
                    let mut tmp = 0.0f64;
                    self.potential_traversal_cached(p, None, None, 0, &mut tmp, &ctx);
                    *out_i = tmp;
                });
        }
    }
}
