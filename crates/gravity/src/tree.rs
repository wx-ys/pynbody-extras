use rayon::prelude::*;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[inline]
fn timing_enabled() -> bool {
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
fn log_timing(label: &str, dt: Duration) {
    eprintln!("[gravity-timing] {label}: {:.3} ms", dt.as_secs_f64() * 1e3);
}

use crate::kernel::{kernel_accel_factor, kernel_potential_per_unit_mass, KernelKind};
use crate::multipole::{
    gravity_accel_multipole_o0_d1, gravity_accel_multipole_o2_d2, gravity_accel_multipole_o3_d3,
    gravity_accel_multipole_o4_d4, gravity_accel_multipole_o5, gravity_potential_multipole_o0_d1,
    gravity_potential_multipole_o2_d2, gravity_potential_multipole_o3_d3,
    gravity_potential_multipole_o4_d4, gravity_potential_multipole_o5, translate_multipole,
    Moment0, Moment2, Moment3, Moment4, Moment5, MultipoleMoment, MultipoleMoments,
    PotentialDerivatives, PotentialDerivatives1, PotentialDerivatives2, PotentialDerivatives3,
    PotentialDerivatives4,
};

// Tiny additive term to avoid division by zero in 1/sqrt(r2).
// Analogous to FLT_MIN in float code, but for f64.
const R2_TINY: f64 = f64::MIN_POSITIVE;
const MIN_SOFTENING: f64 = 0.0;

#[inline]
fn inv_r_from_r2(r2: f64) -> f64 {
    let s2 = r2 + R2_TINY;
    1.0 / s2.sqrt()
}

#[inline]
fn inv_r_and_inv_r3_from_r2(r2: f64) -> (f64, f64) {
    let s2 = r2 + R2_TINY;
    let inv_r = 1.0 / s2.sqrt();
    // Avoid an extra division in hot loops: inv_r3 = inv_r^3.
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    (inv_r, inv_r3)
}

#[inline]
fn node_soft_ok(idx: usize, dist2: f64, target_h_opt: Option<f64>, ctx: &TraversalCtx<'_>) -> bool {
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

#[derive(Clone, Copy)]
struct LeafArgs<'a> {
    positions: &'a [[f64; 3]],
    indices: &'a [usize],
    masses_opt: Option<&'a [f64]>,
    softenings_opt: Option<&'a [f64]>,
}

#[derive(Clone, Copy)]
struct TargetArgs<'a> {
    target: &'a [f64; 3],
    skip_self: Option<usize>,
    target_h_opt: Option<f64>,
    kernel: KernelKind,
}

#[derive(Clone, Copy)]
struct QueryArgs<'a> {
    target: &'a [f64; 3],
    skip_self: Option<usize>,
    target_h_opt: Option<f64>,
    node_idx: usize,
}

#[inline]
fn leaf_potential_sum(leaf: LeafArgs<'_>, targ: TargetArgs<'_>, out: &mut f64) {
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

    // -------------------------
    // Fast path: masses present + constant target softening (no per-particle softenings).
    // This matches common Python usage: Gravity(pos, mass, softening=const).
    // -------------------------
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
                    debug_assert!(pi < positions.len());
                    debug_assert!(pi < masses.len());
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
                // Non-spline kernels: with h>0 we always use kernel potential.
                for &pi in indices {
                    if pi == skip {
                        continue;
                    }
                    debug_assert!(pi < positions.len());
                    debug_assert!(pi < masses.len());
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
                    debug_assert!(pi < positions.len());
                    debug_assert!(pi < masses.len());
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
                    debug_assert!(pi < positions.len());
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
                debug_assert!(pi < positions.len());
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        debug_assert!(pi < masses.len());
                        unsafe { *masses.get_unchecked(pi) }
                    }
                    None => 1.0,
                };

                debug_assert!(pi < hs.len());
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
            // Only target softening applies; h is constant across all particles.
            let h = target_h;
            for &pi in indices {
                if pi == skip {
                    continue;
                }
                debug_assert!(pi < positions.len());
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        debug_assert!(pi < masses.len());
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
fn leaf_acceleration_sum(leaf: LeafArgs<'_>, targ: TargetArgs<'_>, out: &mut [f64; 3]) {
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
                    debug_assert!(pi < positions.len());
                    debug_assert!(pi < masses.len());
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
                    debug_assert!(pi < positions.len());
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
                debug_assert!(pi < positions.len());
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        debug_assert!(pi < masses.len());
                        unsafe { *masses.get_unchecked(pi) }
                    }
                    None => 1.0,
                };

                debug_assert!(pi < hs.len());
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
            // Only target softening applies; h is constant across all particles.
            let h = target_h;
            for &pi in indices {
                if pi == skip {
                    continue;
                }
                debug_assert!(pi < positions.len());
                let p = unsafe { positions.get_unchecked(pi) };
                let ddx = p[0] - tx;
                let ddy = p[1] - ty;
                let ddz = p[2] - tz;
                let r2 = ddx.mul_add(ddx, ddy.mul_add(ddy, ddz * ddz));
                let m = match masses_opt {
                    Some(masses) => {
                        debug_assert!(pi < masses.len());
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

macro_rules! dispatch_multipole_potential_traversal {
    ($tree:expr, $target:expr, $skip_self:expr, $target_h_opt:expr, $node_idx:expr, $out:expr, $ctx:expr) => {{
        let q = QueryArgs {
            target: $target,
            skip_self: $skip_self,
            target_h_opt: $target_h_opt,
            node_idx: $node_idx,
        };
        match $tree.multipoles.as_ref() {
            None => $tree.potential_traversal_cached_no_multipoles(
                $target,
                $skip_self,
                $target_h_opt,
                $node_idx,
                $out,
                $ctx,
            ),
            Some(MultipoleMoments::O0(m)) => $tree.potential_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives1::new(dx, dy, dz, eps2),
                |mm: &Moment0, d: &PotentialDerivatives1| gravity_potential_multipole_o0_d1(mm, d),
            ),
            Some(MultipoleMoments::O2(m)) => $tree.potential_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives2::new(dx, dy, dz, eps2),
                |mm: &Moment2, d: &PotentialDerivatives2| gravity_potential_multipole_o2_d2(mm, d),
            ),
            Some(MultipoleMoments::O3(m)) => $tree.potential_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives3::new(dx, dy, dz, eps2),
                |mm: &Moment3, d: &PotentialDerivatives3| gravity_potential_multipole_o3_d3(mm, d),
            ),
            Some(MultipoleMoments::O4(m)) => $tree.potential_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives4::new(dx, dy, dz, eps2),
                |mm: &Moment4, d: &PotentialDerivatives4| gravity_potential_multipole_o4_d4(mm, d),
            ),
            Some(MultipoleMoments::O5(m)) => $tree.potential_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| {
                    PotentialDerivatives::new(dx, dy, dz, eps2, 5)
                },
                |mm: &Moment5, d: &PotentialDerivatives| gravity_potential_multipole_o5(mm, d),
            ),
        }
    }};
}

macro_rules! dispatch_multipole_acceleration_traversal {
    ($tree:expr, $target:expr, $skip_self:expr, $target_h_opt:expr, $node_idx:expr, $out:expr, $ctx:expr) => {{
        let q = QueryArgs {
            target: $target,
            skip_self: $skip_self,
            target_h_opt: $target_h_opt,
            node_idx: $node_idx,
        };
        match $tree.multipoles.as_ref() {
            None => $tree.acceleration_traversal_cached_no_multipoles(
                $target,
                $skip_self,
                $target_h_opt,
                $node_idx,
                $out,
                $ctx,
            ),
            Some(MultipoleMoments::O0(m)) => $tree.acceleration_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives1::new(dx, dy, dz, eps2),
                |mm: &Moment0, d: &PotentialDerivatives1| gravity_accel_multipole_o0_d1(mm, d),
            ),
            Some(MultipoleMoments::O2(m)) => $tree.acceleration_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives2::new(dx, dy, dz, eps2),
                |mm: &Moment2, d: &PotentialDerivatives2| gravity_accel_multipole_o2_d2(mm, d),
            ),
            Some(MultipoleMoments::O3(m)) => $tree.acceleration_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives3::new(dx, dy, dz, eps2),
                |mm: &Moment3, d: &PotentialDerivatives3| gravity_accel_multipole_o3_d3(mm, d),
            ),
            Some(MultipoleMoments::O4(m)) => $tree.acceleration_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| PotentialDerivatives4::new(dx, dy, dz, eps2),
                |mm: &Moment4, d: &PotentialDerivatives4| gravity_accel_multipole_o4_d4(mm, d),
            ),
            Some(MultipoleMoments::O5(m)) => $tree.acceleration_traversal_cached_with_multipoles(
                q,
                $out,
                $ctx,
                m,
                |dx: f64, dy: f64, dz: f64, eps2: f64| {
                    PotentialDerivatives::new(dx, dy, dz, eps2, 5)
                },
                |mm: &Moment5, d: &PotentialDerivatives| gravity_accel_multipole_o5(mm, d),
            ),
        }
    }};
}

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
    fn compute_accelerations(&self, theta: f64, out: &mut [[f64; 3]]);

    /// Compute gravitational potentials on all particles in-place.
    fn compute_potentials(&self, theta: f64, out: &mut [f64]);

    /// Compute gravitational accelerations at arbitrary query points.
    fn accelerations_at_points(&self, points: &[[f64; 3]], theta: f64, out: &mut [[f64; 3]]);

    /// Compute gravitational potentials at arbitrary query points.
    fn potentials_at_points(&self, points: &[[f64; 3]], theta: f64, out: &mut [f64]);
}

#[derive(Clone)]
pub struct Node {
    pub center: [f64; 3],
    pub half_size: f64,
    /// Cached (2*half_size)^2 for fast opening-criterion checks.
    pub size2: f64,
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
    pub masses: Option<Vec<f64>>,
    /// Optional per-particle softening length h (same length as positions).
    pub softenings: Option<Vec<f64>>,
    pub nodes: Vec<Node>,
    /// Treewalk links for stack-free depth-first traversal.
    ///
    /// For each node index `i`:
    /// - `first_subnode[i]` is the first existing child, or `usize::MAX`.
    /// - `next_branch[i]` is the next node to visit after finishing `i`.
    pub first_subnode: Vec<usize>,
    pub next_branch: Vec<usize>,
    pub bh: Option<Vec<NodeBh>>,
    pub multipoles: Option<MultipoleMoments>,
    /// Optional per-node max softening length h_max (same length as nodes).
    pub hmax: Option<Vec<f64>>,
    pub multipole_order: u8,
    pub leaf_capacity: usize,
    pub kernel: KernelKind,
}

#[derive(Clone, Copy)]
struct TraversalCtx<'a> {
    bh: &'a [NodeBh],
    masses_opt: Option<&'a [f64]>,
    softenings_opt: Option<&'a [f64]>,
    hmax_opt: Option<&'a [f64]>,
    theta2: f64,
    /// tiny term used in opening criterion + multipole derivative builders
    /// to avoid division by zero when r^2 is exactly 0.
    multipole_eps2: f64,
    kernel: KernelKind,
}

impl Octree {
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

    /// Build an Octree from owned positions and optional masses.
    /// If `masses` is None, unit mass is assumed.
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

    fn build_treewalk_links(&mut self) {
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

    fn make_node(&mut self, center: [f64; 3], half_size: f64, indices: Vec<usize>) -> Node {
        // Side length is 2*half_size.
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

    #[inline]
    fn bh(&self) -> &Vec<NodeBh> {
        self.bh
            .as_ref()
            .expect("BH payload not initialized; call build_mass() before gravity queries")
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

    #[inline]
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
            // Safety: idx values are produced by build_treewalk_links and are valid node indices.
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

            // Replace division in hot loop: (s^2)/r^2 < theta^2  <=>  s^2 < theta^2 * r^2
            if soft_ok && node.size2 < ctx.theta2 * dist2 {
                let inv_r = inv_r_from_r2(dist2);
                *out += -node_bh.mass * inv_r;
                idx = unsafe { *self.next_branch.get_unchecked(idx) };
            } else {
                idx = unsafe { *self.first_subnode.get_unchecked(idx) };
            }
        }
    }

    fn potential_traversal_cached_with_multipoles<M, D, FD, FE>(
        &self,
        q: QueryArgs<'_>,
        out: &mut f64,
        ctx: &TraversalCtx<'_>,
        multipoles: &[M],
        build_d: FD,
        eval: FE,
    ) where
        FD: Copy + Fn(f64, f64, f64, f64) -> D,
        FE: Copy + Fn(&M, &D) -> f64,
    {
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
                let d = build_d(dx, dy, dz, ctx.multipole_eps2);
                *out += eval(&multipoles[idx], &d);
                idx = self.next_branch[idx];
            } else {
                idx = self.first_subnode[idx];
            }
        }
    }

    fn potential_traversal_cached(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        target_h_opt: Option<f64>,
        node_idx: usize,
        out: &mut f64,
        ctx: &TraversalCtx<'_>,
    ) {
        dispatch_multipole_potential_traversal!(
            self,
            target,
            skip_self,
            target_h_opt,
            node_idx,
            out,
            ctx
        );
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
            // Safety: idx values are produced by build_treewalk_links and are valid node indices.
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

    fn acceleration_traversal_cached_with_multipoles<M, D, FD, FE>(
        &self,
        q: QueryArgs<'_>,
        out: &mut [f64; 3],
        ctx: &TraversalCtx<'_>,
        multipoles: &[M],
        build_d: FD,
        eval: FE,
    ) where
        FD: Copy + Fn(f64, f64, f64, f64) -> D,
        FE: Copy + Fn(&M, &D) -> [f64; 3],
    {
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
                let d = build_d(dx, dy, dz, ctx.multipole_eps2);
                let acc = eval(&multipoles[idx], &d);
                out[0] += acc[0];
                out[1] += acc[1];
                out[2] += acc[2];
                idx = self.next_branch[idx];
            } else {
                idx = self.first_subnode[idx];
            }
        }
    }

    fn acceleration_traversal_cached(
        &self,
        target: &[f64; 3],
        skip_self: Option<usize>,
        target_h_opt: Option<f64>,
        node_idx: usize,
        out: &mut [f64; 3],
        ctx: &TraversalCtx<'_>,
    ) {
        dispatch_multipole_acceleration_traversal!(
            self,
            target,
            skip_self,
            target_h_opt,
            node_idx,
            out,
            ctx
        );
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

        let mut tree = Octree::from_owned(
            positions_vec,
            masses_vec,
            None,
            leaf_capacity,
            multipole_order,
            KernelKind::Plummer,
        );
        tree.build_mass_payload();
        tree
    }

    fn compute_accelerations(&self, theta: f64, out: &mut [[f64; 3]]) {
        let t_all = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
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

        if let Some(t_all) = t_all {
            log_timing("octree.compute_accelerations.total", t_all.elapsed());
        }
    }

    fn compute_potentials(&self, theta: f64, out: &mut [f64]) {
        let t_all = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
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

        if let Some(t_all) = t_all {
            log_timing("octree.compute_potentials.total", t_all.elapsed());
        }
    }

    fn accelerations_at_points(&self, points: &[[f64; 3]], theta: f64, out: &mut [[f64; 3]]) {
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

    fn potentials_at_points(&self, points: &[[f64; 3]], theta: f64, out: &mut [f64]) {
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
