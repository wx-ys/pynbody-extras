use rayon::prelude::*;

use crate::kernel::{kernel_accel_factor, kernel_potential_per_unit_mass, KernelKind};

// Tiny additive term to avoid division by zero in 1/sqrt(r2).
const R2_TINY: f64 = f64::MIN_POSITIVE;

// ===========================================================================
// Accumulator trait — allows a single generic loop for both accel & potential
// ===========================================================================

/// Abstracts over the output type so the direct-sum inner loops can be shared
/// between acceleration (`[f64; 3]`) and potential (`f64`) computations.
trait DirectAccumulator: Default + Clone + Send + Sync {
    type Output: Default + Send;

    /// Get a mutable reference to the accumulated value for particle `i`.
    fn init() -> Self;

    /// Accumulate the interaction from source `j` onto target `i`.
    ///
    /// `dx, dy, dz` = position difference (source - target).
    /// `m` = source mass.
    /// `invr` = 1/sqrt(r² + ε).
    fn add_newton(&mut self, dx: f64, dy: f64, dz: f64, m: f64, invr: f64);

    /// Accumulate a softened interaction.
    fn add_softened(&mut self, dx: f64, dy: f64, dz: f64, m: f64, r: f64, h: f64, kernel: KernelKind);

    /// Return the accumulated value, consuming self.
    fn finish(self) -> Self::Output;
}

/// Accumulator for gravitational acceleration `[ax, ay, az]`.
#[derive(Default, Clone)]
struct AccelAccumulator {
    ax: f64,
    ay: f64,
    az: f64,
}

impl DirectAccumulator for AccelAccumulator {
    type Output = [f64; 3];

    #[inline]
    fn init() -> Self {
        Self::default()
    }

    #[inline]
    fn add_newton(&mut self, dx: f64, dy: f64, dz: f64, m: f64, invr: f64) {
        // g = m / r^3  ->  g = m * invr * invr^2 = m * invr^3
        let invr2 = invr * invr;
        let invr3 = invr2 * invr;
        let g = m * invr3;
        self.ax += dx * g;
        self.ay += dy * g;
        self.az += dz * g;
    }

    #[inline]
    fn add_softened(&mut self, dx: f64, dy: f64, dz: f64, m: f64, r: f64, h: f64, kernel: KernelKind) {
        let g = m * kernel_accel_factor(kernel, r, h);
        self.ax += dx * g;
        self.ay += dy * g;
        self.az += dz * g;
    }

    #[inline]
    fn finish(self) -> Self::Output {
        [self.ax, self.ay, self.az]
    }
}

/// Accumulator for gravitational potential `phi`.
#[derive(Default, Clone)]
struct PotAccumulator {
    phi: f64,
}

impl DirectAccumulator for PotAccumulator {
    type Output = f64;

    #[inline]
    fn init() -> Self {
        Self::default()
    }

    #[inline]
    fn add_newton(&mut self, _dx: f64, _dy: f64, _dz: f64, m: f64, invr: f64) {
        self.phi += -m * invr;
    }

    #[inline]
    fn add_softened(&mut self, _dx: f64, _dy: f64, _dz: f64, m: f64, r: f64, h: f64, kernel: KernelKind) {
        self.phi += m * kernel_potential_per_unit_mass(kernel, r, h);
    }

    #[inline]
    fn finish(self) -> Self::Output {
        self.phi
    }
}

// ===========================================================================
// Generic direct-sum implementations
// ===========================================================================

/// Self-gravity: compute quantity on each source particle.
///
/// For small N (< 512) uses a symmetric pairwise loop (each pair computed once,
/// both particles updated).  For larger N uses a parallelised per-particle loop.
fn direct_self_impl<A: DirectAccumulator>(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    softenings: Option<&[f64]>,
    kernel: Option<KernelKind>,
) -> Vec<A::Output> {
    let n = positions.len();
    if n == 0 {
        return Vec::new();
    }

    let masses_slice_owned;
    let masses_slice: &[f64] = if let Some(m) = masses {
        m
    } else {
        masses_slice_owned = vec![1.0; n];
        &masses_slice_owned
    };

    let has_softening = softenings.is_some() && kernel.is_some();
    let kernel_kind = kernel.unwrap_or(KernelKind::Plummer);
    let kernel_is_spline = kernel_kind == KernelKind::CubicSplineW2;

    if n < 512 {
        // Symmetric pairwise loop — each (i,j) pair computed once.
        let mut accums: Vec<A> = (0..n).map(|_| A::init()).collect();

        for i in 0..n {
            let pi = positions[i];
            let mi = masses_slice[i];
            let hi = softenings.map(|hs| hs[i]).unwrap_or(0.0);

            for j in (i + 1)..n {
                let pj = positions[j];
                let mj = masses_slice[j];

                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                if has_softening {
                    let hj = softenings.map(|hs| hs[j]).unwrap_or(0.0);
                    let h = hi.max(hj);
                    if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                        let invr = 1.0 / (r2 + R2_TINY).sqrt();
                        accums[i].add_newton(dx, dy, dz, mj, invr);
                        accums[j].add_newton(-dx, -dy, -dz, mi, invr);
                    } else {
                        let r = (r2 + R2_TINY).sqrt();
                        accums[i].add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                        accums[j].add_softened(-dx, -dy, -dz, mi, r, h, kernel_kind);
                    }
                } else {
                    let invr = 1.0 / (r2 + R2_TINY).sqrt();
                    accums[i].add_newton(dx, dy, dz, mj, invr);
                    accums[j].add_newton(-dx, -dy, -dz, mi, invr);
                }
            }
        }

        accums.into_iter().map(|a| a.finish()).collect()
    } else {
        // Parallel per-particle — each particle sums over all others.
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut acc = A::init();
                let pi = positions[i];
                let hi = softenings.map(|hs| hs[i]).unwrap_or(0.0);

                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let pj = positions[j];
                    let mj = masses_slice[j];

                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                    if has_softening {
                        let hj = softenings.map(|hs| hs[j]).unwrap_or(0.0);
                        let h = hi.max(hj);
                        if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                            let invr = 1.0 / (r2 + R2_TINY).sqrt();
                            acc.add_newton(dx, dy, dz, mj, invr);
                        } else {
                            let r = (r2 + R2_TINY).sqrt();
                            acc.add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                        }
                    } else {
                        let invr = 1.0 / (r2 + R2_TINY).sqrt();
                        acc.add_newton(dx, dy, dz, mj, invr);
                    }
                }

                acc.finish()
            })
            .collect()
    }
}

/// Evaluate gravity from source particles at arbitrary target positions.
fn direct_at_points_impl<A: DirectAccumulator>(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    softenings: Option<&[f64]>,
    targets: &[[f64; 3]],
    kernel: Option<KernelKind>,
) -> Vec<A::Output> {
    let n_src = positions.len();
    let n_tgt = targets.len();
    if n_tgt == 0 || n_src == 0 {
        return Vec::new();
    }

    let masses_slice_owned;
    let masses_slice: &[f64] = if let Some(m) = masses {
        m
    } else {
        masses_slice_owned = vec![1.0; n_src];
        &masses_slice_owned
    };

    let has_softening = softenings.is_some() && kernel.is_some();
    let kernel_kind = kernel.unwrap_or(KernelKind::Plummer);
    let kernel_is_spline = kernel_kind == KernelKind::CubicSplineW2;

    if n_tgt < 512 {
        targets
            .iter()
            .map(|tgt| {
                let mut acc = A::init();
                let tx = tgt[0];
                let ty = tgt[1];
                let tz = tgt[2];

                for j in 0..n_src {
                    let pj = positions[j];
                    let mj = masses_slice[j];

                    let dx = pj[0] - tx;
                    let dy = pj[1] - ty;
                    let dz = pj[2] - tz;
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                    if has_softening {
                        let hj = softenings.map(|hs| hs[j]).unwrap_or(0.0);
                        let h = hj.max(0.0);
                        if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                            let invr = 1.0 / (r2 + R2_TINY).sqrt();
                            acc.add_newton(dx, dy, dz, mj, invr);
                        } else {
                            let r = (r2 + R2_TINY).sqrt();
                            acc.add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                        }
                    } else {
                        let invr = 1.0 / (r2 + R2_TINY).sqrt();
                        acc.add_newton(dx, dy, dz, mj, invr);
                    }
                }

                acc.finish()
            })
            .collect()
    } else {
        targets
            .par_iter()
            .map(|tgt| {
                let mut acc = A::init();
                let tx = tgt[0];
                let ty = tgt[1];
                let tz = tgt[2];

                for j in 0..n_src {
                    let pj = positions[j];
                    let mj = masses_slice[j];

                    let dx = pj[0] - tx;
                    let dy = pj[1] - ty;
                    let dz = pj[2] - tz;
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

                    if has_softening {
                        let hj = softenings.map(|hs| hs[j]).unwrap_or(0.0);
                        let h = hj.max(0.0);
                        if h <= 0.0 || (kernel_is_spline && r2 >= h * h) {
                            let invr = 1.0 / (r2 + R2_TINY).sqrt();
                            acc.add_newton(dx, dy, dz, mj, invr);
                        } else {
                            let r = (r2 + R2_TINY).sqrt();
                            acc.add_softened(dx, dy, dz, mj, r, h, kernel_kind);
                        }
                    } else {
                        let invr = 1.0 / (r2 + R2_TINY).sqrt();
                        acc.add_newton(dx, dy, dz, mj, invr);
                    }
                }

                acc.finish()
            })
            .collect()
    }
}

// ===========================================================================
// Public API — thin wrappers over the generic implementations
// ===========================================================================

/// Direct-sum O(N²) gravitational accelerations (Newtonian, no softening).
pub fn direct_accelerations(positions: &[[f64; 3]], masses: Option<&[f64]>) -> Vec<[f64; 3]> {
    direct_self_impl::<AccelAccumulator>(positions, masses, None, None)
}

/// Direct-sum O(N²) gravitational accelerations at arbitrary target points.
pub fn direct_accelerations_at_points(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    targets: &[[f64; 3]],
) -> Vec<[f64; 3]> {
    direct_at_points_impl::<AccelAccumulator>(positions, masses, None, targets, None)
}

/// Direct-sum O(N²) gravitational potentials (Newtonian, no softening).
pub fn direct_potentials(positions: &[[f64; 3]], masses: Option<&[f64]>) -> Vec<f64> {
    direct_self_impl::<PotAccumulator>(positions, masses, None, None)
}

/// Direct-sum O(N²) gravitational potentials at arbitrary target points.
pub fn direct_potentials_at_points(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    targets: &[[f64; 3]],
) -> Vec<f64> {
    direct_at_points_impl::<PotAccumulator>(positions, masses, None, targets, None)
}

/// Direct-sum O(N²) gravitational potentials with softening kernel.
pub fn direct_potentials_kernel(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    softenings: Option<&[f64]>,
    kernel: KernelKind,
) -> Vec<f64> {
    direct_self_impl::<PotAccumulator>(positions, masses, softenings, Some(kernel))
}

/// Direct-sum O(N²) gravitational accelerations with softening kernel.
pub fn direct_accelerations_kernel(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    softenings: Option<&[f64]>,
    kernel: KernelKind,
) -> Vec<[f64; 3]> {
    direct_self_impl::<AccelAccumulator>(positions, masses, softenings, Some(kernel))
}

/// Direct-sum softened gravitational potentials at arbitrary target points.
pub fn direct_potentials_kernel_at_points(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    softenings: Option<&[f64]>,
    targets: &[[f64; 3]],
    kernel: KernelKind,
) -> Vec<f64> {
    direct_at_points_impl::<PotAccumulator>(positions, masses, softenings, targets, Some(kernel))
}

/// Direct-sum softened gravitational accelerations at arbitrary target points.
pub fn direct_accelerations_kernel_at_points(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    softenings: Option<&[f64]>,
    targets: &[[f64; 3]],
    kernel: KernelKind,
) -> Vec<[f64; 3]> {
    direct_at_points_impl::<AccelAccumulator>(positions, masses, softenings, targets, Some(kernel))
}
