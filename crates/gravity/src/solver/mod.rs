use crate::kernel::KernelKind;
use crate::types::ParticleData;

pub mod bh;

/// Common interface for gravity solvers.
///
/// Implementations include direct pairwise (PP), Barnes-Hut tree, FMM, and PM.
/// All solvers operate on pre-allocated output slices.
pub trait GravitySolver {
    /// Compute gravitational accelerations for all source particles (self-gravity).
    fn compute_accelerations(&self, out: &mut [[f64; 3]]);

    /// Compute gravitational potentials for all source particles (self-gravity).
    fn compute_potentials(&self, out: &mut [f64]);

    /// Compute gravitational accelerations at arbitrary target positions.
    fn accelerations_at_points(&self, targets: &[[f64; 3]], out: &mut [[f64; 3]]);

    /// Compute gravitational potentials at arbitrary target positions.
    fn potentials_at_points(&self, targets: &[[f64; 3]], out: &mut [f64]);
}

/// Configuration for choosing and building a gravity solver at runtime.
#[derive(Clone, Debug)]
pub enum GravityMethod {
    /// Direct pairwise O(N²), optionally with softening.
    Direct {
        kernel: KernelKind,
    },
    /// Barnes-Hut tree code with multipole expansion.
    BhTree {
        theta: f64,
        leaf_capacity: usize,
        multipole_order: u8,
        kernel: KernelKind,
    },
    // Future: FmmTree { ... }, Pm { ... }
}

impl GravityMethod {
    /// Build the corresponding solver from particle data.
    pub fn build(&self, particles: &ParticleData) -> Box<dyn GravitySolver + Send + Sync> {
        match self {
            Self::Direct { kernel } => {
                Box::new(bh::DirectSolver::new(particles, *kernel))
            }
            Self::BhTree {
                theta,
                leaf_capacity,
                multipole_order,
                kernel,
            } => Box::new(bh::BhTreeSolver::new(
                particles,
                *theta,
                *leaf_capacity,
                *multipole_order,
                *kernel,
            )),
        }
    }
}
