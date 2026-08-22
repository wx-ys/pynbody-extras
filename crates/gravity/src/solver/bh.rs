use crate::direct;
use crate::kernel::KernelKind;
use crate::octree::Octree;
use crate::solver::GravitySolver;
use crate::types::ParticleData;

// ===========================================================================
// DirectSolver — wraps the direct-sum functions into the GravitySolver trait
// ===========================================================================

/// Direct pairwise O(N²) gravity solver with optional softening.
pub struct DirectSolver {
    positions: Vec<[f64; 3]>,
    masses: Option<Vec<f64>>,
    softenings: Option<Vec<f64>>,
    kernel: KernelKind,
}

impl DirectSolver {
    pub fn new(particles: &ParticleData, kernel: KernelKind) -> Self {
        Self {
            positions: particles.positions.clone(),
            masses: particles.masses.clone(),
            softenings: particles.softenings.clone(),
            kernel,
        }
    }
}

impl GravitySolver for DirectSolver {
    fn compute_accelerations(&self, out: &mut [[f64; 3]]) {
        let acc = if self.softenings.is_some() {
            direct::direct_accelerations_kernel(
                &self.positions,
                self.masses.as_deref(),
                self.softenings.as_deref(),
                self.kernel,
            )
        } else {
            direct::direct_accelerations(&self.positions, self.masses.as_deref())
        };
        out.copy_from_slice(&acc);
    }

    fn compute_potentials(&self, out: &mut [f64]) {
        let pot = if self.softenings.is_some() {
            direct::direct_potentials_kernel(
                &self.positions,
                self.masses.as_deref(),
                self.softenings.as_deref(),
                self.kernel,
            )
        } else {
            direct::direct_potentials(&self.positions, self.masses.as_deref())
        };
        out.copy_from_slice(&pot);
    }

    fn accelerations_at_points(&self, targets: &[[f64; 3]], out: &mut [[f64; 3]]) {
        let acc = if self.softenings.is_some() {
            direct::direct_accelerations_kernel_at_points(
                &self.positions,
                self.masses.as_deref(),
                self.softenings.as_deref(),
                targets,
                self.kernel,
            )
        } else {
            direct::direct_accelerations_at_points(
                &self.positions,
                self.masses.as_deref(),
                targets,
            )
        };
        out.copy_from_slice(&acc);
    }

    fn potentials_at_points(&self, targets: &[[f64; 3]], out: &mut [f64]) {
        let pot = if self.softenings.is_some() {
            direct::direct_potentials_kernel_at_points(
                &self.positions,
                self.masses.as_deref(),
                self.softenings.as_deref(),
                targets,
                self.kernel,
            )
        } else {
            direct::direct_potentials_at_points(
                &self.positions,
                self.masses.as_deref(),
                targets,
            )
        };
        out.copy_from_slice(&pot);
    }
}

// ===========================================================================
// BhTreeSolver — Barnes-Hut tree-based gravity solver
// ===========================================================================

/// Barnes-Hut tree gravity solver with multipole expansion.
pub struct BhTreeSolver {
    tree: Octree,
    theta: f64,
}

impl BhTreeSolver {
    /// Build a BH tree solver from particle data.
    ///
    /// This constructs the octree, precomputes all node-level payloads
    /// (center-of-mass, hmax, multipole moments) so that repeated
    /// calls to compute methods are fast.
    pub fn new(
        particles: &ParticleData,
        theta: f64,
        leaf_capacity: usize,
        multipole_order: u8,
        kernel: KernelKind,
    ) -> Self {
        let mut tree = Octree::from_owned(
            particles.positions.clone(),
            particles.masses.clone(),
            particles.softenings.clone(),
            leaf_capacity,
            multipole_order,
            kernel,
        );
        if tree.masses.is_some() {
            tree.build_mass_payload();
        }
        // Even without masses, call build_mass_payload to initialize bh/hmax.
        if tree.masses.is_none() {
            tree.build_mass_payload();
        }
        Self { tree, theta }
    }

    /// Return a reference to the underlying octree.
    pub fn tree(&self) -> &Octree {
        &self.tree
    }
}

impl GravitySolver for BhTreeSolver {
    fn compute_accelerations(&self, out: &mut [[f64; 3]]) {
        self.tree.compute_accelerations(self.theta, out);
    }

    fn compute_potentials(&self, out: &mut [f64]) {
        self.tree.compute_potentials(self.theta, out);
    }

    fn accelerations_at_points(&self, targets: &[[f64; 3]], out: &mut [[f64; 3]]) {
        self.tree.accelerations_at_points(targets, self.theta, out);
    }

    fn potentials_at_points(&self, targets: &[[f64; 3]], out: &mut [f64]) {
        self.tree.potentials_at_points(targets, self.theta, out);
    }
}
