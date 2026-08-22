/// Physical quantity to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GravityQuantity {
    Acceleration,
    Potential,
}

/// Where to evaluate gravity.
#[derive(Clone, Debug)]
pub enum TargetType<'a> {
    /// Self-gravity: evaluate at each source particle's position.
    Self_,
    /// Evaluate at specified positions (N×M asymmetric).
    AtPoints(&'a [[f64; 3]]),
}

/// Bundled input data for gravity computation.
#[derive(Clone, Debug)]
pub struct ParticleData {
    pub positions: Vec<[f64; 3]>,
    pub masses: Option<Vec<f64>>,
    pub softenings: Option<Vec<f64>>,
}

impl ParticleData {
    /// Number of particles.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns true if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

/// Gravity computation result.
#[derive(Clone, Debug)]
pub enum GravityResult {
    Accelerations(Vec<[f64; 3]>),
    Potentials(Vec<f64>),
}
