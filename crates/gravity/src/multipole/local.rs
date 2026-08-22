//! Local (Taylor) expansion of the gravitational field — FMM support.
//!
//! In FMM, distant multipole sources are converted to a local Taylor expansion
//! about the target cell's center. This expansion can then be evaluated at each
//! particle in the cell (L2P) and translated to child cells (L2L).
//!
//! The M2L operator reuses [`PotentialDerivatives`](super::derivatives::PotentialDerivatives)
//! as the interaction kernel: the multipole moments act as "charges" and the
//! derivatives give the Taylor coefficients.

use super::moment::MultipoleMoment;

/// Local Taylor expansion of the gravitational potential about a center.
///
/// The potential at a point `r` (relative to the expansion center) is:
///
/// ```text
/// phi(r) = sum_{l,m,n} L_{lmn} * x^l * y^m * z^n
/// ```
///
/// where `L_{lmn}` are the Taylor coefficients stored in `coeffs`.
#[derive(Clone, Debug)]
pub struct LocalExpansion {
    /// Taylor coefficients, stored in the same lmn ordering as
    /// [`MultipoleMoment`] but interpreted as powers (not factorial-normalized).
    pub coeffs: Vec<f64>,
    /// Expansion order (max l+m+n).
    pub order: u8,
}

impl LocalExpansion {
    /// Create a zero local expansion of the given order.
    pub fn new(order: u8) -> Self {
        let n_coeffs = num_coefficients(order);
        Self {
            coeffs: vec![0.0; n_coeffs],
            order,
        }
    }

    /// M2L: convert a multipole expansion to a local expansion.
    ///
    /// `shift` = target_center - source_center.  The multipole is expanded
    /// about `source_center`; the resulting local expansion is about `target_center`.
    ///
    /// This uses the same derivative machinery as the M2P evaluators: the
    /// multipole moment `m` acts as a source, and the derivatives of 1/r
    /// evaluated at `shift` become the Taylor coefficients of the local field.
    pub fn from_multipole(m: &MultipoleMoment, shift: [f64; 3], order: u8) -> Self {
        let o = order.min(5);
        let n_coeffs = num_coefficients(o);
        let mut coeffs = vec![0.0; n_coeffs];

        // For each (l,m,n) up to order, the local coefficient L_{lmn} is
        // the lmn-th derivative of the potential from the multipole source.
        // This is computed using the full PotentialDerivatives at the given shift.
        //
        // TODO: implement the actual M2L formula using the derivative tensors.
        // For now, we store the order and leave a placeholder.
        let _ = (m, shift, o);
        coeffs[0] = -m.m000 / (shift[0].powi(2) + shift[1].powi(2) + shift[2].powi(2)).sqrt();

        Self { coeffs, order: o }
    }

    /// L2L: translate this local expansion to a new center.
    ///
    /// `shift` = new_center - old_center.
    pub fn translate(&self, shift: [f64; 3]) -> Self {
        // TODO: implement L2L shift formula
        let _ = shift;
        self.clone()
    }

    /// L2P: evaluate the potential at a position relative to the expansion center.
    pub fn evaluate_potential(&self, pos: &[f64; 3]) -> f64 {
        // TODO: implement L2P via power-series evaluation
        let _ = pos;
        self.coeffs.first().copied().unwrap_or(0.0)
    }

    /// L2P: evaluate the acceleration at a position relative to the expansion center.
    pub fn evaluate_acceleration(&self, pos: &[f64; 3]) -> [f64; 3] {
        // TODO: implement L2P acceleration via gradient of power-series
        let _ = pos;
        [0.0, 0.0, 0.0]
    }

    /// Add another local expansion (same center) to this one.
    pub fn add_assign(&mut self, other: &LocalExpansion) {
        let n = self.coeffs.len().min(other.coeffs.len());
        for i in 0..n {
            self.coeffs[i] += other.coeffs[i];
        }
    }
}

/// Number of independent (l,m,n) coefficients for l+m+n <= order.
fn num_coefficients(order: u8) -> usize {
    let o = order as usize;
    (o + 1) * (o + 2) * (o + 3) / 6
}
