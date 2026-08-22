//! Multipole expansion types for tree-based gravity solvers.
//!
//! # Sub-modules
//!
//! - [`moment`] — full and compact moment storage, P2M, M2M translation
//! - [`derivatives`] — softened 1/r derivatives (full + compact variants)
//! - [`eval`] — per-order evaluator trait + implementations (replaces old macro dispatch)
//! - [`local`] — local Taylor expansion for FMM (M2L, L2L, L2P)

pub mod derivatives;
pub mod eval;
pub mod local;
pub mod moment;

// Re-export the key types used by other modules.
pub use derivatives::{
    PotentialDerivatives, PotentialDerivatives1, PotentialDerivatives2, PotentialDerivatives3,
    PotentialDerivatives4,
};
pub use eval::{gravity_accel_multipole, gravity_potential_multipole};
pub use moment::{
    translate_multipole, Moment0, Moment2, Moment3, Moment4, Moment5, MultipoleMoment,
    MultipoleMoments,
};

// Crate-internal exports (used by traversal.rs and octree.rs).
pub(crate) use eval::{
    MultipoleOrder, Order0Eval, Order2Eval, Order3Eval, Order4Eval, Order5Eval,
};
