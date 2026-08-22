pub mod boundary;
pub mod direct;
pub mod kernel;
pub mod multipole;
pub mod octree;
pub mod solver;
pub mod traversal;
pub mod types;

// Convenience re-exports
pub use octree::Octree;
pub use solver::bh::BhTreeSolver;
pub use solver::GravityMethod;
pub use solver::GravitySolver;
