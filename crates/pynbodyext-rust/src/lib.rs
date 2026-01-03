use pyo3::prelude::*;

mod gravity;

/// Python bindings entry point for pynbodyext Rust extensions.
///
/// This module aggregates Rust-backed functionality from the
/// internal `tree` and `gravity` crates and exposes it under
/// the `pynbodyext._rust` namespace.
#[pymodule]
fn _rust<'py>(_py: Python<'py>, m: &pyo3::prelude::Bound<'py, PyModule>) -> PyResult<()> {
    // Expose the Octree class and direct-sum functions
    // implemented in the internal gravity module.
    m.add_class::<gravity::Octree>()?;
    m.add_function(pyo3::wrap_pyfunction!(gravity::direct_accelerations_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(gravity::direct_potentials_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        gravity::direct_accelerations_at_points_py,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        gravity::direct_potentials_at_points_py,
        m
    )?)?;

    Ok(())
}
