use ndarray::ArrayView2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

use gravity::{direct, Octree as CoreOctree, Tree3D};

/// Helper: run a closure inside a dedicated rayon thread pool
/// when `threads > 0`; otherwise execute it directly (using the
/// global Rayon pool or pure serial code, depending on caller).
fn with_thread_pool<F, R>(threads: usize, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if threads > 0 {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build rayon thread pool");
        pool.install(f)
    } else {
        f()
    }
}

/// Python-facing Octree class backed by gravity::Octree.
#[pyclass]
pub struct Octree {
    inner: CoreOctree,
}

#[pymethods]
impl Octree {
    #[new]
    #[pyo3(signature = (positions, masses=None, leaf_capacity=32, multipole_order=0))]
    fn new(
        positions: &PyArray2<f64>,
        masses: Option<&PyArray1<f64>>, 
        leaf_capacity: usize,
        multipole_order: u8,
    ) -> PyResult<Self> {
        let arr: ArrayView2<'_, f64> = unsafe { positions.as_array() };
        if arr.ndim() != 2 || arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("positions must be (N,3) float64 array"));
        }
        let n = arr.shape()[0];
        let mut pos: Vec<[f64; 3]> = Vec::with_capacity(n);
        for i in 0..n {
            pos.push([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
        }
        let masses_vec: Option<Vec<f64>> = if let Some(m_arr) = masses {
            let slice = unsafe { m_arr.as_slice()? };
            if slice.len() != n {
                return Err(PyValueError::new_err("masses must be length N"));
            }
            Some(slice.to_vec())
        } else {
            None
        };
        let mut inner = CoreOctree::from_owned(pos, masses_vec, leaf_capacity, multipole_order);
        if inner.masses.is_some() {
            inner.build_mass_payload();
        }
        Ok(Octree { inner })
    }

    fn build_mass(&mut self, masses: Option<&PyArray1<f64>>) -> PyResult<()> {
        let n = self.inner.positions.len();
        if let Some(m_arr) = masses {
            let slice = unsafe { m_arr.as_slice()? };
            if slice.len() != n {
                return Err(PyValueError::new_err("masses must be length N"));
            }
            self.inner.set_masses(Some(slice.to_vec()));
        }
        self.inner.build_mass_payload();
        Ok(())
    }

    #[pyo3(signature = (theta, eps, threads=0))]
    fn compute_forces<'py>(
        &self,
        py: Python<'py>,
        theta: f64,
        eps: f64,
        threads: usize,
    ) -> PyResult<&'py PyArray2<f64>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before compute_forces",
            ));
        }
        let n = self.inner.positions.len();
        let mut out = vec![[0.0f64; 3]; n];

        py.allow_threads(|| {
            if threads == 0 {
                // Use global Rayon thread pool (default parallelism).
                <CoreOctree as Tree3D>::compute_forces(&self.inner, theta, eps, &mut out);
            } else {
                // Use a dedicated pool with exactly `threads` workers.
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::compute_forces(&self.inner, theta, eps, &mut out);
                });
            }
        });

        let mut arr = Vec::with_capacity(n * 3);
        for i in 0..n {
            arr.push(out[i][0]);
            arr.push(out[i][1]);
            arr.push(out[i][2]);
        }
        let array = Array2FromVec::from_vec((n, 3), arr);
        Ok(array.into_pyarray(py))
    }

    #[pyo3(signature = (theta, eps, threads=0))]
    fn compute_potentials<'py>(
        &self,
        py: Python<'py>,
        theta: f64,
        eps: f64,
        threads: usize,
    ) -> PyResult<&'py PyArray1<f64>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before compute_potentials",
            ));
        }
        let n = self.inner.positions.len();
        let mut out = vec![0.0f64; n];

        py.allow_threads(|| {
            if threads == 0 {
                <CoreOctree as Tree3D>::compute_potentials(&self.inner, theta, eps, &mut out);
            } else {
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::compute_potentials(&self.inner, theta, eps, &mut out);
                });
            }
        });

        Ok(out.into_pyarray(py))
    }

    #[pyo3(signature = (points, theta, eps, threads=0))]
    fn forces_at_points<'py>(
        &self,
        py: Python<'py>,
        points: &PyArray2<f64>,
        theta: f64,
        eps: f64,
        threads: usize,
    ) -> PyResult<&'py PyArray2<f64>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before forces_at_points",
            ));
        }

        let arr: ArrayView2<'_, f64> = unsafe { points.as_array() };
        if arr.ndim() != 2 || arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("points must be (M,3) float64 array"));
        }
        let m = arr.shape()[0];
        let mut pts: Vec<[f64; 3]> = Vec::with_capacity(m);
        for i in 0..m {
            pts.push([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
        }

        let mut out = vec![[0.0f64; 3]; m];

        py.allow_threads(|| {
            if threads == 0 {
                <CoreOctree as Tree3D>::force_at_points(&self.inner, &pts, theta, eps, &mut out);
            } else {
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::force_at_points(&self.inner, &pts, theta, eps, &mut out);
                });
            }
        });

        let mut flat = Vec::with_capacity(m * 3);
        for i in 0..m {
            flat.push(out[i][0]);
            flat.push(out[i][1]);
            flat.push(out[i][2]);
        }
        let array = Array2FromVec::from_vec((m, 3), flat);
        Ok(array.into_pyarray(py))
    }

    #[pyo3(signature = (points, theta, eps, threads=0))]
    fn potentials_at_points<'py>(
        &self,
        py: Python<'py>,
        points: &PyArray2<f64>,
        theta: f64,
        eps: f64,
        threads: usize,
    ) -> PyResult<&'py PyArray1<f64>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before potentials_at_points",
            ));
        }

        let arr: ArrayView2<'_, f64> = unsafe { points.as_array() };
        if arr.ndim() != 2 || arr.shape()[1] != 3 {
            return Err(PyValueError::new_err("points must be (M,3) float64 array"));
        }
        let m = arr.shape()[0];
        let mut pts: Vec<[f64; 3]> = Vec::with_capacity(m);
        for i in 0..m {
            pts.push([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
        }

        let mut out = vec![0.0f64; m];

        py.allow_threads(|| {
            if threads == 0 {
                <CoreOctree as Tree3D>::potential_at_points(&self.inner, &pts, theta, eps, &mut out);
            } else {
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::potential_at_points(&self.inner, &pts, theta, eps, &mut out);
                });
            }
        });

        Ok(out.into_pyarray(py))
    }
}

/// Direct-sum O(N^2) accelerations.
#[pyfunction]
#[pyo3(signature = (positions, masses=None, eps=0.0, threads=0))]
pub fn direct_accelerations_py<'py>(
    py: Python<'py>,
    positions: &PyArray2<f64>,
    masses: Option<&PyArray1<f64>>,
    eps: f64,
    threads: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let arr: ArrayView2<'_, f64> = unsafe { positions.as_array() };
    if arr.ndim() != 2 || arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must be (N,3) float64 array"));
    }
    let n = arr.shape()[0];
    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        pos.push([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
    }
    let masses_slice: Option<Vec<f64>> = if let Some(m_arr) = masses {
        let slice = unsafe { m_arr.as_slice()? };
        if slice.len() != n {
            return Err(PyValueError::new_err("masses must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let acc = py.allow_threads(|| {
        if threads == 0 {
            direct::direct_accelerations(&pos, masses_slice.as_deref(), eps)
        } else {
            with_thread_pool(threads, || {
                direct::direct_accelerations(&pos, masses_slice.as_deref(), eps)
            })
        }
    });

    let mut flat = Vec::with_capacity(n * 3);
    for i in 0..n {
        flat.push(acc[i][0]);
        flat.push(acc[i][1]);
        flat.push(acc[i][2]);
    }
    let array = Array2FromVec::from_vec((n, 3), flat);
    Ok(array.into_pyarray(py))
}

/// Direct-sum O(N^2) potentials.
#[pyfunction]
#[pyo3(signature = (positions, masses=None, eps=0.0, threads=0))]
pub fn direct_potentials_py<'py>(
    py: Python<'py>,
    positions: &PyArray2<f64>,
    masses: Option<&PyArray1<f64>>,
    eps: f64,
    threads: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let arr: ArrayView2<'_, f64> = unsafe { positions.as_array() };
    if arr.ndim() != 2 || arr.shape()[1] != 3 {
        return Err(PyValueError::new_err("positions must be (N,3) float64 array"));
    }
    let n = arr.shape()[0];
    let mut pos: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        pos.push([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
    }
    let masses_slice: Option<Vec<f64>> = if let Some(m_arr) = masses {
        let slice = unsafe { m_arr.as_slice()? };
        if slice.len() != n {
            return Err(PyValueError::new_err("masses must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let pot = py.allow_threads(|| {
        if threads == 0 {
            direct::direct_potentials(&pos, masses_slice.as_deref(), eps)
        } else {
            with_thread_pool(threads, || {
                direct::direct_potentials(&pos, masses_slice.as_deref(), eps)
            })
        }
    });

    Ok(pot.into_pyarray(py))
}

struct Array2FromVec {
    shape: (usize, usize),
    data: Vec<f64>,
}
impl Array2FromVec {
    fn from_vec(shape: (usize, usize), data: Vec<f64>) -> Self {
        Array2FromVec { shape, data }
    }
    fn into_pyarray<'py>(self, py: Python<'py>) -> &'py PyArray2<f64> {
        let (n, m) = self.shape;
        let arr = ndarray::Array::from_shape_vec((n, m), self.data).unwrap();
        arr.into_pyarray(py)
    }
}
