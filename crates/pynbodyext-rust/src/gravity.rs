use numpy::ndarray::{Array, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use gravity::kernel::KernelKind;
use gravity::{direct, Octree as CoreOctree, Tree3D};

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
    eprintln!(
        "[pynbodyext-timing] {label}: {:.3} ms",
        dt.as_secs_f64() * 1e3
    );
}

fn extract_vec3_from_pyarray2<'py>(
    arr2: PyReadonlyArray2<'py, f64>,
    name: &str,
) -> PyResult<Vec<[f64; 3]>> {
    // Fast path: contiguous (C-order) array -> slice -> chunk.
    if let Ok(slice) = arr2.as_slice() {
        if slice.len() % 3 != 0 {
            return Err(PyValueError::new_err(format!(
                "{name} must be (N,3) float64 array"
            )));
        }
        let n = slice.len() / 3;
        let mut out: Vec<[f64; 3]> = Vec::with_capacity(n);
        for c in slice.chunks_exact(3) {
            out.push([c[0], c[1], c[2]]);
        }
        return Ok(out);
    }

    // Fallback: generic (possibly non-contiguous) path.
    let arr: ArrayView2<'_, f64> = arr2.as_array();
    if arr.ndim() != 2 || arr.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must be (N,3) float64 array"
        )));
    }
    let n = arr.shape()[0];
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        out.push([arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
    }
    Ok(out)
}

fn parse_kernel(kernel: u8) -> PyResult<KernelKind> {
    match kernel {
        0 => Ok(KernelKind::Plummer),
        1 => Ok(KernelKind::CubicSplineW2),
        _ => Err(PyValueError::new_err(
            "kernel must be 0 (Plummer) or 1 (CubicSplineW2)",
        )),
    }
}

fn parse_kernel_opt(kernel: Option<u8>) -> PyResult<KernelKind> {
    match kernel {
        Some(k) => parse_kernel(k),
        None => Ok(KernelKind::Plummer),
    }
}

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

#[inline]
#[allow(deprecated)]
fn release_gil<F, R>(py: Python<'_>, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    py.allow_threads(f)
}

/// Python-facing Octree class backed by gravity::Octree.
#[pyclass]
pub struct Octree {
    inner: CoreOctree,
}

#[pymethods]
impl Octree {
    #[new]
    // Keep positional-arg compatibility: leaf_capacity is still 3rd positional.
    #[pyo3(signature = (positions, masses=None, leaf_capacity=32, multipole_order=0, softenings=None, kernel=None))]
    fn new<'py>(
        positions: PyReadonlyArray2<'py, f64>,
        masses: Option<PyReadonlyArray1<'py, f64>>,
        leaf_capacity: usize,
        multipole_order: u8,
        softenings: Option<PyReadonlyArray1<'py, f64>>,
        kernel: Option<u8>,
    ) -> PyResult<Self> {
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
        let pos = extract_vec3_from_pyarray2(positions, "positions")?;
        if let Some(t0) = t0 {
            log_timing("octree.new.extract_positions", t0.elapsed());
        }
        let n = pos.len();

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let masses_vec: Option<Vec<f64>> = if let Some(m_arr) = masses {
            let slice = m_arr.as_slice()?;
            if slice.len() != n {
                return Err(PyValueError::new_err("masses must be length N"));
            }
            Some(slice.to_vec())
        } else {
            None
        };
        if let Some(t0) = t0 {
            log_timing("octree.new.copy_masses", t0.elapsed());
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let softenings_vec: Option<Vec<f64>> = if let Some(h_arr) = softenings {
            let slice = h_arr.as_slice()?;
            if slice.len() != n {
                return Err(PyValueError::new_err("softenings must be length N"));
            }
            Some(slice.to_vec())
        } else {
            None
        };
        if let Some(t0) = t0 {
            log_timing("octree.new.copy_softenings", t0.elapsed());
        }

        if kernel.is_none() && softenings_vec.is_some() {
            return Err(PyValueError::new_err(
                "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)",
            ));
        }

        let kernel_kind = parse_kernel_opt(kernel)?;

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let mut inner = CoreOctree::from_owned(
            pos,
            masses_vec,
            softenings_vec,
            leaf_capacity,
            multipole_order,
            kernel_kind,
        );
        if let Some(t0) = t0 {
            log_timing("octree.new.core_from_owned", t0.elapsed());
        }

        if inner.masses.is_some() {
            let t0 = if timing_enabled() {
                Some(Instant::now())
            } else {
                None
            };
            inner.build_mass_payload();
            if let Some(t0) = t0 {
                log_timing("octree.new.build_mass_payload", t0.elapsed());
            }
        }

        if let Some(t_all) = t_all {
            log_timing("octree.new.total", t_all.elapsed());
        }
        Ok(Octree { inner })
    }

    fn build_mass<'py>(&mut self, masses: Option<PyReadonlyArray1<'py, f64>>) -> PyResult<()> {
        let n = self.inner.positions.len();
        if let Some(m_arr) = masses {
            let slice = m_arr.as_slice()?;
            if slice.len() != n {
                return Err(PyValueError::new_err("masses must be length N"));
            }
            self.inner.set_masses(Some(slice.to_vec()));
        }
        self.inner.build_mass_payload();
        Ok(())
    }

    #[pyo3(signature = (softenings=None))]
    fn set_softenings<'py>(
        &mut self,
        softenings: Option<PyReadonlyArray1<'py, f64>>,
    ) -> PyResult<()> {
        let n = self.inner.positions.len();
        let softenings_vec: Option<Vec<f64>> = if let Some(h_arr) = softenings {
            let slice = h_arr.as_slice()?;
            if slice.len() != n {
                return Err(PyValueError::new_err("softenings must be length N"));
            }
            Some(slice.to_vec())
        } else {
            None
        };
        self.inner.set_softenings(softenings_vec);
        Ok(())
    }

    #[pyo3(signature = (kernel=None))]
    fn set_kernel(&mut self, kernel: Option<u8>) -> PyResult<()> {
        let kernel_kind = parse_kernel_opt(kernel)?;
        self.inner.set_kernel(kernel_kind);
        Ok(())
    }

    #[pyo3(signature = (theta, threads=0))]
    fn compute_accelerations<'py>(
        &self,
        py: Python<'py>,
        theta: f64,
        threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before compute_accelerations",
            ));
        }
        let n = self.inner.positions.len();
        // Allocate a flat buffer that will become the NumPy array.
        // We temporarily reinterpret it as &mut [[f64;3]] for the core compute.
        let mut flat = vec![0.0f64; n * 3];
        let out: &mut [[f64; 3]] =
            unsafe { std::slice::from_raw_parts_mut(flat.as_mut_ptr() as *mut [f64; 3], n) };

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        release_gil(py, || {
            if threads == 0 {
                // Use global Rayon thread pool (default parallelism).
                <CoreOctree as Tree3D>::compute_accelerations(&self.inner, theta, out);
            } else {
                // Use a dedicated pool with exactly `threads` workers.
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::compute_accelerations(&self.inner, theta, out);
                });
            }
        });
        if let Some(t0) = t0 {
            log_timing("octree.compute_accelerations.core", t0.elapsed());
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let out_arr = Array::from_shape_vec((n, 3), flat)
            .expect("shape mismatch building (n,3) array")
            .into_pyarray(py);
        if let Some(t0) = t0 {
            log_timing("octree.compute_accelerations.to_numpy", t0.elapsed());
        }
        Ok(out_arr)
    }

    #[pyo3(signature = (theta, threads=0))]
    fn compute_potentials<'py>(
        &self,
        py: Python<'py>,
        theta: f64,
        threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before compute_potentials",
            ));
        }
        let n = self.inner.positions.len();
        let mut out = vec![0.0f64; n];

        release_gil(py, || {
            if threads == 0 {
                <CoreOctree as Tree3D>::compute_potentials(&self.inner, theta, &mut out);
            } else {
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::compute_potentials(&self.inner, theta, &mut out);
                });
            }
        });

        Ok(out.into_pyarray(py))
    }

    #[pyo3(signature = (points, theta, threads=0))]
    fn accelerations_at_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        theta: f64,
        threads: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before accelerations_at_points",
            ));
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let pts = extract_vec3_from_pyarray2(points, "points")?;
        if let Some(t0) = t0 {
            log_timing("octree.accel_at_points.extract_points", t0.elapsed());
        }
        let m = pts.len();

        let mut flat = vec![0.0f64; m * 3];
        let out: &mut [[f64; 3]] =
            unsafe { std::slice::from_raw_parts_mut(flat.as_mut_ptr() as *mut [f64; 3], m) };

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        release_gil(py, || {
            if threads == 0 {
                <CoreOctree as Tree3D>::accelerations_at_points(&self.inner, &pts, theta, out);
            } else {
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::accelerations_at_points(&self.inner, &pts, theta, out);
                });
            }
        });
        if let Some(t0) = t0 {
            log_timing("octree.accel_at_points.core", t0.elapsed());
        }

        let t0 = if timing_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        let out_arr = Array::from_shape_vec((m, 3), flat)
            .expect("shape mismatch building (m,3) array")
            .into_pyarray(py);
        if let Some(t0) = t0 {
            log_timing("octree.accel_at_points.to_numpy", t0.elapsed());
        }
        Ok(out_arr)
    }

    #[pyo3(signature = (points, theta, threads=0))]
    fn potentials_at_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
        theta: f64,
        threads: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if self.inner.bh.is_none() {
            return Err(PyValueError::new_err(
                "mass payload not built; call build_mass() before potentials_at_points",
            ));
        }

        let pts = extract_vec3_from_pyarray2(points, "points")?;
        let m = pts.len();

        let mut out = vec![0.0f64; m];

        release_gil(py, || {
            if threads == 0 {
                <CoreOctree as Tree3D>::potentials_at_points(&self.inner, &pts, theta, &mut out);
            } else {
                with_thread_pool(threads, || {
                    <CoreOctree as Tree3D>::potentials_at_points(
                        &self.inner,
                        &pts,
                        theta,
                        &mut out,
                    );
                });
            }
        });

        Ok(out.into_pyarray(py))
    }
}

/// Direct-sum O(N^2) accelerations.
#[pyfunction]
#[pyo3(signature = (positions, masses=None, threads=0, softenings=None, kernel=None))]
pub fn direct_accelerations_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    masses: Option<PyReadonlyArray1<'py, f64>>,
    threads: usize,
    softenings: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<u8>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pos = extract_vec3_from_pyarray2(positions, "positions")?;
    let n = pos.len();
    let masses_slice: Option<Vec<f64>> = if let Some(m_arr) = masses {
        let slice = m_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("masses must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let softenings_slice: Option<Vec<f64>> = if let Some(h_arr) = softenings {
        let slice = h_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("softenings must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    if kernel.is_none() && softenings_slice.is_some() {
        return Err(PyValueError::new_err(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)",
        ));
    }

    let kernel_kind = kernel.map(parse_kernel).transpose()?;

    let acc = release_gil(py, || {
        let run = || match kernel_kind {
            None => direct::direct_accelerations(&pos, masses_slice.as_deref()),
            Some(kind) => direct::direct_accelerations_kernel(
                &pos,
                masses_slice.as_deref(),
                softenings_slice.as_deref(),
                kind,
            ),
        };

        if threads == 0 {
            run()
        } else {
            with_thread_pool(threads, run)
        }
    });

    let mut flat = Vec::with_capacity(n * 3);
    for a in acc.iter() {
        flat.extend(a.iter().copied());
    }
    let arr = Array::from_shape_vec((n, 3), flat).expect("shape mismatch building (n,3) array");
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (positions, targets, masses=None, threads=0, softenings=None, kernel=None))]
pub fn direct_accelerations_at_points_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    masses: Option<PyReadonlyArray1<'py, f64>>,
    threads: usize,
    softenings: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<u8>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pos = extract_vec3_from_pyarray2(positions, "positions")?;
    let n = pos.len();
    let masses_slice: Option<Vec<f64>> = if let Some(m_arr) = masses {
        let slice = m_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("masses must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let softenings_slice: Option<Vec<f64>> = if let Some(h_arr) = softenings {
        let slice = h_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("softenings must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let tgt = extract_vec3_from_pyarray2(targets, "targets")?;
    let m = tgt.len();

    if kernel.is_none() && softenings_slice.is_some() {
        return Err(PyValueError::new_err(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)",
        ));
    }

    let kernel_kind = kernel.map(parse_kernel).transpose()?;

    let acc = release_gil(py, || {
        let run = || match kernel_kind {
            None => direct::direct_accelerations_at_points(&pos, masses_slice.as_deref(), &tgt),
            Some(kind) => direct::direct_accelerations_kernel_at_points(
                &pos,
                masses_slice.as_deref(),
                softenings_slice.as_deref(),
                &tgt,
                kind,
            ),
        };

        if threads == 0 {
            run()
        } else {
            with_thread_pool(threads, run)
        }
    });
    let mut flat = Vec::with_capacity(m * 3);
    for a in acc.iter() {
        flat.extend(a.iter().copied());
    }
    let arr = Array::from_shape_vec((m, 3), flat).expect("shape mismatch building (m,3) array");
    Ok(arr.into_pyarray(py))
}

/// Direct-sum O(N^2) potentials.
#[pyfunction]
#[pyo3(signature = (positions, masses=None, threads=0, softenings=None, kernel=None))]
pub fn direct_potentials_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    masses: Option<PyReadonlyArray1<'py, f64>>,
    threads: usize,
    softenings: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<u8>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pos = extract_vec3_from_pyarray2(positions, "positions")?;
    let n = pos.len();
    let masses_slice: Option<Vec<f64>> = if let Some(m_arr) = masses {
        let slice = m_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("masses must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let softenings_slice: Option<Vec<f64>> = if let Some(h_arr) = softenings {
        let slice = h_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("softenings must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    if kernel.is_none() && softenings_slice.is_some() {
        return Err(PyValueError::new_err(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)",
        ));
    }

    let kernel_kind = kernel.map(parse_kernel).transpose()?;

    let pot = release_gil(py, || {
        let run = || match kernel_kind {
            None => direct::direct_potentials(&pos, masses_slice.as_deref()),
            Some(kind) => direct::direct_potentials_kernel(
                &pos,
                masses_slice.as_deref(),
                softenings_slice.as_deref(),
                kind,
            ),
        };

        if threads == 0 {
            run()
        } else {
            with_thread_pool(threads, run)
        }
    });

    Ok(pot.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (positions, targets, masses=None, threads=0, softenings=None, kernel=None))]
pub fn direct_potentials_at_points_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    masses: Option<PyReadonlyArray1<'py, f64>>,
    threads: usize,
    softenings: Option<PyReadonlyArray1<'py, f64>>,
    kernel: Option<u8>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pos = extract_vec3_from_pyarray2(positions, "positions")?;
    let n = pos.len();
    let masses_slice: Option<Vec<f64>> = if let Some(m_arr) = masses {
        let slice = m_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("masses must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };

    let softenings_slice: Option<Vec<f64>> = if let Some(h_arr) = softenings {
        let slice = h_arr.as_slice()?;
        if slice.len() != n {
            return Err(PyValueError::new_err("softenings must be length N"));
        }
        Some(slice.to_vec())
    } else {
        None
    };
    let tgt = extract_vec3_from_pyarray2(targets, "targets")?;
    //let m = tgt.len();

    if kernel.is_none() && softenings_slice.is_some() {
        return Err(PyValueError::new_err(
            "softenings require an explicit kernel; pass kernel=0/1 (or omit softenings)",
        ));
    }

    let kernel_kind = kernel.map(parse_kernel).transpose()?;

    let pot = release_gil(py, || {
        let run = || match kernel_kind {
            None => direct::direct_potentials_at_points(&pos, masses_slice.as_deref(), &tgt),
            Some(kind) => direct::direct_potentials_kernel_at_points(
                &pos,
                masses_slice.as_deref(),
                softenings_slice.as_deref(),
                &tgt,
                kind,
            ),
        };

        if threads == 0 {
            run()
        } else {
            with_thread_pool(threads, run)
        }
    });

    Ok(pot.into_pyarray(py))
}
