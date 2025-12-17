
use rayon::prelude::*;

/// Simple O(N^2) direct-sum gravity solver (reference implementation).
pub fn direct_accelerations(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    eps: f64,
) -> Vec<[f64; 3]> {
    let n = positions.len();
    let mut acc = vec![[0.0f64; 3]; n];
    if n == 0 {
        return acc;
    }
    let eps2 = eps * eps;
    // If masses are not provided, assume unit mass for all particles.
    let masses_slice_owned;
    let masses_slice: &[f64] = if let Some(m) = masses {
        m
    } else {
        masses_slice_owned = vec![1.0; n];
        &masses_slice_owned
    };
    // For small N, keep it serial to avoid threading overhead.
    if n < 512 {
        for i in 0..n {
            let pi = positions[i];
            let mut ax = 0.0f64;
            let mut ay = 0.0f64;
            let mut az = 0.0f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr3 = 1.0 / (r2.sqrt() * r2);
                let m = masses_slice[j];
                ax += m * dx * invr3;
                ay += m * dy * invr3;
                az += m * dz * invr3;
            }
            acc[i] = [ax, ay, az];
        }
    } else {
        acc.par_iter_mut().enumerate().for_each(|(i, acc_i)| {
            let pi = positions[i];
            let mut ax = 0.0f64;
            let mut ay = 0.0f64;
            let mut az = 0.0f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr3 = 1.0 / (r2.sqrt() * r2);
                let m = masses_slice[j];
                ax += m * dx * invr3;
                ay += m * dy * invr3;
                az += m * dz * invr3;
            }
            *acc_i = [ax, ay, az];
        });
    }
    acc
}

/// Simple O(N^2) direct-sum gravitational potential solver
/// (reference implementation, mainly for testing).
pub fn direct_potentials(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    eps: f64,
) -> Vec<f64> {
    let n = positions.len();
    let mut pot = vec![0.0f64; n];
    if n == 0 {
        return pot;
    }
    let eps2 = eps * eps;
    // If masses are not provided, assume unit mass for all particles.
    let masses_slice_owned;
    let masses_slice: &[f64] = if let Some(m) = masses {
        m
    } else {
        masses_slice_owned = vec![1.0; n];
        &masses_slice_owned
    };
    if n < 512 {
        for i in 0..n {
            let pi = positions[i];
            let mut phi = 0.0f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr = 1.0 / r2.sqrt();
                let m = masses_slice[j];
                phi += -m * invr;
            }
            pot[i] = phi;
        }
    } else {
        pot.par_iter_mut().enumerate().for_each(|(i, pot_i)| {
            let pi = positions[i];
            let mut phi = 0.0f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr = 1.0 / r2.sqrt();
                let m = masses_slice[j];
                phi += -m * invr;
            }
            *pot_i = phi;
        });
    }
    pot
}