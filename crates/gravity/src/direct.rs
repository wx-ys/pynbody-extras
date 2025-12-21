use rayon::prelude::*;

// Wrapper types for raw pointers used in parallel regions.
#[allow(dead_code)]
#[derive(Copy, Clone)]
struct AccPtr(*mut [f64; 3]);
unsafe impl Send for AccPtr {}
unsafe impl Sync for AccPtr {} // we guarantee no concurrent writes to same index per round

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct PotPtr(*mut f64);
unsafe impl Send for PotPtr {}
unsafe impl Sync for PotPtr {} // same reasoning for potentials

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[inline]
unsafe fn accumulate_pair_acc(
    acc: AccPtr,
    i: usize,
    j: usize,
    mi: f64,
    mj: f64,
    dx: f64,
    dy: f64,
    dz: f64,
    invr3: f64,
) {
    let ai = acc.0.add(i);
    let aj = acc.0.add(j);

    (*ai)[0] += mj * dx * invr3;
    (*ai)[1] += mj * dy * invr3;
    (*ai)[2] += mj * dz * invr3;

    (*aj)[0] -= mi * dx * invr3;
    (*aj)[1] -= mi * dy * invr3;
    (*aj)[2] -= mi * dz * invr3;
}

#[allow(dead_code)]
#[inline]
unsafe fn accumulate_pair_pot(pot: PotPtr, i: usize, j: usize, mi: f64, mj: f64, phi_pair: f64) {
    let pi_ptr = pot.0.add(i);
    let pj_ptr = pot.0.add(j);

    *pi_ptr += phi_pair * mj;
    *pj_ptr += phi_pair * mi;
}

// Generates all unique pairs (i, j) for i < j in a round-robin fashion,
// and applies the provided function `f` to each batch of pairs per round.
// but actually this is slower than just parallelizing over all pairs directly.
// Keeping it here for future optimization attempts.
#[allow(dead_code)]
fn for_each_pair_round<F>(n: usize, mut f: F)
where
    F: FnMut(&[(usize, usize)]),
{
    if n <= 1 {
        return;
    }

    let has_dummy = n % 2 == 1;
    let m = if has_dummy { n + 1 } else { n };
    let dummy = m - 1;

    let rounds = m - 1;
    let pairs_per_round = m / 2;

    let mut pairs: Vec<(usize, usize)> = vec![(0, 0); pairs_per_round];

    for r in 0..rounds {
        pairs.par_iter_mut().enumerate().for_each(|(k, pair)| {
            let a = (r + k) % (m - 1);
            let b = if k == 0 {
                m - 1
            } else {
                (r + m - 1 - k) % (m - 1)
            };

            if has_dummy && (a == dummy || b == dummy) {
                *pair = (usize::MAX, usize::MAX);
            } else {
                let (i, j) = if a < b { (a, b) } else { (b, a) };
                *pair = (i, j);
            }
        });

        let valid_len = if has_dummy {
            if let Some(pos) = pairs.iter().position(|&(i, _)| i == usize::MAX) {
                pairs.swap_remove(pos);
            }
            pairs.len()
        } else {
            pairs_per_round
        };

        f(&pairs[..valid_len]);

        if has_dummy {
            pairs.resize(pairs_per_round, (0, 0));
        }
    }
}

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
        // pair-wise loop, compute each interaction once and update both particles
        for i in 0..n {
            let pi = positions[i];
            let mi = masses_slice[i];
            for j in (i + 1)..n {
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr3 = 1.0 / (r2.sqrt() * r2);

                let mj = masses_slice[j];

                // acceleration on i due to j
                acc[i][0] += mj * dx * invr3;
                acc[i][1] += mj * dy * invr3;
                acc[i][2] += mj * dz * invr3;

                // acceleration on j due to i  (opposite direction)
                acc[j][0] -= mi * dx * invr3;
                acc[j][1] -= mi * dy * invr3;
                acc[j][2] -= mi * dz * invr3;
            }
        }
    } else {
        // Parallel version for per particles, N*(N-1) interactions,
        // how to parallel over unique pairs more efficiently? that N(N-1)/2 interactions.
        acc.par_iter_mut().enumerate().for_each(|(i, acc_i)| {
            let pi = positions[i];
            let mut ax = 0.0f64;
            let mut ay = 0.0f64;
            let mut az = 0.0f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr3 = 1.0 / (r2.sqrt() * r2);
                let mj = masses_slice[j];
                ax += mj * dx * invr3;
                ay += mj * dy * invr3;
                az += mj * dz * invr3;
            }
            *acc_i = [ax, ay, az];
        });
    }
    acc
}

pub fn direct_accelerations_at_points(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    targets: &[[f64; 3]],
    eps: f64,
) -> Vec<[f64; 3]> {
    let n_src = positions.len();
    let n_tgt = targets.len();
    let mut acc = vec![[0.0f64; 3]; n_tgt];
    if n_tgt == 0 || n_src == 0 {
        return acc;
    }
    let eps2 = eps * eps;
    // If masses are not provided, assume unit mass for all particles.
    let masses_slice_owned;
    let masses_slice: &[f64] = if let Some(m) = masses {
        m
    } else {
        masses_slice_owned = vec![1.0; n_src];
        &masses_slice_owned
    };
    // For small N, keep it serial to avoid threading overhead.
    if n_tgt < 512 {
        for i in 0..n_tgt {
            let pi = targets[i];
            let mut ax = 0.0f64;
            let mut ay = 0.0f64;
            let mut az = 0.0f64;
            for j in 0..n_src {
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
            let pi = targets[i];
            let mut ax = 0.0f64;
            let mut ay = 0.0f64;
            let mut az = 0.0f64;
            for j in 0..n_src {
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
pub fn direct_potentials(positions: &[[f64; 3]], masses: Option<&[f64]>, eps: f64) -> Vec<f64> {
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
        // pair-wise loop, compute each interaction once and update both particles
        for i in 0..n {
            let pi = positions[i];
            let mi = masses_slice[i];
            for j in (i + 1)..n {
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr = 1.0 / r2.sqrt();

                let mj = masses_slice[j];
                let phi_pair = -invr;

                // potential at i due to j
                pot[i] += phi_pair * mj;
                // potential at j due to i
                pot[j] += phi_pair * mi;
            }
        }
    } else {
        // Parallel version, but memory intensive if large number of threads
        pot.par_iter_mut().enumerate().for_each(|(i, pot_i)| {
            let pi = positions[i];
            let mut phi = 0.0f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let pj = positions[j];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx * dx + dy * dy + dz * dz + eps2;
                let invr = 1.0 / r2.sqrt();
                let mj = masses_slice[j];
                phi += -mj * invr;
            }
            *pot_i = phi;
        });
    }
    pot
}

pub fn direct_potentials_at_points(
    positions: &[[f64; 3]],
    masses: Option<&[f64]>,
    targets: &[[f64; 3]],
    eps: f64,
) -> Vec<f64> {
    let n_tgt = targets.len();
    let n_src = positions.len();
    let mut pot = vec![0.0f64; n_tgt];
    if n_tgt == 0 || n_src == 0 {
        return pot;
    }
    let eps2 = eps * eps;
    // If masses are not provided, assume unit mass for all particles.
    let masses_slice_owned;
    let masses_slice: &[f64] = if let Some(m) = masses {
        m
    } else {
        masses_slice_owned = vec![1.0; n_src];
        &masses_slice_owned
    };
    if n_tgt < 512 {
        for i in 0..n_tgt {
            let pi = targets[i];
            let mut phi = 0.0f64;
            for j in 0..n_src {
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
            let pi = targets[i];
            let mut phi = 0.0f64;
            for j in 0..n_src {
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
