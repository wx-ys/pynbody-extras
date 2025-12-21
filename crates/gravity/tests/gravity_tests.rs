use gravity::{direct, Octree, Tree3D};

fn gen_points(seed: u64, n: usize) -> Vec<[f64; 3]> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.gen::<f64>() - 0.5;
        let y = rng.gen::<f64>() - 0.5;
        let z = rng.gen::<f64>() - 0.5;
        pts.push([x, y, z]);
    }
    pts
}

fn gen_masses(seed: u64, n: usize) -> Vec<f64> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| 0.5 + rng.gen::<f64>()).collect()
}

fn max_abs3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

fn rms_vec3(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let mut s = 0.0f64;
    for i in 0..n {
        let dx = a[i][0] - b[i][0];
        let dy = a[i][1] - b[i][1];
        let dz = a[i][2] - b[i][2];
        s += dx*dx + dy*dy + dz*dz;
    }
    (s / (n as f64)).sqrt()
}

#[test]
fn accelerations_match_direct_small_n() {
    let n = 256;
    let eps = 1e-3f64;
    let theta = 0.0f64; // force full traversal down to leaves

    let pts = gen_points(1, n);
    let masses = gen_masses(2, n);

    let tree = Octree::build(&pts, Some(&masses), 32, 2);

    let mut acc_tree = vec![[0.0f64; 3]; n];
    tree.compute_accelerations(theta, eps, &mut acc_tree);

    let acc_direct = direct::direct_accelerations(&pts, Some(&masses), eps);

    for i in 0..n {
        let diff = max_abs3(&acc_tree[i], &acc_direct[i]);
        assert!(diff < 1e-10, "acc mismatch at {}: diff {}", i, diff);
    }
}

#[test]
fn potentials_match_direct_small_n() {
    let n = 256;
    let eps = 1e-3f64;
    let theta = 0.0f64; // force full traversal down to leaves

    let pts = gen_points(3, n);
    let masses = gen_masses(4, n);

    let tree = Octree::build(&pts, Some(&masses), 32, 2);

    let mut pot_tree = vec![0.0f64; n];
    tree.compute_potentials(theta, eps, &mut pot_tree);

    let pot_direct = direct::direct_potentials(&pts, Some(&masses), eps);

    for i in 0..n {
        let diff = (pot_tree[i] - pot_direct[i]).abs();
        assert!(diff < 1e-10, "pot mismatch at {}: diff {}", i, diff);
    }
}

#[test]
fn queries_match_direct_at_points() {
    let n_src = 512;
    let n_tgt = 128;
    let eps = 1e-3f64;
    let theta = 0.0f64; // force full traversal down to leaves

    let src = gen_points(11, n_src);
    let masses = gen_masses(12, n_src);
    let queries = gen_points(13, n_tgt);

    let tree = Octree::build(&src, Some(&masses), 32, 2);

    let mut acc_tree = vec![[0.0f64; 3]; n_tgt];
    let mut pot_tree = vec![0.0f64; n_tgt];

    tree.accelerations_at_points(&queries, theta, eps, &mut acc_tree);
    tree.potentials_at_points(&queries, theta, eps, &mut pot_tree);

    let acc_direct = direct::direct_accelerations_at_points(&src, Some(&masses), &queries, eps);
    let pot_direct = direct::direct_potentials_at_points(&src, Some(&masses), &queries, eps);

    for i in 0..n_tgt {
        let d_acc = max_abs3(&acc_tree[i], &acc_direct[i]);
        let d_pot = (pot_tree[i] - pot_direct[i]).abs();
        assert!(d_acc < 1e-10, "acc query mismatch at {}: diff {}", i, d_acc);
        assert!(d_pot < 1e-10, "pot query mismatch at {}: diff {}", i, d_pot);
    }
}

#[test]
fn error_decreases_with_multipole_order_accel() {
    // For a fixed theta, higher multipole order should reduce error.
    // Use moderate N to keep test fast yet meaningful.
    let n = 800;
    let eps = 1e-3f64;
    let theta = 0.7f64; // approximate regime to expose truncation error

    let pts = gen_points(21, n);
    let masses = gen_masses(22, n);

    // Reference (exact) accelerations
    let acc_ref = direct::direct_accelerations(&pts, Some(&masses), eps);

    let orders = [0u8, 3u8, 4u8, 5u8];
    let mut errs = Vec::with_capacity(orders.len());
    for &ord in &orders {
        let tree = Octree::build(&pts, Some(&masses), 64, ord);
        let mut acc_tree = vec![[0.0f64; 3]; n];
        tree.compute_accelerations(theta, eps, &mut acc_tree);
        let e = rms_vec3(&acc_tree, &acc_ref);
        errs.push(e);
    }

    // Ensure non-increasing with some slack for numerical noise
    for i in 1..errs.len() {
        assert!(errs[i] <= errs[i-1], "error did not decrease sufficiently: order {} vs {} -> {} > {}", orders[i-1], orders[i], errs[i], errs[i-1]);
    }
    // And significant improvement from order 0 to highest order
    assert!(errs.last().unwrap() <= &(errs[0] * 0.8), "insufficient improvement: err_o0={} err_o5={}", errs[0], errs.last().unwrap());
}

fn rms_scalar(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let mut s = 0.0f64;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d*d;
    }
    (s / (n as f64)).sqrt()
}

#[test]
fn error_decreases_with_multipole_order_potential() {
    // For a fixed theta, higher multipole order should reduce potential error.
    let n = 800;
    let eps = 1e-3f64;
    let theta = 0.7f64;

    let pts = gen_points(31, n);
    let masses = gen_masses(32, n);

    let pot_ref = direct::direct_potentials(&pts, Some(&masses), eps);

    let orders = [0u8, 2u8, 3u8, 4u8, 5u8];
    let mut errs = Vec::with_capacity(orders.len());
    for &ord in &orders {
        let tree = Octree::build(&pts, Some(&masses), 64, ord);
        let mut pot_tree = vec![0.0f64; n];
        tree.compute_potentials(theta, eps, &mut pot_tree);
        let e = rms_scalar(&pot_tree, &pot_ref);
        errs.push(e);
    }

    for i in 1..errs.len() {
        assert!(errs[i] <= errs[i-1], "potential error did not decrease sufficiently: order {} -> {}: {} > {}", orders[i-1], orders[i], errs[i], errs[i-1]);
    }
}
