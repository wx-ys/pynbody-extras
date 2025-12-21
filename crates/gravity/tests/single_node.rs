use gravity::multipole::{gravity_potential_multipole, MultipoleMoment, PotentialDerivatives};
use rand::Rng;

fn direct_potential(positions: &[[f64; 3]], masses: &[f64], target: [f64; 3], eps2: f64) -> f64 {
    let mut phi = 0.0;
    for (p, &m) in positions.iter().zip(masses.iter()) {
        let dx = p[0] - target[0];
        let dy = p[1] - target[1];
        let dz = p[2] - target[2];
        let r2 = dx * dx + dy * dy + dz * dz + eps2;
        if r2 == 0.0 {
            continue;
        }
        phi += -m / r2.sqrt();
    }
    phi
}

#[test]
fn single_node_multipole_vs_direct() {
    let mut rng = rand::thread_rng();
    let n = 4000usize;

    // Random positions in a small cube around origin (source size a ~ 0.1)
    let mut positions = Vec::with_capacity(n);
    let mut masses = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.gen_range(-0.1..0.1);
        let y = rng.gen_range(-0.1..0.1);
        let z = rng.gen_range(-0.1..0.1);
        let m = rng.gen_range(0.1..1.0);
        positions.push([x, y, z]);
        masses.push(m);
    }

    // Center-of-mass of this single node
    let mut com = [0.0f64; 3];
    let mut mtot = 0.0f64;
    for (p, &m) in positions.iter().zip(masses.iter()) {
        mtot += m;
        com[0] += p[0] * m;
        com[1] += p[1] * m;
        com[2] += p[2] * m;
    }
    com[0] /= mtot;
    com[1] /= mtot;
    com[2] /= mtot;

    let indices: Vec<usize> = (0..n).collect();
    let m_mom = MultipoleMoment::from_points(&positions, Some(&masses), &indices, com, 5);

    let eps2 = 0.0f64;

    // Choose a set of target points far away from com (r ~ 20-30, so a/r ~ 0.005)
    let n_targets = 400usize;
    let mut err_per_order: [Vec<f64>; 6] = Default::default();

    for _ in 0..n_targets {
        // random direction on sphere, radius in [20,30]
        let mut vx: f64;
        let mut vy: f64;
        let mut vz: f64;
        loop {
            vx = rng.gen_range(-1.0..1.0);
            vy = rng.gen_range(-1.0..1.0);
            vz = rng.gen_range(-1.0..1.0);
            let r2 = vx * vx + vy * vy + vz * vz;
            if r2 > 1e-6 && r2 <= 1.0 {
                break;
            }
        }
        let norm = (vx * vx + vy * vy + vz * vz).sqrt();
        vx /= norm;
        vy /= norm;
        vz /= norm;
        let r = rng.gen_range(20.0..30.0);
        let target = [com[0] + r * vx, com[1] + r * vy, com[2] + r * vz];

        let phi_direct = direct_potential(&positions, &masses, target, eps2);

        let dx = com[0] - target[0];
        let dy = com[1] - target[1];
        let dz = com[2] - target[2];
        let d = PotentialDerivatives::new(dx, dy, dz, eps2, 5);

        for order in 0u8..=5 {
            let phi_mp = gravity_potential_multipole(&m_mom, &d, order);
            let err = if phi_direct != 0.0 {
                ((phi_mp - phi_direct) / phi_direct).abs()
            } else {
                (phi_mp - phi_direct).abs()
            };
            err_per_order[order as usize].push(err);
        }
    }

    // For each order compute 90th percentile of relative error
    for order in 0usize..=5 {
        let errs = &mut err_per_order[order];
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ((errs.len() as f64) * 0.9) as usize;
        let k = k.min(errs.len() - 1);
        let p90 = errs[k];
        println!("order {}: p90 relative error = {:e}", order, p90);

        // Sanity check: all orders should be reasonably accurate in this far-field setup
        assert!(p90 < 1.0e-2, "order {} too inaccurate: p90={}", order, p90);
    }
}
