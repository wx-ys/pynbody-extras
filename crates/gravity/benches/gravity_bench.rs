use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gravity::Octree;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn gen_points(n: usize) -> Vec<[f64; 3]> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        // Uniform cube [-0.5, 0.5]^3
        let x = rng.gen::<f64>() - 0.5;
        let y = rng.gen::<f64>() - 0.5;
        let z = rng.gen::<f64>() - 0.5;
        pts.push([x, y, z]);
    }
    pts
}

fn gen_masses(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(123);
    (0..n).map(|_| 0.5 + rng.gen::<f64>()).collect()
}

fn bench_octree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("octree_build");
    let sizes: Vec<usize> = match std::env::var("BENCH_N").ok().and_then(|v| v.parse().ok()) {
        Some(n) => vec![n],
        None => vec![10_000, 50_000], // keep defaults moderate; set BENCH_N for larger
    };

    for &n in &sizes {
        let pts = gen_points(n);
        let masses = gen_masses(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| {
                let tree = Octree::build(black_box(&pts), black_box(Some(&masses)), 16, 2);
                black_box(tree);
            });
        });
    }

    group.finish();
}

fn bench_accels_and_pots(c: &mut Criterion) {
    let mut group = c.benchmark_group("gravity_eval");
    let sizes: Vec<usize> = match std::env::var("BENCH_N").ok().and_then(|v| v.parse().ok()) {
        Some(n) => vec![n],
        None => vec![20_000],
    };

    let theta = 0.6f64;

    for &n in &sizes {
        let pts = gen_points(n);
        let masses = gen_masses(n);
        let tree = Octree::build(&pts, Some(&masses), 16, 2);
        let mut acc = vec![[0.0f64; 3]; n];
        let mut pot = vec![0.0f64; n];

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("accelerations", n), &n, |b, &_n| {
            b.iter(|| {
                let mut out = acc.clone();
                black_box(tree.compute_accelerations(theta, &mut out));
                black_box(out);
            });
        });
        group.bench_with_input(BenchmarkId::new("potentials", n), &n, |b, &_n| {
            b.iter(|| {
                let mut out = pot.clone();
                black_box(tree.compute_potentials(theta, &mut out));
                black_box(out);
            });
        });
    }

    group.finish();
}

fn bench_queries_at_points(c: &mut Criterion) {
    let mut group = c.benchmark_group("gravity_queries");
    let n_sources: usize = std::env::var("BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50_000);
    let n_targets: usize = std::env::var("BENCH_Q")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    let theta = 0.6f64;

    let src = gen_points(n_sources);
    let masses = gen_masses(n_sources);
    let queries = gen_points(n_targets);
    let tree = Octree::build(&src, Some(&masses), 64, 2);

    let mut acc = vec![[0.0f64; 3]; n_targets];
    let mut pot = vec![0.0f64; n_targets];

    group.throughput(Throughput::Elements(n_targets as u64));
    group.bench_function("accelerations_at_points", |b| {
        b.iter(|| {
            let mut out = acc.clone();
            black_box(tree.accelerations_at_points(&queries, theta, &mut out));
            black_box(out);
        });
    });
    group.bench_function("potentials_at_points", |b| {
        b.iter(|| {
            let mut out = pot.clone();
            black_box(tree.potentials_at_points(&queries, theta, &mut out));
            black_box(out);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_octree_build,
    bench_accels_and_pots,
    bench_queries_at_points
);
criterion_main!(benches);
