use gravity::multipole::{MultipoleMoment, translate_multipole};
use rand::Rng;

#[test]
fn compare_translate_vs_direct() {
    let mut rng = rand::thread_rng();
    let n = 200usize;
    let mut positions = Vec::with_capacity(n);
    for _ in 0..n {
        positions.push([rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()]);
    }
    let masses: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();

    let center_b = [0.3, 0.4, 0.5];
    let center_a = [0.8, -0.2, 0.1];
    let delta = [center_a[0] - center_b[0], center_a[1] - center_b[1], center_a[2] - center_b[2]];
    let indices: Vec<usize> = (0..n).collect();

    let order = 5u8;
    let m_b = MultipoleMoment::from_points(&positions, Some(&masses), &indices, center_b, order);
    let m_trans = translate_multipole(&m_b, delta, order);
    let m_direct = MultipoleMoment::from_points(&positions, Some(&masses), &indices, center_a, order);

    let tol = 1e-10;
    let mut maxdiff = 0.0f64;
    let mut bad = Vec::new();

    macro_rules! chk {
        ($name:expr, $a:expr, $b:expr) => {{
            let d = ($a - $b).abs();
            if d > maxdiff { maxdiff = d; }
            if d > tol {
                bad.push(($name, $a, $b, d));
            }
        }};
    }

    chk!("m000", m_trans.m000, m_direct.m000);
    chk!("m100", m_trans.m100, m_direct.m100);
    chk!("m010", m_trans.m010, m_direct.m010);
    chk!("m001", m_trans.m001, m_direct.m001);
    chk!("m200", m_trans.m200, m_direct.m200);
    chk!("m020", m_trans.m020, m_direct.m020);
    chk!("m002", m_trans.m002, m_direct.m002);
    chk!("m110", m_trans.m110, m_direct.m110);
    chk!("m101", m_trans.m101, m_direct.m101);
    chk!("m011", m_trans.m011, m_direct.m011);
    chk!("m300", m_trans.m300, m_direct.m300);
    chk!("m030", m_trans.m030, m_direct.m030);
    chk!("m003", m_trans.m003, m_direct.m003);
    chk!("m210", m_trans.m210, m_direct.m210);
    chk!("m201", m_trans.m201, m_direct.m201);
    chk!("m120", m_trans.m120, m_direct.m120);
    chk!("m102", m_trans.m102, m_direct.m102);
    chk!("m021", m_trans.m021, m_direct.m021);
    chk!("m012", m_trans.m012, m_direct.m012);
    chk!("m111", m_trans.m111, m_direct.m111);
    chk!("m400", m_trans.m400, m_direct.m400);
    chk!("m040", m_trans.m040, m_direct.m040);
    chk!("m004", m_trans.m004, m_direct.m004);
    chk!("m310", m_trans.m310, m_direct.m310);
    chk!("m301", m_trans.m301, m_direct.m301);
    chk!("m130", m_trans.m130, m_direct.m130);
    chk!("m103", m_trans.m103, m_direct.m103);
    chk!("m031", m_trans.m031, m_direct.m031);
    chk!("m013", m_trans.m013, m_direct.m013);
    chk!("m220", m_trans.m220, m_direct.m220);
    chk!("m202", m_trans.m202, m_direct.m202);
    chk!("m022", m_trans.m022, m_direct.m022);
    chk!("m211", m_trans.m211, m_direct.m211);
    chk!("m121", m_trans.m121, m_direct.m121);
    chk!("m112", m_trans.m112, m_direct.m112);
    chk!("m500", m_trans.m500, m_direct.m500);
    chk!("m050", m_trans.m050, m_direct.m050);
    chk!("m005", m_trans.m005, m_direct.m005);
    chk!("m410", m_trans.m410, m_direct.m410);
    chk!("m401", m_trans.m401, m_direct.m401);
    chk!("m140", m_trans.m140, m_direct.m140);
    chk!("m104", m_trans.m104, m_direct.m104);
    chk!("m041", m_trans.m041, m_direct.m041);
    chk!("m014", m_trans.m014, m_direct.m014);
    chk!("m320", m_trans.m320, m_direct.m320);
    chk!("m302", m_trans.m302, m_direct.m302);
    chk!("m230", m_trans.m230, m_direct.m230);
    chk!("m203", m_trans.m203, m_direct.m203);
    chk!("m032", m_trans.m032, m_direct.m032);
    chk!("m023", m_trans.m023, m_direct.m023);
    chk!("m221", m_trans.m221, m_direct.m221);
    chk!("m212", m_trans.m212, m_direct.m212);
    chk!("m122", m_trans.m122, m_direct.m122);
    chk!("m311", m_trans.m311, m_direct.m311);
    chk!("m131", m_trans.m131, m_direct.m131);
    chk!("m113", m_trans.m113, m_direct.m113);

    if !bad.is_empty() {
        eprintln!("translate vs direct: maxdiff = {}", maxdiff);
        for (name, a, b, d) in bad.iter() {
            eprintln!("{}: trans = {:e}, direct = {:e}, diff = {:e}", name, a, b, d);
        }
    }

    assert!(bad.is_empty(), "maxdiff = {}", maxdiff);
}