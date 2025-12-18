use std::f64;


/// Cartesian multipole moments up to order 5.
///
/// The notation follows example_multipole.py: M_lmn stores
/// 1/(l! m! n!) * sum m x^l y^m z^n about a chosen origin.
#[derive(Clone, Copy, Default)]
pub struct MultipoleMoment {
    // order 0
    pub m000: f64,
    // order 1
    pub m100: f64,
    pub m010: f64,
    pub m001: f64,
    // order 2
    pub m200: f64,
    pub m020: f64,
    pub m002: f64,
    pub m110: f64,
    pub m101: f64,
    pub m011: f64,
    // order 3
    pub m300: f64,
    pub m030: f64,
    pub m003: f64,
    pub m210: f64,
    pub m201: f64,
    pub m120: f64,
    pub m102: f64,
    pub m021: f64,
    pub m012: f64,
    pub m111: f64,
    // order 4
    pub m400: f64,
    pub m040: f64,
    pub m004: f64,
    pub m310: f64,
    pub m301: f64,
    pub m130: f64,
    pub m103: f64,
    pub m031: f64,
    pub m013: f64,
    pub m220: f64,
    pub m202: f64,
    pub m022: f64,
    pub m211: f64,
    pub m121: f64,
    pub m112: f64,
    // order 5
    pub m500: f64,
    pub m050: f64,
    pub m005: f64,
    pub m410: f64,
    pub m401: f64,
    pub m140: f64,
    pub m104: f64,
    pub m041: f64,
    pub m014: f64,
    pub m320: f64,
    pub m302: f64,
    pub m230: f64,
    pub m203: f64,
    pub m032: f64,
    pub m023: f64,
    pub m221: f64,
    pub m212: f64,
    pub m122: f64,
    pub m311: f64,
    pub m131: f64,
    pub m113: f64,
}

impl MultipoleMoment {
    pub fn zero() -> Self {
        Self::default()
    }

    /// In-place addition of another multipole moment (same expansion center).
    pub fn add_assign(&mut self, other: &MultipoleMoment) {
        self.m000 += other.m000;
        self.m100 += other.m100;
        self.m010 += other.m010;
        self.m001 += other.m001;
        self.m200 += other.m200;
        self.m020 += other.m020;
        self.m002 += other.m002;
        self.m110 += other.m110;
        self.m101 += other.m101;
        self.m011 += other.m011;
        self.m300 += other.m300;
        self.m030 += other.m030;
        self.m003 += other.m003;
        self.m210 += other.m210;
        self.m201 += other.m201;
        self.m120 += other.m120;
        self.m102 += other.m102;
        self.m021 += other.m021;
        self.m012 += other.m012;
        self.m111 += other.m111;
        self.m400 += other.m400;
        self.m040 += other.m040;
        self.m004 += other.m004;
        self.m310 += other.m310;
        self.m301 += other.m301;
        self.m130 += other.m130;
        self.m103 += other.m103;
        self.m031 += other.m031;
        self.m013 += other.m013;
        self.m220 += other.m220;
        self.m202 += other.m202;
        self.m022 += other.m022;
        self.m211 += other.m211;
        self.m121 += other.m121;
        self.m112 += other.m112;
        self.m500 += other.m500;
        self.m050 += other.m050;
        self.m005 += other.m005;
        self.m410 += other.m410;
        self.m401 += other.m401;
        self.m140 += other.m140;
        self.m104 += other.m104;
        self.m041 += other.m041;
        self.m014 += other.m014;
        self.m320 += other.m320;
        self.m302 += other.m302;
        self.m230 += other.m230;
        self.m203 += other.m203;
        self.m032 += other.m032;
        self.m023 += other.m023;
        self.m221 += other.m221;
        self.m212 += other.m212;
        self.m122 += other.m122;
        self.m311 += other.m311;
        self.m131 += other.m131;
        self.m113 += other.m113;
    }

    /// Build multipole moments for a set of points relative to a given center.
    ///
    /// `positions` and `masses_opt` are full arrays; `indices` selects the
    /// particles belonging to this node. Coordinates are taken relative to
    /// `center`, which should normally be the node's center-of-mass.
    pub fn from_points(
        positions: &[[f64; 3]],
        masses_opt: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
        order: u8,
    ) -> Self {
        let mut m = MultipoleMoment::default();
        if indices.is_empty() {
            return m;
        }

        for &pi in indices {
            let p = positions[pi];
            let mass = masses_opt.map(|mm| mm[pi]).unwrap_or(1.0);
            let x = p[0] - center[0];
            let y = p[1] - center[1];
            let z = p[2] - center[2];

            m.m000 += mass;

            if order >= 1 {
                m.m100 += mass * x;
                m.m010 += mass * y;
                m.m001 += mass * z;
            }
            if order >= 2 {
                m.m200 += 0.5 * mass * x * x;
                m.m020 += 0.5 * mass * y * y;
                m.m002 += 0.5 * mass * z * z;
                m.m110 += mass * x * y;
                m.m101 += mass * x * z;
                m.m011 += mass * y * z;
            }
            if order >= 3 {
                m.m300 += (1.0 / 6.0) * mass * x.powi(3);
                m.m030 += (1.0 / 6.0) * mass * y.powi(3);
                m.m003 += (1.0 / 6.0) * mass * z.powi(3);
                m.m210 += 0.5 * mass * x * x * y;
                m.m201 += 0.5 * mass * x * x * z;
                m.m120 += 0.5 * mass * y * y * x;
                // (l,m,n) = (1,0,2): x^1 z^2
                m.m102 += 0.5 * mass * x * z * z;
                // (0,2,1): y^2 z
                m.m021 += 0.5 * mass * y * y * z;
                // (0,1,2): y z^2
                m.m012 += 0.5 * mass * y * z * z;
                m.m111 += mass * x * y * z;
            }
            if order >= 4 {
                m.m400 += (1.0 / 24.0) * mass * x.powi(4);
                m.m040 += (1.0 / 24.0) * mass * y.powi(4);
                m.m004 += (1.0 / 24.0) * mass * z.powi(4);
                m.m310 += (1.0 / 6.0) * mass * x.powi(3) * y;
                m.m301 += (1.0 / 6.0) * mass * x.powi(3) * z;
                m.m130 += (1.0 / 6.0) * mass * y.powi(3) * x;
                m.m103 += (1.0 / 6.0) * mass * x * z.powi(3);
                m.m031 += (1.0 / 6.0) * mass * y.powi(3) * z;
                m.m013 += (1.0 / 6.0) * mass * y * z.powi(3);
                m.m220 += 0.25 * mass * x * x * y * y;
                m.m202 += 0.25 * mass * x * x * z * z;
                m.m022 += 0.25 * mass * y * y * z * z;
                m.m211 += 0.5 * mass * x * x * y * z;
                m.m121 += 0.5 * mass * y * y * x * z;
                m.m112 += 0.5 * mass * z * z * x * y;
            }
            if order >= 5 {
                m.m500 += (1.0 / 120.0) * mass * x.powi(5);
                m.m050 += (1.0 / 120.0) * mass * y.powi(5);
                m.m005 += (1.0 / 120.0) * mass * z.powi(5);
                m.m410 += (1.0 / 24.0) * mass * x.powi(4) * y;
                m.m401 += (1.0 / 24.0) * mass * x.powi(4) * z;
                m.m140 += (1.0 / 24.0) * mass * y.powi(4) * x;
                m.m104 += (1.0 / 24.0) * mass * z.powi(4) * x;
                m.m041 += (1.0 / 24.0) * mass * y.powi(4) * z;
                m.m014 += (1.0 / 24.0) * mass * z.powi(4) * y;
                m.m320 += (1.0 / 12.0) * mass * x.powi(3) * y.powi(2);
                m.m302 += (1.0 / 12.0) * mass * x.powi(3) * z.powi(2);
                m.m230 += (1.0 / 12.0) * mass * x.powi(2) * y.powi(3);
                m.m203 += (1.0 / 12.0) * mass * x.powi(2) * z.powi(3);
                m.m032 += (1.0 / 12.0) * mass * y.powi(3) * z.powi(2);
                m.m023 += (1.0 / 12.0) * mass * y.powi(2) * z.powi(3);
                m.m221 += 0.25 * mass * x * x * y * y * z;
                m.m212 += 0.25 * mass * x * x * z * z * y;
                m.m122 += 0.25 * mass * y * y * z * z * x;
                m.m311 += (1.0 / 6.0) * mass * x.powi(3) * y * z;
                m.m131 += (1.0 / 6.0) * mass * y.powi(3) * x * z;
                m.m113 += (1.0 / 6.0) * mass * z.powi(3) * x * y;
            }
        }

        m
    }
}

// Factorials up to 5! as f64.
const FACT: [f64; 6] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0];

fn get_moment(m: &MultipoleMoment, l: usize, mm: usize, n: usize) -> f64 {
    match (l, mm, n) {
        (0, 0, 0) => m.m000,
        (1, 0, 0) => m.m100,
        (0, 1, 0) => m.m010,
        (0, 0, 1) => m.m001,
        (2, 0, 0) => m.m200,
        (0, 2, 0) => m.m020,
        (0, 0, 2) => m.m002,
        (1, 1, 0) => m.m110,
        (1, 0, 1) => m.m101,
        (0, 1, 1) => m.m011,
        (3, 0, 0) => m.m300,
        (0, 3, 0) => m.m030,
        (0, 0, 3) => m.m003,
        (2, 1, 0) => m.m210,
        (2, 0, 1) => m.m201,
        (1, 2, 0) => m.m120,
        (1, 0, 2) => m.m102,
        (0, 2, 1) => m.m021,
        (0, 1, 2) => m.m012,
        (1, 1, 1) => m.m111,
        (4, 0, 0) => m.m400,
        (0, 4, 0) => m.m040,
        (0, 0, 4) => m.m004,
        (3, 1, 0) => m.m310,
        (3, 0, 1) => m.m301,
        (1, 3, 0) => m.m130,
        (1, 0, 3) => m.m103,
        (0, 3, 1) => m.m031,
        (0, 1, 3) => m.m013,
        (2, 2, 0) => m.m220,
        (2, 0, 2) => m.m202,
        (0, 2, 2) => m.m022,
        (2, 1, 1) => m.m211,
        (1, 2, 1) => m.m121,
        (1, 1, 2) => m.m112,
        (5, 0, 0) => m.m500,
        (0, 5, 0) => m.m050,
        (0, 0, 5) => m.m005,
        (4, 1, 0) => m.m410,
        (4, 0, 1) => m.m401,
        (1, 4, 0) => m.m140,
        (1, 0, 4) => m.m104,
        (0, 4, 1) => m.m041,
        (0, 1, 4) => m.m014,
        (3, 2, 0) => m.m320,
        (3, 0, 2) => m.m302,
        (2, 3, 0) => m.m230,
        (0, 3, 2) => m.m032,
        (0, 2, 3) => m.m023,
        (2, 0, 3) => m.m203,
        (2, 2, 1) => m.m221,
        (2, 1, 2) => m.m212,
        (1, 2, 2) => m.m122,
        (3, 1, 1) => m.m311,
        (1, 3, 1) => m.m131,
        (1, 1, 3) => m.m113,
        _ => 0.0,
    }
}

fn set_moment(m: &mut MultipoleMoment, l: usize, mm: usize, n: usize, value: f64) {
    match (l, mm, n) {
        (0, 0, 0) => m.m000 = value,
        (1, 0, 0) => m.m100 = value,
        (0, 1, 0) => m.m010 = value,
        (0, 0, 1) => m.m001 = value,
        (2, 0, 0) => m.m200 = value,
        (0, 2, 0) => m.m020 = value,
        (0, 0, 2) => m.m002 = value,
        (1, 1, 0) => m.m110 = value,
        (1, 0, 1) => m.m101 = value,
        (0, 1, 1) => m.m011 = value,
        (3, 0, 0) => m.m300 = value,
        (0, 3, 0) => m.m030 = value,
        (0, 0, 3) => m.m003 = value,
        (2, 1, 0) => m.m210 = value,
        (2, 0, 1) => m.m201 = value,
        (1, 2, 0) => m.m120 = value,
        (1, 0, 2) => m.m102 = value,
        (0, 2, 1) => m.m021 = value,
        (0, 1, 2) => m.m012 = value,
        (1, 1, 1) => m.m111 = value,
        (4, 0, 0) => m.m400 = value,
        (0, 4, 0) => m.m040 = value,
        (0, 0, 4) => m.m004 = value,
        (3, 1, 0) => m.m310 = value,
        (3, 0, 1) => m.m301 = value,
        (1, 3, 0) => m.m130 = value,
        (1, 0, 3) => m.m103 = value,
        (0, 3, 1) => m.m031 = value,
        (0, 1, 3) => m.m013 = value,
        (2, 2, 0) => m.m220 = value,
        (2, 0, 2) => m.m202 = value,
        (0, 2, 2) => m.m022 = value,
        (2, 1, 1) => m.m211 = value,
        (1, 2, 1) => m.m121 = value,
        (1, 1, 2) => m.m112 = value,
        (5, 0, 0) => m.m500 = value,
        (0, 5, 0) => m.m050 = value,
        (0, 0, 5) => m.m005 = value,
        (4, 1, 0) => m.m410 = value,
        (4, 0, 1) => m.m401 = value,
        (1, 4, 0) => m.m140 = value,
        (1, 0, 4) => m.m104 = value,
        (0, 4, 1) => m.m041 = value,
        (0, 1, 4) => m.m014 = value,
        (3, 2, 0) => m.m320 = value,
        (3, 0, 2) => m.m302 = value,
        (2, 3, 0) => m.m230 = value,
        (0, 3, 2) => m.m032 = value,
        (0, 2, 3) => m.m023 = value,
        (2, 0, 3) => m.m203 = value,
        (2, 2, 1) => m.m221 = value,
        (2, 1, 2) => m.m212 = value,
        (1, 2, 2) => m.m122 = value,
        (3, 1, 1) => m.m311 = value,
        (1, 3, 1) => m.m131 = value,
        (1, 1, 3) => m.m113 = value,
        _ => {},
    }
}

/// Derivatives of softened 1/r with respect to x, y, z up to order 5.
#[derive(Clone, Copy, Default)]
pub struct PotentialDerivatives {
    pub d000: f64,
    pub d100: f64,
    pub d010: f64,
    pub d001: f64,
    pub d200: f64,
    pub d020: f64,
    pub d002: f64,
    pub d110: f64,
    pub d101: f64,
    pub d011: f64,
    pub d300: f64,
    pub d030: f64,
    pub d003: f64,
    pub d210: f64,
    pub d201: f64,
    pub d120: f64,
    pub d102: f64,
    pub d021: f64,
    pub d012: f64,
    pub d111: f64,
    pub d400: f64,
    pub d040: f64,
    pub d004: f64,
    pub d310: f64,
    pub d301: f64,
    pub d130: f64,
    pub d103: f64,
    pub d031: f64,
    pub d013: f64,
    pub d220: f64,
    pub d202: f64,
    pub d022: f64,
    pub d211: f64,
    pub d121: f64,
    pub d112: f64,
    pub d500: f64,
    pub d050: f64,
    pub d005: f64,
    pub d410: f64,
    pub d401: f64,
    pub d140: f64,
    pub d104: f64,
    pub d041: f64,
    pub d014: f64,
    pub d320: f64,
    pub d302: f64,
    pub d230: f64,
    pub d203: f64,
    pub d032: f64,
    pub d023: f64,
    pub d221: f64,
    pub d212: f64,
    pub d122: f64,
    pub d311: f64,
    pub d131: f64,
    pub d113: f64,
}

impl PotentialDerivatives {
    pub fn new(dx: f64, dy: f64, dz: f64, eps2: f64, _order: u8) -> Self {
        let r2 = dx * dx + dy * dy + dz * dz + eps2;
        let r = r2.sqrt();
        let r_inv = 1.0 / r;

        let dt_1 = r_inv; // 1/r
        let mut dt_2 = -1.0 * dt_1 * r_inv; // -1/r^2
        let mut dt_3 = -3.0 * dt_2 * r_inv; // 3/r^3
        let mut dt_4 = -5.0 * dt_3 * r_inv; // -15/r^4
        let mut dt_5 = -7.0 * dt_4 * r_inv; // 105/r^5
        let dt_6 = -9.0 * dt_5 * r_inv; // -945/r^6

        let rx_r = dx * r_inv;
        let ry_r = dy * r_inv;
        let rz_r = dz * r_inv;

        let rx_r2 = rx_r * rx_r;
        let ry_r2 = ry_r * ry_r;
        let rz_r2 = rz_r * rz_r;

        let rx_r3 = rx_r2 * rx_r;
        let ry_r3 = ry_r2 * ry_r;
        let rz_r3 = rz_r2 * rz_r;

        let rx_r4 = rx_r3 * rx_r;
        let ry_r4 = ry_r3 * ry_r;
        let rz_r4 = rz_r3 * rz_r;

        let rx_r5 = rx_r4 * rx_r;
        let ry_r5 = ry_r4 * ry_r;
        let rz_r5 = rz_r4 * rz_r;

        let mut d = PotentialDerivatives::default();

        d.d000 = dt_1;

        d.d100 = dt_2 * rx_r;
        d.d010 = dt_2 * ry_r;
        d.d001 = dt_2 * rz_r;

        dt_2 *= r_inv;
        d.d200 = dt_3 * rx_r2 + dt_2;
        d.d020 = dt_3 * ry_r2 + dt_2;
        d.d002 = dt_3 * rz_r2 + dt_2;
        d.d110 = dt_3 * rx_r * ry_r;
        d.d101 = dt_3 * rx_r * rz_r;
        d.d011 = dt_3 * ry_r * rz_r;

        dt_3 *= r_inv;
        d.d300 = dt_4 * rx_r3 + 3.0 * dt_3 * rx_r;
        d.d030 = dt_4 * ry_r3 + 3.0 * dt_3 * ry_r;
        d.d003 = dt_4 * rz_r3 + 3.0 * dt_3 * rz_r;
        d.d210 = dt_4 * rx_r2 * ry_r + dt_3 * ry_r;
        d.d201 = dt_4 * rx_r2 * rz_r + dt_3 * rz_r;
        d.d120 = dt_4 * ry_r2 * rx_r + dt_3 * rx_r;
        d.d102 = dt_4 * rz_r2 * rx_r + dt_3 * rx_r;
        d.d021 = dt_4 * ry_r2 * rz_r + dt_3 * rz_r;
        d.d012 = dt_4 * rz_r2 * ry_r + dt_3 * ry_r;
        d.d111 = dt_4 * rx_r * ry_r * rz_r;

        dt_3 *= r_inv;
        dt_4 *= r_inv;
        d.d400 = dt_5 * rx_r4 + 6.0 * dt_4 * rx_r2 + 3.0 * dt_3;
        d.d040 = dt_5 * ry_r4 + 6.0 * dt_4 * ry_r2 + 3.0 * dt_3;
        d.d004 = dt_5 * rz_r4 + 6.0 * dt_4 * rz_r2 + 3.0 * dt_3;
        d.d310 = dt_5 * rx_r3 * ry_r + 3.0 * dt_4 * rx_r * ry_r;
        d.d301 = dt_5 * rx_r3 * rz_r + 3.0 * dt_4 * rx_r * rz_r;
        d.d130 = dt_5 * ry_r3 * rx_r + 3.0 * dt_4 * ry_r * rx_r;
        d.d103 = dt_5 * rz_r3 * rx_r + 3.0 * dt_4 * rx_r * rz_r;
        d.d031 = dt_5 * ry_r3 * rz_r + 3.0 * dt_4 * rz_r * ry_r;
        d.d013 = dt_5 * rz_r3 * ry_r + 3.0 * dt_4 * rz_r * ry_r;
        d.d220 = dt_5 * rx_r2 * ry_r2 + dt_4 * (rx_r2 + ry_r2) + dt_3;
        d.d202 = dt_5 * rx_r2 * rz_r2 + dt_4 * (rx_r2 + rz_r2) + dt_3;
        d.d022 = dt_5 * ry_r2 * rz_r2 + dt_4 * (ry_r2 + rz_r2) + dt_3;
        d.d211 = dt_5 * rx_r2 * ry_r * rz_r + dt_4 * ry_r * rz_r;
        d.d121 = dt_5 * ry_r2 * rx_r * rz_r + dt_4 * rx_r * rz_r;
        d.d112 = dt_5 * rz_r2 * rx_r * ry_r + dt_4 * rx_r * ry_r;

        dt_4 *= r_inv;
        dt_5 *= r_inv;
        d.d500 = dt_6 * rx_r5 + 10.0 * dt_5 * rx_r3 + 15.0 * dt_4 * rx_r;
        d.d050 = dt_6 * ry_r5 + 10.0 * dt_5 * ry_r3 + 15.0 * dt_4 * ry_r;
        d.d005 = dt_6 * rz_r5 + 10.0 * dt_5 * rz_r3 + 15.0 * dt_4 * rz_r;
        d.d410 = dt_6 * rx_r4 * ry_r + 6.0 * dt_5 * rx_r2 * ry_r + 3.0 * dt_4 * ry_r;
        d.d401 = dt_6 * rx_r4 * rz_r + 6.0 * dt_5 * rx_r2 * rz_r + 3.0 * dt_4 * rz_r;
        d.d140 = dt_6 * ry_r4 * rx_r + 6.0 * dt_5 * ry_r2 * rx_r + 3.0 * dt_4 * rx_r;
        d.d041 = dt_6 * ry_r4 * rz_r + 6.0 * dt_5 * ry_r2 * rz_r + 3.0 * dt_4 * rz_r;
        d.d104 = dt_6 * rz_r4 * rx_r + 6.0 * dt_5 * rz_r2 * rx_r + 3.0 * dt_4 * rx_r;
        d.d014 = dt_6 * rz_r4 * ry_r + 6.0 * dt_5 * rz_r2 * ry_r + 3.0 * dt_4 * ry_r;
        d.d320 = dt_6 * rx_r3 * ry_r2 + dt_5 * rx_r3 + 3.0 * dt_5 * rx_r * ry_r2 + 3.0 * dt_4 * rx_r;
        d.d302 = dt_6 * rx_r3 * rz_r2 + dt_5 * rx_r3 + 3.0 * dt_5 * rx_r * rz_r2 + 3.0 * dt_4 * rx_r;
        d.d230 = dt_6 * ry_r3 * rx_r2 + dt_5 * ry_r3 + 3.0 * dt_5 * ry_r * rx_r2 + 3.0 * dt_4 * ry_r;
        d.d032 = dt_6 * ry_r3 * rz_r2 + dt_5 * ry_r3 + 3.0 * dt_5 * ry_r * rz_r2 + 3.0 * dt_4 * ry_r;
        d.d203 = dt_6 * rz_r3 * rx_r2 + dt_5 * rz_r3 + 3.0 * dt_5 * rz_r * rx_r2 + 3.0 * dt_4 * rz_r;
        d.d023 = dt_6 * rz_r3 * ry_r2 + dt_5 * rz_r3 + 3.0 * dt_5 * rz_r * ry_r2 + 3.0 * dt_4 * rz_r;
        d.d311 = dt_6 * rx_r3 * ry_r * rz_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
        d.d131 = dt_6 * ry_r3 * rx_r * rz_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
        d.d113 = dt_6 * rz_r3 * rx_r * ry_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
        d.d122 = dt_6 * rx_r * ry_r2 * rz_r2 + dt_5 * rx_r * ry_r2 + dt_5 * rx_r * rz_r2 + dt_4 * rx_r;
        d.d212 = dt_6 * ry_r * rx_r2 * rz_r2 + dt_5 * ry_r * rx_r2 + dt_5 * ry_r * rz_r2 + dt_4 * ry_r;
        d.d221 = dt_6 * rz_r * rx_r2 * ry_r2 + dt_5 * rz_r * rx_r2 + dt_5 * rz_r * ry_r2 + dt_4 * rz_r;
        d.d311 = dt_6 * rx_r3 * ry_r * rz_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
        d.d131 = dt_6 * ry_r3 * rx_r * rz_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;
        d.d113 = dt_6 * rz_r3 * rx_r * ry_r + 3.0 * dt_5 * rx_r * ry_r * rz_r;

        d
    }
}

/// Compute gravitational potential from multipole moments.
pub fn gravity_potential_multipole(
    m: &MultipoleMoment,
    d: &PotentialDerivatives,
    order: u8,
) -> f64 {
    let mut phi = 0.0;

    // Monopole
    phi -= m.m000 * d.d000;

    // higher-order terms (about center-of-mass) follow example_multipole.py
    if order >= 1 {
        // should be zero for center-of-mass expansion
       // phi -= m.m100 * d.d100 + m.m010 * d.d010 + m.m001 * d.d001;
    }

    if order >= 2 {
        phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
        phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
    }

    if order >= 3 {
        phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
        phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
        phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
        phi -= m.m111 * d.d111;
    }

    if order >= 4 {
        phi -= m.m400 * d.d400 + m.m040 * d.d040 + m.m004 * d.d004;
        phi -= m.m310 * d.d310 + m.m301 * d.d301 + m.m130 * d.d130;
        phi -= m.m103 * d.d103 + m.m031 * d.d031 + m.m013 * d.d013;
        phi -= m.m220 * d.d220 + m.m202 * d.d202 + m.m022 * d.d022;
        phi -= m.m211 * d.d211 + m.m121 * d.d121 + m.m112 * d.d112;
    }

    if order >= 5 {
        phi -= m.m500 * d.d500 + m.m050 * d.d050 + m.m005 * d.d005;
        phi -= m.m410 * d.d410 + m.m401 * d.d401 + m.m140 * d.d140;
        phi -= m.m104 * d.d104 + m.m041 * d.d041 + m.m014 * d.d014;
        phi -= m.m320 * d.d320 + m.m302 * d.d302 + m.m230 * d.d230;
        phi -= m.m203 * d.d203 + m.m032 * d.d032 + m.m023 * d.d023;
        phi -= m.m221 * d.d221 + m.m212 * d.d212 + m.m122 * d.d122;
        phi -= m.m311 * d.d311 + m.m131 * d.d131 + m.m113 * d.d113;
    }

    phi
}

/// Compute gravitational acceleration vector from multipole moments.
pub fn gravity_accel_multipole(
    m: &MultipoleMoment,
    d: &PotentialDerivatives,
    order: u8,
) -> [f64; 3] {

    let mut ax = 0.0f64;
    let mut ay = 0.0f64;
    let mut az = 0.0f64;

    ax -= m.m000 * d.d100;
    ay -= m.m000 * d.d010;
    az -= m.m000 * d.d001;

    if order >=2 {
        // should be zero for center-of-mass expansion
        ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
        ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
        az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
    }
    if order >= 3{
        ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
        ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
        ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
        ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
        az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
        az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
    }
    if order >= 4 {
        ax -= m.m003 * d.d103 + m.m012 * d.d112 + m.m021 * d.d121 + m.m030 * d.d130 +
              m.m102 * d.d202 + m.m111 * d.d211 + m.m120 * d.d220 + m.m201 * d.d301 +
              m.m210 * d.d310 + m.m300 * d.d400;
        ay -= m.m003 * d.d013 + m.m012 * d.d022 + m.m021 * d.d031 + m.m030 * d.d040 +
              m.m102 * d.d112 + m.m111 * d.d121 + m.m120 * d.d130 + m.m201 * d.d211 +
              m.m210 * d.d220 + m.m300 * d.d310;
        az -= m.m003 * d.d004 + m.m012 * d.d013 + m.m021 * d.d022 + m.m030 * d.d031 +
              m.m102 * d.d103 + m.m111 * d.d112 + m.m120 * d.d121 + m.m201 * d.d202 +
              m.m210 * d.d211 + m.m300 * d.d301;
    }
    if order >= 5{
        ax -= m.m004 * d.d104 + m.m013 * d.d113 + m.m022 * d.d122 + m.m031 * d.d131 +
              m.m040 * d.d140 + m.m103 * d.d203 + m.m112 * d.d212 + m.m121 * d.d221 +
              m.m130 * d.d230 + m.m202 * d.d302 + m.m211 * d.d311 + m.m220 * d.d320 +
              m.m301 * d.d401 + m.m310 * d.d410 + m.m400 * d.d500;
        ay -= m.m004 * d.d014 + m.m013 * d.d023 + m.m022 * d.d032 + m.m031 * d.d041 +
              m.m040 * d.d050 + m.m103 * d.d113 + m.m112 * d.d122 + m.m121 * d.d131 +
              m.m130 * d.d140 + m.m202 * d.d212 + m.m211 * d.d221 + m.m220 * d.d230 +
              m.m301 * d.d311 + m.m310 * d.d320 + m.m400 * d.d410;
        az -= m.m004 * d.d005 + m.m013 * d.d014 + m.m022 * d.d023 + m.m031 * d.d032 +
              m.m040 * d.d041 + m.m103 * d.d104 + m.m112 * d.d113 + m.m121 * d.d122 +
              m.m130 * d.d131 + m.m202 * d.d203 + m.m211 * d.d212 + m.m220 * d.d221 +
              m.m301 * d.d302 + m.m310 * d.d311 + m.m400 * d.d401;
    }

    [ax, ay, az]
}

/// Translate a multipole expansion from one center to another.
///
/// Moments `m_child` are defined about some center `C_c`. Given a new
/// center `C_p`, with `shift = C_p - C_c`, this returns moments about
/// `C_p`, using the factorial-normalized translation formula up to
/// the requested `order` (max 5).
pub fn translate_multipole(
    m_child: &MultipoleMoment,
    shift: [f64; 3],
    order: u8,
) -> MultipoleMoment {
    let o = (order as usize).min(5);
    let mut out = MultipoleMoment::zero();

    for l in 0..=o {
        for mm in 0..=o {
            for n in 0..=o {
                if l + mm + n > o {
                    continue;
                }

                let mut sum = 0.0f64;
                for i in 0..=l {
                    for j in 0..=mm {
                        for k in 0..=n {
                            let base = get_moment(m_child, i, j, k);
                            if base == 0.0 {
                                continue;
                            }
                            let dl = l - i;
                            let dm = mm - j;
                            let dn = n - k;
                            let pow = if dl + dm + dn == 0 {
                                1.0
                            } else {
                                let sx = if dl > 0 { shift[0].powi(dl as i32) } else { 1.0 };
                                let sy = if dm > 0 { shift[1].powi(dm as i32) } else { 1.0 };
                                let sz = if dn > 0 { shift[2].powi(dn as i32) } else { 1.0 };
                                sx * sy * sz
                            };
                            let sign = if (dl + dm + dn) % 2 == 0 { 1.0 } else { -1.0 };
                            let coeff = sign * pow
                                / (FACT[dl] * FACT[dm] * FACT[dn]);
                            sum += coeff * base;
                        }
                    }
                }

                set_moment(&mut out, l, mm, n, sum);
            }
        }
    }

    out
}
