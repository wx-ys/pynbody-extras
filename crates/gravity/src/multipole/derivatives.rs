//! Derivatives of the softened 1/r potential, full and compact variants.
//!
//! Each variant stores only the terms needed for a given multipole order.

const R2_TINY: f64 = f64::MIN_POSITIVE;

// ---- Order-1 derivatives (enough for monopole) ----

#[derive(Clone, Copy, Default)]
pub struct PotentialDerivatives1 {
    pub d000: f64,
    pub d100: f64,
    pub d010: f64,
    pub d001: f64,
}

impl PotentialDerivatives1 {
    #[inline]
    pub fn new(dx: f64, dy: f64, dz: f64, eps2: f64) -> Self {
        let r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
        let r = r2.sqrt();
        let r_inv = 1.0 / r;
        let dt_1 = r_inv;
        let dt_2 = -dt_1 * r_inv;
        let rx_r = dx * r_inv;
        let ry_r = dy * r_inv;
        let rz_r = dz * r_inv;
        Self {
            d000: dt_1,
            d100: dt_2 * rx_r,
            d010: dt_2 * ry_r,
            d001: dt_2 * rz_r,
        }
    }
}

// ---- Order-2 derivatives ----

#[derive(Clone, Copy, Default)]
pub struct PotentialDerivatives2 {
    pub d000: f64,
    pub d100: f64, pub d010: f64, pub d001: f64,
    pub d200: f64, pub d020: f64, pub d002: f64,
    pub d110: f64, pub d101: f64, pub d011: f64,
}

impl PotentialDerivatives2 {
    #[inline]
    pub fn new(dx: f64, dy: f64, dz: f64, eps2: f64) -> Self {
        let r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
        let r = r2.sqrt();
        let r_inv = 1.0 / r;
        let dt_1 = r_inv;
        let mut dt_2 = -dt_1 * r_inv;
        let dt_3 = -3.0 * dt_2 * r_inv;
        let rx_r = dx * r_inv;
        let ry_r = dy * r_inv;
        let rz_r = dz * r_inv;
        let rx_r2 = rx_r * rx_r;
        let ry_r2 = ry_r * ry_r;
        let rz_r2 = rz_r * rz_r;
        let d100 = dt_2 * rx_r;
        let d010 = dt_2 * ry_r;
        let d001 = dt_2 * rz_r;
        dt_2 *= r_inv;
        Self {
            d000: dt_1,
            d100, d010, d001,
            d200: dt_3 * rx_r2 + dt_2,
            d020: dt_3 * ry_r2 + dt_2,
            d002: dt_3 * rz_r2 + dt_2,
            d110: dt_3 * rx_r * ry_r,
            d101: dt_3 * rx_r * rz_r,
            d011: dt_3 * ry_r * rz_r,
        }
    }
}

// ---- Order-3 derivatives ----

#[derive(Clone, Copy, Default)]
pub struct PotentialDerivatives3 {
    pub d000: f64,
    pub d100: f64, pub d010: f64, pub d001: f64,
    pub d200: f64, pub d020: f64, pub d002: f64,
    pub d110: f64, pub d101: f64, pub d011: f64,
    pub d300: f64, pub d030: f64, pub d003: f64,
    pub d210: f64, pub d201: f64, pub d120: f64,
    pub d102: f64, pub d021: f64, pub d012: f64,
    pub d111: f64,
}

impl PotentialDerivatives3 {
    #[inline]
    pub fn new(dx: f64, dy: f64, dz: f64, eps2: f64) -> Self {
        let r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
        let r = r2.sqrt();
        let r_inv = 1.0 / r;
        let dt_1 = r_inv;
        let mut dt_2 = -dt_1 * r_inv;
        let mut dt_3 = -3.0 * dt_2 * r_inv;
        let dt_4 = -5.0 * dt_3 * r_inv;
        let rx_r = dx * r_inv;
        let ry_r = dy * r_inv;
        let rz_r = dz * r_inv;
        let rx_r2 = rx_r * rx_r;
        let ry_r2 = ry_r * ry_r;
        let rz_r2 = rz_r * rz_r;
        let rx_r3 = rx_r2 * rx_r;
        let ry_r3 = ry_r2 * ry_r;
        let rz_r3 = rz_r2 * rz_r;
        let d100 = dt_2 * rx_r;
        let d010 = dt_2 * ry_r;
        let d001 = dt_2 * rz_r;
        dt_2 *= r_inv;
        let d200 = dt_3 * rx_r2 + dt_2;
        let d020 = dt_3 * ry_r2 + dt_2;
        let d002 = dt_3 * rz_r2 + dt_2;
        let d110 = dt_3 * rx_r * ry_r;
        let d101 = dt_3 * rx_r * rz_r;
        let d011 = dt_3 * ry_r * rz_r;
        dt_3 *= r_inv;
        Self {
            d000: dt_1,
            d100, d010, d001,
            d200, d020, d002, d110, d101, d011,
            d300: dt_4 * rx_r3 + 3.0 * dt_3 * rx_r,
            d030: dt_4 * ry_r3 + 3.0 * dt_3 * ry_r,
            d003: dt_4 * rz_r3 + 3.0 * dt_3 * rz_r,
            d210: dt_4 * rx_r2 * ry_r + dt_3 * ry_r,
            d201: dt_4 * rx_r2 * rz_r + dt_3 * rz_r,
            d120: dt_4 * ry_r2 * rx_r + dt_3 * rx_r,
            d102: dt_4 * rz_r2 * rx_r + dt_3 * rx_r,
            d021: dt_4 * ry_r2 * rz_r + dt_3 * rz_r,
            d012: dt_4 * rz_r2 * ry_r + dt_3 * ry_r,
            d111: dt_4 * rx_r * ry_r * rz_r,
        }
    }
}

// ---- Order-4 derivatives ----

#[derive(Clone, Copy, Default)]
pub struct PotentialDerivatives4 {
    pub d000: f64,
    pub d100: f64, pub d010: f64, pub d001: f64,
    pub d200: f64, pub d020: f64, pub d002: f64,
    pub d110: f64, pub d101: f64, pub d011: f64,
    pub d300: f64, pub d030: f64, pub d003: f64,
    pub d210: f64, pub d201: f64, pub d120: f64,
    pub d102: f64, pub d021: f64, pub d012: f64,
    pub d111: f64,
    pub d400: f64, pub d040: f64, pub d004: f64,
    pub d310: f64, pub d301: f64, pub d130: f64,
    pub d103: f64, pub d031: f64, pub d013: f64,
    pub d220: f64, pub d202: f64, pub d022: f64,
    pub d211: f64, pub d121: f64, pub d112: f64,
}

impl PotentialDerivatives4 {
    #[inline]
    pub fn new(dx: f64, dy: f64, dz: f64, eps2: f64) -> Self {
        // Delegate to the full implementation and truncate.
        let d = PotentialDerivatives::new(dx, dy, dz, eps2, 4);
        Self {
            d000: d.d000,
            d100: d.d100, d010: d.d010, d001: d.d001,
            d200: d.d200, d020: d.d020, d002: d.d002,
            d110: d.d110, d101: d.d101, d011: d.d011,
            d300: d.d300, d030: d.d030, d003: d.d003,
            d210: d.d210, d201: d.d201, d120: d.d120,
            d102: d.d102, d021: d.d021, d012: d.d012,
            d111: d.d111,
            d400: d.d400, d040: d.d040, d004: d.d004,
            d310: d.d310, d301: d.d301, d130: d.d130,
            d103: d.d103, d031: d.d031, d013: d.d013,
            d220: d.d220, d202: d.d202, d022: d.d022,
            d211: d.d211, d121: d.d121, d112: d.d112,
        }
    }
}

// ---- Full order-5 derivatives ----

#[derive(Clone, Copy, Default)]
pub struct PotentialDerivatives {
    pub d000: f64,
    pub d100: f64, pub d010: f64, pub d001: f64,
    pub d200: f64, pub d020: f64, pub d002: f64,
    pub d110: f64, pub d101: f64, pub d011: f64,
    pub d300: f64, pub d030: f64, pub d003: f64,
    pub d210: f64, pub d201: f64, pub d120: f64,
    pub d102: f64, pub d021: f64, pub d012: f64,
    pub d111: f64,
    pub d400: f64, pub d040: f64, pub d004: f64,
    pub d310: f64, pub d301: f64, pub d130: f64,
    pub d103: f64, pub d031: f64, pub d013: f64,
    pub d220: f64, pub d202: f64, pub d022: f64,
    pub d211: f64, pub d121: f64, pub d112: f64,
    pub d500: f64, pub d050: f64, pub d005: f64,
    pub d410: f64, pub d401: f64, pub d140: f64,
    pub d104: f64, pub d041: f64, pub d014: f64,
    pub d320: f64, pub d302: f64, pub d230: f64,
    pub d203: f64, pub d032: f64, pub d023: f64,
    pub d221: f64, pub d212: f64, pub d122: f64,
    pub d311: f64, pub d131: f64, pub d113: f64,
}

impl PotentialDerivatives {
    pub fn new(dx: f64, dy: f64, dz: f64, eps2: f64, order: u8) -> Self {
        let max = (order as usize).min(5);
        let r2 = dx * dx + dy * dy + dz * dz + eps2 + R2_TINY;
        let r = r2.sqrt();
        let r_inv = 1.0 / r;

        let dt_1 = r_inv;
        let mut dt_2 = -dt_1 * r_inv;
        let mut dt_3 = -3.0 * dt_2 * r_inv;
        let mut dt_4 = -5.0 * dt_3 * r_inv;
        let mut dt_5 = -7.0 * dt_4 * r_inv;
        let dt_6 = -9.0 * dt_5 * r_inv;

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

        let mut d = PotentialDerivatives { d000: dt_1, ..Default::default() };
        if max == 0 { return d; }

        d.d100 = dt_2 * rx_r;
        d.d010 = dt_2 * ry_r;
        d.d001 = dt_2 * rz_r;
        if max == 1 { return d; }

        dt_2 *= r_inv;
        d.d200 = dt_3 * rx_r2 + dt_2;
        d.d020 = dt_3 * ry_r2 + dt_2;
        d.d002 = dt_3 * rz_r2 + dt_2;
        d.d110 = dt_3 * rx_r * ry_r;
        d.d101 = dt_3 * rx_r * rz_r;
        d.d011 = dt_3 * ry_r * rz_r;
        if max == 2 { return d; }

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
        if max == 3 { return d; }

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
        if max == 4 { return d; }

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

        d
    }
}
