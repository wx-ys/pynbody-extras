//! Multipole moments — full and compact storage, construction, and M2M translation.
//!
//! This module provides:
//! - [`MultipoleMoment`] — full 56-field moment (up to order 5), used temporarily during tree build
//! - `Moment0` through `Moment5` — compact storage with only the fields needed for a given order
//! - [`MultipoleMoments`] — enum wrapping the compact variants for dispatch

use std::f64;

/// Full Cartesian multipole moment up to order 5 (56 coefficients).
///
/// Notation follows `example_multipole.py`: `M_lmn` stores
/// `1/(l! m! n!) * sum m_i x_i^l y_i^m z_i^n` about a chosen origin.
///
/// This is used as a temporary working type during tree building — the
/// final node-level storage uses the compact [`MultipoleMoments`] enum.
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
        match order.min(5) {
            0 => Self::from_points_const::<0>(positions, masses_opt, indices, center),
            1 => Self::from_points_const::<1>(positions, masses_opt, indices, center),
            2 => Self::from_points_const::<2>(positions, masses_opt, indices, center),
            3 => Self::from_points_const::<3>(positions, masses_opt, indices, center),
            4 => Self::from_points_const::<4>(positions, masses_opt, indices, center),
            _ => Self::from_points_const::<5>(positions, masses_opt, indices, center),
        }
    }

    #[inline]
    fn from_points_const<const O: u8>(
        positions: &[[f64; 3]],
        masses_opt: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
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

            if O >= 1 {
                m.m100 += mass * x;
                m.m010 += mass * y;
                m.m001 += mass * z;
            }
            if O >= 2 {
                m.m200 += 0.5 * mass * x * x;
                m.m020 += 0.5 * mass * y * y;
                m.m002 += 0.5 * mass * z * z;
                m.m110 += mass * x * y;
                m.m101 += mass * x * z;
                m.m011 += mass * y * z;
            }
            if O >= 3 {
                m.m300 += (1.0 / 6.0) * mass * x.powi(3);
                m.m030 += (1.0 / 6.0) * mass * y.powi(3);
                m.m003 += (1.0 / 6.0) * mass * z.powi(3);
                m.m210 += 0.5 * mass * x * x * y;
                m.m201 += 0.5 * mass * x * x * z;
                m.m120 += 0.5 * mass * y * y * x;
                m.m102 += 0.5 * mass * x * z * z;
                m.m021 += 0.5 * mass * y * y * z;
                m.m012 += 0.5 * mass * y * z * z;
                m.m111 += mass * x * y * z;
            }
            if O >= 4 {
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
            if O >= 5 {
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
}

// ===========================================================================
// Compact moment storage — only the fields needed for a given order
// ===========================================================================

macro_rules! define_moment_struct {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        #[derive(Clone, Copy, Default)]
        pub struct $name {
            $(pub $field: f64,)+
        }

        impl From<MultipoleMoment> for $name {
            fn from(value: MultipoleMoment) -> Self {
                Self { $($field: value.$field,)+ }
            }
        }
    };
}

define_moment_struct!(Moment0 { m000 });

define_moment_struct!(Moment2 {
    m000, m100, m010, m001,
    m200, m020, m002,
    m110, m101, m011
});

define_moment_struct!(Moment3 {
    m000, m100, m010, m001,
    m200, m020, m002,
    m110, m101, m011,
    m300, m030, m003,
    m210, m201, m120, m102, m021, m012,
    m111
});

define_moment_struct!(Moment4 {
    m000, m100, m010, m001,
    m200, m020, m002,
    m110, m101, m011,
    m300, m030, m003,
    m210, m201, m120, m102, m021, m012,
    m111,
    m400, m040, m004,
    m310, m301, m130, m103, m031, m013,
    m220, m202, m022,
    m211, m121, m112
});

/// Order-5 storage is identical to the full moment (all 56 coefficients).
pub type Moment5 = MultipoleMoment;

// ===========================================================================
// Compact multipole storage enum — one variant per order
// ===========================================================================

/// Compact multipole storage for a fixed runtime order.
///
/// The tree uses a single multipole order, so we store one compact variant
/// for all nodes and dispatch once per traversal (not per node).
#[derive(Clone)]
pub enum MultipoleMoments {
    /// Order 0 (monopole only) — also used for order 1.
    O0(Vec<Moment0>),
    O2(Vec<Moment2>),
    O3(Vec<Moment3>),
    O4(Vec<Moment4>),
    O5(Vec<Moment5>),
}

impl MultipoleMoments {
    /// Build compact moments from a vector of full [`MultipoleMoment`]s.
    pub fn from_full(full: Vec<MultipoleMoment>, order: u8) -> Self {
        match order.min(5) {
            0 | 1 => Self::O0(full.into_iter().map(Moment0::from).collect()),
            2 => Self::O2(full.into_iter().map(Moment2::from).collect()),
            3 => Self::O3(full.into_iter().map(Moment3::from).collect()),
            4 => Self::O4(full.into_iter().map(Moment4::from).collect()),
            _ => Self::O5(full),
        }
    }
}

// ===========================================================================
// translate_multipole — M2M operator
// ===========================================================================

const FACT: [f64; 6] = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0];

#[inline]
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
        _ => {}
    }
}

/// M2M: translate a multipole expansion from one center to another.
///
/// `shift = C_parent - C_child`.  Returns moments about `C_parent` up to `order` (max 5).
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
                                let sx = if dl > 0 {
                                    shift[0].powi(dl as i32)
                                } else {
                                    1.0
                                };
                                let sy = if dm > 0 {
                                    shift[1].powi(dm as i32)
                                } else {
                                    1.0
                                };
                                let sz = if dn > 0 {
                                    shift[2].powi(dn as i32)
                                } else {
                                    1.0
                                };
                                sx * sy * sz
                            };
                            let sign = if (dl + dm + dn) % 2 == 0 { 1.0 } else { -1.0 };
                            let coeff = sign * pow / (FACT[dl] * FACT[dm] * FACT[dn]);
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
