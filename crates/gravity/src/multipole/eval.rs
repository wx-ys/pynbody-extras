//! Multipole evaluators — trait + per-order implementations.
//!
//! Each order gets a unit struct implementing [`MultipoleOrder`]. This replaces
//! the old combinatorial explosion of `gravity_potential_multipole_oN_dM` functions.
//! The compiler monomorphizes one copy of the traversal per order via generic dispatch.

use crate::multipole::derivatives::{
    PotentialDerivatives, PotentialDerivatives1, PotentialDerivatives2, PotentialDerivatives3,
    PotentialDerivatives4,
};
use crate::multipole::moment::{
    Moment0, Moment2, Moment3, Moment4, Moment5, MultipoleMoment, MultipoleMoments,
};

// ===========================================================================
// MultipoleOrder trait
// ===========================================================================

/// Trait abstracting over multipole expansion order.
///
/// Each implementor provides the compact moment storage type, the compact
/// derivative type, and all operations needed for tree building and traversal.
#[allow(dead_code)]
pub(crate) trait MultipoleOrder {
    type Moment: Clone + Default;
    type Derivatives;

    const ORDER: u8;

    /// P2M: build moments from a set of particles about a center.
    fn from_points(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
    ) -> Self::Moment;

    /// M2M: translate child moment to parent center.
    fn translate(moment: &Self::Moment, shift: [f64; 3]) -> Self::Moment;

    /// Accumulate another moment at the same order and center.
    fn add_assign(acc: &mut Self::Moment, other: &Self::Moment);

    /// M2P: evaluate gravitational potential.
    fn potential(m: &Self::Moment, d: &Self::Derivatives) -> f64;

    /// M2P: evaluate gravitational acceleration.
    fn acceleration(m: &Self::Moment, d: &Self::Derivatives) -> [f64; 3];

    /// Build derivatives at a given displacement vector.
    fn derivatives(dx: f64, dy: f64, dz: f64, eps2: f64) -> Self::Derivatives;

    /// Build compact moments from a full MultipoleMoment for all nodes.
    fn compact_from_full(full: Vec<MultipoleMoment>) -> MultipoleMoments;
}

// ===========================================================================
// Evaluator structs (one per order)
// ===========================================================================

pub(crate) struct Order0Eval;
pub(crate) struct Order2Eval;
pub(crate) struct Order3Eval;
pub(crate) struct Order4Eval;
pub(crate) struct Order5Eval;

// ------- Order 0 (also used for order 1 — dipole vanishes about COM) -------

impl MultipoleOrder for Order0Eval {
    type Moment = Moment0;
    type Derivatives = PotentialDerivatives1;
    const ORDER: u8 = 0;

    fn from_points(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
    ) -> Moment0 {
        let full = MultipoleMoment::from_points(positions, masses, indices, center, 0);
        Moment0::from(full)
    }

    fn translate(moment: &Moment0, _shift: [f64; 3]) -> Moment0 {
        // Monopole is invariant under translation at order 0.
        // (Higher-order moments would be generated if the target order were >0,
        // but at order 0 we truncate them.)
        *moment
    }

    fn add_assign(acc: &mut Moment0, other: &Moment0) {
        acc.m000 += other.m000;
    }

    fn potential(m: &Moment0, d: &PotentialDerivatives1) -> f64 {
        -m.m000 * d.d000
    }

    fn acceleration(m: &Moment0, d: &PotentialDerivatives1) -> [f64; 3] {
        [-m.m000 * d.d100, -m.m000 * d.d010, -m.m000 * d.d001]
    }

    fn derivatives(dx: f64, dy: f64, dz: f64, eps2: f64) -> PotentialDerivatives1 {
        PotentialDerivatives1::new(dx, dy, dz, eps2)
    }

    fn compact_from_full(full: Vec<MultipoleMoment>) -> MultipoleMoments {
        MultipoleMoments::O0(full.into_iter().map(Moment0::from).collect())
    }
}

// ------- Order 2 -------

impl MultipoleOrder for Order2Eval {
    type Moment = Moment2;
    type Derivatives = PotentialDerivatives2;
    const ORDER: u8 = 2;

    fn from_points(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
    ) -> Moment2 {
        let full = MultipoleMoment::from_points(positions, masses, indices, center, 2);
        Moment2::from(full)
    }

    fn translate(moment: &Moment2, shift: [f64; 3]) -> Moment2 {
        // Convert to full for translation, then compact back.
        let full = moment_to_full_o2(moment);
        let translated =
            crate::multipole::moment::translate_multipole(&full, shift, 2);
        Moment2::from(translated)
    }

    fn add_assign(acc: &mut Moment2, other: &Moment2) {
        acc.m000 += other.m000;
        acc.m100 += other.m100; acc.m010 += other.m010; acc.m001 += other.m001;
        acc.m200 += other.m200; acc.m020 += other.m020; acc.m002 += other.m002;
        acc.m110 += other.m110; acc.m101 += other.m101; acc.m011 += other.m011;
    }

    fn potential(m: &Moment2, d: &PotentialDerivatives2) -> f64 {
        let mut phi = -m.m000 * d.d000;
        phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
        phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
        phi
    }

    fn acceleration(m: &Moment2, d: &PotentialDerivatives2) -> [f64; 3] {
        let mut ax = -m.m000 * d.d100;
        let mut ay = -m.m000 * d.d010;
        let mut az = -m.m000 * d.d001;
        ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
        ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
        az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
        [ax, ay, az]
    }

    fn derivatives(dx: f64, dy: f64, dz: f64, eps2: f64) -> PotentialDerivatives2 {
        PotentialDerivatives2::new(dx, dy, dz, eps2)
    }

    fn compact_from_full(full: Vec<MultipoleMoment>) -> MultipoleMoments {
        MultipoleMoments::O2(full.into_iter().map(Moment2::from).collect())
    }
}

// ------- Order 3 -------

impl MultipoleOrder for Order3Eval {
    type Moment = Moment3;
    type Derivatives = PotentialDerivatives3;
    const ORDER: u8 = 3;

    fn from_points(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
    ) -> Moment3 {
        let full = MultipoleMoment::from_points(positions, masses, indices, center, 3);
        Moment3::from(full)
    }

    fn translate(moment: &Moment3, shift: [f64; 3]) -> Moment3 {
        let full = moment_to_full_o3(moment);
        let translated = crate::multipole::moment::translate_multipole(&full, shift, 3);
        Moment3::from(translated)
    }

    fn add_assign(acc: &mut Moment3, other: &Moment3) {
        acc.m000 += other.m000;
        acc.m100 += other.m100; acc.m010 += other.m010; acc.m001 += other.m001;
        acc.m200 += other.m200; acc.m020 += other.m020; acc.m002 += other.m002;
        acc.m110 += other.m110; acc.m101 += other.m101; acc.m011 += other.m011;
        acc.m300 += other.m300; acc.m030 += other.m030; acc.m003 += other.m003;
        acc.m210 += other.m210; acc.m201 += other.m201; acc.m120 += other.m120;
        acc.m102 += other.m102; acc.m021 += other.m021; acc.m012 += other.m012;
        acc.m111 += other.m111;
    }

    fn potential(m: &Moment3, d: &PotentialDerivatives3) -> f64 {
        let mut phi = -m.m000 * d.d000;
        phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
        phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
        phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
        phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
        phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
        phi -= m.m111 * d.d111;
        phi
    }

    fn acceleration(m: &Moment3, d: &PotentialDerivatives3) -> [f64; 3] {
        let mut ax = -m.m000 * d.d100;
        let mut ay = -m.m000 * d.d010;
        let mut az = -m.m000 * d.d001;
        ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
        ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
        az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
        ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
        ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
        ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
        ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
        az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
        az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
        [ax, ay, az]
    }

    fn derivatives(dx: f64, dy: f64, dz: f64, eps2: f64) -> PotentialDerivatives3 {
        PotentialDerivatives3::new(dx, dy, dz, eps2)
    }

    fn compact_from_full(full: Vec<MultipoleMoment>) -> MultipoleMoments {
        MultipoleMoments::O3(full.into_iter().map(Moment3::from).collect())
    }
}

// ------- Order 4 -------

impl MultipoleOrder for Order4Eval {
    type Moment = Moment4;
    type Derivatives = PotentialDerivatives4;
    const ORDER: u8 = 4;

    fn from_points(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
    ) -> Moment4 {
        let full = MultipoleMoment::from_points(positions, masses, indices, center, 4);
        Moment4::from(full)
    }

    fn translate(moment: &Moment4, shift: [f64; 3]) -> Moment4 {
        let full = moment_to_full_o4(moment);
        let translated = crate::multipole::moment::translate_multipole(&full, shift, 4);
        Moment4::from(translated)
    }

    fn add_assign(acc: &mut Moment4, other: &Moment4) {
        acc.m000 += other.m000;
        acc.m100 += other.m100; acc.m010 += other.m010; acc.m001 += other.m001;
        acc.m200 += other.m200; acc.m020 += other.m020; acc.m002 += other.m002;
        acc.m110 += other.m110; acc.m101 += other.m101; acc.m011 += other.m011;
        acc.m300 += other.m300; acc.m030 += other.m030; acc.m003 += other.m003;
        acc.m210 += other.m210; acc.m201 += other.m201; acc.m120 += other.m120;
        acc.m102 += other.m102; acc.m021 += other.m021; acc.m012 += other.m012;
        acc.m111 += other.m111;
        acc.m400 += other.m400; acc.m040 += other.m040; acc.m004 += other.m004;
        acc.m310 += other.m310; acc.m301 += other.m301; acc.m130 += other.m130;
        acc.m103 += other.m103; acc.m031 += other.m031; acc.m013 += other.m013;
        acc.m220 += other.m220; acc.m202 += other.m202; acc.m022 += other.m022;
        acc.m211 += other.m211; acc.m121 += other.m121; acc.m112 += other.m112;
    }

    fn potential(m: &Moment4, d: &PotentialDerivatives4) -> f64 {
        let mut phi = -m.m000 * d.d000;
        phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
        phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
        phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
        phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
        phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
        phi -= m.m111 * d.d111;
        phi -= m.m400 * d.d400 + m.m040 * d.d040 + m.m004 * d.d004;
        phi -= m.m310 * d.d310 + m.m301 * d.d301 + m.m130 * d.d130;
        phi -= m.m103 * d.d103 + m.m031 * d.d031 + m.m013 * d.d013;
        phi -= m.m220 * d.d220 + m.m202 * d.d202 + m.m022 * d.d022;
        phi -= m.m211 * d.d211 + m.m121 * d.d121 + m.m112 * d.d112;
        phi
    }

    fn acceleration(m: &Moment4, d: &PotentialDerivatives4) -> [f64; 3] {
        let mut ax = -m.m000 * d.d100;
        let mut ay = -m.m000 * d.d010;
        let mut az = -m.m000 * d.d001;
        ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
        ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
        az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
        ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
        ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
        ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
        ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
        az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
        az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
        ax -= m.m003 * d.d103 + m.m012 * d.d112 + m.m021 * d.d121 + m.m030 * d.d130
            + m.m102 * d.d202 + m.m111 * d.d211 + m.m120 * d.d220 + m.m201 * d.d301
            + m.m210 * d.d310 + m.m300 * d.d400;
        ay -= m.m003 * d.d013 + m.m012 * d.d022 + m.m021 * d.d031 + m.m030 * d.d040
            + m.m102 * d.d112 + m.m111 * d.d121 + m.m120 * d.d130 + m.m201 * d.d211
            + m.m210 * d.d220 + m.m300 * d.d310;
        az -= m.m003 * d.d004 + m.m012 * d.d013 + m.m021 * d.d022 + m.m030 * d.d031
            + m.m102 * d.d103 + m.m111 * d.d112 + m.m120 * d.d121 + m.m201 * d.d202
            + m.m210 * d.d211 + m.m300 * d.d301;
        [ax, ay, az]
    }

    fn derivatives(dx: f64, dy: f64, dz: f64, eps2: f64) -> PotentialDerivatives4 {
        PotentialDerivatives4::new(dx, dy, dz, eps2)
    }

    fn compact_from_full(full: Vec<MultipoleMoment>) -> MultipoleMoments {
        MultipoleMoments::O4(full.into_iter().map(Moment4::from).collect())
    }
}

// ------- Order 5 -------

impl MultipoleOrder for Order5Eval {
    type Moment = Moment5;
    type Derivatives = PotentialDerivatives;
    const ORDER: u8 = 5;

    fn from_points(
        positions: &[[f64; 3]],
        masses: Option<&[f64]>,
        indices: &[usize],
        center: [f64; 3],
    ) -> Moment5 {
        MultipoleMoment::from_points(positions, masses, indices, center, 5)
    }

    fn translate(moment: &Moment5, shift: [f64; 3]) -> Moment5 {
        crate::multipole::moment::translate_multipole(moment, shift, 5)
    }

    fn add_assign(acc: &mut Moment5, other: &Moment5) {
        acc.add_assign(other);
    }

    fn potential(m: &Moment5, d: &PotentialDerivatives) -> f64 {
        crate::multipole::eval::gravity_potential_multipole_o5(m, d)
    }

    fn acceleration(m: &Moment5, d: &PotentialDerivatives) -> [f64; 3] {
        crate::multipole::eval::gravity_accel_multipole_o5(m, d)
    }

    fn derivatives(dx: f64, dy: f64, dz: f64, eps2: f64) -> PotentialDerivatives {
        PotentialDerivatives::new(dx, dy, dz, eps2, 5)
    }

    fn compact_from_full(full: Vec<MultipoleMoment>) -> MultipoleMoments {
        MultipoleMoments::O5(full)
    }
}

// ===========================================================================
// Helpers: convert compact moments back to full for translation
// ===========================================================================

#[allow(dead_code)]
fn moment_to_full_o2(m: &Moment2) -> MultipoleMoment {
    MultipoleMoment {
        m000: m.m000,
        m100: m.m100, m010: m.m010, m001: m.m001,
        m200: m.m200, m020: m.m020, m002: m.m002,
        m110: m.m110, m101: m.m101, m011: m.m011,
        ..Default::default()
    }
}

#[allow(dead_code)]
fn moment_to_full_o3(m: &Moment3) -> MultipoleMoment {
    MultipoleMoment {
        m000: m.m000,
        m100: m.m100, m010: m.m010, m001: m.m001,
        m200: m.m200, m020: m.m020, m002: m.m002,
        m110: m.m110, m101: m.m101, m011: m.m011,
        m300: m.m300, m030: m.m030, m003: m.m003,
        m210: m.m210, m201: m.m201, m120: m.m120,
        m102: m.m102, m021: m.m021, m012: m.m012,
        m111: m.m111,
        ..Default::default()
    }
}

#[allow(dead_code)]
fn moment_to_full_o4(m: &Moment4) -> MultipoleMoment {
    MultipoleMoment {
        m000: m.m000,
        m100: m.m100, m010: m.m010, m001: m.m001,
        m200: m.m200, m020: m.m020, m002: m.m002,
        m110: m.m110, m101: m.m101, m011: m.m011,
        m300: m.m300, m030: m.m030, m003: m.m003,
        m210: m.m210, m201: m.m201, m120: m.m120,
        m102: m.m102, m021: m.m021, m012: m.m012,
        m111: m.m111,
        m400: m.m400, m040: m.m040, m004: m.m004,
        m310: m.m310, m301: m.m301, m130: m.m130,
        m103: m.m103, m031: m.m031, m013: m.m013,
        m220: m.m220, m202: m.m202, m022: m.m022,
        m211: m.m211, m121: m.m121, m112: m.m112,
        ..Default::default()
    }
}

// ===========================================================================
// Legacy evaluator functions — kept for order-5 (full) compatibility
// ===========================================================================

/// Gravitational potential from order-5 multipole.
#[inline]
pub fn gravity_potential_multipole_o5(m: &Moment5, d: &PotentialDerivatives) -> f64 {
    let mut phi = -m.m000 * d.d000;
    phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
    phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
    phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
    phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
    phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
    phi -= m.m111 * d.d111;
    phi -= m.m400 * d.d400 + m.m040 * d.d040 + m.m004 * d.d004;
    phi -= m.m310 * d.d310 + m.m301 * d.d301 + m.m130 * d.d130;
    phi -= m.m103 * d.d103 + m.m031 * d.d031 + m.m013 * d.d013;
    phi -= m.m220 * d.d220 + m.m202 * d.d202 + m.m022 * d.d022;
    phi -= m.m211 * d.d211 + m.m121 * d.d121 + m.m112 * d.d112;
    phi -= m.m500 * d.d500 + m.m050 * d.d050 + m.m005 * d.d005;
    phi -= m.m410 * d.d410 + m.m401 * d.d401 + m.m140 * d.d140;
    phi -= m.m104 * d.d104 + m.m041 * d.d041 + m.m014 * d.d014;
    phi -= m.m320 * d.d320 + m.m302 * d.d302 + m.m230 * d.d230;
    phi -= m.m203 * d.d203 + m.m032 * d.d032 + m.m023 * d.d023;
    phi -= m.m221 * d.d221 + m.m212 * d.d212 + m.m122 * d.d122;
    phi -= m.m311 * d.d311 + m.m131 * d.d131 + m.m113 * d.d113;
    phi
}

/// Gravitational acceleration from order-5 multipole.
#[inline]
pub fn gravity_accel_multipole_o5(m: &Moment5, d: &PotentialDerivatives) -> [f64; 3] {
    let mut ax = -m.m000 * d.d100;
    let mut ay = -m.m000 * d.d010;
    let mut az = -m.m000 * d.d001;
    ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
    ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
    az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
    ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
    ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
    ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
    ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
    az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
    az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
    ax -= m.m003 * d.d103 + m.m012 * d.d112 + m.m021 * d.d121 + m.m030 * d.d130
        + m.m102 * d.d202 + m.m111 * d.d211 + m.m120 * d.d220 + m.m201 * d.d301
        + m.m210 * d.d310 + m.m300 * d.d400;
    ay -= m.m003 * d.d013 + m.m012 * d.d022 + m.m021 * d.d031 + m.m030 * d.d040
        + m.m102 * d.d112 + m.m111 * d.d121 + m.m120 * d.d130 + m.m201 * d.d211
        + m.m210 * d.d220 + m.m300 * d.d310;
    az -= m.m003 * d.d004 + m.m012 * d.d013 + m.m021 * d.d022 + m.m030 * d.d031
        + m.m102 * d.d103 + m.m111 * d.d112 + m.m120 * d.d121 + m.m201 * d.d202
        + m.m210 * d.d211 + m.m300 * d.d301;
    ax -= m.m004 * d.d104 + m.m013 * d.d113 + m.m022 * d.d122 + m.m031 * d.d131
        + m.m040 * d.d140 + m.m103 * d.d203 + m.m112 * d.d212 + m.m121 * d.d221
        + m.m130 * d.d230 + m.m202 * d.d302 + m.m211 * d.d311 + m.m220 * d.d320
        + m.m301 * d.d401 + m.m310 * d.d410 + m.m400 * d.d500;
    ay -= m.m004 * d.d014 + m.m013 * d.d023 + m.m022 * d.d032 + m.m031 * d.d041
        + m.m040 * d.d050 + m.m103 * d.d113 + m.m112 * d.d122 + m.m121 * d.d131
        + m.m130 * d.d140 + m.m202 * d.d212 + m.m211 * d.d221 + m.m220 * d.d230
        + m.m301 * d.d311 + m.m310 * d.d320 + m.m400 * d.d410;
    az -= m.m004 * d.d005 + m.m013 * d.d014 + m.m022 * d.d023 + m.m031 * d.d032
        + m.m040 * d.d041 + m.m103 * d.d104 + m.m112 * d.d113 + m.m121 * d.d122
        + m.m130 * d.d131 + m.m202 * d.d203 + m.m211 * d.d212 + m.m220 * d.d221
        + m.m301 * d.d302 + m.m310 * d.d311 + m.m400 * d.d401;
    [ax, ay, az]
}

// ===========================================================================
// Runtime-dispatch evaluators (useful for tests and interactive use)
// ===========================================================================

/// Evaluate gravitational potential from full multipole moments at a given order (0..5).
pub fn gravity_potential_multipole(
    m: &MultipoleMoment,
    d: &PotentialDerivatives,
    order: u8,
) -> f64 {
    match order.min(5) {
        0 | 1 => -m.m000 * d.d000,
        2 => {
            let mut phi = -m.m000 * d.d000;
            phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
            phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
            phi
        }
        3 => {
            let mut phi = -m.m000 * d.d000;
            phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
            phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
            phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
            phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
            phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
            phi -= m.m111 * d.d111;
            phi
        }
        4 => {
            let mut phi = -m.m000 * d.d000;
            phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
            phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
            phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
            phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
            phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
            phi -= m.m111 * d.d111;
            phi -= m.m400 * d.d400 + m.m040 * d.d040 + m.m004 * d.d004;
            phi -= m.m310 * d.d310 + m.m301 * d.d301 + m.m130 * d.d130;
            phi -= m.m103 * d.d103 + m.m031 * d.d031 + m.m013 * d.d013;
            phi -= m.m220 * d.d220 + m.m202 * d.d202 + m.m022 * d.d022;
            phi -= m.m211 * d.d211 + m.m121 * d.d121 + m.m112 * d.d112;
            phi
        }
        _ => gravity_potential_multipole_o5(m, d),
    }
}

/// Evaluate gravitational acceleration from full multipole moments at a given order (0..5).
pub fn gravity_accel_multipole(
    m: &MultipoleMoment,
    d: &PotentialDerivatives,
    order: u8,
) -> [f64; 3] {
    let mut ax = -m.m000 * d.d100;
    let mut ay = -m.m000 * d.d010;
    let mut az = -m.m000 * d.d001;
    match order.min(5) {
        0 | 1 => {}
        2 => {
            ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
            ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
            az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
        }
        3 => {
            ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
            ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
            az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
            ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
            ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
            ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
            ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
            az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
            az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
        }
        4 => {
            let a3 = gravity_accel_multipole(m, d, 3);
            ax = a3[0]; ay = a3[1]; az = a3[2];
            ax -= m.m003 * d.d103 + m.m012 * d.d112 + m.m021 * d.d121 + m.m030 * d.d130
                + m.m102 * d.d202 + m.m111 * d.d211 + m.m120 * d.d220 + m.m201 * d.d301
                + m.m210 * d.d310 + m.m300 * d.d400;
            ay -= m.m003 * d.d013 + m.m012 * d.d022 + m.m021 * d.d031 + m.m030 * d.d040
                + m.m102 * d.d112 + m.m111 * d.d121 + m.m120 * d.d130 + m.m201 * d.d211
                + m.m210 * d.d220 + m.m300 * d.d310;
            az -= m.m003 * d.d004 + m.m012 * d.d013 + m.m021 * d.d022 + m.m030 * d.d031
                + m.m102 * d.d103 + m.m111 * d.d112 + m.m120 * d.d121 + m.m201 * d.d202
                + m.m210 * d.d211 + m.m300 * d.d301;
        }
        _ => {
            let a5 = gravity_accel_multipole_o5(m, d);
            return a5;
        }
    }
    [ax, ay, az]
}
