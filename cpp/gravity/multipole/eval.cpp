#include "gravity/multipole/eval.hpp"

namespace gravity {

// ===========================================================================
// Helpers: convert compact moments back to full for translation
// ===========================================================================

namespace {

MultipoleMoment moment_to_full_o2(const Moment2& m) {
    MultipoleMoment full{};
    full.m000 = m.m000;
    full.m100 = m.m100;
    full.m010 = m.m010;
    full.m001 = m.m001;
    full.m200 = m.m200;
    full.m020 = m.m020;
    full.m002 = m.m002;
    full.m110 = m.m110;
    full.m101 = m.m101;
    full.m011 = m.m011;
    return full;
}

MultipoleMoment moment_to_full_o3(const Moment3& m) {
    MultipoleMoment full{};
    full.m000 = m.m000;
    full.m100 = m.m100;
    full.m010 = m.m010;
    full.m001 = m.m001;
    full.m200 = m.m200;
    full.m020 = m.m020;
    full.m002 = m.m002;
    full.m110 = m.m110;
    full.m101 = m.m101;
    full.m011 = m.m011;
    full.m300 = m.m300;
    full.m030 = m.m030;
    full.m003 = m.m003;
    full.m210 = m.m210;
    full.m201 = m.m201;
    full.m120 = m.m120;
    full.m102 = m.m102;
    full.m021 = m.m021;
    full.m012 = m.m012;
    full.m111 = m.m111;
    return full;
}

MultipoleMoment moment_to_full_o4(const Moment4& m) {
    MultipoleMoment full{};
    full.m000 = m.m000;
    full.m100 = m.m100;
    full.m010 = m.m010;
    full.m001 = m.m001;
    full.m200 = m.m200;
    full.m020 = m.m020;
    full.m002 = m.m002;
    full.m110 = m.m110;
    full.m101 = m.m101;
    full.m011 = m.m011;
    full.m300 = m.m300;
    full.m030 = m.m030;
    full.m003 = m.m003;
    full.m210 = m.m210;
    full.m201 = m.m201;
    full.m120 = m.m120;
    full.m102 = m.m102;
    full.m021 = m.m021;
    full.m012 = m.m012;
    full.m111 = m.m111;
    full.m400 = m.m400;
    full.m040 = m.m040;
    full.m004 = m.m004;
    full.m310 = m.m310;
    full.m301 = m.m301;
    full.m130 = m.m130;
    full.m103 = m.m103;
    full.m031 = m.m031;
    full.m013 = m.m013;
    full.m220 = m.m220;
    full.m202 = m.m202;
    full.m022 = m.m022;
    full.m211 = m.m211;
    full.m121 = m.m121;
    full.m112 = m.m112;
    return full;
}

// ===========================================================================
// Legacy evaluator functions — kept for order-5 (full) compatibility
// ===========================================================================

// Gravitational potential from order-5 multipole.
double gravity_potential_multipole_o5(const MultipoleMoment& m, const PotentialDerivatives& d) {
    double phi = -m.m000 * d.d000;
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
    return phi;
}

// Gravitational acceleration from order-5 multipole.
Vec3 gravity_accel_multipole_o5(const MultipoleMoment& m, const PotentialDerivatives& d) {
    double ax = -m.m000 * d.d100;
    double ay = -m.m000 * d.d010;
    double az = -m.m000 * d.d001;
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
    return Vec3(ax, ay, az);
}

} // namespace

// ===========================================================================
// Order 0 (also used for order 1 — dipole vanishes about COM)
// ===========================================================================

MultipoleEval<0>::Moment MultipoleEval<0>::from_points(const std::vector<Vec3>& positions,
                                                       const double* masses,
                                                       const std::vector<size_t>& indices,
                                                       const Vec3& center) {
    MultipoleMoment full = MultipoleMoment::from_points(positions, masses, indices, center, 0);
    return Moment(full);
}

MultipoleEval<0>::Moment MultipoleEval<0>::translate(const Moment& m, const Vec3& /*shift*/) {
    // Monopole is invariant under translation at order 0.
    return m;
}

void MultipoleEval<0>::add_assign(Moment& acc, const Moment& other) {
    acc.m000 += other.m000;
}

double MultipoleEval<0>::potential(const Moment& m, const Derivatives& d) {
    return -m.m000 * d.d000;
}

Vec3 MultipoleEval<0>::acceleration(const Moment& m, const Derivatives& d) {
    return Vec3(-m.m000 * d.d100, -m.m000 * d.d010, -m.m000 * d.d001);
}

MultipoleEval<0>::Derivatives MultipoleEval<0>::derivatives(double dx, double dy, double dz, double eps2) {
    return Derivatives::new_derivatives(dx, dy, dz, eps2);
}

// ===========================================================================
// Order 2
// ===========================================================================

MultipoleEval<2>::Moment MultipoleEval<2>::from_points(const std::vector<Vec3>& positions,
                                                       const double* masses,
                                                       const std::vector<size_t>& indices,
                                                       const Vec3& center) {
    MultipoleMoment full = MultipoleMoment::from_points(positions, masses, indices, center, 2);
    return Moment(full);
}

MultipoleEval<2>::Moment MultipoleEval<2>::translate(const Moment& m, const Vec3& shift) {
    MultipoleMoment full = moment_to_full_o2(m);
    MultipoleMoment translated = translate_multipole(full, shift, 2);
    return Moment(translated);
}

void MultipoleEval<2>::add_assign(Moment& acc, const Moment& other) {
    acc.m000 += other.m000;
    acc.m100 += other.m100;
    acc.m010 += other.m010;
    acc.m001 += other.m001;
    acc.m200 += other.m200;
    acc.m020 += other.m020;
    acc.m002 += other.m002;
    acc.m110 += other.m110;
    acc.m101 += other.m101;
    acc.m011 += other.m011;
}

double MultipoleEval<2>::potential(const Moment& m, const Derivatives& d) {
    double phi = -m.m000 * d.d000;
    phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
    phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
    return phi;
}

Vec3 MultipoleEval<2>::acceleration(const Moment& m, const Derivatives& d) {
    double ax = -m.m000 * d.d100;
    double ay = -m.m000 * d.d010;
    double az = -m.m000 * d.d001;
    ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
    ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
    az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
    return Vec3(ax, ay, az);
}

MultipoleEval<2>::Derivatives MultipoleEval<2>::derivatives(double dx, double dy, double dz, double eps2) {
    return Derivatives::new_derivatives(dx, dy, dz, eps2);
}

// ===========================================================================
// Order 3
// ===========================================================================

MultipoleEval<3>::Moment MultipoleEval<3>::from_points(const std::vector<Vec3>& positions,
                                                       const double* masses,
                                                       const std::vector<size_t>& indices,
                                                       const Vec3& center) {
    MultipoleMoment full = MultipoleMoment::from_points(positions, masses, indices, center, 3);
    return Moment(full);
}

MultipoleEval<3>::Moment MultipoleEval<3>::translate(const Moment& m, const Vec3& shift) {
    MultipoleMoment full = moment_to_full_o3(m);
    MultipoleMoment translated = translate_multipole(full, shift, 3);
    return Moment(translated);
}

void MultipoleEval<3>::add_assign(Moment& acc, const Moment& other) {
    acc.m000 += other.m000;
    acc.m100 += other.m100;
    acc.m010 += other.m010;
    acc.m001 += other.m001;
    acc.m200 += other.m200;
    acc.m020 += other.m020;
    acc.m002 += other.m002;
    acc.m110 += other.m110;
    acc.m101 += other.m101;
    acc.m011 += other.m011;
    acc.m300 += other.m300;
    acc.m030 += other.m030;
    acc.m003 += other.m003;
    acc.m210 += other.m210;
    acc.m201 += other.m201;
    acc.m120 += other.m120;
    acc.m102 += other.m102;
    acc.m021 += other.m021;
    acc.m012 += other.m012;
    acc.m111 += other.m111;
}

double MultipoleEval<3>::potential(const Moment& m, const Derivatives& d) {
    double phi = -m.m000 * d.d000;
    phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
    phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
    phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
    phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
    phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
    phi -= m.m111 * d.d111;
    return phi;
}

Vec3 MultipoleEval<3>::acceleration(const Moment& m, const Derivatives& d) {
    double ax = -m.m000 * d.d100;
    double ay = -m.m000 * d.d010;
    double az = -m.m000 * d.d001;
    ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
    ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
    az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
    ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
    ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
    ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
    ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
    az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
    az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
    return Vec3(ax, ay, az);
}

MultipoleEval<3>::Derivatives MultipoleEval<3>::derivatives(double dx, double dy, double dz, double eps2) {
    return Derivatives::new_derivatives(dx, dy, dz, eps2);
}

// ===========================================================================
// Order 4
// ===========================================================================

MultipoleEval<4>::Moment MultipoleEval<4>::from_points(const std::vector<Vec3>& positions,
                                                       const double* masses,
                                                       const std::vector<size_t>& indices,
                                                       const Vec3& center) {
    MultipoleMoment full = MultipoleMoment::from_points(positions, masses, indices, center, 4);
    return Moment(full);
}

MultipoleEval<4>::Moment MultipoleEval<4>::translate(const Moment& m, const Vec3& shift) {
    MultipoleMoment full = moment_to_full_o4(m);
    MultipoleMoment translated = translate_multipole(full, shift, 4);
    return Moment(translated);
}

void MultipoleEval<4>::add_assign(Moment& acc, const Moment& other) {
    acc.m000 += other.m000;
    acc.m100 += other.m100;
    acc.m010 += other.m010;
    acc.m001 += other.m001;
    acc.m200 += other.m200;
    acc.m020 += other.m020;
    acc.m002 += other.m002;
    acc.m110 += other.m110;
    acc.m101 += other.m101;
    acc.m011 += other.m011;
    acc.m300 += other.m300;
    acc.m030 += other.m030;
    acc.m003 += other.m003;
    acc.m210 += other.m210;
    acc.m201 += other.m201;
    acc.m120 += other.m120;
    acc.m102 += other.m102;
    acc.m021 += other.m021;
    acc.m012 += other.m012;
    acc.m111 += other.m111;
    acc.m400 += other.m400;
    acc.m040 += other.m040;
    acc.m004 += other.m004;
    acc.m310 += other.m310;
    acc.m301 += other.m301;
    acc.m130 += other.m130;
    acc.m103 += other.m103;
    acc.m031 += other.m031;
    acc.m013 += other.m013;
    acc.m220 += other.m220;
    acc.m202 += other.m202;
    acc.m022 += other.m022;
    acc.m211 += other.m211;
    acc.m121 += other.m121;
    acc.m112 += other.m112;
}

double MultipoleEval<4>::potential(const Moment& m, const Derivatives& d) {
    double phi = -m.m000 * d.d000;
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
    return phi;
}

Vec3 MultipoleEval<4>::acceleration(const Moment& m, const Derivatives& d) {
    double ax = -m.m000 * d.d100;
    double ay = -m.m000 * d.d010;
    double az = -m.m000 * d.d001;
    ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
    ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
    az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
    ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
    ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
    ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
    ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
    az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
    az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
    ax -= m.m003 * d.d103
        + m.m012 * d.d112
        + m.m021 * d.d121
        + m.m030 * d.d130
        + m.m102 * d.d202
        + m.m111 * d.d211
        + m.m120 * d.d220
        + m.m201 * d.d301
        + m.m210 * d.d310
        + m.m300 * d.d400;
    ay -= m.m003 * d.d013
        + m.m012 * d.d022
        + m.m021 * d.d031
        + m.m030 * d.d040
        + m.m102 * d.d112
        + m.m111 * d.d121
        + m.m120 * d.d130
        + m.m201 * d.d211
        + m.m210 * d.d220
        + m.m300 * d.d310;
    az -= m.m003 * d.d004
        + m.m012 * d.d013
        + m.m021 * d.d022
        + m.m030 * d.d031
        + m.m102 * d.d103
        + m.m111 * d.d112
        + m.m120 * d.d121
        + m.m201 * d.d202
        + m.m210 * d.d211
        + m.m300 * d.d301;
    return Vec3(ax, ay, az);
}

MultipoleEval<4>::Derivatives MultipoleEval<4>::derivatives(double dx, double dy, double dz, double eps2) {
    return Derivatives::new_derivatives(dx, dy, dz, eps2);
}

// ===========================================================================
// Order 5
// ===========================================================================

MultipoleEval<5>::Moment MultipoleEval<5>::from_points(const std::vector<Vec3>& positions,
                                                       const double* masses,
                                                       const std::vector<size_t>& indices,
                                                       const Vec3& center) {
    return MultipoleMoment::from_points(positions, masses, indices, center, 5);
}

MultipoleEval<5>::Moment MultipoleEval<5>::translate(const Moment& m, const Vec3& shift) {
    return translate_multipole(m, shift, 5);
}

void MultipoleEval<5>::add_assign(Moment& acc, const Moment& other) {
    acc.add_assign(other);
}

double MultipoleEval<5>::potential(const Moment& m, const Derivatives& d) {
    return gravity_potential_multipole_o5(m, d);
}

Vec3 MultipoleEval<5>::acceleration(const Moment& m, const Derivatives& d) {
    return gravity_accel_multipole_o5(m, d);
}

MultipoleEval<5>::Derivatives MultipoleEval<5>::derivatives(double dx, double dy, double dz, double eps2) {
    return Derivatives::new_derivatives(dx, dy, dz, eps2, 5);
}

// ===========================================================================
// Runtime-dispatch evaluators (useful for tests and interactive use)
// ===========================================================================

double gravity_potential_multipole(const MultipoleMoment& m, const PotentialDerivatives& d,
                                   unsigned char order) {
    unsigned char o = (order < 5) ? order : 5;
    switch (o) {
        case 0:
        case 1:
            return -m.m000 * d.d000;
        case 2: {
            double phi = -m.m000 * d.d000;
            phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
            phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
            return phi;
        }
        case 3: {
            double phi = -m.m000 * d.d000;
            phi -= m.m200 * d.d200 + m.m020 * d.d020 + m.m002 * d.d002;
            phi -= m.m110 * d.d110 + m.m101 * d.d101 + m.m011 * d.d011;
            phi -= m.m300 * d.d300 + m.m030 * d.d030 + m.m003 * d.d003;
            phi -= m.m210 * d.d210 + m.m201 * d.d201 + m.m120 * d.d120;
            phi -= m.m102 * d.d102 + m.m021 * d.d021 + m.m012 * d.d012;
            phi -= m.m111 * d.d111;
            return phi;
        }
        case 4: {
            double phi = -m.m000 * d.d000;
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
            return phi;
        }
        default:
            return gravity_potential_multipole_o5(m, d);
    }
}

Vec3 gravity_accel_multipole(const MultipoleMoment& m, const PotentialDerivatives& d,
                             unsigned char order) {
    double ax = -m.m000 * d.d100;
    double ay = -m.m000 * d.d010;
    double az = -m.m000 * d.d001;
    unsigned char o = (order < 5) ? order : 5;
    switch (o) {
        case 0:
        case 1:
            break;
        case 2: {
            ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
            ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
            az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
            break;
        }
        case 3: {
            ax -= m.m100 * d.d200 + m.m010 * d.d110 + m.m001 * d.d101;
            ay -= m.m100 * d.d110 + m.m010 * d.d020 + m.m001 * d.d011;
            az -= m.m100 * d.d101 + m.m010 * d.d011 + m.m001 * d.d002;
            ax -= m.m200 * d.d300 + m.m020 * d.d120 + m.m002 * d.d102;
            ax -= m.m110 * d.d210 + m.m101 * d.d201 + m.m011 * d.d111;
            ay -= m.m200 * d.d210 + m.m020 * d.d030 + m.m002 * d.d012;
            ay -= m.m110 * d.d120 + m.m101 * d.d111 + m.m011 * d.d021;
            az -= m.m200 * d.d201 + m.m020 * d.d021 + m.m002 * d.d003;
            az -= m.m110 * d.d111 + m.m101 * d.d102 + m.m011 * d.d012;
            break;
        }
        case 4: {
            Vec3 a3 = gravity_accel_multipole(m, d, 3);
            ax = a3.x;
            ay = a3.y;
            az = a3.z;
            ax -= m.m003 * d.d103
                + m.m012 * d.d112
                + m.m021 * d.d121
                + m.m030 * d.d130
                + m.m102 * d.d202
                + m.m111 * d.d211
                + m.m120 * d.d220
                + m.m201 * d.d301
                + m.m210 * d.d310
                + m.m300 * d.d400;
            ay -= m.m003 * d.d013
                + m.m012 * d.d022
                + m.m021 * d.d031
                + m.m030 * d.d040
                + m.m102 * d.d112
                + m.m111 * d.d121
                + m.m120 * d.d130
                + m.m201 * d.d211
                + m.m210 * d.d220
                + m.m300 * d.d310;
            az -= m.m003 * d.d004
                + m.m012 * d.d013
                + m.m021 * d.d022
                + m.m030 * d.d031
                + m.m102 * d.d103
                + m.m111 * d.d112
                + m.m120 * d.d121
                + m.m201 * d.d202
                + m.m210 * d.d211
                + m.m300 * d.d301;
            break;
        }
        default:
            return gravity_accel_multipole_o5(m, d);
    }
    return Vec3(ax, ay, az);
}

// Explicit instantiations (consumers link against these).
template struct MultipoleEval<0>;
template struct MultipoleEval<2>;
template struct MultipoleEval<3>;
template struct MultipoleEval<4>;
template struct MultipoleEval<5>;

} // namespace gravity
