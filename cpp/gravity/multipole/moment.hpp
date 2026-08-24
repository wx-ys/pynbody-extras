#pragma once
#include <cstddef>
#include <variant>
#include <vector>
#include "gravity/vec3.hpp"

namespace gravity {

// Full Cartesian multipole moment up to order 5 (56 coefficients).
// M_lmn stores 1/(l! m! n!) * sum m_i x_i^l y_i^m z_i^n about a chosen origin.
struct MultipoleMoment {
    // order 0
    double m000;
    // order 1
    double m100, m010, m001;
    // order 2
    double m200, m020, m002, m110, m101, m011;
    // order 3
    double m300, m030, m003, m210, m201, m120, m102, m021, m012, m111;
    // order 4
    double m400, m040, m004, m310, m301, m130, m103, m031, m013;
    double m220, m202, m022, m211, m121, m112;
    // order 5
    double m500, m050, m005, m410, m401, m140, m104, m041, m014;
    double m320, m302, m230, m203, m032, m023, m221, m212, m122, m311, m131, m113;

    static MultipoleMoment zero();
    static MultipoleMoment from_points(const std::vector<Vec3>& positions,
                                       const double* masses,
                                       const std::vector<size_t>& indices,
                                       const Vec3& center,
                                       unsigned char order);
    void add_assign(const MultipoleMoment& o);
};

// Compact storage — only the fields needed for a given order.

struct Moment0 {
    double m000;
    explicit Moment0(const MultipoleMoment& v) : m000(v.m000) {}
};

struct Moment2 {
    double m000, m100, m010, m001, m200, m020, m002, m110, m101, m011;
    explicit Moment2(const MultipoleMoment& v)
        : m000(v.m000), m100(v.m100), m010(v.m010), m001(v.m001),
          m200(v.m200), m020(v.m020), m002(v.m002),
          m110(v.m110), m101(v.m101), m011(v.m011) {}
};

struct Moment3 {
    double m000, m100, m010, m001, m200, m020, m002, m110, m101, m011;
    double m300, m030, m003, m210, m201, m120, m102, m021, m012, m111;
    explicit Moment3(const MultipoleMoment& v)
        : m000(v.m000), m100(v.m100), m010(v.m010), m001(v.m001),
          m200(v.m200), m020(v.m020), m002(v.m002),
          m110(v.m110), m101(v.m101), m011(v.m011),
          m300(v.m300), m030(v.m030), m003(v.m003),
          m210(v.m210), m201(v.m201), m120(v.m120),
          m102(v.m102), m021(v.m021), m012(v.m012), m111(v.m111) {}
};

struct Moment4 {
    double m000, m100, m010, m001, m200, m020, m002, m110, m101, m011;
    double m300, m030, m003, m210, m201, m120, m102, m021, m012, m111;
    double m400, m040, m004, m310, m301, m130, m103, m031, m013;
    double m220, m202, m022, m211, m121, m112;
    explicit Moment4(const MultipoleMoment& v)
        : m000(v.m000), m100(v.m100), m010(v.m010), m001(v.m001),
          m200(v.m200), m020(v.m020), m002(v.m002),
          m110(v.m110), m101(v.m101), m011(v.m011),
          m300(v.m300), m030(v.m030), m003(v.m003),
          m210(v.m210), m201(v.m201), m120(v.m120),
          m102(v.m102), m021(v.m021), m012(v.m012), m111(v.m111),
          m400(v.m400), m040(v.m040), m004(v.m004),
          m310(v.m310), m301(v.m301), m130(v.m130),
          m103(v.m103), m031(v.m031), m013(v.m013),
          m220(v.m220), m202(v.m202), m022(v.m022),
          m211(v.m211), m121(v.m121), m112(v.m112) {}
};

// Order-5 storage is identical to the full moment (all 56 coefficients).
using Moment5 = MultipoleMoment;

// Compact multipole storage for a fixed runtime order.
using MultipoleMoments = std::variant<std::vector<Moment0>, std::vector<Moment2>,
                                      std::vector<Moment3>, std::vector<Moment4>,
                                      std::vector<Moment5>>;

MultipoleMoments multipole_moments_from_full(std::vector<MultipoleMoment> full, unsigned char order);

// M2M: translate a multipole expansion from one center to another.
// shift = C_parent - C_child.
MultipoleMoment translate_multipole(const MultipoleMoment& m_child, const Vec3& shift, unsigned char order);

} // namespace gravity
