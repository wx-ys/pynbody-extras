#include "gravity/multipole/moment.hpp"
#include "gravity/vec3.hpp"
#include <cmath>
#include <cstddef>
#include <utility>

namespace gravity {

MultipoleMoment MultipoleMoment::zero() {
    return MultipoleMoment{};
}

MultipoleMoment MultipoleMoment::from_points(const std::vector<Vec3>& positions,
                                             const double* masses,
                                             const std::vector<size_t>& indices,
                                             const Vec3& center,
                                             unsigned char order) {
    MultipoleMoment m{};  // zero-init all 56 fields
    if (indices.empty()) {
        return m;
    }

    unsigned char o = (order < 5) ? order : 5;

    for (size_t pi : indices) {
        const Vec3& p = positions[pi];
        double mass = (masses != nullptr) ? masses[pi] : 1.0;
        double x = p.x - center.x;
        double y = p.y - center.y;
        double z = p.z - center.z;

        double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
        double y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y;
        double z2 = z * z, z3 = z2 * z, z4 = z3 * z, z5 = z4 * z;

        m.m000 += mass;

        if (o >= 1) {
            m.m100 += mass * x;
            m.m010 += mass * y;
            m.m001 += mass * z;
        }
        if (o >= 2) {
            m.m200 += 0.5 * mass * x2;
            m.m020 += 0.5 * mass * y2;
            m.m002 += 0.5 * mass * z2;
            m.m110 += mass * x * y;
            m.m101 += mass * x * z;
            m.m011 += mass * y * z;
        }
        if (o >= 3) {
            m.m300 += (1.0 / 6.0) * mass * x3;
            m.m030 += (1.0 / 6.0) * mass * y3;
            m.m003 += (1.0 / 6.0) * mass * z3;
            m.m210 += 0.5 * mass * x2 * y;
            m.m201 += 0.5 * mass * x2 * z;
            m.m120 += 0.5 * mass * y2 * x;
            m.m102 += 0.5 * mass * x * z2;
            m.m021 += 0.5 * mass * y2 * z;
            m.m012 += 0.5 * mass * y * z2;
            m.m111 += mass * x * y * z;
        }
        if (o >= 4) {
            m.m400 += (1.0 / 24.0) * mass * x4;
            m.m040 += (1.0 / 24.0) * mass * y4;
            m.m004 += (1.0 / 24.0) * mass * z4;
            m.m310 += (1.0 / 6.0) * mass * x3 * y;
            m.m301 += (1.0 / 6.0) * mass * x3 * z;
            m.m130 += (1.0 / 6.0) * mass * y3 * x;
            m.m103 += (1.0 / 6.0) * mass * x * z3;
            m.m031 += (1.0 / 6.0) * mass * y3 * z;
            m.m013 += (1.0 / 6.0) * mass * y * z3;
            m.m220 += 0.25 * mass * x2 * y2;
            m.m202 += 0.25 * mass * x2 * z2;
            m.m022 += 0.25 * mass * y2 * z2;
            m.m211 += 0.5 * mass * x2 * y * z;
            m.m121 += 0.5 * mass * y2 * x * z;
            m.m112 += 0.5 * mass * z2 * x * y;
        }
        if (o >= 5) {
            m.m500 += (1.0 / 120.0) * mass * x5;
            m.m050 += (1.0 / 120.0) * mass * y5;
            m.m005 += (1.0 / 120.0) * mass * z5;
            m.m410 += (1.0 / 24.0) * mass * x4 * y;
            m.m401 += (1.0 / 24.0) * mass * x4 * z;
            m.m140 += (1.0 / 24.0) * mass * y4 * x;
            m.m104 += (1.0 / 24.0) * mass * z4 * x;
            m.m041 += (1.0 / 24.0) * mass * y4 * z;
            m.m014 += (1.0 / 24.0) * mass * z4 * y;
            m.m320 += (1.0 / 12.0) * mass * x3 * y2;
            m.m302 += (1.0 / 12.0) * mass * x3 * z2;
            m.m230 += (1.0 / 12.0) * mass * x2 * y3;
            m.m203 += (1.0 / 12.0) * mass * x2 * z3;
            m.m032 += (1.0 / 12.0) * mass * y3 * z2;
            m.m023 += (1.0 / 12.0) * mass * y2 * z3;
            m.m221 += 0.25 * mass * x2 * y2 * z;
            m.m212 += 0.25 * mass * x2 * z2 * y;
            m.m122 += 0.25 * mass * y2 * z2 * x;
            m.m311 += (1.0 / 6.0) * mass * x3 * y * z;
            m.m131 += (1.0 / 6.0) * mass * y3 * x * z;
            m.m113 += (1.0 / 6.0) * mass * z3 * x * y;
        }
    }

    return m;
}

void MultipoleMoment::add_assign(const MultipoleMoment& o) {
    m000 += o.m000;
    m100 += o.m100;
    m010 += o.m010;
    m001 += o.m001;
    m200 += o.m200;
    m020 += o.m020;
    m002 += o.m002;
    m110 += o.m110;
    m101 += o.m101;
    m011 += o.m011;
    m300 += o.m300;
    m030 += o.m030;
    m003 += o.m003;
    m210 += o.m210;
    m201 += o.m201;
    m120 += o.m120;
    m102 += o.m102;
    m021 += o.m021;
    m012 += o.m012;
    m111 += o.m111;
    m400 += o.m400;
    m040 += o.m040;
    m004 += o.m004;
    m310 += o.m310;
    m301 += o.m301;
    m130 += o.m130;
    m103 += o.m103;
    m031 += o.m031;
    m013 += o.m013;
    m220 += o.m220;
    m202 += o.m202;
    m022 += o.m022;
    m211 += o.m211;
    m121 += o.m121;
    m112 += o.m112;
    m500 += o.m500;
    m050 += o.m050;
    m005 += o.m005;
    m410 += o.m410;
    m401 += o.m401;
    m140 += o.m140;
    m104 += o.m104;
    m041 += o.m041;
    m014 += o.m014;
    m320 += o.m320;
    m302 += o.m302;
    m230 += o.m230;
    m203 += o.m203;
    m032 += o.m032;
    m023 += o.m023;
    m221 += o.m221;
    m212 += o.m212;
    m122 += o.m122;
    m311 += o.m311;
    m131 += o.m131;
    m113 += o.m113;
}

MultipoleMoments multipole_moments_from_full(std::vector<MultipoleMoment> full, unsigned char order) {
    unsigned char o = (order < 5) ? order : 5;
    switch (o) {
        case 0:
        case 1: {
            std::vector<Moment0> v;
            v.reserve(full.size());
            for (const MultipoleMoment& m : full) v.emplace_back(m);
            return MultipoleMoments(std::in_place_index<0>, std::move(v));
        }
        case 2: {
            std::vector<Moment2> v;
            v.reserve(full.size());
            for (const MultipoleMoment& m : full) v.emplace_back(m);
            return MultipoleMoments(std::in_place_index<1>, std::move(v));
        }
        case 3: {
            std::vector<Moment3> v;
            v.reserve(full.size());
            for (const MultipoleMoment& m : full) v.emplace_back(m);
            return MultipoleMoments(std::in_place_index<2>, std::move(v));
        }
        case 4: {
            std::vector<Moment4> v;
            v.reserve(full.size());
            for (const MultipoleMoment& m : full) v.emplace_back(m);
            return MultipoleMoments(std::in_place_index<3>, std::move(v));
        }
        default: {
            return MultipoleMoments(std::in_place_index<4>, std::move(full));
        }
    }
}

// ===========================================================================
// translate_multipole — M2M operator
// ===========================================================================

namespace {

double get_moment(const MultipoleMoment& m, int l, int mm, int n) {
    switch (l * 100 + mm * 10 + n) {
        case 0: return m.m000;
        case 100: return m.m100;
        case 10: return m.m010;
        case 1: return m.m001;
        case 200: return m.m200;
        case 20: return m.m020;
        case 2: return m.m002;
        case 110: return m.m110;
        case 101: return m.m101;
        case 11: return m.m011;
        case 300: return m.m300;
        case 30: return m.m030;
        case 3: return m.m003;
        case 210: return m.m210;
        case 201: return m.m201;
        case 120: return m.m120;
        case 102: return m.m102;
        case 21: return m.m021;
        case 12: return m.m012;
        case 111: return m.m111;
        case 400: return m.m400;
        case 40: return m.m040;
        case 4: return m.m004;
        case 310: return m.m310;
        case 301: return m.m301;
        case 130: return m.m130;
        case 103: return m.m103;
        case 31: return m.m031;
        case 13: return m.m013;
        case 220: return m.m220;
        case 202: return m.m202;
        case 22: return m.m022;
        case 211: return m.m211;
        case 121: return m.m121;
        case 112: return m.m112;
        case 500: return m.m500;
        case 50: return m.m050;
        case 5: return m.m005;
        case 410: return m.m410;
        case 401: return m.m401;
        case 140: return m.m140;
        case 104: return m.m104;
        case 41: return m.m041;
        case 14: return m.m014;
        case 320: return m.m320;
        case 302: return m.m302;
        case 230: return m.m230;
        case 32: return m.m032;
        case 23: return m.m023;
        case 203: return m.m203;
        case 221: return m.m221;
        case 212: return m.m212;
        case 122: return m.m122;
        case 311: return m.m311;
        case 131: return m.m131;
        case 113: return m.m113;
        default: return 0.0;
    }
}

void set_moment(MultipoleMoment& m, int l, int mm, int n, double value) {
    switch (l * 100 + mm * 10 + n) {
        case 0: m.m000 = value; break;
        case 100: m.m100 = value; break;
        case 10: m.m010 = value; break;
        case 1: m.m001 = value; break;
        case 200: m.m200 = value; break;
        case 20: m.m020 = value; break;
        case 2: m.m002 = value; break;
        case 110: m.m110 = value; break;
        case 101: m.m101 = value; break;
        case 11: m.m011 = value; break;
        case 300: m.m300 = value; break;
        case 30: m.m030 = value; break;
        case 3: m.m003 = value; break;
        case 210: m.m210 = value; break;
        case 201: m.m201 = value; break;
        case 120: m.m120 = value; break;
        case 102: m.m102 = value; break;
        case 21: m.m021 = value; break;
        case 12: m.m012 = value; break;
        case 111: m.m111 = value; break;
        case 400: m.m400 = value; break;
        case 40: m.m040 = value; break;
        case 4: m.m004 = value; break;
        case 310: m.m310 = value; break;
        case 301: m.m301 = value; break;
        case 130: m.m130 = value; break;
        case 103: m.m103 = value; break;
        case 31: m.m031 = value; break;
        case 13: m.m013 = value; break;
        case 220: m.m220 = value; break;
        case 202: m.m202 = value; break;
        case 22: m.m022 = value; break;
        case 211: m.m211 = value; break;
        case 121: m.m121 = value; break;
        case 112: m.m112 = value; break;
        case 500: m.m500 = value; break;
        case 50: m.m050 = value; break;
        case 5: m.m005 = value; break;
        case 410: m.m410 = value; break;
        case 401: m.m401 = value; break;
        case 140: m.m140 = value; break;
        case 104: m.m104 = value; break;
        case 41: m.m041 = value; break;
        case 14: m.m014 = value; break;
        case 320: m.m320 = value; break;
        case 302: m.m302 = value; break;
        case 230: m.m230 = value; break;
        case 32: m.m032 = value; break;
        case 23: m.m023 = value; break;
        case 203: m.m203 = value; break;
        case 221: m.m221 = value; break;
        case 212: m.m212 = value; break;
        case 122: m.m122 = value; break;
        case 311: m.m311 = value; break;
        case 131: m.m131 = value; break;
        case 113: m.m113 = value; break;
        default: break;
    }
}

} // namespace

MultipoleMoment translate_multipole(const MultipoleMoment& m_child, const Vec3& shift, unsigned char order) {
    const double FACT[6] = {1.0, 1.0, 2.0, 6.0, 24.0, 120.0};
    int o = (order < 5) ? static_cast<int>(order) : 5;
    MultipoleMoment out = MultipoleMoment::zero();

    for (int l = 0; l <= o; ++l) {
        for (int mm = 0; mm <= o; ++mm) {
            for (int n = 0; n <= o; ++n) {
                if (l + mm + n > o) {
                    continue;
                }

                double sum = 0.0;
                for (int i = 0; i <= l; ++i) {
                    for (int j = 0; j <= mm; ++j) {
                        for (int k = 0; k <= n; ++k) {
                            double base = get_moment(m_child, i, j, k);
                            if (base == 0.0) {
                                continue;
                            }
                            int dl = l - i;
                            int dm = mm - j;
                            int dn = n - k;
                            double pow_val;
                            if (dl + dm + dn == 0) {
                                pow_val = 1.0;
                            } else {
                                double sx = (dl > 0) ? std::pow(shift.x, dl) : 1.0;
                                double sy = (dm > 0) ? std::pow(shift.y, dm) : 1.0;
                                double sz = (dn > 0) ? std::pow(shift.z, dn) : 1.0;
                                pow_val = sx * sy * sz;
                            }
                            double sign = ((dl + dm + dn) % 2 == 0) ? 1.0 : -1.0;
                            double coeff = sign * pow_val / (FACT[dl] * FACT[dm] * FACT[dn]);
                            sum += coeff * base;
                        }
                    }
                }

                set_moment(out, l, mm, n, sum);
            }
        }
    }

    return out;
}

} // namespace gravity
