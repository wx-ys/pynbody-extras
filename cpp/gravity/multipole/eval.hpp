#pragma once

#include <cstddef>
#include <vector>

#include "gravity/vec3.hpp"
#include "gravity/multipole/moment.hpp"
#include "gravity/multipole/derivatives.hpp"

namespace gravity {

// Primary template (intentionally undefined). Only the full specializations for
// orders 0, 2, 3, 4, 5 are provided; instantiating any other order is an error.
template<int Order>
struct MultipoleEval;

// ------- Order 0 (also used for order 1 — dipole vanishes about COM) -------

template<>
struct MultipoleEval<0> {
    using Moment = Moment0;
    using Derivatives = PotentialDerivatives1;

    static Moment from_points(const std::vector<Vec3>& positions,
                              const double* masses,
                              const std::vector<size_t>& indices,
                              const Vec3& center);
    static Moment translate(const Moment& m, const Vec3& shift);
    static void add_assign(Moment& acc, const Moment& other);
    static double potential(const Moment& m, const Derivatives& d);
    static Vec3 acceleration(const Moment& m, const Derivatives& d);
    static Derivatives derivatives(double dx, double dy, double dz, double eps2);
};

// ------- Order 2 -------

template<>
struct MultipoleEval<2> {
    using Moment = Moment2;
    using Derivatives = PotentialDerivatives2;

    static Moment from_points(const std::vector<Vec3>& positions,
                              const double* masses,
                              const std::vector<size_t>& indices,
                              const Vec3& center);
    static Moment translate(const Moment& m, const Vec3& shift);
    static void add_assign(Moment& acc, const Moment& other);
    static double potential(const Moment& m, const Derivatives& d);
    static Vec3 acceleration(const Moment& m, const Derivatives& d);
    static Derivatives derivatives(double dx, double dy, double dz, double eps2);
};

// ------- Order 3 -------

template<>
struct MultipoleEval<3> {
    using Moment = Moment3;
    using Derivatives = PotentialDerivatives3;

    static Moment from_points(const std::vector<Vec3>& positions,
                              const double* masses,
                              const std::vector<size_t>& indices,
                              const Vec3& center);
    static Moment translate(const Moment& m, const Vec3& shift);
    static void add_assign(Moment& acc, const Moment& other);
    static double potential(const Moment& m, const Derivatives& d);
    static Vec3 acceleration(const Moment& m, const Derivatives& d);
    static Derivatives derivatives(double dx, double dy, double dz, double eps2);
};

// ------- Order 4 -------

template<>
struct MultipoleEval<4> {
    using Moment = Moment4;
    using Derivatives = PotentialDerivatives4;

    static Moment from_points(const std::vector<Vec3>& positions,
                              const double* masses,
                              const std::vector<size_t>& indices,
                              const Vec3& center);
    static Moment translate(const Moment& m, const Vec3& shift);
    static void add_assign(Moment& acc, const Moment& other);
    static double potential(const Moment& m, const Derivatives& d);
    static Vec3 acceleration(const Moment& m, const Derivatives& d);
    static Derivatives derivatives(double dx, double dy, double dz, double eps2);
};

// ------- Order 5 -------

template<>
struct MultipoleEval<5> {
    using Moment = Moment5;
    using Derivatives = PotentialDerivatives;

    static Moment from_points(const std::vector<Vec3>& positions,
                              const double* masses,
                              const std::vector<size_t>& indices,
                              const Vec3& center);
    static Moment translate(const Moment& m, const Vec3& shift);
    static void add_assign(Moment& acc, const Moment& other);
    static double potential(const Moment& m, const Derivatives& d);
    static Vec3 acceleration(const Moment& m, const Derivatives& d);
    static Derivatives derivatives(double dx, double dy, double dz, double eps2);
};

// Runtime-dispatch evaluators (useful for tests and interactive use).
double gravity_potential_multipole(const MultipoleMoment& m, const PotentialDerivatives& d,
                                   unsigned char order);
Vec3 gravity_accel_multipole(const MultipoleMoment& m, const PotentialDerivatives& d,
                             unsigned char order);

} // namespace gravity
