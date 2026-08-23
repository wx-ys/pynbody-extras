#pragma once
#include <cstddef>

namespace gravity {

// Derivatives of the softened 1/r potential, full and compact variants.
// Each variant stores only the terms needed for a given multipole order.
// All structs are zero-initialized when value-initialized (`PotentialDerivativesX{}`).

// ---- Order-1 derivatives (enough for monopole) ----

struct PotentialDerivatives1 {
    double d000;
    double d100;
    double d010;
    double d001;

    static PotentialDerivatives1 new_derivatives(double dx, double dy, double dz, double eps2);
};

// ---- Order-2 derivatives ----

struct PotentialDerivatives2 {
    double d000;
    double d100;
    double d010;
    double d001;
    double d200;
    double d020;
    double d002;
    double d110;
    double d101;
    double d011;

    static PotentialDerivatives2 new_derivatives(double dx, double dy, double dz, double eps2);
};

// ---- Order-3 derivatives ----

struct PotentialDerivatives3 {
    double d000;
    double d100;
    double d010;
    double d001;
    double d200;
    double d020;
    double d002;
    double d110;
    double d101;
    double d011;
    double d300;
    double d030;
    double d003;
    double d210;
    double d201;
    double d120;
    double d102;
    double d021;
    double d012;
    double d111;

    static PotentialDerivatives3 new_derivatives(double dx, double dy, double dz, double eps2);
};

// ---- Order-4 derivatives ----

struct PotentialDerivatives4 {
    double d000;
    double d100;
    double d010;
    double d001;
    double d200;
    double d020;
    double d002;
    double d110;
    double d101;
    double d011;
    double d300;
    double d030;
    double d003;
    double d210;
    double d201;
    double d120;
    double d102;
    double d021;
    double d012;
    double d111;
    double d400;
    double d040;
    double d004;
    double d310;
    double d301;
    double d130;
    double d103;
    double d031;
    double d013;
    double d220;
    double d202;
    double d022;
    double d211;
    double d121;
    double d112;

    static PotentialDerivatives4 new_derivatives(double dx, double dy, double dz, double eps2);
};

// ---- Full order-5 derivatives ----

struct PotentialDerivatives {
    double d000;
    double d100;
    double d010;
    double d001;
    double d200;
    double d020;
    double d002;
    double d110;
    double d101;
    double d011;
    double d300;
    double d030;
    double d003;
    double d210;
    double d201;
    double d120;
    double d102;
    double d021;
    double d012;
    double d111;
    double d400;
    double d040;
    double d004;
    double d310;
    double d301;
    double d130;
    double d103;
    double d031;
    double d013;
    double d220;
    double d202;
    double d022;
    double d211;
    double d121;
    double d112;
    double d500;
    double d050;
    double d005;
    double d410;
    double d401;
    double d140;
    double d104;
    double d041;
    double d014;
    double d320;
    double d302;
    double d230;
    double d203;
    double d032;
    double d023;
    double d221;
    double d212;
    double d122;
    double d311;
    double d131;
    double d113;

    static PotentialDerivatives new_derivatives(double dx, double dy, double dz, double eps2, int order);
};

} // namespace gravity
