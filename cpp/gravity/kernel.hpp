#pragma once
namespace gravity {
enum class KernelKind { Plummer = 0, CubicSplineW2 = 1 };
double multipole_min_separation_factor(KernelKind k);
bool multipole_soft_ok(KernelKind k, double r, double h);
double kernel_potential_per_unit_mass(KernelKind k, double r, double h);
double kernel_accel_factor(KernelKind k, double r, double h);
} // namespace gravity
