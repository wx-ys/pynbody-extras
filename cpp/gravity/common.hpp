#pragma once
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace gravity {
constexpr double R2_TINY = std::numeric_limits<double>::min();
constexpr double MIN_SOFTENING = 0.0;
constexpr size_t NO_INDEX = std::numeric_limits<size_t>::max();

inline double inv_r_from_r2(double r2) { return 1.0 / std::sqrt(r2 + R2_TINY); }

inline void inv_r_and_inv_r3_from_r2(double r2, double& inv_r, double& inv_r3) {
    inv_r = inv_r_from_r2(r2);
    inv_r3 = inv_r * inv_r * inv_r;
}

// Port of octree.rs timing_enabled()/log_timing(): prints "[gravity-timing] ..."
// to stderr only when GRAVITY_TIMING env var is truthy. Use a function-local
// static bool initialized once from std::getenv.
inline bool timing_enabled();
inline void log_timing(const char* label, double dt_ms);
} // namespace gravity
