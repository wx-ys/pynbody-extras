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
inline bool timing_enabled() {
    // Function-local static, initialized exactly once (mirrors Rust's OnceLock).
    static const bool enabled = [] {
        const char* v = std::getenv("GRAVITY_TIMING");
        if (v == nullptr) return false;                       // unset -> disabled
        // Trim leading/trailing ASCII whitespace (mirrors Rust's str::trim()).
        const char* s = v;
        const char* e = s;
        while (*e) ++e;
        while (s < e && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r' || *s == '\f' || *s == '\v')) ++s;
        while (e > s && (*(e - 1) == ' ' || *(e - 1) == '\t' || *(e - 1) == '\n' || *(e - 1) == '\r' || *(e - 1) == '\f' || *(e - 1) == '\v')) --e;
        const std::size_t n = static_cast<std::size_t>(e - s);
        if (n == 0) return false;                             // empty -> disabled
        if (n == 1 && *s == '0') return false;                // "0" -> disabled
        if (n == 5 &&
            (s[0] == 'f' || s[0] == 'F') && (s[1] == 'a' || s[1] == 'A') &&
            (s[2] == 'l' || s[2] == 'L') && (s[3] == 's' || s[3] == 'S') &&
            (s[4] == 'e' || s[4] == 'E')) return false;       // "false" (case-insensitive) -> disabled
        return true;
    }();
    return enabled;
}
inline void log_timing(const char* label, double dt_ms) {
    std::fprintf(stderr, "[gravity-timing] %s: %.3f ms\n", label, dt_ms);
}
} // namespace gravity
