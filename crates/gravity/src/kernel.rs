// ...existing code...
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KernelKind {
    /// Plummer softening: phi = -1/sqrt(r^2 + eps^2)
    #[default]
    Plummer,
    /// Springel-style cubic spline gravitational softening (W2 kernel), eq.(71).
    /// Potential per unit mass: phi(r;h) = (1/h) * W2(r/h)
    /// where W2(u>=1) = -1/u -> phi = -1/r.
    CubicSplineW2,
}

impl KernelKind {
    /// Minimal separation requirement for using a node-level (BH/multipole) approximation
    /// when particle softening is present.
    ///
    /// This is expressed as a factor `c` such that the approximation is allowed only if
    /// `r > c * h`, where `h` is a conservative softening scale for the node (typically hmax).
    #[inline]
    pub fn multipole_min_separation_factor(self) -> f64 {
        match self {
            // Plummer needs to be well outside the softening scale.
            KernelKind::Plummer => 2.8,
            // spline becomes Newtonian outside ~h.
            KernelKind::CubicSplineW2 => 1.0,
        }
    }

    /// Returns true if it is safe to approximate a softened node at distance `r`
    /// with a single multipole/BH interaction.
    #[inline]
    pub fn multipole_soft_ok(self, r: f64, h: f64) -> bool {
        if h <= 0.0 {
            return true;
        }
        r > self.multipole_min_separation_factor() * h
    }
}

#[inline]
pub fn kernel_potential_per_unit_mass(kind: KernelKind, r: f64, h: f64) -> f64 {
    if r == 0.0 {
        return 0.0;
    }
    match kind {
        KernelKind::Plummer => -1.0 / (r * r + h * h).sqrt(),
        KernelKind::CubicSplineW2 => {
            if h <= 0.0 {
                return -1.0 / r;
            }
            let h_inv = 1.0 / h;
            let u = r * h_inv;
            w2(u) * h_inv
        }
    }
}

/// Returns accel factor g(r;h) such that:
/// a_vec = m * r_vec * g
/// where r_vec = (source - target), r = |r_vec|.
#[inline]
pub fn kernel_accel_factor(kind: KernelKind, r: f64, h: f64) -> f64 {
    if r == 0.0 {
        return 0.0;
    }
    match kind {
        KernelKind::Plummer => {
            let s2 = r * r + h * h;
            1.0 / (s2.sqrt() * s2) // 1/(r^2+eps^2)^(3/2)
        }
        KernelKind::CubicSplineW2 => {
            if h <= 0.0 {
                return 1.0 / (r * r * r);
            }
            let h_inv = 1.0 / h;
            let u = r * h_inv;
            // phi = W2(u)/h => K'(r) = W2'(u)/h^2
            // g = K'(r)/r
            w2_prime(u) * (h_inv * h_inv) / r
        }
    }
}

#[inline]
fn w2(u: f64) -> f64 {
    // Springel et al. W2(u), eq.(71)
    if u < 0.5 {
        // 16/3 u^2 - 48/5 u^4 + 32/5 u^5 - 14/5
        let u2 = u * u;
        let u4 = u2 * u2;
        let u5 = u4 * u;
        (16.0 / 3.0) * u2 - (48.0 / 5.0) * u4 + (32.0 / 5.0) * u5 - 14.0 / 5.0
    } else if u < 1.0 {
        // 1/(15u) + 32/3 u^2 - 16 u^3 + 48/5 u^4 - 32/15 u^5 - 16/5
        let inv_u = 1.0 / u;
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        let u5 = u4 * u;
        (1.0 / 15.0) * inv_u + (32.0 / 3.0) * u2 - 16.0 * u3 + (48.0 / 5.0) * u4
            - (32.0 / 15.0) * u5
            - 16.0 / 5.0
    } else {
        -1.0 / u
    }
}

#[inline]
fn w2_prime(u: f64) -> f64 {
    // d/du W2(u)
    if u < 0.5 {
        // = 32/3 u - 192/5 u^3 + 32 u^4
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        (32.0 / 3.0) * u - (192.0 / 5.0) * u3 + 32.0 * u4
    } else if u < 1.0 {
        // = -1/(15 u^2) + 64/3 u - 48 u^2 + 192/5 u^3 - 32/3 u^4
        let u2 = u * u;
        let u3 = u2 * u;
        let u4 = u2 * u2;
        -(1.0 / 15.0) * (1.0 / u2) + (64.0 / 3.0) * u - 48.0 * u2 + (192.0 / 5.0) * u3
            - (32.0 / 3.0) * u4
    } else {
        // d/du[-1/u] = 1/u^2
        1.0 / (u * u)
    }
}
