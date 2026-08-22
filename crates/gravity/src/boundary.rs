/// Boundary condition for gravity computations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// Isolated / vacuum boundary — no periodic images.
    Vacuum,
    /// Periodic box. The box runs from 0 to `box_size` in each dimension.
    Periodic { box_size: [f64; 3] },
}

impl Default for BoundaryCondition {
    fn default() -> Self {
        Self::Vacuum
    }
}

impl BoundaryCondition {
    /// Compute the minimum-image displacement vector and its squared length
    /// from `p1` to `p2`, respecting the boundary condition.
    ///
    /// For `Vacuum`, this is simply `p2 - p1`.
    /// For `Periodic`, each component is folded into `[-L/2, L/2]`.
    #[inline]
    pub fn displacement(&self, p1: &[f64; 3], p2: &[f64; 3]) -> ([f64; 3], f64) {
        match self {
            Self::Vacuum => {
                let dx = p2[0] - p1[0];
                let dy = p2[1] - p1[1];
                let dz = p2[2] - p1[2];
                let r2 = dx * dx + dy * dy + dz * dz;
                ([dx, dy, dz], r2)
            }
            Self::Periodic { box_size } => {
                let mut dx = p2[0] - p1[0];
                let mut dy = p2[1] - p1[1];
                let mut dz = p2[2] - p1[2];

                let hx = box_size[0] * 0.5;
                let hy = box_size[1] * 0.5;
                let hz = box_size[2] * 0.5;

                if dx > hx {
                    dx -= box_size[0];
                } else if dx < -hx {
                    dx += box_size[0];
                }
                if dy > hy {
                    dy -= box_size[1];
                } else if dy < -hy {
                    dy += box_size[1];
                }
                if dz > hz {
                    dz -= box_size[2];
                } else if dz < -hz {
                    dz += box_size[2];
                }

                let r2 = dx * dx + dy * dy + dz * dz;
                ([dx, dy, dz], r2)
            }
        }
    }

    /// Returns true if this is a periodic boundary.
    #[inline]
    pub fn is_periodic(&self) -> bool {
        matches!(self, Self::Periodic { .. })
    }
}
