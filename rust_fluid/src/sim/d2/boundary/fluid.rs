//! Normal two-dimensional fluid-cell streaming.

use super::Boundary2D;

/// Marks cells that use unmodified pull streaming and collision.
pub struct Fluid;

impl Boundary2D for Fluid {
    fn type_id(&self) -> u32 {
        0
    }

    fn name(&self) -> &'static str {
        "Fluid"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(
            r#"
    pulled_f = load_fa(neighbour_idx + i * TOTAL_CELLS);
"#,
        )
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(
            r#"
    pulled_f = load_fa(neighbour_idx + OPP[i] * TOTAL_CELLS);
"#,
        )
    }
}
