//! Normal three-dimensional fluid-cell streaming.

use super::Boundary3D;

/// Marks cells that use unmodified pull streaming and collision.
pub struct Fluid;

impl Boundary3D for Fluid {
    fn type_id(&self) -> u32 {
        0
    }

    fn name(&self) -> &'static str {
        "Fluid"
    }

    fn pull_even(&self) -> Option<&'static str> {
        None
    }

    fn pull_odd(&self) -> Option<&'static str> {
        None
    }
}
