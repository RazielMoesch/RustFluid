//! No-slip halfway bounce-back for three-dimensional solid cells.

use super::Boundary3D;

/// Reflects incoming populations into their opposite directions.
pub struct BounceBack;

impl Boundary3D for BounceBack {
    fn type_id(&self) -> u32 {
        1
    }

    fn name(&self) -> &'static str {
        "Bounce Back"
    }

    fn pull_even(&self) -> Option<&'static str> {
        None
    }

    fn pull_odd(&self) -> Option<&'static str> {
        None
    }
}
