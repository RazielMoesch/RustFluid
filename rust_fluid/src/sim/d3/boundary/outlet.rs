//! High-X zero-gradient outlet reconstruction for D3Q19.

use super::Boundary3D;

/// Right-face zero-gradient outlet.
///
/// Its populations are extrapolated by the solver's separate outlet pass.
/// Keeping that copy out of the collide/stream dispatch prevents a neighbor
/// from reading an address while its owning invocation overwrites it.
/// Copies the adjacent interior state into an outlet cell.
pub struct ZeroGradientOutlet;

impl Boundary3D for ZeroGradientOutlet {
    fn type_id(&self) -> u32 {
        3
    }

    fn name(&self) -> &'static str {
        "Zero Gradient Outlet"
    }
}

#[cfg(test)]
mod tests {
    use super::ZeroGradientOutlet;
    use crate::sim::d3::boundary::Boundary3D;

    #[test]
    fn outlet_does_not_inject_cross_cell_reads_into_the_main_step() {
        assert!(ZeroGradientOutlet.pull_even().is_none());
        assert!(ZeroGradientOutlet.pull_odd().is_none());
        assert!(ZeroGradientOutlet.post_streaming().is_none());
    }
}
