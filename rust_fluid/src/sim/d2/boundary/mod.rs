//! Two-dimensional boundary components selected by encoded cell flags.

/// No-slip halfway bounce-back wall.
pub mod bounce_back;
/// Inlet reconstructed from a configured equilibrium state.
pub mod equilibrium_inlet;
/// Ordinary fluid-cell streaming.
pub mod fluid;
/// Axis-aligned specular reflection walls.
pub mod free_slip;
/// Zero-gradient outlet reconstruction.
pub mod outlet;
/// Left-face Zou-He velocity boundary.
pub mod zou_he;

/// Supplies a flag ID and WGSL fragments for one 2D cell-boundary type.
pub trait Boundary2D {
    fn type_id(&self) -> u32;
    fn name(&self) -> &'static str;
    fn pull_even(&self) -> Option<&'static str> {
        None
    }
    fn pull_odd(&self) -> Option<&'static str> {
        None
    }
    fn post_streaming(&self) -> Option<&'static str> {
        None
    }
}
