//! Three-dimensional collision operators injected into generated WGSL.

/// Single-relaxation-time BGK collision.
pub mod bgk;
/// Multiple-relaxation-time collision in moment space.
pub mod mrt;
/// BGK collision with local Smagorinsky sub-grid viscosity.
pub mod smagorinsky;
/// Two-relaxation-time collision.
pub mod trt;

/// Provides the local WGSL population update for a 3D solver.
pub trait Collision3D {
    fn name(&self) -> &'static str;
    fn wgsl(&self) -> &'static str;
}
