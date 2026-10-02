//! Two-dimensional collision operators injected into generated WGSL.

/// Single-relaxation-time BGK collision.
pub mod bgk;
/// Multiple-relaxation-time collision in moment space.
pub mod mrt;

/// Provides the local WGSL population update for a 2D solver.
pub trait Collision2D {
    fn name(&self) -> &'static str;
    fn wgsl(&self) -> &'static str;
}
