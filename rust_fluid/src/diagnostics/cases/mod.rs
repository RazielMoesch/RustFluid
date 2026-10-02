//! Solver scenarios used by the diagnostic runner.

/// Boundary-specific checks.
pub mod boundary;
/// Channel-flow checks.
pub mod channel;
/// Closed-box conservation checks.
pub mod closed_box;
/// Alternating-access phase checks.
pub mod parity;
/// Renderer-facing data checks.
pub mod renderer;
/// Taylor-Green vortex decay checks.
pub mod taylor_green;
/// Uniform-equilibrium preservation checks.
pub mod uniform;
/// Wake outlet and sponge-layer checks.
pub mod wake;
