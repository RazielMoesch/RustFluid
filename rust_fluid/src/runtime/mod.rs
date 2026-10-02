//! Generic execution layers shared by simulations and renderers.

/// Borrowed GPU contexts passed to user factories.
pub mod context;
/// Window creation, event handling, and frame submission.
pub mod graphics;
/// Surface-free simulation execution.
pub mod headless;
/// Renderer lifecycle contract.
pub mod renderer;
/// Solver lifecycle contract.
pub mod simulation;

pub use context::{GraphicsContext, HeadlessContext};
pub use graphics::{Graphics, GraphicsBuilder, GraphicsConfig};
pub use headless::{Headless, HeadlessBuilder, HeadlessConfig, HeadlessResult};
pub use renderer::Renderer;
pub use simulation::Simulation;
