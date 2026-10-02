//! D2Q9 solver components and GPU orchestration.

/// Cell-boundary shader fragments.
pub mod boundary;
/// GPU population and macroscopic buffers.
pub mod buffers;
/// Local collision operators.
pub mod collision;
/// Domain and dispatch configuration.
pub mod config;
/// Reserved namespace for force models.
pub mod force;
/// Discrete velocity lattices.
pub mod lattice;
/// WGSL templates and source assembly.
pub mod shader;
/// The D2Q9 `wgpu` solver.
pub mod solver;
