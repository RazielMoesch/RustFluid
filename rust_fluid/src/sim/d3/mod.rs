//! D3Q19 solver components and GPU orchestration.

/// Cell-boundary shader fragments.
pub mod boundary;
/// GPU buffers and byte accounting.
pub mod buffers;
/// Local collision operators.
pub mod collision;
/// Domain, forcing, periodicity, and sponge configuration.
pub mod config;
/// Reserved namespace for force models.
pub mod force;
/// Discrete velocity lattices.
pub mod lattice;
/// WGSL templates and source assembly.
pub mod shader;
/// The D3Q19 `wgpu` solver.
pub mod solver;
