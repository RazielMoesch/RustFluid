//! GPU-accelerated lattice Boltzmann solvers, runtimes, diagnostics, and renderers.

/// Benchmark statistics and workgroup-sweep helpers.
pub mod benchmark;
/// Orbit and pan cameras used by the visualizers.
pub mod camera;
/// Repeatable GPU correctness and stability cases.
pub mod diagnostics;
/// Complete headless and interactive simulation configurations.
pub mod examples;
/// Adapter discovery and common `wgpu` resource helpers.
pub mod gpu;
/// Two- and three-dimensional visualization pipelines.
pub mod render;
/// Windowed and headless execution orchestration.
pub mod runtime;
/// Domain construction and SVG/STL geometry loading.
pub mod setup;
/// D2Q9 and D3Q19 solver implementations.
pub mod sim;
/// CPU-side field, mass, and stability comparisons.
pub mod validation;
