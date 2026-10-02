//! Structured GPU diagnostics for solver invariants and flow behavior.

/// Individual diagnostic scenarios.
pub mod cases;
/// Macroscopic field reductions used by diagnostics.
pub mod metrics;
/// Pass, warning, and failure records.
pub mod result;
/// Shared adapter context and result collection.
pub mod runner;
