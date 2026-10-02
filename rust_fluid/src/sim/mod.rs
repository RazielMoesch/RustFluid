//! Dimension-specific LBM solvers and their shared support code.

/// Precision and output helpers shared by both dimensions.
pub mod common;
/// Two-dimensional D2Q9 implementation.
pub mod d2;
/// Three-dimensional D3Q19 implementation.
pub mod d3;
