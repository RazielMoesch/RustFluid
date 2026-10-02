//! Three-dimensional discrete-velocity lattice definitions.

/// Standard nineteen-direction cubic lattice.
pub mod d3q19;

/// Describes lattice constants in WGSL and Rust forms for shader generation.
pub trait Lattice3D {
    fn name(&self) -> &'static str;
    fn q(&self) -> u32;
    fn ex(&self) -> &'static str; // Array string in WGSL
    fn ey(&self) -> &'static str;
    fn ez(&self) -> &'static str;
    fn weights(&self) -> &'static str;
    fn opp(&self) -> &'static str;
    fn reflect_x(&self) -> Option<&'static str> {
        None
    }
    fn reflect_y(&self) -> Option<&'static str> {
        None
    }
    fn reflect_z(&self) -> Option<&'static str> {
        None
    }

    // Arrays for Rust-side unrolling
    fn ex_array(&self) -> &[i32];
    fn ey_array(&self) -> &[i32];
    fn ez_array(&self) -> &[i32];
    fn opp_array(&self) -> &[u32];

    // Constants definition string for WGSL
    fn wgsl_constants(&self) -> String {
        format!(
            "const Q: u32 = {}u;\n{}\n{}\n{}\n{}\n{}",
            self.q(),
            self.ex(),
            self.ey(),
            self.ez(),
            self.weights(),
            self.opp()
        )
    }
}
