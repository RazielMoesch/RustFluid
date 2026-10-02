//! Specular-reflection walls that remove normal velocity without tangential drag.

use super::Boundary3D;

/// Free-slip wall whose normal is the Y axis.
pub struct FreeSlipY;

impl Boundary3D for FreeSlipY {
    fn type_id(&self) -> u32 {
        4
    }

    fn name(&self) -> &'static str {
        "Free Slip Y"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(
            r#"
    // Specular reflection for top/bottom walls (flips Y, preserves X and Z).
    pulled_f = load_streamed_at(i32(x), i32(y), i32(z), REFLECT_Y[i]);
"#,
        )
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(
            r#"
    pulled_f = load_streamed_at(i32(x), i32(y), i32(z), REFLECT_Y[i]);
"#,
        )
    }
}

/// Free-slip wall whose normal is the X axis.
pub struct FreeSlipX;

impl Boundary3D for FreeSlipX {
    fn type_id(&self) -> u32 {
        6
    }

    fn name(&self) -> &'static str {
        "Free Slip X"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(
            r#"
    // Specular reflection for left/right walls (flips X, preserves Y and Z).
    pulled_f = load_streamed_at(i32(x), i32(y), i32(z), REFLECT_X[i]);
"#,
        )
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(
            r#"
    pulled_f = load_streamed_at(i32(x), i32(y), i32(z), REFLECT_X[i]);
"#,
        )
    }
}

/// Free-slip wall whose normal is the Z axis.
pub struct FreeSlipZ;

impl Boundary3D for FreeSlipZ {
    fn type_id(&self) -> u32 {
        7
    }

    fn name(&self) -> &'static str {
        "Free Slip Z"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(
            r#"
    // Specular reflection for front/back walls (flips Z, preserves X and Y).
    pulled_f = load_streamed_at(i32(x), i32(y), i32(z), REFLECT_Z[i]);
"#,
        )
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(
            r#"
    pulled_f = load_streamed_at(i32(x), i32(y), i32(z), REFLECT_Z[i]);
"#,
        )
    }
}
