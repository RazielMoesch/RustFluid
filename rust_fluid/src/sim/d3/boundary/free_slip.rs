use super::Boundary3D;

pub struct FreeSlipY;

impl Boundary3D for FreeSlipY {
    fn type_id(&self) -> u32 {
        4
    }

    fn name(&self) -> &'static str {
        "Free Slip Y"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(r#"
    // Specular reflection for top/bottom walls (Flips Y, preserves X, Z)
    pulled_f = load_fa(cell_idx + REFLECT_Y[i] * TOTAL_CELLS);
"#)
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(r#"
    // Inverted memory read for the specular reflection
    pulled_f = load_fa(cell_idx + OPP[REFLECT_Y[i]] * TOTAL_CELLS);
"#)
    }
}

pub struct FreeSlipX;

impl Boundary3D for FreeSlipX {
    fn type_id(&self) -> u32 {
        6
    }

    fn name(&self) -> &'static str {
        "Free Slip X"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(r#"
    // Specular reflection for left/right walls (Flips X, preserves Y, Z)
    pulled_f = load_fa(cell_idx + REFLECT_X[i] * TOTAL_CELLS);
"#)
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(r#"
    // Inverted memory read for the specular reflection
    pulled_f = load_fa(cell_idx + OPP[REFLECT_X[i]] * TOTAL_CELLS);
"#)
    }
}

pub struct FreeSlipZ;

impl Boundary3D for FreeSlipZ {
    fn type_id(&self) -> u32 {
        7
    }

    fn name(&self) -> &'static str {
        "Free Slip Z"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(r#"
    // Specular reflection for front/back walls (Flips Z, preserves X, Y)
    pulled_f = load_fa(cell_idx + REFLECT_Z[i] * TOTAL_CELLS);
"#)
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(r#"
    // Inverted memory read for the specular reflection
    pulled_f = load_fa(cell_idx + OPP[REFLECT_Z[i]] * TOTAL_CELLS);
"#)
    }
}
