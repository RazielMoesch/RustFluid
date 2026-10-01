use super::Boundary3D;

pub struct BounceBack;

impl Boundary3D for BounceBack {
    fn type_id(&self) -> u32 {
        1
    }

    fn name(&self) -> &'static str {
        "Bounce Back"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(r#"
    // Pull from our own cell's opposite direction 
    pulled_f = load_fa(cell_idx + OPP[i] * TOTAL_CELLS);
"#)
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(r#"
    // Pull from our own cell's opposite direction (stored non-inverted in this step)
    pulled_f = load_fa(cell_idx + i * TOTAL_CELLS);
"#)
    }
}
