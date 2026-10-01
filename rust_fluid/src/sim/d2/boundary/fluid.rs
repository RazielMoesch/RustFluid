use super::Boundary2D;

pub struct Fluid;

impl Boundary2D for Fluid {
    fn type_id(&self) -> u32 {
        0
    }

    fn name(&self) -> &'static str {
        "Fluid"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(r#"
    pulled_f = load_fa(neighbour_idx + i * TOTAL_CELLS);
"#)
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(r#"
    pulled_f = load_fa(neighbour_idx + OPP[i] * TOTAL_CELLS);
"#)
    }
}
