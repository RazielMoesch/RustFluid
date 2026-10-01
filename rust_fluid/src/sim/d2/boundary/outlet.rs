use super::Boundary2D;

pub struct ZeroGradientOutlet;

impl Boundary2D for ZeroGradientOutlet {
    fn type_id(&self) -> u32 {
        3
    }

    fn name(&self) -> &'static str {
        "Zero Gradient Outlet"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(r#"
    // Pull from our own cell in the same direction (zero gradient extrapolation)
    pulled_f = load_fa(cell_idx + i * TOTAL_CELLS);
"#)
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(r#"
    // Pull from our own cell in the same direction
    pulled_f = load_fa(cell_idx + OPP[i] * TOTAL_CELLS);
"#)
    }
}
