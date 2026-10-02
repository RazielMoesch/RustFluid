//! Two-dimensional zero-gradient outlet reconstruction.

use super::Boundary2D;

/// Copies the adjacent interior state into an outlet cell.
pub struct ZeroGradientOutlet;

impl Boundary2D for ZeroGradientOutlet {
    fn type_id(&self) -> u32 {
        3
    }

    fn name(&self) -> &'static str {
        "Zero Gradient Outlet"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(
            r#"
    // Zero-gradient (Neumann) extrapolation from the adjacent outlet-plane
    // cell.  The 2D step shader switches on the *neighbour's* boundary type,
    // so `neighbour_idx` is the outlet cell and `cell_idx` is the fluid cell
    // being streamed.  Reading `cell_idx` here would copy the fluid cell onto
    // itself and freeze the outflow populations.
    pulled_f = load_fa(neighbour_idx + i * TOTAL_CELLS);
"#,
        )
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(
            r#"
    // Odd phase reads the direction-inverted slot (even writes OPP[i]).
    pulled_f = load_fa(neighbour_idx + OPP[i] * TOTAL_CELLS);
"#,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ZeroGradientOutlet;
    use crate::sim::d2::boundary::Boundary2D;

    #[test]
    fn zero_gradient_reads_the_outlet_neighbour_not_the_fluid_cell() {
        let even = ZeroGradientOutlet.pull_even().expect("even pull");
        let odd = ZeroGradientOutlet.pull_odd().expect("odd pull");

        for pull in [even, odd] {
            // The 2D step shader switches on the neighbour's boundary type, so
            // `neighbour_idx` is the outlet cell.  Using `cell_idx` here would
            // copy the fluid cell onto itself and freeze the outflow.
            assert!(pull.contains("neighbour_idx"));
            // Regression guard: copying the fluid cell onto itself would freeze
            // the outflow populations.
            assert!(!pull.contains("load_fa(cell_idx"));
        }

        assert!(even.contains("neighbour_idx + i * TOTAL_CELLS"));
        assert!(odd.contains("neighbour_idx + OPP[i] * TOTAL_CELLS"));
    }
}
