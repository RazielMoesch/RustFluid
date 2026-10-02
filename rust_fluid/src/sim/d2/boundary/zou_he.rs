//! Zou-He velocity reconstruction on the minimum-X face.

use super::Boundary2D;
use crate::sim::d2::boundary::fluid::Fluid;

/// Prescribes left-face velocity while recovering density locally.
pub struct ZouHeLeftVelocity;

impl Boundary2D for ZouHeLeftVelocity {
    fn type_id(&self) -> u32 {
        5
    }

    fn name(&self) -> &'static str {
        "Zou-He Left Velocity"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Fluid.pull_even()
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Fluid.pull_odd()
    }

    fn post_streaming(&self) -> Option<&'static str> {
        Some(
            r#"
    let my_type = my_flag >> FLAG_TYPE_SHIFT;
    
    if (my_type == 5u) {
        let cfg_id = my_flag & FLAG_ID_MASK;
        let cfg = boundary_configs[cfg_id];
        let ux = cfg.vel.x;
        let uy = cfg.vel.y;
        
        // Compute density at the left wall
        let rho_in = (f_local[0] + f_local[2] + f_local[4] + 2.0 * (f_local[3] + f_local[6] + f_local[7])) / (1.0 - ux);
        
        // Solve for the unknown incoming populations
        f_local[1] = f_local[3] + (2.0 / 3.0) * rho_in * ux;
        
        let diff_2_4 = f_local[2] - f_local[4];
        f_local[5] = f_local[7] - 0.5 * diff_2_4 + (1.0 / 6.0) * rho_in * ux + 0.5 * rho_in * uy;
        f_local[8] = f_local[6] + 0.5 * diff_2_4 + (1.0 / 6.0) * rho_in * ux - 0.5 * rho_in * uy;
        
        // Update macroscopics for the collision step
        rho = rho_in;
        u = cfg.vel * rho_in;
    }
"#,
        )
    }
}
