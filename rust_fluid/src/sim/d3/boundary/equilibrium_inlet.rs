//! Three-dimensional inlet populated from a configured equilibrium state.

use super::Boundary3D;

/// Reconstructs all populations from target velocity and density.
pub struct EquilibriumInlet;

impl Boundary3D for EquilibriumInlet {
    fn type_id(&self) -> u32 {
        2
    }

    fn name(&self) -> &'static str {
        "Equilibrium Inlet"
    }

    fn pull_even(&self) -> Option<&'static str> {
        Some(
            r#"
    let cfg = boundary_configs[cfg_id];
    let eu = (f32(ex) * cfg.vel.x) + (f32(ey) * cfg.vel.y) + (f32(ez) * cfg.vel.z);
    let u_sq = dot(cfg.vel, cfg.vel);
    pulled_f = WEIGHTS[i] * cfg.density * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * u_sq);
"#,
        )
    }

    fn pull_odd(&self) -> Option<&'static str> {
        Some(
            r#"
    let cfg = boundary_configs[cfg_id];
    let eu = (f32(ex) * cfg.vel.x) + (f32(ey) * cfg.vel.y) + (f32(ez) * cfg.vel.z);
    let u_sq = dot(cfg.vel, cfg.vel);
    pulled_f = WEIGHTS[i] * cfg.density * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - 1.5 * u_sq);
"#,
        )
    }
}
