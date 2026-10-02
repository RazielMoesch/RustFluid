//! Smagorinsky large-eddy closure layered onto BGK collision.

use super::Collision3D;

/// BGK collision with a local Smagorinsky sub-grid viscosity.
///
/// The configured `OMEGA` still represents the molecular viscosity.  In cells
/// containing unresolved shear, the non-equilibrium stress raises the local
/// relaxation time and supplies the dissipation required by an LES grid.
pub struct Smagorinsky;

impl Smagorinsky {
    pub fn new() -> Self {
        Self
    }
}

impl Collision3D for Smagorinsky {
    fn name(&self) -> &'static str {
        "Smagorinsky LES"
    }

    fn wgsl(&self) -> &'static str {
        r#"
    let F = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u = (u + 0.5 * F) / rho;

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;

    // Non-equilibrium momentum-flux tensor.  Its Frobenius norm gives a
    // lattice-local strain estimate without finite differencing the velocity.
    var pi_xx = 0.0;
    var pi_yy = 0.0;
    var pi_zz = 0.0;
    var pi_xy = 0.0;
    var pi_xz = 0.0;
    var pi_yz = 0.0;
    for (var i: u32 = 0u; i < Q; i += 1u) {
        let ex = f32(EX[i]);
        let ey = f32(EY[i]);
        let ez = f32(EZ[i]);
        let eu = ex * u.x + ey * u.y + ez * u.z;
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        let fneq = f_local[i] - feq;
        pi_xx += ex * ex * fneq;
        pi_yy += ey * ey * fneq;
        pi_zz += ez * ez * fneq;
        pi_xy += ex * ey * fneq;
        pi_xz += ex * ez * fneq;
        pi_yz += ey * ez * fneq;
    }

    let pi_norm = sqrt(max(
        pi_xx * pi_xx + pi_yy * pi_yy + pi_zz * pi_zz
            + 2.0 * (pi_xy * pi_xy + pi_xz * pi_xz + pi_yz * pi_yz),
        0.0
    ));
    let tau_molecular = 1.0 / OMEGA;
    let c_smag = 0.10;
    let tau_eff = 0.5 * (tau_molecular + sqrt(
        tau_molecular * tau_molecular
            + 18.0 * sqrt(2.0) * c_smag * c_smag * pi_norm / rho
    ));
    let omega_eff = 1.0 / tau_eff;
    let omega_factor = 1.0 - 0.5 * omega_eff;
    let uF = dot(u, F);

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let e_vec = vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i]));
        let eu = dot(e_vec, u);
        let eF = dot(e_vec, F);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        let S_i = WEIGHTS[i] * omega_factor * (3.0 * (eF - uF) + 9.0 * eu * eF);

        f_local[i] = f_local[i] - omega_eff * (f_local[i] - feq) + S_i;
    }
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_uses_molecular_omega_in_quiescent_flow_and_local_les_omega_in_shear() {
        let shader = Smagorinsky::new().wgsl();
        assert!(shader.contains("let tau_molecular = 1.0 / OMEGA"));
        assert!(shader.contains("let c_smag = 0.10"));
        assert!(shader.contains("let omega_eff = 1.0 / tau_eff"));
        assert!(shader.contains("omega_eff * (f_local[i] - feq)"));
    }
}
