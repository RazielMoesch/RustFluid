//! Single-relaxation-time BGK collision with Guo body forcing.

use super::Collision3D;

/// Relaxes every D3Q19 population with the configured `OMEGA`.
pub struct Bgk;

impl Bgk {
    pub fn new() -> Self {
        Self
    }
}

impl Collision3D for Bgk {
    fn name(&self) -> &'static str {
        "BGK"
    }

    fn wgsl(&self) -> &'static str {
        r#"
    let F = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u = (u + 0.5 * F) / rho;

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;
    let omega_factor = 1.0 - 0.5 * OMEGA;
    let uF = dot(u, F);

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let e_vec = vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i]));
        let eu = dot(e_vec, u);
        let eF = dot(e_vec, F);

        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        let S_i = WEIGHTS[i] * omega_factor * (3.0 * (eF - uF) + 9.0 * eu * eF);

        f_local[i] = f_local[i] - OMEGA * (f_local[i] - feq) + S_i;
    }
"#
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn guo_force_has_zero_mass_and_requested_first_moment() {
        let ex = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
        let ey = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1];
        let ez = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1];
        let w = [
            1.0 / 3.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ];
        let u = [0.0577f64, -0.003, 0.002];
        let force = [1.8496058e-7f64, -2.0e-8, 3.0e-8];
        let omega = 1.9824302f64;
        let prefactor = 1.0 - 0.5 * omega;
        let mut mass = 0.0;
        let mut momentum = [0.0; 3];
        for i in 0..19 {
            let c = [ex[i] as f64, ey[i] as f64, ez[i] as f64];
            let eu = c[0] * u[0] + c[1] * u[1] + c[2] * u[2];
            let ef = c[0] * force[0] + c[1] * force[1] + c[2] * force[2];
            let uf = u[0] * force[0] + u[1] * force[1] + u[2] * force[2];
            let source = w[i] * prefactor * (3.0 * (ef - uf) + 9.0 * eu * ef);
            mass += source;
            for a in 0..3 {
                momentum[a] += c[a] * source;
            }
        }
        assert!(mass.abs() < 1e-21, "mass moment = {mass:e}");
        for a in 0..3 {
            let expected = prefactor * force[a];
            assert!((momentum[a] - expected).abs() < 1e-21);
        }
    }
}
