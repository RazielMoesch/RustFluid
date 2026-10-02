//! Two-relaxation-time collision for opposite D3Q19 populations.

use super::Collision3D;

/// Separately relaxes symmetric and antisymmetric population components.
pub struct Trt;

impl Trt {
    pub fn new() -> Self {
        Self
    }
}

impl Collision3D for Trt {
    fn name(&self) -> &'static str {
        "TRT"
    }

    fn wgsl(&self) -> &'static str {
        r#"
    let F = vec3<f32>(FORCE_X, FORCE_Y, FORCE_Z);
    u = (u + 0.5 * F) / rho;
    let c = 0.57735026919;
    u = clamp(u, vec3<f32>(-c), vec3<f32>(c));

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;
    let OMEGA_EFF = OMEGA;
    let omega_factor = 1.0 - 0.5 * OMEGA;

    // Two-Relaxation-Time (TRT) approximation
    let s_plus = OMEGA_EFF;
    let s_minus = 8.0 * (2.0 - s_plus) / (8.0 - s_plus);
    
    var f_pre = f_local;

    // This moment pairing is specifically ordered for D3Q19.
    // Actually, we can use `Q` since it's defined as constant.
    for (var i: u32 = 0u; i < Q; i += 1u) {
        let e_vec = vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i]));
        let eu = dot(e_vec, u);
        let eF = dot(e_vec, F);
        let uF = dot(u, F);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        let S_i = WEIGHTS[i] * omega_factor * (3.0 * (eF - uF) + 9.0 * eu * eF);
        
        let opp = OPP[i];
        let e_vec_opp = vec3<f32>(f32(EX[opp]), f32(EY[opp]), f32(EZ[opp]));
        let eu_opp = dot(e_vec_opp, u);
        let feq_opp = WEIGHTS[opp] * rho * (1.0 + 3.0 * eu_opp + 4.5 * (eu_opp * eu_opp) - u_sq_term);
        
        let f_plus = 0.5 * (f_pre[i] + f_pre[opp]);
        let f_minus = 0.5 * (f_pre[i] - f_pre[opp]);
        
        let feq_plus = 0.5 * (feq + feq_opp);
        let feq_minus = 0.5 * (feq - feq_opp);
        
        f_local[i] = f_local[i] - s_plus * (f_plus - feq_plus) - s_minus * (f_minus - feq_minus) + S_i;
    }
"#
    }
}
