use super::Collision3D;

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
    let c = 0.57735026919;
    u = clamp(u, vec3<f32>(-c), vec3<f32>(c));

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;
    let tau_0 = 1.0 / OMEGA;
    let C_smag = 0.16;
    
    var Q_xx = 0.0; var Q_yy = 0.0; var Q_zz = 0.0;
    var Q_xy = 0.0; var Q_yz = 0.0; var Q_zx = 0.0;

    for (var i: u32 = 1u; i < Q; i += 1u) {
        let e_vec = vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i]));
        let eu = dot(e_vec, u);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        let fneq = f_local[i] - feq;
        
        Q_xx += e_vec.x * e_vec.x * fneq;
        Q_yy += e_vec.y * e_vec.y * fneq;
        Q_zz += e_vec.z * e_vec.z * fneq;
        Q_xy += e_vec.x * e_vec.y * fneq;
        Q_yz += e_vec.y * e_vec.z * fneq;
        Q_zx += e_vec.z * e_vec.x * fneq;
    }

    let Pi_mag = sqrt(Q_xx*Q_xx + Q_yy*Q_yy + Q_zz*Q_zz + 2.0 * (Q_xy*Q_xy + Q_yz*Q_yz + Q_zx*Q_zx));
    let tau_total = 0.5 * (tau_0 + sqrt(max(tau_0 * tau_0 + 18.0 * C_smag * C_smag * Pi_mag / rho, 0.0)));
    let OMEGA_EFF = 1.0 / tau_total;
    let omega_factor = 1.0 - 0.5 * OMEGA_EFF;

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let e_vec = vec3<f32>(f32(EX[i]), f32(EY[i]), f32(EZ[i]));
        let eu = dot(e_vec, u);
        let eF = dot(e_vec, F);
        let uF = dot(u, F);

        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        let S_i = WEIGHTS[i] * omega_factor * (3.0 * (eF - uF) + 9.0 * eu * eF);

        f_local[i] = f_local[i] - OMEGA_EFF * (f_local[i] - feq) + S_i;
    }
"#
    }
}
