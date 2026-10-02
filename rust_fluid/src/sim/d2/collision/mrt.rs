//! Multiple-relaxation-time collision for the D2Q9 moment basis.

use super::Collision2D;

/// Relaxes D2Q9 moments at individually selected rates.
pub struct Mrt;

impl Mrt {
    pub fn new() -> Self {
        Self
    }
}

impl Collision2D for Mrt {
    fn name(&self) -> &'static str {
        "MRT"
    }

    fn wgsl(&self) -> &'static str {
        r#"
    if (rho > 0.0) {
        u = u / rho;
    }

    let ux = u.x;
    let uy = u.y;
    let ux2 = ux * ux;
    let uy2 = uy * uy;
    let u2 = ux2 + uy2;

    let f0 = f_local[0]; let f1 = f_local[1]; let f2 = f_local[2];
    let f3 = f_local[3]; let f4 = f_local[4]; let f5 = f_local[5];
    let f6 = f_local[6]; let f7 = f_local[7]; let f8 = f_local[8];

    // 1. Compute Moments (m = M * f)
    var m: array<f32, 9>;
    m[0] = rho;
    m[1] = -4.0*f0 - (f1+f2+f3+f4) + 2.0*(f5+f6+f7+f8);
    m[2] = 4.0*f0 - 2.0*(f1+f2+f3+f4) + (f5+f6+f7+f8);
    m[3] = f1 - f3 + f5 - f6 - f7 + f8;
    m[4] = -2.0*(f1 - f3) + f5 - f6 - f7 + f8;
    m[5] = f2 - f4 + f5 + f6 - f7 - f8;
    m[6] = -2.0*(f2 - f4) + f5 + f6 - f7 - f8;
    m[7] = f1 - f2 + f3 - f4;
    m[8] = f5 - f6 + f7 - f8;

    // 2. Compute Equilibrium Moments (meq)
    var meq: array<f32, 9>;
    meq[0] = rho;
    meq[1] = rho * (-2.0 + 3.0 * u2);
    meq[2] = rho * (1.0 - 3.0 * u2);
    meq[3] = rho * ux;
    meq[4] = -rho * ux;
    meq[5] = rho * uy;
    meq[6] = -rho * uy;
    meq[7] = rho * (ux2 - uy2);
    meq[8] = rho * ux * uy;

    let m7_neq = m[7] - meq[7];
    let m8_neq = m[8] - meq[8];
    
    let OMEGA_EFF = OMEGA;

    // 3. Relax Moments
    m[1] = m[1] - 1.63 * (m[1] - meq[1]); // e
    m[2] = m[2] - 1.14 * (m[2] - meq[2]); // epsilon 
    m[4] = m[4] - 1.92 * (m[4] - meq[4]); // q_x 
    m[6] = m[6] - 1.92 * (m[6] - meq[6]); // q_y 
    m[7] = m[7] - OMEGA_EFF * m7_neq;     // p_xx
    m[8] = m[8] - OMEGA_EFF * m8_neq;     // p_xy

    // 4. Inverse Transformation -> Write to f_local
    let m0 = m[0]; let m1 = m[1]; let m2 = m[2]; let m3 = m[3]; 
    let m4 = m[4]; let m5 = m[5]; let m6 = m[6]; let m7 = m[7]; let m8 = m[8];

    f_local[0] = (1.0/9.0)  * (m0 - m1 + m2);
    f_local[1] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 + 6.0*m3 - 6.0*m4 + 9.0*m7);
    f_local[2] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 + 6.0*m5 - 6.0*m6 - 9.0*m7);
    f_local[3] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 - 6.0*m3 + 6.0*m4 + 9.0*m7);
    f_local[4] = (1.0/36.0) * (4.0*m0 - m1 - 2.0*m2 - 6.0*m5 + 6.0*m6 - 9.0*m7);
    f_local[5] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 + 6.0*m3 + 3.0*m4 + 6.0*m5 + 3.0*m6 + 9.0*m8);
    f_local[6] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 - 6.0*m3 - 3.0*m4 + 6.0*m5 + 3.0*m6 - 9.0*m8);
    f_local[7] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 - 6.0*m3 - 3.0*m4 - 6.0*m5 - 3.0*m6 + 9.0*m8);
    f_local[8] = (1.0/36.0) * (4.0*m0 + 2.0*m1 + m2 + 6.0*m3 + 3.0*m4 - 6.0*m5 - 3.0*m6 - 9.0*m8);
"#
    }
}
