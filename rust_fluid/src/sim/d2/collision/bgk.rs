use super::Collision2D;

pub struct Bgk;

impl Bgk {
    pub fn new() -> Self {
        Self
    }
}

impl Collision2D for Bgk {
    fn name(&self) -> &'static str {
        "BGK"
    }

    fn wgsl(&self) -> &'static str {
        r#"
    if (rho > 0.0) {
        u = u / rho;
    }

    let u_sq = dot(u, u);
    let u_sq_term = 1.5 * u_sq;

    for (var i: u32 = 0u; i < Q; i += 1u) {
        let eu = (f32(EX[i]) * u.x) + (f32(EY[i]) * u.y);
        let feq = WEIGHTS[i] * rho * (1.0 + 3.0 * eu + 4.5 * (eu * eu) - u_sq_term);
        f_local[i] = f_local[i] - OMEGA * (f_local[i] - feq);
    }
"#
    }
}
