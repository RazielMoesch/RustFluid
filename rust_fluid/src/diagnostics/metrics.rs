//! Reductions over downloaded `[u_x, u_y, u_z, rho]` values.

/// Aggregate density, velocity, and validity measures for a field.
pub struct MacroMetrics {
    pub rho_min: f32,
    pub rho_max: f32,
    pub rho_mean: f32,
    pub max_u: f32,
    pub rms_u: f32,
    pub mass: f32,
    pub nan_count: usize,
    pub inf_count: usize,
}

impl MacroMetrics {
    pub fn compute(data: &[[f32; 4]]) -> Self {
        let mut rho_min = f32::MAX;
        let mut rho_max = f32::MIN;
        let mut rho_sum = 0.0;
        let mut max_u: f32 = 0.0;
        let mut u2_sum = 0.0;
        let mut nan_count = 0;
        let mut inf_count = 0;

        for cell in data {
            let ux = cell[0];
            let uy = cell[1];
            let uz = cell[2];
            let rho = cell[3];

            if rho.is_nan() || ux.is_nan() || uy.is_nan() || uz.is_nan() {
                nan_count += 1;
                continue;
            }
            if rho.is_infinite() || ux.is_infinite() || uy.is_infinite() || uz.is_infinite() {
                inf_count += 1;
                continue;
            }

            rho_min = rho_min.min(rho);
            rho_max = rho_max.max(rho);
            rho_sum += rho;

            let u2 = ux * ux + uy * uy + uz * uz;
            let u = u2.sqrt();
            max_u = max_u.max(u);
            u2_sum += u2;
        }

        let valid_n = (data.len() - nan_count - inf_count) as f32;

        Self {
            rho_min,
            rho_max,
            rho_mean: if valid_n > 0.0 {
                rho_sum / valid_n
            } else {
                f32::NAN
            },
            max_u,
            rms_u: if valid_n > 0.0 {
                (u2_sum / valid_n).sqrt()
            } else {
                f32::NAN
            },
            mass: rho_sum,
            nan_count,
            inf_count,
        }
    }
}
