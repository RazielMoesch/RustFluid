//! User-controlled constants used to specialize a D2Q9 solver.

/// Grid, initialization, relaxation, and dispatch settings for `Lbm2D`.
#[derive(Debug, Clone, Copy)]
pub struct SimulationConfig2D {
    pub nx: u32,
    pub ny: u32,
    pub rho_init: f32,
    pub u_x_init: f32,
    pub u_y_init: f32,
    pub wgs_x: u32,
    pub wgs_y: u32,
    pub omega: f32,
    pub num_boundary_configs: u32,
}

impl Default for SimulationConfig2D {
    fn default() -> Self {
        Self {
            nx: 100,
            ny: 100,
            rho_init: 1.0,
            u_x_init: 0.0,
            u_y_init: 0.0,
            wgs_x: 8,
            wgs_y: 8,
            omega: 1.0,
            num_boundary_configs: 1,
        }
    }
}
