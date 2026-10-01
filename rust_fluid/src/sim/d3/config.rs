#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitType {
    Uniform,
    TaylorGreen,
}

#[derive(Debug, Clone, Copy)]
pub struct SimulationConfig3D {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub init_type: InitType,
    pub rho_init: f32,
    pub u_x_init: f32,
    pub u_y_init: f32,
    pub u_z_init: f32,
    pub wgs_x: u32,
    pub wgs_y: u32,
    pub wgs_z: u32,
    pub omega: f32,
    pub force_x: f32,
    pub force_y: f32,
    pub force_z: f32,
    pub periodic_x: bool,
    pub pure_fluid: bool,
    pub num_boundary_configs: u32,
}

impl Default for SimulationConfig3D {
    fn default() -> Self {
        Self {
            nx: 100,
            ny: 100,
            nz: 100,
            init_type: InitType::Uniform,
            rho_init: 1.0,
            u_x_init: 0.0,
            u_y_init: 0.0,
            u_z_init: 0.0,
            wgs_x: 8,
            wgs_y: 8,
            wgs_z: 1,
            omega: 1.0,
            force_x: 0.0,
            force_y: 0.0,
            force_z: 0.0,
            periodic_x: false,
            pure_fluid: false,
            num_boundary_configs: 1,
        }
    }
}
