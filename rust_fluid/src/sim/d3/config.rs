//! User-controlled constants used to specialize a D3Q19 solver.

/// Selects the analytical field used to initialize populations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitType {
    /// Constant density and velocity in every fluid cell.
    Uniform,
    /// Periodic Taylor-Green vortex used by decay diagnostics.
    TaylorGreen,
}

/// Grid, physics, boundary, sponge, and dispatch settings for `Lbm3D`.
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
    pub periodic_y: bool,
    pub periodic_z: bool,
    pub pure_fluid: bool,
    pub num_boundary_configs: u32,
    /// Number of cells at the high-X end of the domain that form a sponge
    /// (damping) layer. `0` disables it. The layer blends the populations
    /// toward the far-field equilibrium so residual wake vorticity is absorbed
    /// before it can pile up as spurious vorticity at the outlet plane.
    /// Requires the boundary-aware step shader, i.e. `pure_fluid == false`.
    pub sponge_len: u32,
    /// Maximum per-step blend fraction applied at the very last cell of the
    /// sponge. The ramp is quadratic, so it is zero at the sponge entrance.
    pub sponge_strength: f32,
    /// Index into `boundary_configs` holding the sponge target
    /// `(velocity, density)`. Reuses the far-field inlet configuration.
    pub sponge_cfg: u32,
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
            periodic_y: false,
            periodic_z: false,
            pure_fluid: false,
            num_boundary_configs: 1,
            sponge_len: 0,
            sponge_strength: 0.0,
            sponge_cfg: 0,
        }
    }
}
